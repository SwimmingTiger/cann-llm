"""`--large-mem`：用 lldb 自动给推理进程打「大模型补丁」。

## 为什么需要它

鸿蒙侧引擎在**两个地方**用 securec 的安全函数拷贝权重，而 `memcpy_s` 一族的
`destMax` 有硬上限 `0x7fffffff`（2 GiB − 1，见 `docs/maintainer-notes.md` §32/§32.8）：

1. `libhiai_adapter.so` 通过 **动态导入** 的 `memcpy_s`（`libsec_shared.z.so`）；
2. `libhiai_ir.so` 里 **静态链入** 的那份 securec `memcpy_s`（符号插桩够不着 ✗）。

适配层把**整段模型的权重缓冲**（≈ 3.94 × `.ms` 文件大小，权重按 fp32 展开）当
`destMax` 传进去；一旦超过 2 GiB 就直接返回 `ERANGE` ⇒ Build 失败（`Set Data Fail.`）。
于是**不补丁**时每段 `.ms` 只能到 ≈545 MB（≈520 MiB）。

把这几条检查在**运行时改成 `nop`**（或反极性的那条改成无条件跳转）之后，
上限上移到模型头里的 **u32 长度字段**（§38）：每段 `.ms` 可到 ≈1.09 GB（≈1.02 GiB）。

## 实现方式

本模块只负责**宿主侧**：补丁表 + 组装 argv。真正的动作在
- `large_mem_lldb.py` —— 由 lldb 的 python 载入，安装断点并打补丁；
- `scripts/large_mem_run.sh` —— 编排 `gdbserver`（起被调试进程）+ `lldb`（批处理接入）。

★ 为什么必须走 `huawei-debug-lldb-server`：本机 `lldb -- <bin>` 直接拉起进程会报
`error: 'A' packet returned an error: 8`（平台限制），只有这份 gdbserver 能调试 ✓

★ 定位用**文件内偏移**而不是绝对地址：ASLR 每次不同，偏移固定（受 lib 版本约束）。
补丁前会先读回 4 字节与期望值比对，**不一致就跳过并告警**，避免改错地方。
"""
from __future__ import annotations

import os
import shutil
from typing import List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "LargeMemPatch", "PATCHES", "LLDB_SCRIPT_NAME", "DRIVER_NAME",
    "RENDEZVOUS_ENV", "WAIT_ENV", "PRELOAD_LIBS",
    "preload_targets", "rendezvous",
    "DEFAULT_LARGE_MEM_PORT", "LIMIT_STOCK", "LIMIT_PATCHED",
    "large_mem_script_path", "large_mem_driver_path", "strip_large_mem",
    "build_large_mem_argv", "describe_patches",
]


class LargeMemPatch(NamedTuple):
    """一处「2 GiB 上限」检查的补丁。"""

    module: str        #: .so 文件名（按名字在已加载模块里找）
    offset: int        #: 文件内偏移（= vaddr，与模块基址相加即运行期地址）
    expect: bytes      #: 期望读到的原始指令字节（补丁前校验，防止改错）
    patch: bytes       #: 要写入的字节
    what: str          #: 这处检查属于哪个函数
    note: str = ""     #: 额外说明


#: ★补丁表★ —— 偏移与字节来自本机实测（`docs/maintainer-notes.md` §33/§35/§38）。
#: 这些值与**系统库版本**绑定：换版本后偏移可能变 ✗
#: 好在补丁前会校验 `expect`，对不上会明确报"跳过"而不是把别处改坏 ✓
PATCHES: Tuple[LargeMemPatch, ...] = (
    LargeMemPatch(
        "libhiai_ir.so", 0xC16EC,
        b"\xe8\x02\x00\xb5", b"\x1f\x20\x03\xd5",
        "securec memcpy_s（静态链入）",
        "cbnz x8（x8 = destMax>>31）：destMax ≥ 2 GiB 即拒，nop 即可 ✓ "
        "★符号表把这段归在 AI_Log_Print 名下（本地符号），其实是一份 memcpy_s 克隆★",
    ),
    LargeMemPatch(
        "libsec_shared.z.so", 0x3E50,
        b"\x09\x03\x00\xb5", b"\x1f\x20\x03\xd5",
        "securec memcpy_s（导出）",
        "适配层搬权重走的就是它 ✓（0x3E4C: lsr x9,x8,#31 → 0x3E50: cbnz x9）",
    ),
    LargeMemPatch(
        "libsec_shared.z.so", 0x51DC,
        b"\x89\x01\x00\xb5", b"\x1f\x20\x03\xd5",
        "securec memset_s",
        "同一族的上限门（x9 = destMax>>31）。★编码与 memcpy_s 不同★（立即数不一样），"
        "所以必须按实测字节校验，不能照抄 ✓",
    ),
    LargeMemPatch(
        "libsec_shared.z.so", 0x5138,
        b"\xc2\x00\x00\x54", b"\x06\x00\x00\x14",
        "securec memmove_s",
        "★极性相反★：它是 b.hs 跳向【正常】路径（0x5150），nop 反而会掉进 ERANGE ✗ "
        "⇒ 必须改成无条件跳转 b 0x5150 ✓",
    ),
    # ---- 下面两处：hiai 后端（引擎自带 securec 克隆）★2026-10-06 新增★
    # ★为什么必须加★：hiai 后端只加载 libhiai_llm_engine.so，**不加载** libhiai_ir.so
    #   ⇒ 上面第一条对它是死代码 ✗；而引擎自己静态链入了一份 securec，
    #   权力缓冲（SubGraph_0.weight 4.4 GB）就是走它拷的 ⇒ 它才是这条路径的真天花板 ✓
    # 证据：/proc/<pid>/maps 339 个库；IDA Pro 9.3 + hexarm 反编译；与设备库 sha256 一致 ✓
    LargeMemPatch(
        "libhiai_llm_engine.so", 0x3057F0,
        b"\xe8\x02\x00\xb5", b"\x1f\x20\x03\xd5",
        "securec memcpy_s（静态链入引擎 —— hiai 后端真正在用的拷贝）",
        "★IDA 反编译确认★ 函数 sub_3057E8(dest, destMax, src, count)："
        "0x3057EC `lsr x8, x1, #31` → 0x3057F0 `cbnz x8`；伪代码 "
        "`if (n - 0x80000000 >= 0xFFFFFFFF80000001) … return 34;`（34 = ERANGE ✓）"
        "⇒ 与 libsec_shared 的 memcpy_s 同源（门前后 64 字节里 50 字节逐字节相同，仅寄存器分配不同）✓",
    ),
    LargeMemPatch(
        "libhiai_llm_engine.so", 0x305910,
        b"\x69\x01\x00\xb5", b"\x1f\x20\x03\xd5",
        "securec memset_s（静态链入引擎）",
        "同源确认：sub_3058F4(s, n, c, destMax) 的伪代码 "
        "`if (!s || n >> 31 || …) { … else return 34; }`；"
        "0x30590C `lsr x9, x8, #31` → 0x305910 `cbnz x9` ✓",
    ),
)

#: lldb 侧脚本名（与 `large_mem_lldb.py` 同目录）
LLDB_SCRIPT_NAME = "large_mem_lldb.py"

#: 编排脚本名（在 `scripts/` 下）
DRIVER_NAME = "large_mem_run.sh"

#: 默认端口（与 `--lldb` 的 5091 错开，避免撞车）
DEFAULT_LARGE_MEM_PORT = 5092

#: ★"报到—放行"握手★：值是**放行文件**的路径
#:   · runner 在第一次 build 前，如果这个变量有值，就停下来等这个文件出现；
#:   · lldb 侧脚本打完补丁就把它创建出来 ⇒ 取消等待 ✓
#:   （一个变量给两边用：驱动 export，app 与 lldb 都继承 ✓）
RENDEZVOUS_ENV = "CANN_LLM_LARGE_MEM_RENDEZVOUS"

#: 等待上限（秒）—— 调试器要是没来，程序自己走，绝不永久卡住 ✗
WAIT_ENV = "CANN_LLM_LARGE_MEM_WAIT"

#: 报到前先 dlopen 的库
#: ★为什么★：这些库在补丁时刻可能还没加载 ——
#:   · `libhiai_ir.so`：nnrt 路径下**第一次 build 中途**才加载 ⇒ 那一处赶不上 ✗
#:   · `libhiai_llm_engine.so`：hiai 后端在 attach 之后才 dlopen ⇒ 引擎里那两处赶不上 ✗
#:   先拉起来 ⇒ 补丁一次到位 ✓（§40.7 与 2026-10-06 hiai 实测的遗留短板）
PRELOAD_LIBS = ("libhiai_ir.so", "libsec_shared.z.so", "libhiai_llm_engine.so")

_rendezvous_done = False


def preload_targets(verbose: bool = True) -> List[str]:
    """把补丁目标的库先 dlopen 起来；返回成功的库名。失败不报错（尽力而为）。"""
    import ctypes
    got = []
    for name in PRELOAD_LIBS:
        try:
            ctypes.CDLL(name)
            got.append(name)
        except OSError:
            pass
    if verbose and got:
        print("[large-mem] 已预加载 %s（让 4 处补丁一次到位）" % "、".join(got), flush=True)
    return got


def rendezvous(log=None) -> bool:
    """★第一次 build 之前"报到—等放行"★（`--large-mem` 专用；幂等，可重复调用）

    没有设 :data:`RENDEZVOUS_ENV` 时**立刻返回**（普通运行零开销 ✓）。

    有值时的顺序（顺序很重要）：

    1. 先 :func:`preload_targets` —— 让 ``libhiai_ir.so`` 这时就在 ✓
    2. 等 lldb 侧脚本创建"放行文件"（它是在 `command script import` 时打完补丁后
       创建的 ✓）—— 最多等 :data:`WAIT_ENV` 秒，超时就自己走（绝不永久卡住 ✗）

    返回 ``True`` 表示这次确实走了报到流程。
    """
    global _rendezvous_done
    release = os.environ.get(RENDEZVOUS_ENV)
    if not release or _rendezvous_done:
        return False
    _rendezvous_done = True
    say = log or (lambda msg: print(msg, flush=True))

    if os.path.exists(release):
        # 调试器已经先跑过了（补丁就位）⇒ 不必等，直接走 ✓
        say("[large-mem] 调试器已就位（放行标记已存在），直接继续 ✓")
        preload_targets(verbose=False)
        return True

    preload_targets()
    wait = float(os.environ.get(WAIT_ENV) or 30)
    import time
    t0 = time.time()
    say("[large-mem] 等调试器打补丁后放行（最多 %.0f 秒）…" % wait)
    while time.time() - t0 < wait:
        if os.path.exists(release):
            say("[large-mem] 已放行（补丁就位）✓ 用时 %.1f 秒" % (time.time() - t0))
            return True
        time.sleep(0.05)
    say("[large-mem] ⚠ 等调试器超时（%.0f 秒）—— 继续运行（这次可能没打上补丁）" % wait)
    return True

#: 两种模式下的单段 `.ms` 上限（§38 实测 + 推算）
LIMIT_STOCK = "≈545 MB（≈520 MiB）"
LIMIT_PATCHED = "≈1.09 GB（≈1.02 GiB）"


def large_mem_script_path() -> str:
    """lldb 侧脚本的绝对路径（本模块同目录）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), LLDB_SCRIPT_NAME)


def large_mem_driver_path() -> str:
    """编排脚本的绝对路径（仓库 `scripts/` 下，由本模块位置反推）。"""
    here = os.path.dirname(os.path.abspath(__file__))     # <root>/src/cann_llm
    root = os.path.dirname(os.path.dirname(here))         # <root>
    return os.path.join(root, "scripts", DRIVER_NAME)


def strip_large_mem(argv: Sequence[str]) -> Tuple[List[str], bool]:
    """把 ``--large-mem`` 从参数表里摘出来（``--large-mem=<x>`` 也一并摘）。"""
    out, found = [], False
    for a in argv:
        if a == "--large-mem" or a.startswith("--large-mem="):
            found = True
            continue
        out.append(a)
    return out, found


def describe_patches() -> List[str]:
    """给用户看的补丁清单（启动时打印）。"""
    lines = []
    for p in PATCHES:
        lines.append("  · %-20s +0x%05X  %-28s %s"
                     % (p.module, p.offset, p.what, p.note))
    return lines


def build_large_mem_argv(py: str, module: str, args: Sequence[str],
                         env: "Optional[dict]" = None
                         ) -> Tuple[Optional[List[str]], List[str], str]:
    """构造「自动打补丁地运行 ``py -m module args…``」的 argv。

    实际交给编排脚本的是 ``(python, 模块名, 模块参数…)``：脚本要用 ``runpy``
    在**同一个进程**里把模块跑起来（先自停等调试器，见 ``large_mem_run.sh``）✓

    返回 ``(argv, 提示行, 错误信息)``；失败时 ``argv`` 为 ``None``。
    """
    env = os.environ if env is None else env

    # gdbserver 用 execve 直接拉起进程、不解析 shebang ⇒ 解释器必须是真 ELF
    from .lldb_launch import find_gdbserver, is_real_executable
    if not is_real_executable(py):
        return None, [], (
            "--large-mem 需要真正的可执行文件，但 %s 不是 ELF（看起来是脚本包装器）。\n"
            "      gdbserver 用 execve 直接拉起进程、不解析 shebang ✓\n"
            "      请改用同目录下带版本号的那个解释器" % py)

    driver = large_mem_driver_path()
    if not os.path.exists(driver):
        return None, [], ("找不到 %s（--large-mem 的编排脚本）" % driver)

    script = large_mem_script_path()
    if not os.path.exists(script):
        return None, [], ("找不到 %s（--large-mem 的 lldb 脚本）" % script)

    gdbserver = find_gdbserver(env)
    if not gdbserver:
        return None, [], (
            "找不到 huawei-debug-lldb-server —— --large-mem 必须经它来改内存\n"
            "      （本机 `lldb -- <bin>` 直接拉起进程会报 'A' packet returned an error: 8）\n"
            "      可用 CANN_LLM_LLDB_SERVER=/path/to/huawei-debug-lldb-server 指定")

    lldb = (env.get("CANN_LLM_LLDB") or shutil.which("lldb")
            or env.get("LLDB") or "")
    if not lldb or not os.path.exists(lldb):
        return None, [], (
            "找不到 lldb —— 可用 CANN_LLM_LLDB=/path/to/lldb 指定\n"
            "      （发布包里它在 bin/lldb，并需要 lib/liblldb.so 在 LD_LIBRARY_PATH 里）")

    port = int(env.get("CANN_LLM_LLDB_PORT") or DEFAULT_LARGE_MEM_PORT)
    argv = ["/bin/sh", driver, py, module] + list(args)

    hints = [
        "── --large-mem：自动给推理进程打「大模型补丁」（经 ptrace 改内存）──",
        "  单段 .ms 上限：不补丁 %s  →  补丁后 %s" % (LIMIT_STOCK, LIMIT_PATCHED),
        "  将改这几处 2 GiB 上限检查（先校验原字节，对不上就跳过并告警）：",
    ] + describe_patches() + [
        "  接法：程序先自停 → %s 附着 → lldb 打补丁（程序的终端自始至终是它自己的）"
        % os.path.basename(gdbserver),
        "        ★输入能进得去★：不打补丁时 gdbserver 会给被调试进程另开一个 pty，"
        "键盘输入永远进不去 ✗；",
        "        这里改成 attach ⇒ 程序用【自己的终端】，交互式对话照常可用 ✓",
        "  端口 %d；等补丁的超时可用 CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT 调（默认 60 秒）" % port,
        "  ★这是【验证手段】不是交付方案：改的是系统库的进程内副本（内存），磁盘不动 ✓",
    ]
    return argv, hints, ""
