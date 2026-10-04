"""一键脚本的**实际逻辑**都在这里；`scripts/*.sh` 只负责找到解释器再转发过来。

为什么这样分
------------
外壳脚本只需要做一件事：**发现一个与引擎 libc 兼容的 python，然后把参数原样转过来**。
那件事在 sh 里只有十来行、纯 POSIX，不会再碰数组 / `[[ ]]` / `shopt` 这些
bash 专有语法（目标环境是只有 python 的 HarmonyOS，`/bin/sh` 是 mksh）。

其余逻辑（参数解析、引擎库选择、模型目录探测、预检提示、后台管理）放在 Python 里：
有 argparse、有异常、能单测。

用法（由外壳转过来）::

    python -m cann_llm.launcher chat   -d <模型目录> [其它 chat 参数]
    python -m cann_llm.launcher server -d <模型目录> [其它 server 参数]
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

__all__ = ["main"]

#: 引擎库（按后端分开）
LIBS = {
    "cann": ("CANN_LLM_LIB", "/system/lib64/ndk/libcann_llm_engine.so", "cann NDK 库"),
    # ★ nnrt 后端【不用】系统里的 hiai 引擎 ✓ ——
    #   它自己 ctypes 直调 MindSpore Lite 的 NDK，并选 OH_AI_DEVICETYPE_NNRT(=60) ✓。
    #   以前这张表里没有 nnrt ⇒ LIBS.get("nnrt", LIBS["hiai"]) 回落到 hiai ✗，
    #   于是 -b nnrt 会误导性地打印 "✓ hiai 引擎: /system/lib64/libhiai_llm_engine.so" ✗
    "nnrt": ("CANN_LLM_NNRT_LIB", "/system/lib64/ndk/libmindspore_lite_ndk.so",
             "nnrt 引擎 (MindSpore Lite NDK)"),
    "hiai": ("CANN_LLM_HIAI_LIB", "/system/lib64/libhiai_llm_engine.so", "hiai 引擎"),
}

#: 找到候选解释器后，用它自检"与引擎 libc 是否兼容"。
#: glibc 构建的 python 靠 libmusl_compat 垫片跑，加载 musl 版引擎会段错误。
_LIBC_PROBE = (
    "import sys\n"
    "try:\n"
    "    maps = open('/proc/self/maps').read()\n"
    "except OSError:\n"
    "    raise SystemExit(0)\n"
    "raise SystemExit(1 if 'libmusl_compat' in maps else 0)\n"
)


def _say(mark: str, text: str, color: str = "") -> None:
    c = f"\033[{color}m" if color else ""
    r = "\033[0m" if color else ""
    print(f"{c}{mark}{r} {text}")


def ok(text: str) -> None:
    _say("✓", text, "32")


def info(text: str) -> None:
    _say("›", text, "36")


def warn(text: str) -> None:
    _say("!", text, "33")


def die(text: str, code: int = 1) -> "None":
    _say("错误", text, "31")
    raise SystemExit(code)


# ---------------------------------------------------------------- 解释器（外壳用同一个逻辑做，这里只做查错）

def current_python() -> str:
    """返回要用来重跑/启动的解释器路径。

    ★ 直接用 ``sys.executable``：它一定是**真二进制**。用 `#!/bin/sh` 包装器启动时，
      Python 自己会把它解析成实际的可执行文件（实测：`python3` 包装器 →
      `…/bin/python3.12`）。也就是说外壳挑好的那个解释器，这里本来就知道，
      不必再靠环境变量传一遍。

      这比读 ``CANN_LLM_RESOLVED_PYTHON`` 可靠：外壳传进来的可能就是那个包装器，
      而调试器（gdbserver）用 execve 直接拉起进程、**不解析 shebang**，
      传包装器进去会得到 "execve failed: Operation not permitted"。

    ``CANN_LLM_RESOLVED_PYTHON`` 保留为兜底（极少数取不到 ``sys.executable`` 的场合）。
    """
    return sys.executable or os.environ.get("CANN_LLM_RESOLVED_PYTHON") or "python3"


def libc_compatible(py: str) -> bool:
    """该解释器能否安全加载 musl 版引擎（glibc 构建的会段错误）。"""
    try:
        return subprocess.run([py, "-c", _LIBC_PROBE],
                              capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return True          # 判不了就别拦


# ---------------------------------------------------------------- 引擎库 / 模型目录

def pick_engine_lib(backend: str) -> "tuple[str, str, str]":
    """按后端选引擎库：返回 ``(环境变量名, 路径, 人读名称)``。"""
    var, default, kind = LIBS.get(backend, LIBS["hiai"])
    return var, os.environ.get(var) or default, kind


def detect_backend(model_dir: Optional[str]) -> Optional[str]:
    """★ 按模型目录的内容判断该用哪个后端（用户没显式给 -b 时）★

    判据来自各后端真正需要的东西，不是猜：
      · nnrt —— 本仓库自装配的布局：有段图目录（seg*/mseg*/dec*/pre*）
                且带 weights/ 或 graphP/；它只吃这些，不吃 api_config.json
      · hiai —— 官方打包布局：有 api_config.json
      · cann —— 官方 OMC 包解压：有 executor.json / context.json
    都不像就返回 None，由调用方回落到默认值 ✓
    """
    if not model_dir:
        return None
    try:
        names = set(os.listdir(model_dir))
    except OSError:
        return None
    if any(n.startswith(("seg", "mseg", "dec", "pre")) for n in names) and (
            "weights" in names or "graphP" in names):
        return "nnrt"
    if "api_config.json" in names:
        return "hiai"
    if "executor.json" in names or "context.json" in names:
        return "cann"
    return None


def find_model_dir(root: str, given: Optional[str]) -> Optional[str]:
    """模型目录：显式给的优先；否则在 ``models/*/`` 里找一个含 executor.json 的。"""
    if given:
        d = os.path.abspath(os.path.expanduser(given))
        if not os.path.isdir(d):
            die(f"模型目录不存在: {given}")
        return d
    import glob
    for pat in (os.path.join(root, "models", "*"),
                os.path.join(root, "..", "models", "*")):
        for d in sorted(glob.glob(pat)):
            if os.path.isfile(os.path.join(d, "executor.json")):
                return os.path.abspath(d)
    return None


def model_info_line(model_dir: str) -> Optional[int]:
    """读 kv_cache_max_len（复用 cann_llm.modelcfg，不猜默认值）。"""
    try:
        from .modelcfg import read_kv_cache_max_len
        val, _src = read_kv_cache_max_len(model_dir)
        return val
    except Exception:            # noqa: BLE001
        return None


# ---------------------------------------------------------------- 参数解析（只挑它认识的，其余原样透传）

_TAKES_VALUE = {"-d", "--model-dir", "-b", "--backend"}


def split_known(argv: "List[str]") -> "tuple[Optional[str], str, List[str]]":
    """把 ``-d`` / ``-b`` 摘出来（引擎库与预检要用），其余**原样保留**给下游。"""
    model_dir: Optional[str] = None
    # ★ 不写死默认后端：None = 交给 detect_backend() 按模型目录判断 ✓
    backend = os.environ.get("CANN_LLM_BACKEND")
    rest: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-d", "--model-dir") and i + 1 < len(argv):
            model_dir = argv[i + 1]
            rest += [a, argv[i + 1]]
            i += 2
            continue
        if a in ("-b", "--backend") and i + 1 < len(argv):
            backend = argv[i + 1]
            rest += [a, argv[i + 1]]
            i += 2
            continue
        if a.startswith("--model-dir="):
            model_dir = a.split("=", 1)[1]
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
        rest.append(a)
        i += 1
    return model_dir, backend, rest


# ---------------------------------------------------------------- 两条入口

#: ★引擎库的依赖搜索路径★ ——
#  libmindspore_lite_ndk.so 的 DT_NEEDED 里有 libmindspore-lite.so /
#  libmindspore-lite-train.so / libhilog.so / libsec_shared.z.so，
#  它们【不在】/system/lib64/ndk ✓ 而在 ★/system/lib64/platformsdk★ ✓
#  ⇒ 搜索路径少一个目录时，musl 的 dlopen 会在解析失败的分支上★段错误★ ✗
#    （实测：只有 /system/lib64/ndk 时，python-3.14 一 CDLL 就 core dump ✗；
#      补上 platformsdk 立刻正常 ✓ —— 这正是"3.14 兼容性问题"的真身 ✓）
ENGINE_LIB_DIRS = ("/system/lib64/ndk", "/system/lib64/platformsdk")


def ensure_engine_lib_path(env: "dict") -> None:
    """把引擎库需要的目录补进 LD_LIBRARY_PATH（缺了会段错误 ✗，不是"找不到"那么温和 ✓）"""
    cur = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d]
    add = [d for d in ENGINE_LIB_DIRS if d not in cur]
    if add:
        env["LD_LIBRARY_PATH"] = ":".join(add + cur)


def run_chat(root: str, argv: "List[str]") -> int:
    model_dir, backend, rest = split_known(argv)
    # ★ 没显式给 -b（也没设 CANN_LLM_BACKEND）时，按模型目录自动判断后端 ★
    #   否则会拿默认的 hiai 去校验 nnrt 的模型目录 ⇒ 报"缺少 api_config.json" ✗
    if backend is None:
        backend = detect_backend(find_model_dir(root, model_dir)) or "hiai"
    # ★ 把判定结果【显式】变成 -b 传下去 ★ ——
    #   子进程的 ModelConfig.backend 默认是 "cann"，而且它不读 CANN_LLM_BACKEND ✗，
    #   所以只导出环境变量是不够的（实测仍然走错后端 ✗）。
    #   用户自己给了 -b/--backend 时不覆盖 ✓
    if not any(a in ("-b", "--backend") or a.startswith("--backend=") for a in rest):
        rest = ["-b", backend] + rest
    # ★ cann 后端只读 executor.json / context.json；官方 OMC 包解压出来没有这两个，
    #   这里就地导入（只补缺的，幂等）—— 让"解压即用"对 cann 也成立。
    if model_dir:
        from .omcimport import ensure_model_dir
        ensure_model_dir(model_dir, backend)
    # --lldb / --large-mem：本启动器的选项，不属于 chat CLI 的参数，先摘掉
    from .lldb_launch import build_debug_argv, strip_flag
    from .large_mem import build_large_mem_argv, strip_large_mem
    rest, want_lldb = strip_flag(rest)
    rest, want_large_mem = strip_large_mem(rest)
    var, lib, kind = pick_engine_lib(backend)

    py = current_python()
    if not libc_compatible(py):
        warn("解释器兼容性提示")
        print(f"    {py} 是 glibc 构建（带 libmusl_compat 垫片），"
              f"与按 musl 编译的引擎不兼容，加载引擎时大概率段错误。\n"
              f"    请改用 /data/service/hnp/bin/python3，或用 PYTHON=… 指定。", file=sys.stderr)

    # 与 start_server.sh 保持一致：把最终选中的解释器和版本打出来 ——
    # 否则用户看不出"到底用了哪一个 python"，而"为什么不是更靠前的那个"
    # 只能从 shell 打印的跳过行反推。
    ok(f"Python {sys.version.split()[0]}  ·  {py}")

    if not os.path.isfile(lib):
        die(f"找不到 {kind} {lib}\n    后端 {backend} 需要鸿蒙设备（NPU）环境。")
    ok(f"{kind}: {lib}")

    d = find_model_dir(root, model_dir)
    if d is None:
        info("未指定模型目录，将交由程序提示。用 -d 指定，"
             f"或把模型放到 {root}/models/<名字>/ 下。")
    else:
        ok(f"模型目录: {d}")
        kv = model_info_line(d)
        if kv:
            ok(f"上下文窗口: {kv} token  (= kv_cache_max_len，输入 + 输出之和)")
            info("最大输出没有固定值：= 窗口 − 本次输入长度")
            info("改单轮上限：--max-tokens <n>   例：scripts/start_chat.sh -d … --max-tokens 512")

    env = dict(os.environ)
    ensure_engine_lib_path(env)
    env["PYTHONPATH"] = os.path.join(root, "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env[var] = lib
    # ★ 把【判定出的后端名】也传下去 ★ ——
    #   真正创建后端的是子进程（chat CLI / server），它读 CANN_LLM_BACKEND；
    #   不传的话子进程会用自己的默认值 hiai ✗ ⇒ 于是刚才"判定成 nnrt 却仍走 hiai" ✗
    env["CANN_LLM_BACKEND"] = backend
    # ★ execve 会把整个进程换掉，**Python 的 stdout 缓冲区不会自动 flush**。
    #   终端里是行缓冲所以看着没问题；一旦重定向到文件或接管道（块缓冲），
    #   上面那些 ✓/› 提示就会【全部丢失】。必须手动刷。
    sys.stdout.flush()
    sys.stderr.flush()
    if want_large_mem:
        # ★ 大模型补丁：gdbserver + lldb 全自动（不需要人工敲命令）✓
        #   和 --lldb 同时给时以本项为准 —— 它本身就是"带补丁的调试启动"
        argv_lm, hints, err = build_large_mem_argv(
            py, "cann_llm.cli.chat", rest, env)
        if argv_lm is None:
            die(err)
        for line in hints:
            info(line) if line else print()
        print()
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(argv_lm[0], argv_lm, env)
        return 0
    if want_lldb:
        argv_dbg, hints, err = build_debug_argv(
            py, ["-m", "cann_llm.cli.chat"] + rest)
        if argv_dbg is None:
            die(err)
        for line in hints:
            info(line) if line else print()
        print()
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(argv_dbg[0], argv_dbg, env)
        return 0
    os.execve(py, [py, "-X", "faulthandler", "-m", "cann_llm.cli.chat"] + rest, env)
    return 0


def main(argv: "Optional[List[str]]" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("用法: python -m cann_llm.launcher {chat|server} [参数…]")
        print("      --lldb   在调试器下前台运行（抓崩溃现场）")
        print("               本机只有 huawei-debug-lldb-server 能正常调试，")
        print("               找不到会警告并建议从 CodeArts IDE 的终端运行")
        print("      --large-mem  自动打「大模型补丁」（经 gdbserver+lldb 改内存）：")
        print("               单段 .ms 上限从 ≈545 MB 提到 ≈1.09 GB；全自动，")
        print("               只改内存不动磁盘（验证手段，不是交付方案）")
        return 0
    what, rest = argv[0], argv[1:]
    # 脚本所在仓库根：src/cann_llm/launcher.py → 上溯三级
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if what == "chat":
        return run_chat(root, rest)
    if what == "server":
        from .launcher_server import run_server
        return run_server(root, rest)
    die(f"未知子命令 {what!r}（可用：chat / server）")
    return 1


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
