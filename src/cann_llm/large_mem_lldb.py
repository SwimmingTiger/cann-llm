"""lldb 侧脚本：在推理进程里把「2 GiB 上限」的几处检查改成 nop。

**这个文件是给 lldb 的内嵌 python 用的**，不是给引擎跑的。正常用法由
``scripts/large_mem_run.sh`` 自动完成（`--large-mem` 会调它）：

    (lldb) gdb-remote 127.0.0.1:<port>
    (lldb) command script import <root>/src/cann_llm/large_mem_lldb.py
    (lldb) large_mem_run        # 装断点 + 放行 + 等结束 + 记退出码（一条命令做完）

补丁表在 :mod:`cann_llm.large_mem`（单一来源，避免两处漂移 ✗）——
所以 **lldb 进程也要能 import 到它**：编排脚本会导出 ``PYTHONPATH=<root>/src`` ✓。

三个实测踩出来的设计要点：

1. ★批处理模式下只能有一条"python 实现的命令"★
   实测 ``lldb --batch -o … -o …``：一旦执行了 `script` 或 `command script add`
   注册的命令，**后续 `-o` 就不再生效** ✗（于是 ``process continue`` 根本没跑、
   进程一直停着）。所以把全流程塞进 ``large_mem_run`` 一条命令里 ✓
2. ★同一条命令内要继续跑进程，得用 SBListener 泵事件★
   ``SBProcess.Continue()`` 是异步的；直接轮询 ``GetState()`` 拿不到更新 ✗，
   而 ``SBListener.WaitForEvent()`` 会泵事件循环 ⇒ 状态才会动 ✓
3. ★打补丁前先读回 4 字节比对★（已经打过的也算通过，幂等）；
   既不是原字节、也不是补丁字节就告警并跳过 —— 换库版本时不会改错地方 ✓
"""
from __future__ import annotations

import os
import sys
import time

try:
    import lldb
except ImportError:                       # 被普通 python 误导入时给个清楚的话
    raise SystemExit("large_mem_lldb.py 只能由 lldb 的内嵌 python 导入")


def _p(*args):
    """带 flush 的 print。

    ★必须 flush★：lldb 的 stdout 与 gdbserver 转发的被调试进程输出混在一起，
      不刷就会看到"补丁日志跑到程序输出后面"的错乱顺序 ✗
    """
    print(*args, flush=True)


def _table():
    """拿补丁表；**必须**能从 cann_llm.large_mem 导入（单一来源）。"""
    try:
        from cann_llm.large_mem import PATCHES
    except Exception as exc:              # noqa: BLE001 - 要把原因原样告诉用户
        _p("[large-mem] ✗ 导入补丁表失败：%s" % exc)
        _p("[large-mem]   请设 PYTHONPATH=<仓库根>/src（编排脚本会自动设）")
        return None
    return PATCHES


def _module_base(target, name):
    """已加载模块的基址（= 其 vaddr 0 的运行期地址）；没有则 None。"""
    for mod in target.module_iter():
        if mod.GetFileSpec().GetFilename() != name:
            continue
        addr = mod.GetObjectFileHeaderAddress()
        if not addr.IsValid():
            continue
        load = addr.GetLoadAddress(target)
        if load != lldb.LLDB_INVALID_ADDRESS:
            return load
    return None


def _read(process, addr, n):
    err = lldb.SBError()
    data = process.ReadMemory(addr, n, err)
    if not err.Success():
        return None
    return bytes(bytearray(data))


def _write(debugger, addr, data):
    """写内存（走 lldb 命令 —— 已验证可穿透只读代码段 ✓）。"""
    vals = " ".join("0x%02x" % b for b in bytearray(data))
    debugger.HandleCommand("memory write -s 1 0x%x %s" % (addr, vals))


_STATE = {"pending": 0,      # 还有几处"模块没加载"而没补上（模块一加载就重试）
          "patched": False}   # 首次 Build 命中、补丁处理过一次


def patch_all(debugger, quiet=False):
    """按表打补丁；返回 ``(成功数, 跳过数)``。

    ``quiet=True``：只在**这次真的补上了东西**时才打印（用于轮询重试，避免刷屏）。
    """
    table = _table()
    if table is None:
        return 0, 0

    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    done = skipped = missing = 0
    applied_now = []

    for p in table:
        base = _module_base(target, p.module)
        if base is None:
            # ★不是错误★：nnrt 后端下 libhiai_ir.so 是**之后**才随 DDK 加载的，
            #   这里先记下，等 modules-loaded 事件再补（见 run）✓
            missing += 1
            continue

        addr = base + p.offset
        cur = _read(process, addr, len(p.expect))
        if cur is None:
            _p("[large-mem] ✗ 读不了 %s+0x%X —— 跳过 %s" % (p.module, p.offset, p.what))
            skipped += 1
            continue

        if cur == p.patch:                       # 幂等：已经打过了 ✓
            if not quiet:
                _p("[large-mem] = %-20s +0x%05X 已补丁（跳过）" % (p.module, p.offset))
            done += 1
            continue
        if cur != p.expect:
            _p("[large-mem] ✗ %s+0x%05X 原字节不符：读到 %s，期望 %s ⇒ 跳过 %s"
               % (p.module, p.offset,
                  " ".join("%02x" % b for b in cur),
                  " ".join("%02x" % b for b in bytearray(p.expect)),
                  p.what))
            _p("[large-mem]   多半是系统库版本不同（换版本后偏移会变）—— 别硬改 ✗")
            skipped += 1
            continue

        _write(debugger, addr, p.patch)
        back = _read(process, addr, len(p.patch))
        if back is not None and back == p.patch:
            applied_now.append("%s+0x%05X %s" % (p.module, p.offset, p.what))
            if not quiet:
                _p("[large-mem] ✓ %-20s +0x%05X  %s → %s 已就位  （%s）"
                   % (p.module, p.offset,
                      " ".join("%02x" % b for b in cur),
                      " ".join("%02x" % b for b in bytearray(p.patch)), p.what))
        else:
            _p("[large-mem] ✗ %s+0x%05X 写入后回读不一致 ⇒ 这处没生效 ✗"
               % (p.module, p.offset))
            skipped += 1
            continue
        done += 1

    _STATE["pending"] = missing
    if applied_now and quiet:
        _p("[large-mem] ✓ 新加载的模块上补了 %d 处：%s"
           % (len(applied_now), "；".join(applied_now)))
    if not quiet:
        _p("[large-mem] 补丁结果：成功 %d / 跳过 %d / 等模块加载 %d（共 %d 处）"
           % (done, skipped, missing, len(table)))
    return done, skipped


def on_build(frame, bp_loc, internal_dict):
    """`OH_AI_ModelBuildFromFile` 命中时打补丁，然后放行（返回 False = 继续）。"""
    proc = frame.GetThread().GetProcess()
    debugger = proc.GetTarget().GetDebugger()
    _STATE["patched"] = True
    _p("")
    _p("[large-mem] 命中 OH_AI_ModelBuildFromFile —— 开始打补丁")
    patch_all(debugger)

    if _STATE["pending"]:
        # ★还有模块没加载★（nnrt 后端下 libhiai_ir.so 就是之后才来的）——
        #   不放行 detach，留着断点等**下一次 Build** 再补一轮 ✓
        #   （真卡大段的 libsec_shared 那三处在首次 Build 前就补好了 ✓）
        _p("[large-mem] 还有 %d 处要等模块加载 —— 下次 Build 时自动补 ✓"
           % _STATE["pending"])
        return False

    # ★不做 detach★：在断点回调里调 Detach() 会把 lldb 的 resume 逻辑搞坏——
    #   实测报 `Failed to resume process: process not in stopped state after
    #   synchronous resume: detached`，之后程序活着却**卡住不再往下跑** ✗
    #   改由驱动侧把 lldb 的 stdin 指到 /dev/null（它就不会去抢键盘），
    #   程序本来就用【它自己的终端】✓（attach 模式下它的 stdio 一直是自己的）✓
    bp_loc.GetBreakpoint().SetEnabled(False)
    _p("[large-mem] 补丁已就绪；程序带着自己的终端继续 ✓")
    return False


def _install_bp(debugger):
    target = debugger.GetSelectedTarget()
    bp = target.BreakpointCreateByName("OH_AI_ModelBuildFromFile")
    bp.SetScriptCallbackFunction("large_mem_lldb.on_build")
    if bp.GetNumLocations() == 0:
        _p("[large-mem] 断点已挂起（等 libmindspore_lite_ndk.so 加载后自动生效）✓")
    else:
        _p("[large-mem] 断点已装 ✓")
    return bp


def _release():
    """创建"放行文件" ⇒ 取消 runner 在第一次 build 前的等待（★顺序：先补丁后放行★）"""
    path = os.environ.get("CANN_LLM_LARGE_MEM_RENDEZVOUS")
    if not path:
        return
    try:
        with open(path, "w") as fh:
            fh.write("go\n")
        _p("[large-mem] 已放行 runner ✓（它可以继续做第一次 build 了）")
    except OSError as exc:
        _p("[large-mem] ✗ 写放行文件失败：%s" % exc)


def _write_status(status):
    """把退出码交给编排脚本（写状态文件，不动 stdout）。

    ★为什么不打到 stdout★：lldb 的 stdout 必须直通终端（对话要流式输出），
    一旦为了让 shell 取退出码去捕获它，就变成"跑完才出字"了 ✗
    """
    path = os.environ.get("CANN_LLM_LARGE_MEM_STATUS_FILE")
    if not path:
        return
    try:
        with open(path, "w") as fh:
            fh.write("%d\n" % status)
    except OSError as exc:
        _p("[large-mem] ✗ 写状态文件失败：%s" % exc)


def run(debugger=None, command=None, exe_ctx=None, result=None, internal_dict=None):
    """`large_mem_run`：装断点 → 放行 → 等进程结束 → 记下退出码（**一条命令做完全部**）。

    为什么要一条命令：见模块 docstring 第 1 点（批处理模式下 python 命令会截断 `-o` 链）。
    为什么要 SBListener：见第 2 点（直接轮询拿不到状态更新）。
    """
    target = debugger.GetSelectedTarget()
    proc = target.GetProcess()
    listener = lldb.SBListener("large-mem-watch")
    proc.GetBroadcaster().AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
    # ★模块加载事件★：nnrt 后端下 libhiai_ir.so 在【第一次 BuildFromFile 之后】
    #   才随 DDK 载入 ⇒ 必须在它出现时立刻补上，否则权重拷贝那步已经过了 ✗
    target.GetBroadcaster().AddListener(listener, lldb.SBTarget.eBroadcastBitModulesLoaded)

    _install_bp(debugger)
    _p("[large-mem] 放行…")
    proc.Continue()

    done = {lldb.eStateExited, lldb.eStateDetached,
            lldb.eStateCrashed, lldb.eStateInvalid}
    event = lldb.SBEvent()
    while proc.GetState() not in done:
        # ★第 2 个参数是 uint32_t（**整数**秒）★
        #   传 1.0（float）会直接
        #     TypeError: in method 'SBListener_WaitForEvent', argument 2 of type 'uint32_t'
        #   —— 实测：短命进程可能还没轮到这行就退出了，长驻的对话/服务进程必踩 ✗
        listener.WaitForEvent(1, event)
        if _STATE["pending"]:
            # 有新模块了？重试一下（幂等；只有真补上才会打印）
            patch_all(debugger, quiet=True)

    status = proc.GetExitStatus()
    _p("[large-mem] 被调试进程退出码 = %d" % status)
    _write_status(status)


def run_attach(debugger=None, command=None, exe_ctx=None, result=None, internal_dict=None):
    """`large_mem_attach`：**附着式**流程 —— 放行 → 等补丁打完 → detach。

    ★为什么是"打完就 detach"而不是一直挂着★
      程序的终端必须始终是它自己的：`gdbserver -- prog` 那种"由调试器拉起"的模式会给
      被调试进程**另开一个 pty** ✗（实测 fd0/fd1 指向 /dev/pts/N，不是当前终端），
      程序输出靠 lldb 转发还能看见，但**键盘输入永远进不去** ⇒ 交互式对话卡死在提示符 ✗
      attach 模式下程序本来就是被"我们"正常启动的、终端是自己的 ✓，
      补丁一打完就 detach ⇒ 调试器彻底走人（连带它那个 `(lldb)` 提示符也不会抢输入）✓

    超时（``CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT``，默认 60 秒）到点也 detach，
    免得程序被永远停着（这时打印警告，说明补丁可能没上全）。
    """
    target = debugger.GetSelectedTarget()
    proc = target.GetProcess()
    listener = lldb.SBListener("large-mem-attach")
    proc.GetBroadcaster().AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
    target.GetBroadcaster().AddListener(listener, lldb.SBTarget.eBroadcastBitModulesLoaded)

    deadline = time.time() + float(os.environ.get("CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT", "60"))
    _p("[large-mem] 已附着 pid=%d，放行…" % proc.GetProcessID())
    proc.Continue()

    done = {lldb.eStateExited, lldb.eStateDetached,
            lldb.eStateCrashed, lldb.eStateInvalid}
    event = lldb.SBEvent()
    while True:
        state = proc.GetState()
        if state in done:
            _p("[large-mem] 程序已结束/已分离，无需再 detach")
            return
        if _STATE["patched"] and _STATE["pending"] == 0:
            break
        if time.time() > deadline:
            _p("[large-mem] ⚠ 等补丁打上超时（%s 秒）—— 直接 detach 让程序继续跑"
               % os.environ.get("CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT", "60"))
            break
        listener.WaitForEvent(1, event)
        if _STATE["pending"]:
            patch_all(debugger, quiet=True)

    proc.Detach()
    _p("[large-mem] 已 detach：程序带着【自己的终端】继续运行 ✓（键盘输入照常可用）")


def install(debugger, command=None, exe_ctx=None, result=None, internal_dict=None):
    """`large_mem_install`：只装断点（手工调试用；自动化请用 `large_mem_run`）。"""
    _install_bp(debugger)


def report_exit(debugger=None, command=None, exe_ctx=None, result=None, internal_dict=None):
    """`large_mem_report_exit`：手工调试用 —— 把当前已知的退出码写进状态文件。"""
    status = -1
    try:
        status = lldb.debugger.GetSelectedTarget().GetProcess().GetExitStatus()
    except Exception:                     # noqa: BLE001
        pass
    _write_status(status)
    _p("[large-mem] 被调试进程退出码 = %d" % status)


def __lldb_init_module(debugger, internal_dict):
    # ★在 import 时就装好断点★
    #   为什么：批处理模式下，**注册的 python 命令**会让后续 `-o` 失效 ✗，
    #   而 `command script import` 不会 ⇒ 把"装断点"放这里，驱动脚本就能用
    #   lldb 原生的 `process continue` 放行 ✓
    #   这一点很关键：`process continue` 跑的是 lldb 自己的事件循环，
    #   ★它会把终端的键盘输入转发给被调试进程★（`target.process.disable-stdio`
    #   默认 false）—— gdbserver 给被调试进程另开的那个 pty 就是靠这个打通的 ✓
    #   而用 python 命令里的 SBListener 自己等，事件循环不在跑 ⇒
    #   输入根本转发不过去，交互式程序（对话）会卡死 ✗
    _install_bp(debugger)
    # ★attach 时进程是停着的 ⇒ 立刻把已加载模块上的补丁打掉，然后放行 runner★
    #   runner 在第一次 build 前会先 dlopen 补丁目标的库（含 libhiai_ir.so）
    #   ⇒ 这一步通常就能一次到位 4/4 ✓
    #   若 runner 还没来得及预加载（我们 attach 得更早），这里先补能补的，
    #   剩下那处由 Build 断点（on_build，安全网）在该次 build 入口补上 ✓
    try:
        patch_all(debugger)
    finally:
        _release()
    for name, fn in (("large_mem_attach", "run_attach"),
                     ("large_mem_install", "install"),
                     ("large_mem_run", "run"),
                     ("large_mem_report_exit", "report_exit")):
        debugger.HandleCommand(
            "command script add -f large_mem_lldb.%s %s" % (fn, name))
    _p("[large-mem] lldb 侧脚本已载入（断点已装；手工调试另有 large_mem_install/run/report_exit）")


if __name__ == "__main__":
    print("这是 lldb 侧脚本，请用 command script import 载入", file=sys.stderr)
