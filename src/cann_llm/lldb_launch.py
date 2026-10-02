"""在调试器下运行「python -m …」的公共逻辑（server 与 chat 共用）。

为什么单独一个模块：`launcher.py`（chat）与 `launcher_server.py`（server）都要
支持 `--lldb`，逻辑必须一致 —— 两处各写一份迟早会漂移。

★ 本机（HarmonyOS）不能用 `lldb -- <binary>` 直接拉起进程：

    (lldb) run
    error: 'A' packet returned an error: 8

这是 lldb 经 gdb-remote 协议设置 argv 失败（平台限制），不是被调试程序的问题。
所以改为在系统自带的 gdbserver 里起进程，再用 lldb 的 `gdb-remote` 接上去。

★ 本机目前**只有 huawei-debug-lldb-server 能正常调试**（鸿蒙 7 **没有**系统自带的
`lldb` / `lldb-server`；会 `ptrace failed: Permission denied` 的是 DevBox、Harmonybrew
和 OHOS-SDK 那几份）。它来自应用商店里的 CodeArts IDE（`com.huawei.codearts`，与
`com.huawei.codearts.agent` 是两个应用），躺在 IDE 自己的沙箱里，要在 **CodeArts IDE
的终端**里拷出来：

    mkdir -p ~/.local/bin
    cp /data/storage/el2/base/files/huawei-debug-lldb-server ~/.local/bin/

找不到它时这里会给出同样的提示（普通 `lldb -- <python>` 会报上面那个 `'A' packet`
错误）；随后仍会尝试普通 lldb —— 别的平台这样是可行的。
"""
from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence, Tuple

__all__ = ["build_debug_argv", "is_real_executable", "find_gdbserver",
           "GDBSERVER_NAME", "GDBSERVER_DIRS", "DEFAULT_GDBSERVER", "DEFAULT_LLDB_PORT"]

#: 这份 gdbserver 的可执行名（来自 CodeArts IDE，不是系统自带）
GDBSERVER_NAME = "huawei-debug-lldb-server"

#: 按名字找不到时，顺带尝试的目录 —— **追加到 PATH 末尾**再查。
#: 追加而不是前插：免得盖掉用户自己 PATH 里已有的同名程序。
#:   · ~/.local/bin                  —— 文档推荐的拷贝目标
#:   · /data/storage/el2/base/files  —— CodeArts IDE 沙箱里那份的原地
GDBSERVER_DIRS = ("~/.local/bin", "/data/storage/el2/base/files")

#: 兼容旧名字（调用方/测试可能引用过）
DEFAULT_GDBSERVER = "/data/storage/el2/base/files/" + GDBSERVER_NAME


def find_gdbserver(env: "Optional[dict]" = None) -> str:
    """找 ``huawei-debug-lldb-server``，返回路径；找不到返回 ``""``。

    顺序：① ``CANN_LLM_LLDB_SERVER`` 显式指定（给了就用它，不存在也算没找到）；
    ② 把 :data:`GDBSERVER_DIRS` 追加到 ``PATH`` 末尾后按名字找。
    """
    env = os.environ if env is None else env
    explicit = env.get("CANN_LLM_LLDB_SERVER")
    if explicit:
        return explicit if os.path.exists(explicit) else ""
    # 把两个常见位置【追加到 PATH 末尾】，再按名字逐个目录找。
    # 这里用 os.path.exists 而不是 shutil.which：单测会 mock 掉 os.path.exists
    # 来模拟"某处存在/不存在"，走 which 会绕过 mock。
    path = env.get("PATH", "")
    dirs = [d for d in path.split(os.pathsep) if d]
    for d in GDBSERVER_DIRS:
        full = os.path.expanduser(d)
        if full not in dirs:
            dirs.append(full)
    for d in dirs:
        cand = os.path.join(d, GDBSERVER_NAME)
        if os.path.exists(cand):
            return cand
    return ""

DEFAULT_LLDB_PORT = 5091


def is_real_executable(path: str) -> bool:
    """是不是能直接 execve 的**真二进制**（ELF）。

    有的解释器入口是 `#!/bin/sh` 包装器（`python3` → `exec python3.12 "$@"`）。
    shell 跑它没问题（内核解析 shebang），但 **gdbserver 用 execve 直接拉起进程、
    不解析 shebang** —— 传包装器进去只会得到

        execve failed: Operation not permitted

    （实测）。所以调试启动前必须先确认这一点，把问题在启动时说清楚。
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def build_debug_argv(py: str, args: Sequence[str]
                     ) -> Tuple[Optional[List[str]], List[str], str]:
    """构造「在调试器下运行 ``py`` + ``args``」的 argv。

    返回 ``(argv, 提示行, 错误信息)``：

    * 成功：``argv`` 非空、``错误信息`` 为空；
    * 失败（找不到调试器 / 解释器不是 ELF）：``argv`` 为 ``None``。

    提示行由调用方打印 —— 保持本函数纯粹，便于单测。
    """
    env = os.environ

    # ★ 不做启发式替换：调试器要的是真正的可执行文件，包装器直接报错，
    #   由调用方换成真二进制（通常是同目录下带版本号的那个）。
    if not is_real_executable(py):
        return None, [], (
            f"--lldb 需要真正的可执行文件，但 {py} 不是 ELF（看起来是脚本包装器）。\n"
            "      gdbserver 用 execve 直接拉起进程、不解析 shebang，会报\n"
            "      execve failed: Operation not permitted。\n"
            "      请改用真正的解释器，例如同目录下带版本号的那个：\n"
            "      PYTHON=<同一目录>/python3.12 ./scripts/start_server.sh … --lldb")

    gdbserver = find_gdbserver(env)
    lldb = (env.get("CANN_LLM_LLDB")
            or shutil.which("lldb")
            or "/storage/Users/currentUser/.harmonybrew/bin/lldb")

    if gdbserver:
        port = int(env.get("CANN_LLM_LLDB_PORT") or DEFAULT_LLDB_PORT)
        argv = [gdbserver, "gdbserver", "--native-regs",
                f"127.0.0.1:{port}", "--", py, "-X", "faulthandler"] + list(args)
        hints = [
            f"在 gdbserver 下启动（进程会先停住，等调试器接入）：{gdbserver}",
            "",
            "另开一个终端接上去：",
            f"    {lldb} -o 'gdb-remote 127.0.0.1:{port}'",
            "（接上后敲 continue 让它跑起来；崩溃时会停住，用 bt 看栈）",
        ]
        return argv, hints, ""

    if os.path.exists(lldb):
        # ★ 实测：本机目前**只有 huawei-debug-lldb-server 能正常调试** —— 普通 lldb
        #   直接拉起进程会报 "error: 'A' packet returned an error: 8"。
        #   这里仍然回退（别的平台没问题），但必须把话说清楚，免得白折腾。
        argv = [lldb, "--", py, "-X", "faulthandler"] + list(args)
        return argv, [
            f"⚠️ 没找到 {GDBSERVER_NAME}"
            "（已查 CANN_LLM_LLDB_SERVER、PATH，以及 "
            + "、".join(GDBSERVER_DIRS) + "）",
            "     本机目前只有它能正常调试；普通 lldb 直接拉起进程会报",
            "     error: 'A' packet returned an error: 8。",
            "     建议从 CodeArts IDE 的终端运行。",
            f"     仍然尝试：{lldb} -- …",
        ], ""

    return None, [], (
        f"找不到调试器（试过 {gdbserver} 与 {lldb}）。\n"
        "      本机目前只有 huawei-debug-lldb-server 能正常调试 ——\n"
        "      建议从 CodeArts IDE 的终端运行；\n"
        "      或用 CANN_LLM_LLDB_SERVER / CANN_LLM_LLDB 指定路径")


def strip_flag(argv: List[str]) -> Tuple[List[str], bool]:
    """把 ``--lldb`` 从参数表里摘出来（返回 ``(剩余参数, 是否出现)``）。"""
    found = "--lldb" in argv
    return [a for a in argv if a != "--lldb"], found
