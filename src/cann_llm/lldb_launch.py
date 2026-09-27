"""在调试器下运行「python -m …」的公共逻辑（server 与 chat 共用）。

为什么单独一个模块：`launcher.py`（chat）与 `launcher_server.py`（server）都要
支持 `--lldb`，逻辑必须一致 —— 两处各写一份迟早会漂移。

★ 本机（HarmonyOS）不能用 `lldb -- <binary>` 直接拉起进程：

    (lldb) run
    error: 'A' packet returned an error: 8

这是 lldb 经 gdb-remote 协议设置 argv 失败（平台限制），不是被调试程序的问题。
所以改为在系统自带的 gdbserver 里起进程，再用 lldb 的 `gdb-remote` 接上去。
其它平台（没有那个 gdbserver）回落到普通的 `lldb -- <python>`。
"""
from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence, Tuple

__all__ = ["build_debug_argv", "DEFAULT_GDBSERVER", "DEFAULT_LLDB_PORT"]

#: HarmonyOS 上系统自带的 lldb-server（gdbserver 模式）
DEFAULT_GDBSERVER = "/data/storage/el2/base/files/huawei-debug-lldb-server"

DEFAULT_LLDB_PORT = 5091


def build_debug_argv(py: str, args: Sequence[str]
                     ) -> Tuple[Optional[List[str]], List[str], str]:
    """构造「在调试器下运行 ``py`` + ``args``」的 argv。

    返回 ``(argv, 提示行, 错误信息)``：

    * 成功：``argv`` 非空、``错误信息`` 为空；
    * 失败（找不到调试器）：``argv`` 为 ``None``，由调用方决定怎么退出。

    提示行由调用方打印 —— 保持本函数纯粹，便于单测。
    """
    env = os.environ
    gdbserver = env.get("CANN_LLM_LLDB_SERVER") or DEFAULT_GDBSERVER
    lldb = (env.get("CANN_LLM_LLDB")
            or shutil.which("lldb")
            or "/storage/Users/currentUser/.harmonybrew/bin/lldb")

    if os.path.exists(gdbserver):
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
        argv = [lldb, "--", py, "-X", "faulthandler"] + list(args)
        return argv, [f"在 lldb 下启动：{lldb}"], ""

    return None, [], (f"找不到调试器（试过 {gdbserver} 与 {lldb}）——"
                      "用 CANN_LLM_LLDB_SERVER / CANN_LLM_LLDB 指定路径")


def strip_flag(argv: List[str]) -> Tuple[List[str], bool]:
    """把 ``--lldb`` 从参数表里摘出来（返回 ``(剩余参数, 是否出现)``）。"""
    found = "--lldb" in argv
    return [a for a in argv if a != "--lldb"], found
