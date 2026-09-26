"""出错时从 hilog 取回**引擎自己说的话**。

为什么需要它
------------
引擎的失败原因**不在返回码里**（`GenerateAsync` 失败也返回 0，`GetGenerateStatus`
只给出一个笼统的值）。真正的原因由引擎写进 hilog，例如实测抓到的原话：

    mask_and_pos_manager.cpp SetInitTokenLen(26)::
      "set init token len = 2048 error. it should smaller than kvCache…"

所以出错时读一次 hilog，把引擎的原话附在错误信息里，比我们罗列"可能原因"准确得多。

前提与降级
----------
- **读 hilog 需要权限**（进程要在 ``log`` 组里）。读不到时返回 ``None``，
  调用方保持原有的"可能原因"文案即可 —— **读不到本身也是一个信号**
  （换个系统终端往往就好）。
- hilog 是**全局缓冲**，必须按**本进程 pid** 过滤，否则会捞到别人的日志。
- 只在出错时调用，不为正常路径增加开销。
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import List, Optional, Tuple

__all__ = ["recent_engine_errors", "hilog_available", "describe_probe"]

#: 我们只关心引擎自己的这几类 tag
_TAGS = ("AI_INFRA", "HIAI_DDK_MSG")
#: 形如 ... E A0FFFF/python3.12/AI_INFRA: file.cpp Func(123)::"message"
_LINE = re.compile(
    r"^\s*\S+\s+\S+\s+(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<lvl>[EWID])\s+"
    r"\S*/(?P<tag>[A-Z_]+):\s*(?P<body>.*)$")
_QUOTED = re.compile(r'"([^"]{6,300})"')
#: 只保留"看起来像失败"的消息，避免把一堆 is not in json file 也带上
_NOISE = re.compile(
    r"is not in json file|use mhc|use ndk|supported on this plat"
    r"|io uring reader is init failed"          # load 阶段的 warning，与本次失败无关
    r"|can not be used\."
)


def hilog_available() -> bool:
    """本终端能否读 hilog（在 ``log`` 组里，且存在 hilog 命令）。"""
    if not any(os.path.exists(p) for p in ("/usr/bin/hilog", "/system/bin/hilog")):
        return False
    try:
        out = subprocess.run(["id"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "(log)" in out


def describe_probe() -> str:
    """一句话说明为什么读不到（供提示用）。"""
    if not any(os.path.exists(p) for p in ("/usr/bin/hilog", "/system/bin/hilog")):
        return "本机没有 hilog 命令"
    try:
        out = subprocess.run(["id"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"无法执行 id：{e}"
    if "(log)" not in out:
        return "当前终端不在 log 组里，读不到 hilog（换一个系统终端通常可以）"
    return ""


def recent_engine_errors(pid: Optional[int] = None, limit: int = 6,
                         timeout: float = 8.0) -> "Optional[List[str]]":
    """读 hilog 缓冲，返回本进程最近若干条**引擎报错**（已去掉引号外的装饰）。

    :return: 消息列表；**读不到 hilog 时返回 ``None``**（与"读到但没有错误"区分开）。
    """
    pid = pid or os.getpid()
    exe = "/usr/bin/hilog" if os.path.exists("/usr/bin/hilog") else "/system/bin/hilog"
    if not os.path.exists(exe):
        return None
    try:
        # -x：非阻塞读一遍缓冲即退出（不会挂住）
        proc = subprocess.run([exe, "-x"], capture_output=True, text=True,
                              timeout=timeout, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 and not proc.stdout:
        return None

    msgs: List[str] = []
    for line in proc.stdout.splitlines():
        m = _LINE.match(line)
        if not m or m.group("pid") != str(pid):
            continue
        if m.group("tag") not in _TAGS:
            continue
        body = m.group("body")
        for q in _QUOTED.findall(body):
            if _NOISE.search(q):
                continue
            # 去掉 <|...|> 之类无关字符，保留引擎原话
            if q not in msgs:
                msgs.append(q)
    return msgs[-limit:] if msgs else []


def format_engine_errors(pid: Optional[int] = None, limit: int = 6) -> str:
    """把引擎最近的报错整理成可直接拼进异常消息的文本（读不到时给一句原因）。"""
    got = recent_engine_errors(pid=pid, limit=limit)
    if got is None:
        why = describe_probe()
        return (f"\n（读不到引擎日志：{why}）" if why else
                "\n（读不到引擎日志）")
    if not got:
        return "\n（hilog 里没有本进程的引擎报错 —— 可能是静默失败）"
    lines = "\n".join(f"    · {m}" for m in got)
    return f"\n引擎日志（原话）：\n{lines}"
