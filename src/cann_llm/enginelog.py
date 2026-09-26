"""出错时把 hilog 里**本进程的原始日志**取回来。

为什么需要它
------------
引擎的失败原因**不在返回码里**：``GenerateAsync`` 失败也返回 0，
``GetGenerateStatus`` 只给一个笼统的值。真正的原因由引擎逐层写进 hilog，
例如实测抓到的原话：

    mask_and_pos_manager.cpp SetInitTokenLen(26)::
      "set init token len = 2048 error. it should smaller than kvCacheMaxLen 2048"

所以出错时读一次 hilog，把日志原样附在错误信息里。

**唯一做的筛选是按 pid** ——
hilog 是全局缓冲，不过滤 pid 会把别的进程的日志一起捞进来（实测 270 行里
混着 ``1149/hiaiserver`` 等其它进程的行）。除此之外：
不提取引号、不去重、不限长度、不按 tag/级别过滤、不按内容过滤。

（曾经加过白名单/黑名单，都删掉了：过滤器是按【已知原因】设计的，
只能留下我已经想到的那几类，恰好把"没想到的"全丢了 —— 而那才是诊断价值所在。）
"""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional

__all__ = ["recent_engine_log", "hilog_available", "describe_probe"]


def _hilog_exe() -> Optional[str]:
    for p in ("/usr/bin/hilog", "/system/bin/hilog"):
        if os.path.exists(p):
            return p
    return None


def hilog_available() -> bool:
    """本终端能否读 hilog（存在命令，且进程在 ``log`` 组里）。"""
    if not _hilog_exe():
        return False
    try:
        out = subprocess.run(["id"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "(log)" in out


def describe_probe() -> str:
    """一句话说明为什么读不到（供提示用）。"""
    if not _hilog_exe():
        return "本机没有 hilog 命令"
    try:
        out = subprocess.run(["id"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"无法执行 id：{e}"
    if "(log)" not in out:
        return "当前终端不在 log 组里，读不到 hilog（换一个系统终端通常可以）"
    return ""


def recent_engine_log(pid: Optional[int] = None,
                      timeout: float = 20.0) -> "Optional[List[str]]":
    """读 hilog 缓冲，返回**本进程的原始日志行**（只按 pid 过滤）。

    :return: 原始行列表（顺序与 hilog 一致）；**读不到 hilog 时返回 ``None``**
             —— 与"读到了但没有内容"区分开。
    """
    pid = pid or os.getpid()
    exe = _hilog_exe()
    if not exe:
        return None
    try:
        # -x：非阻塞读一遍缓冲即退出（不会挂住）
        proc = subprocess.run([exe, "-x"], capture_output=True, text=True,
                              timeout=timeout, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 and not proc.stdout:
        return None
    # ★ 按 hilog 的【pid 字段】比较，不是子串匹配 ——
    #   子串会把提到该 pid 的别的进程日志也捞进来，例如
    #   "… 1723 2827 E utils_base: path (/proc/51255/status) to realpath error …"
    #   行首三列是 日期 时间 PID，所以取第 3 列。
    key = str(pid)
    out: List[str] = []
    for ln in proc.stdout.splitlines():
        parts = ln.split(None, 4)
        if len(parts) >= 3 and parts[2] == key:
            out.append(ln)
    return out


def format_engine_log(pid: Optional[int] = None) -> str:
    """把本进程的原始 hilog 行整理成可直接拼进异常消息的文本。"""
    lines = recent_engine_log(pid=pid)
    if lines is None:
        why = describe_probe()
        return (f"\n（读不到引擎日志：{why}）" if why else
                "\n（读不到引擎日志）")
    if not lines:
        return "\n（hilog 里没有本进程的日志）"
    body = "\n".join(f"    {ln}" for ln in lines)
    return f"\n引擎日志（原始，{len(lines)} 行，仅按 pid 过滤）：\n{body}"
