"""NPU 访问权限探测。

为什么需要它
------------
在鸿蒙上，**不是每个终端都有权限访问 NPU**。例如 MKCode 的终端跑在
``u:r:develop_tools_hap:s0`` 域、uid 20020325，连 ``stat("/dev/npu0")`` 都会被拒
（``PermissionError``，errno 13）。此时引擎加载可能正常，但一推理就失败，
而引擎只回报一个笼统的 ``Generate 返回 1`` / ``OnGenerateAsyncFailed``，
让人误以为是模型或参数问题。

我们**能观测到**的判据是设备节点本身：有权限时 ``/dev/npu0`` 等可以 stat；
没权限时连 stat 都拿不到。据此给出**准确的**提示。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

__all__ = ["NPU_DEVICES", "probe_npu_access", "npu_hint", "npu_unavailable_reason"]

#: 候选设备节点（命中任一即认为可访问）
NPU_DEVICES: Tuple[str, ...] = (
    "/dev/npu0",
    "/dev/npu_manager",
    "/dev/npu_direct",
    "/dev/npu_freq_uplimit",
    "/dev/davinci0",
)


def _selinux_domain() -> str:
    """读当前进程的 SELinux 域。

    ★ ``/proc/self/attr/current`` 的内容可能带 NUL 结尾（实测 ``u:r:xxx:s0\0``）——
    必须把 NUL 和其它不可见字符去掉，否则拼进提示文本后会让 shell 的 ``$(...)``
    报 "ignored null byte in input"。
    """
    try:
        with open("/proc/self/attr/current", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return ""
    return "".join(ch for ch in raw if ch.isprintable()).strip()


def probe_npu_access() -> "Tuple[bool, Optional[str]]":
    """探测 NPU 可访问性。

    :return: ``(可访问?, 失败原因)``。可访问时原因为 ``None``。

    判据：任一候选节点能 ``os.stat`` 即视为可访问。全部被拒时，用实际拿到的
    errno 判断是"权限问题"还是"节点不存在"——两者提示不同。
    """
    denied: List[str] = []
    missing = 0
    for dev in NPU_DEVICES:
        try:
            os.stat(dev)
            return True, None
        except PermissionError:
            denied.append(dev)
        except FileNotFoundError:
            missing += 1
        except OSError:
            denied.append(dev)

    if denied:
        dom = _selinux_domain()
        why = (f"当前终端没有访问 NPU 的权限"
               f"（{'、'.join(denied)} 全部被拒：Permission denied）")
        if dom:
            why += f"\n    SELinux 域: {dom}"
        why += (f"\n    这类终端（如 MKCode 的终端）通常拿不到 NPU 设备节点；"
                f"请换一个【有 NPU 权限】的终端运行，或用系统的终端应用。")
        return False, why
    return True, (f"找不到 NPU 设备节点（{'、'.join(NPU_DEVICES)} 都不存在）—— "
                  f"这台机器可能不是鸿蒙设备，或驱动未加载")


def npu_unavailable_reason() -> Optional[str]:
    """不可访问时返回原因文本，可访问时返回 ``None``（供后端拼错误信息用）。"""
    ok, why = probe_npu_access()
    return None if ok else (why or "")


def npu_hint() -> str:
    """一行式提示，用于"失败原因待定"的场景。"""
    ok, why = probe_npu_access()
    return "" if ok else (why or "")
