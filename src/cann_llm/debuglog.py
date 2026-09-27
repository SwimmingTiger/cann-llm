"""``--debug`` 用的诊断打点。

只做一件事：把**HTTP 层看到的**和**引擎层看到的**原样记下来，让事后能回答
"这一次请求到底发生了什么"：

* 客户端给了什么（含被忽略的字段、max_tokens 到底多少）
* 模板渲染出的**原文**（送给引擎的 prompt）
* 引擎吐出的**原始文本**与最后上报的 ``finish_reason``

为什么单独一个模块：``api/server.py`` 与 ``backends/hiai.py`` 都要打点，
而它们不该互相 import。默认关闭 —— 不开关就不产生任何输出。

★ **不做任何过滤、拦截、改写、限制**：请求体、请求头、每一帧 SSE、引擎的原始
  输入输出，全部原样写入。诊断的价值就在完整原文，任何"帮你收一收"都是在藏东西。
  ⚠️ 因此日志里**会包含**请求头里的 `Authorization` / API key 等敏感值 ——
  这是刻意的（用户要求），别把 debug 日志随手贴出去或提交进版本库。

开启方式（任一）::

    server --debug
    CANN_LLM_DEBUG=1
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

__all__ = ["enabled", "enable", "log", "kv", "log_file"]

_LOGGER_NAME = "cann_llm.debug"

def _logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def _text(data: bytes) -> str:
    r"""bytes → str，**不丢任何字节**。

    用 ``backslashreplace``：真正的 UTF-8 文本原样通过；万一有坏字节，
    也变成 ``\xNN`` 留在日志里，而不是被替换成 ``?`` 抹掉。
    """
    return data.decode("utf-8", "backslashreplace")


def _as_text(v: Any) -> str:
    return _text(v) if isinstance(v, (bytes, bytearray)) else (v if isinstance(v, str) else str(v))


def enabled() -> bool:
    """是否已开启（``--debug`` 或 ``CANN_LLM_DEBUG=1``）。"""
    return _logger().handlers != [] or os.environ.get("CANN_LLM_DEBUG", "") not in ("", "0")


def enable(stream: Optional[Any] = None, path: Optional[str] = None) -> None:
    """开启打点。重复调用无害。

    ``path`` 给了就写文件，否则写 stderr。

    ★ 为什么默认建议写文件：**引擎自己的 SELinux 噪声也走 stderr**
    （``Unknown class perfgenius_interface``），项目文档里让大家用 ``2>/dev/null``
    屏蔽它 —— 那样会把我们的诊断日志一起屏蔽掉。写文件就没这个问题。
    """
    lg = _logger()
    lg.setLevel(logging.DEBUG)
    target = path or getattr(stream or sys.stderr, "name", None) or id(stream or sys.stderr)
    # ★ 目标变了就要换 handler。原来写成 `if not lg.handlers:` —— 已经挂过就直接
    #   返回，于是 enable(stream=别的流) 变成空操作（单元测试里被这条坑到：
    #   缓冲一直是空的，因为日志写进了先前那个 handler）。
    if lg.handlers and getattr(lg, "_cann_llm_target", None) == target:
        lg.propagate = False
        return
    for h in list(lg.handlers):
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:            # noqa: BLE001
            pass
    if path:
        h: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    else:
        h = logging.StreamHandler(stream or sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [debug] %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    lg._cann_llm_target = target       # type: ignore[attr-defined]
    lg.propagate = False


def log_file() -> Optional[str]:
    """当前写的是哪个文件（写 stderr 时返回 None）。"""
    for h in _logger().handlers:
        if isinstance(h, logging.FileHandler):
            return getattr(h, "baseFilename", None)
    return None


def log(section: str, text: Any = "") -> None:
    """打一条。未开启时什么都不做（调用方不必先判断）。

    ★ 内容**原样写出**：不截断、不过滤、不改写。
      诊断的价值就在原文 —— 任何"帮你收一收"都是在藏东西。
    """
    if not enabled():
        return
    if isinstance(text, (bytes, bytearray)):
        text = _text(bytes(text))
    _logger().debug("── %s ──\n%s", section, text)


def kv(section: str, **items: Any) -> None:
    """把一组键值打成 ``k = v`` 行。**值原样写出，不截断。**"""
    if not enabled():
        return
    body = "\n".join(f"    {k} = {_as_text(v)}" for k, v in items.items())
    _logger().debug("── %s ──\n%s", section, body)


