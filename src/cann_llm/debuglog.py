"""``--debug`` 用的诊断打点。

只做一件事：把**HTTP 层看到的**和**引擎层看到的**原样记下来，让事后能回答
"这一次请求到底发生了什么"：

* 客户端给了什么（含被忽略的字段、max_tokens 到底多少）
* 模板渲染出的**原文**（送给引擎的 prompt）
* 引擎吐出的**原始文本**与最后上报的 ``finish_reason``

为什么单独一个模块：``api/server.py`` 与 ``backends/hiai.py`` 都要打点，
而它们不该互相 import。默认关闭 —— 不开关就不产生任何输出。

开启方式（任一）::

    server --debug
    CANN_LLM_DEBUG=1
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

__all__ = ["enabled", "enable", "log", "kv", "clamp", "redact", "summary"]

_LOGGER_NAME = "cann_llm.debug"

#: 单条打点最大字符数 —— 日志是给人看的，不要把一个 4 万字的 prompt 全塞进去
CLAMP = 4000

#: 这些请求头不能进日志（密钥）
_SECRET_HEADERS = {"authorization", "x-api-key", "api-key", "cookie", "proxy-authorization"}


def _logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


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
    if not lg.handlers:
        if path:
            h: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        else:
            h = logging.StreamHandler(stream or sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s [debug] %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    lg.propagate = False


def log_file() -> Optional[str]:
    """当前写的是哪个文件（写 stderr 时返回 None）。"""
    for h in _logger().handlers:
        if isinstance(h, logging.FileHandler):
            return getattr(h, "baseFilename", None)
    return None


def log(section: str, text: Any = "") -> None:
    """打一条。未开启时什么都不做（调用方不必先判断）。"""
    if not enabled():
        return
    _logger().debug("── %s ──\n%s", section, clamp(text))


def kv(section: str, **items: Any) -> None:
    """把一组键值打成 ``k=v`` 行（值过长会被截断）。"""
    if not enabled():
        return
    body = "\n".join(f"    {k} = {clamp(v, 400)}" for k, v in items.items())
    _logger().debug("── %s ──\n%s", section, body)


def clamp(text: Any, limit: int = CLAMP) -> str:
    """截断到 ``limit`` 字符（并标出原长度）—— 日志要能看，不能把内存写爆。"""
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace")   # 引擎的 prompt 是 bytes
    s = text if isinstance(text, str) else str(text)
    if len(s) <= limit:
        return s
    return f"{s[:limit]}…[已截断，原文 {len(s)} 字符]"


def redact(headers: Any) -> Dict[str, str]:
    """请求头进日志前把密钥抹掉。

    打印原始头是**故意**的（诊断时需要），但密钥不能落盘 —— 这条不能省。
    """
    out: Dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        return out
    for k, v in items:
        out[str(k)] = "***已隐去***" if str(k).lower() in _SECRET_HEADERS else str(v)
    return out


def summary(d: Dict[str, Any]) -> str:
    """把字典收成一行（用于 SSE 帧那样的短打点）。"""
    return " ".join(f"{k}={clamp(v, 120)}" for k, v in d.items())
