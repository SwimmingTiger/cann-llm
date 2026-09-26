"""内置工具。

设计取舍
--------
默认集合**只包含无副作用的工具**（时间、计算器）。任何能触达外部世界的
（HTTP 抓取）都标记 ``dangerous=True``，默认不进 ``select()`` 结果，
必须由使用方显式打开。

刻意**不提供** shell / 读写文件 / 任意网络请求这类工具：那些能力一旦默认
打开，一个会写 prompt 的攻击面就能变成 RCE。需要的话请在应用侧自己注册，
并自行评估风险（``ToolRegistry.register`` 就是为此准备的）。

``http_get`` 做了基本的 SSRF 防护：只允许 http/https、解析后的地址不能是
私有/回环/链路本地网段（除非显式放行）、限制响应体大小与超时。
"""

from __future__ import annotations

import ast
import datetime as _dt
import ipaddress
import json
import operator
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .registry import ToolError, ToolRegistry

__all__ = ["register_into", "BUILTIN_NAMES"]

#: 默认启用（无副作用）
SAFE_NAMES = ("get_current_time", "calculator")
#: 需要显式打开
DANGEROUS_NAMES = ("http_get",)

BUILTIN_NAMES = SAFE_NAMES + DANGEROUS_NAMES


# ------------------------------------------------------------------ 时间


def _get_current_time(timezone_offset_hours: float = 8.0,
                      fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    tz = _dt.timezone(_dt.timedelta(hours=timezone_offset_hours))
    now = _dt.datetime.now(tz)
    return json.dumps({
        "datetime": now.strftime(fmt),
        "iso8601": now.isoformat(timespec="seconds"),
        "timezone_offset_hours": timezone_offset_hours,
        "weekday": now.strftime("%A"),
    }, ensure_ascii=False)


# ------------------------------------------------------------------ 计算器

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "pow": pow,
}
_MAX_POW = 10 ** 6


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError("表达式里只允许数字常量")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op is operator.pow and abs(right) > 1000:
            raise ToolError("指数过大")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的一元运算: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolError("只允许调用白名单里的普通函数，不允许属性调用")
        if node.func.id not in _ALLOWED_FUNCS:
            raise ToolError(f"不允许调用 {node.func.id!r}；"
                            f"可用: {', '.join(sorted(_ALLOWED_FUNCS))}")
        if node.keywords:
            raise ToolError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e) for e in node.elts]
    raise ToolError(f"表达式里不允许出现 {type(node).__name__}")


def _calculator(expression: str) -> str:
    """安全的算术求值（走 AST 白名单，不用 eval）。"""
    if len(expression) > 500:
        raise ToolError("表达式过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"表达式语法错误: {e.msg}") from e
    try:
        value = _eval_node(tree)
    except (ArithmeticError, ValueError, TypeError) as e:
        # 转成 ToolError，让 agent 循环把错误回给模型自我修正，而不是崩掉
        raise ToolError(f"计算失败: {type(e).__name__}: {e}") from e
    return json.dumps({"expression": expression, "result": value}, ensure_ascii=False)


# ------------------------------------------------------------------ HTTP（危险）

_ALLOWED_SCHEMES = ("http", "https")


def _is_forbidden_host(host: str, allow_private: bool) -> bool:
    """SSRF 防护：私有/回环/链路本地/保留地址一律拒绝。"""
    if allow_private:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True                      # 解析不了就当危险
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def _http_get(url: str, max_bytes: int = 65536, timeout_s: float = 8.0,
              allow_private_network: bool = False) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ToolError(f"只允许 {_ALLOWED_SCHEMES}，收到 {parsed.scheme!r}")
    if not parsed.hostname:
        raise ToolError("URL 缺少主机名")
    if _is_forbidden_host(parsed.hostname, allow_private_network):
        raise ToolError(
            f"拒绝访问 {parsed.hostname}：解析到私有/回环/保留地址（SSRF 防护）。"
            "确需访问内网时把 allow_private_network 设为 true。")

    # 不允许跟随到非 http(s) 的重定向
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
            if urllib.parse.urlparse(newurl).scheme not in _ALLOWED_SCHEMES:
                raise ToolError(f"重定向到不允许的协议: {newurl}")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "cann-llm/agent"})
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            raw = raw[:max_bytes]
            charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, "replace")
            return json.dumps({
                "url": url,
                "status": resp.status,
                "content_type": resp.headers.get("Content-Type", ""),
                "truncated": truncated,
                "text": text,
            }, ensure_ascii=False)
    except urllib.error.HTTPError as e:
        raise ToolError(f"HTTP {e.code}: {e.reason}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ToolError(f"请求失败: {e}") from e


# ------------------------------------------------------------------ 注册

def register_into(registry: ToolRegistry) -> None:
    """把内置工具注册进给定注册表。"""
    registry.register(
        name="get_current_time",
        description="获取当前日期与时间。",
        parameters={
            "type": "object",
            "properties": {
                "timezone_offset_hours": {
                    "type": "number",
                    "description": "相对 UTC 的时区偏移小时数，中国为 8，默认 8",
                    "minimum": -12, "maximum": 14,
                },
                "fmt": {
                    "type": "string",
                    "description": "strftime 格式串，默认 %Y-%m-%d %H:%M:%S",
                },
            },
        },
        tags=("time",),
    )(_get_current_time)

    registry.register(
        name="calculator",
        description="计算一个算术表达式，支持 + - * / // % ** 与 "
                    "abs/round/min/max/sum/int/float/pow。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的表达式，例如 (23*7+11)/4",
                    "maxLength": 500,
                },
            },
            "required": ["expression"],
        },
        notes="只做算术，不能访问变量或调用其它函数。"
              "注意：百分数请写成 15/100 这样的除法，不要写 15%（后者不是合法表达式）。",
        tags=("math",),
    )(_calculator)

    registry.register(
        name="http_get",
        description="抓取一个 http/https URL 并返回文本内容（截断到 64KB）。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的 URL"},
                "max_bytes": {"type": "integer", "minimum": 256,
                              "maximum": 262144, "description": "最多读取的字节数"},
                "timeout_s": {"type": "number", "minimum": 0.5, "maximum": 30,
                              "description": "超时秒数"},
                "allow_private_network": {
                    "type": "boolean",
                    "description": "允许访问私有网段（默认 false，存在 SSRF 风险）",
                },
            },
            "required": ["url"],
        },
        dangerous=True,
        timeout_s=30.0,
        notes="会发起真实网络请求；默认不启用。私有网段默认被拒绝。",
        tags=("network",),
    )(_http_get)
