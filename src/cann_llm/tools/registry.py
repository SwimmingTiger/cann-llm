"""工具注册表。

一个「工具」= JSON Schema 描述 + 一个 Python 可调用对象。描述部分直接就是
OpenAI 的 ``tools`` 条目格式，因此可以原样塞进 prompt 或返回给客户端。

安全约定（重要）
----------------
* :class:`Tool` 有 ``dangerous`` 标记；默认只启用 ``dangerous=False`` 的工具。
* **本框架不内置任何能执行 shell / 写文件 / 访问任意网络的工具。**
  需要的话请自己在应用侧注册，并显式通过 ``enabled`` 打开 —— 这是个有意的
  设计选择：让「能干什么」永远由使用方明确决定，而不是框架默认给一堆能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .schema import SchemaError, validate

__all__ = ["Tool", "ToolRegistry", "default_registry", "ToolError"]


class ToolError(RuntimeError):
    """工具执行失败（会被包成 tool_response 回给模型，而不是中断对话）。"""


@dataclass
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    #: True 表示有副作用或能触达外部世界，默认不启用
    dangerous: bool = False
    #: 单次调用的超时（秒），None 表示不限制
    timeout_s: Optional[float] = None
    #: 给模型看的额外约束说明（会写进 description）
    notes: str = ""
    tags: Sequence[str] = field(default_factory=tuple)

    def schema(self) -> Dict[str, Any]:
        """转成 OpenAI 的 ``tools`` 条目。"""
        desc = self.description
        if self.notes:
            desc = f"{desc}\n\n{self.notes}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: Dict[str, Any]) -> Any:
        """校验参数并执行。参数不符 schema 时抛 :class:`SchemaError`。"""
        res = validate(arguments, self.parameters)
        res.raise_if_bad()
        return self.handler(**arguments)


class ToolRegistry:
    """工具集合。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    # ---- 增删查 ----
    def add(self, tool: Tool) -> Tool:
        if not tool.name:
            raise ValueError("工具必须有 name")
        self._tools[tool.name] = tool
        return tool

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 *, dangerous: bool = False, timeout_s: Optional[float] = None,
                 notes: str = "", tags: Sequence[str] = ()) -> Callable:
        """装饰器：把函数注册成工具。"""

        def deco(fn: Callable) -> Callable:
            self.add(Tool(name=name, description=description, parameters=parameters,
                          handler=fn, dangerous=dangerous, timeout_s=timeout_s,
                          notes=notes, tags=tags))
            return fn

        return deco

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # ---- 选择与导出 ----
    def select(self, names: Optional[Sequence[str]] = None, *,
               include_dangerous: bool = False) -> "ToolRegistry":
        """挑出一个子集。

        :param names: 指定工具名；``None`` 表示「全部安全的」。
        :param include_dangerous: 为 ``True`` 时安全过滤不生效
            （仅在 names 显式列出的工具上允许）。
        """
        if names is None:
            picked = [t for t in self._tools.values()
                      if include_dangerous or not t.dangerous]
        else:
            picked = []
            for n in names:
                t = self._tools.get(n)
                if t is None:
                    raise KeyError(f"未注册的工具: {n}（可用: {self.names()}）")
                if t.dangerous and not include_dangerous:
                    raise PermissionError(
                        f"工具 {n!r} 被标记为 dangerous，需显式 include_dangerous=True")
                picked.append(t)
        return ToolRegistry(picked)

    def openai_schemas(self) -> List[Dict[str, Any]]:
        """转成 OpenAI ``tools`` 数组。"""
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"没有名为 {name!r} 的工具；可用: {self.names()}")
        return tool.call(arguments)


# ------------------------------------------------------------------ 默认集合访问

_default: Optional[ToolRegistry] = None


def default_registry() -> ToolRegistry:
    """进程级默认注册表（内置工具在首次访问时注册）。"""
    global _default
    if _default is None:
        _default = ToolRegistry()
        # 内置工具由包入口（tools/__init__.py）负责注册，这里不反向依赖，
        # 避免 registry 与具体工具耦合。
    return _default


__all__ += ["validate", "SchemaError"]
