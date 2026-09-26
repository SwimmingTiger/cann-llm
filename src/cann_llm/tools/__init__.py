"""工具（function calling）支持。

导入本包会注册内置工具（见 ``builtin.py``）。内置集合刻意保守，只包含
无副作用的工具；能触达外部世界的（如 HTTP 抓取）标记为 ``dangerous``，
默认不启用，需要显式选择。
"""

from .registry import Tool, ToolError, ToolRegistry, default_registry  # noqa: F401
from .schema import SchemaError, validate  # noqa: F401

__all__ = ["Tool", "ToolRegistry", "ToolError", "default_registry",
           "SchemaError", "validate"]
