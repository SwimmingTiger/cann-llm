"""对话模板。

模板负责把「消息列表」渲染成模型能吃的纯文本。CANN LLM Engine 只接受文本，
不接受结构化消息，所以这一步必须在客户端完成。

内置：
* ``chatml`` —— Qwen2.5 / Qwen3 系列（``<|im_start|>role\\n…<|im_end|>``），当前默认
* ``plain``  —— 不做任何包装，直接把消息内容拼起来（调试 / 续写用）

要加新模板（GLM、HunYuan 等）：实现 :class:`ChatTemplate` 并用
:func:`register_template` 注册即可，上层零改动。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..errors import InvalidRequestError
from ..types import ToolCall

#: 允许的角色（与 OpenAI 对齐；tool 先放行，等接入工具调用再细化）
ROLES = ("system", "user", "assistant", "tool")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class Message:
    """一条对话消息。

    ``tool_calls`` 只对 assistant 有意义（模型请求调用的工具）；
    ``tool_call_id`` / ``name`` 只对 role=tool 有意义（工具结果回填）。
    """

    role: str
    content: str
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise InvalidRequestError(f"不支持的角色 {self.role!r}；可用：{', '.join(ROLES)}")


class ChatTemplate(ABC):
    """对话模板协议。"""

    name: str = "base"

    @abstractmethod
    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        """渲染成提示文本。

        :param add_generation_prompt: 是否在末尾追加「轮到 assistant 说话」的引导段。
        :param tools: OpenAI 格式的工具声明；非空时会把「可用工具 + 调用格式」
            注入 system 段（与 Qwen 官方 chat template 的行为一致）。
        """

    def stop_strings(self) -> Sequence[str]:
        """该模板对应的停止串（会额外传给引擎的 stop_sequence）。"""
        return ()

    @property
    def default_system(self) -> str:
        return "You are a helpful assistant."


#: Qwen 官方 chat template 里注入的工具说明（逐字对齐，改动需谨慎）
TOOLS_PREAMBLE = (
    "\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
    "\n<tools>"
)
TOOLS_EPILOGUE = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def render_tools_block(tools: Sequence[Dict[str, Any]]) -> str:
    """把工具声明渲染成官方模板要求的 ``<tools>…</tools>`` 段。"""
    lines = [TOOLS_PREAMBLE]
    for t in tools:
        lines.append("\n" + json.dumps(t, ensure_ascii=False))
    lines.append(TOOLS_EPILOGUE)
    return "".join(lines)


def render_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    """assistant 消息里的工具调用段。"""
    out = []
    for tc in tool_calls:
        payload = {"name": tc.name, "arguments": tc.arguments}
        out.append(f"\n{TOOL_CALL_OPEN}\n"
                   f"{json.dumps(payload, ensure_ascii=False)}"
                   f"\n{TOOL_CALL_CLOSE}")
    return "".join(out)


class ChatMLTemplate(ChatTemplate):
    """Qwen2.5 / Qwen3 的 ChatML 模板（含 function calling 支持）。"""

    name = "chatml"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        parts: List[str] = []
        tools_block = render_tools_block(tools) if tools else ""
        tools_emitted = False
        n = len(messages)
        i = 0
        while i < n:
            m = messages[i]

            if m.role == "tool":
                # 连续多条 tool 结果合并进同一个 user 轮次（对齐官方模板）
                parts.append(f"{IM_START}user")
                while i < n and messages[i].role == "tool":
                    parts.append(f"\n{TOOL_RESPONSE_OPEN}\n"
                                 f"{messages[i].content}\n{TOOL_RESPONSE_CLOSE}")
                    i += 1
                parts.append(f"{IM_END}\n")
                continue

            content = m.content
            is_first_system = m.role == "system" and not tools_emitted
            if is_first_system and tools_block:
                content = content + tools_block
                tools_emitted = True
            if m.role == "assistant" and m.tool_calls:
                content = content + render_tool_calls(m.tool_calls)
            parts.append(f"{IM_START}{m.role}\n{content}{IM_END}\n")
            i += 1

        # 没有 system 消息但带了工具 → 补一个 system 轮次承载工具说明
        if tools_block and not tools_emitted:
            parts.insert(0, f"{IM_START}system\n{tools_block.lstrip(chr(10))}{IM_END}\n")

        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)

    def stop_strings(self) -> Sequence[str]:
        return (IM_END,)

    @property
    def tool_call_tags(self) -> Sequence[str]:
        return (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)


class PlainTemplate(ChatTemplate):
    """不包装：把各条消息内容直接拼接。"""

    name = "plain"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        return "".join(m.content for m in messages)


_REGISTRY: Dict[str, Callable[[], ChatTemplate]] = {}


def register_template(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


# 注册内置模板
register_template("chatml")(ChatMLTemplate)
register_template("plain")(PlainTemplate)


def get_template(name: str) -> ChatTemplate:
    if name not in _REGISTRY:
        raise InvalidRequestError(
            f"未知对话模板 {name!r}；可用：{', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]()


def available_templates() -> List[str]:
    return sorted(_REGISTRY)
