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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from ..errors import InvalidRequestError

#: 允许的角色（与 OpenAI 对齐；tool 先放行，等接入工具调用再细化）
ROLES = ("system", "user", "assistant", "tool")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise InvalidRequestError(f"不支持的角色 {self.role!r}；可用：{', '.join(ROLES)}")


class ChatTemplate(ABC):
    """对话模板协议。"""

    name: str = "base"

    @abstractmethod
    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True) -> str:
        """渲染成提示文本。

        :param add_generation_prompt: 是否在末尾追加「轮到 assistant 说话」的引导段。
        """

    def stop_strings(self) -> Sequence[str]:
        """该模板对应的停止串（会额外传给引擎的 stop_sequence）。"""
        return ()

    @property
    def default_system(self) -> str:
        return "You are a helpful assistant."


class ChatMLTemplate(ChatTemplate):
    """Qwen2.5 / Qwen3 的 ChatML 模板。"""

    name = "chatml"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True) -> str:
        parts: List[str] = []
        for m in messages:
            parts.append(f"{IM_START}{m.role}\n{m.content}{IM_END}\n")
        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)

    def stop_strings(self) -> Sequence[str]:
        return (IM_END,)


class PlainTemplate(ChatTemplate):
    """不包装：把各条消息内容直接拼接。"""

    name = "plain"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True) -> str:
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
