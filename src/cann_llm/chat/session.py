"""多轮会话。

引擎本身**不保存对话历史**：每次生成都要把完整 transcript 重新发过去。
好处是引擎会复用公共前缀的 KV 缓存（实测多轮时第二轮明显更快）。

本模块负责：
* 维护消息列表；
* 渲染时按 ``max_prompt_tokens`` 裁剪过老的历史（从最旧的 user/assistant 对开始丢）；
* 流式生成并在结束后把回复写回历史。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

from ..backends.base import EngineBackend
from ..types import (
    GenerationChunk,
    GenerationParams,
    GenerationRequest,
    GenerationStats,
)
from .template import ChatTemplate, Message, get_template


@dataclass
class ChatSession:
    """一次对话（可多轮）。"""

    backend: EngineBackend
    template: ChatTemplate
    system_prompt: Optional[str] = None
    params: GenerationParams = field(default_factory=GenerationParams)
    #: 渲染时的输入上限；超出就丢最旧的一轮
    max_prompt_tokens: int = 1800

    messages: List[Message] = field(default_factory=list)

    #: 最近一轮的用量与耗时（由 ask() 填充）
    last_stats: Optional[GenerationStats] = None
    #: 最近一轮的结束原因（stop / length / error）
    last_finish_reason: Optional[str] = None

    # ------------------------------------------------------------ 历史

    def reset(self) -> None:
        self.messages.clear()

    def set_system(self, text: Optional[str]) -> None:
        self.system_prompt = text
        self.reset()

    def _effective(self) -> List[Message]:
        msgs: List[Message] = list(self.messages)
        if self.system_prompt:
            msgs.insert(0, Message("system", self.system_prompt))
        return msgs

    def _trim(self, msgs: List[Message]) -> List[Message]:
        """从最旧的一轮开始丢，直到估算长度落回上限内。"""
        if self.max_prompt_tokens <= 0:
            return msgs
        while len(msgs) > 1:
            est = self.backend.count_prompt_tokens(
                self.template.render(msgs, add_generation_prompt=True))
            if est <= self.max_prompt_tokens:
                break
            # 保留 system，从第二条开始丢
            drop = 1 if msgs[0].role == "system" else 0
            del msgs[drop]
        return msgs

    def render(self, add_generation_prompt: bool = True) -> str:
        return self.template.render(self._trim(self._effective()),
                                    add_generation_prompt=add_generation_prompt)

    # ------------------------------------------------------------ 生成

    def ask(self, user_text: str,
            params: Optional[GenerationParams] = None) -> Iterator[GenerationChunk]:
        """流式提问。

        迭代过程中 ``yield`` 的是 :class:`GenerationChunk`；
        **迭代结束（或提前 break）后**，本轮问答会被写入历史 ——
        提前 break 时保存的是已收到的部分，避免历史与实际不符。
        """
        params = params or self.params
        self.messages.append(Message("user", user_text))
        prompt = self.render()
        collected: List[str] = []
        try:
            for chunk in self.backend.generate(GenerationRequest(prompt=prompt, params=params)):
                if chunk.text:
                    collected.append(chunk.text)
                if chunk.stats is not None:
                    self.last_stats = chunk.stats
                if chunk.finish_reason:
                    self.last_finish_reason = chunk.finish_reason
                yield chunk
        finally:
            self.messages.append(Message("assistant", "".join(collected)))

    def ask_sync(self, user_text: str,
                 params: Optional[GenerationParams] = None) -> str:
        """非流式封装：收干分块后返回完整回复。"""
        last = None
        for chunk in self.ask(user_text, params=params):
            last = chunk
        # 历史里已写入完整回复，直接取最后一条 assistant
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""

    @property
    def turn_count(self) -> int:
        """已完成的问答轮数。"""
        return sum(1 for m in self.messages if m.role == "user")
