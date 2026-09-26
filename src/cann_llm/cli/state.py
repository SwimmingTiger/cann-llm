"""CLI 会话状态。

单独成模块，避免 chat.py（主循环）与 _commands.py（命令处理）循环 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..chat.template import Message
from ..types import GenerationParams, GenerationStats


@dataclass
class CliState:
    """一次交互式会话的全部可变状态。"""

    messages: List[Message] = field(default_factory=list)
    params: GenerationParams = field(default_factory=GenerationParams)
    system_prompt: Optional[str] = None
    #: 启用的工具名（空 = 退化成普通对话）
    tool_names: List[str] = field(default_factory=list)
    stream: bool = True
    model_id: str = ""
    backend: str = ""
    #: 最近一轮的用量与结束原因
    last_stats: Optional[GenerationStats] = None
    last_finish_reason: Optional[str] = None
    last_steps: int = 0

    def reset(self) -> None:
        self.messages.clear()
        self.last_stats = None
        self.last_finish_reason = None
        self.last_steps = 0

    def set_system(self, text: Optional[str]) -> None:
        self.system_prompt = text
        self.reset()

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")
