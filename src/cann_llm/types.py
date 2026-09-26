"""后端与上层之间的数据契约。

设计原则
--------
* **流式优先**：后端只实现 ``generate() -> Iterator[GenerationChunk]``；
  非流式就是"把分块收干"。这样 OpenAI 接口的 ``stream=true/false`` 两条路径
  共用同一套逻辑，不需要各写一遍。
* **不可变 + 纯数据**：全部是 dataclass，方便序列化、比较与测试。
* **面向未来**：``GenerationChunk.token_id`` / ``index`` 为将来接
  token 级 API（logprobs、tokenize 接口）预留；``GenerationRequest`` 预留
  ``prompt_token_ids`` / ``lora`` 等字段位置。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, List, Optional, Sequence, Tuple

# 结束原因（与 OpenAI 对齐）
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_ERROR = "error"


@dataclass(frozen=True)
class GenerationParams:
    """采样参数。默认值与官方示例 context.json 的推荐值保持一致。"""

    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    seed: Optional[int] = None
    #: 额外停止串（引擎自身的 stop_sequence 之外）
    stop: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens 必须 >= 1")
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if self.top_k < 0:
            raise ValueError("top_k 必须 >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须落在 (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty 必须 > 0")

    @property
    def greedy(self) -> bool:
        """温度为 0 视为贪心解码。"""
        return self.temperature <= 0


@dataclass(frozen=True)
class GenerationRequest:
    """一次生成请求。

    当前只用到 ``prompt``（纯文本）；``prompt_token_ids`` 为将来接
    预分词路径预留。
    """

    prompt: str
    params: GenerationParams = field(default_factory=GenerationParams)
    prompt_token_ids: Optional[Sequence[int]] = None

    def with_params(self, **kw) -> "GenerationRequest":
        return replace(self, params=replace(self.params, **kw))


@dataclass(frozen=True)
class GenerationChunk:
    """流式输出的一个增量。

    ``text`` 是**新增的文本**（不是累积文本）—— 上层直接写出去即可。
    最后一个分块的 ``finish_reason`` 非空。
    """

    text: str = ""
    index: int = 0
    token_id: Optional[int] = None
    finish_reason: Optional[str] = None
    #: 仅最终分块携带（用量与耗时），中间分块为 None
    stats: Optional["GenerationStats"] = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


@dataclass(frozen=True)
class GenerationStats:
    """一次推理的用量与耗时。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    wall_s: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        dec = self.decode_ms / 1000.0
        return self.completion_tokens / dec if dec > 0 else 0.0

    def as_openai_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass(frozen=True)
class GenerationResult:
    """非流式结果（由分块聚合而来）。"""

    text: str
    finish_reason: str = FINISH_STOP
    stats: GenerationStats = field(default_factory=GenerationStats)
    chunk_count: int = 0


def aggregate(chunks: Iterator[GenerationChunk], stats: GenerationStats | None = None) -> GenerationResult:
    """把分块流收干成一个 :class:`GenerationResult`。

    这是"非流式"路径的唯一实现，保证与流式路径行为一致。
    """
    parts: List[str] = []
    reason = FINISH_STOP
    n = 0
    seen_stats: Optional[GenerationStats] = None
    for c in chunks:
        parts.append(c.text)
        n += 1
        if c.stats is not None:
            seen_stats = c.stats
        if c.finish_reason:
            reason = c.finish_reason
    return GenerationResult(text="".join(parts), finish_reason=reason,
                            stats=seen_stats or stats or GenerationStats(),
                            chunk_count=n)


@dataclass(frozen=True)
class ToolCall:
    """模型请求的一次工具调用。

    ``id`` 由本服务生成（模型的输出里没有 id），用于把 tool 结果与请求对应起来。
    """

    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""
    #: 模型原始输出片段，出问题时便于排查
    raw: str = ""

    def to_openai(self) -> dict:
        """转成 OpenAI ``message.tool_calls`` 的条目。"""
        import json as _json

        return {
            "id": self.id or "call_0",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": _json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的结果。``ok=False`` 时内容里是错误说明。"""

    tool_call_id: str
    name: str
    content: str
    ok: bool = True

    def to_openai(self) -> dict:
        return {"role": "tool", "tool_call_id": self.tool_call_id,
                "name": self.name, "content": self.content}


@dataclass(frozen=True)
class ModelInfo:
    """后端对外声明的模型元信息，用于 /v1/models 与日志。"""

    id: str
    backend: str
    path: str = ""
    context_length: int = 0
    max_tokens: int = 0
    chat_template: str = "chatml"
    vocab_size: int = 0
    extra: dict = field(default_factory=dict)
