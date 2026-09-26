"""测试用的假后端，避免依赖 NPU / 模型文件。"""

from __future__ import annotations

from typing import Iterator, List, Optional

from cann_llm.backends.base import EngineBackend, register_backend
from cann_llm.types import (
    FINISH_STOP,
    GenerationChunk,
    GenerationRequest,
    GenerationStats,
    ModelInfo,
)


class FakeBackend(EngineBackend):
    """把输入的 prompt 按固定规则「回答」，逐字符产出分块。"""

    name = "fake"

    def __init__(self, reply: str = "hello world", *, model_dir: str = "",
                 raise_on_generate: Optional[BaseException] = None,
                 stats: Optional[GenerationStats] = None, **kwargs):
        self.reply = reply
        self.model_dir = model_dir or "/tmp/fake"
        self.raise_on_generate = raise_on_generate
        self.stats = stats or GenerationStats(prompt_tokens=3, completion_tokens=len(reply),
                                              prefill_ms=10.0, decode_ms=100.0,
                                              total_ms=110.0, wall_s=0.11)
        self.loaded = False
        self.closed = False
        self.requests: List[GenerationRequest] = []

    def load(self) -> ModelInfo:
        self.loaded = True
        self._info = ModelInfo(id="fake-model", backend="fake", path=self.model_dir,
                               context_length=2048, chat_template="chatml")
        return self._info

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        if not self.loaded:
            self.load()
        self.requests.append(request)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        for i, ch in enumerate(self.reply):
            yield GenerationChunk(text=ch, index=i)
        yield GenerationChunk(index=len(self.reply), finish_reason=FINISH_STOP,
                              stats=self.stats)

    def close(self) -> None:
        self.closed = True

    def count_prompt_tokens(self, text: str) -> int:
        # 便于测试裁剪逻辑：按空格切
        return len(text.split())


class ScriptedBackend(FakeBackend):
    """按预设脚本依次返回回复 —— 用来模拟「先发起工具调用，再给出回答」。"""

    name = "scripted"

    def __init__(self, replies, **kwargs):
        super().__init__("", **kwargs)
        self.replies = list(replies)
        self.call_count = 0
        self.prompts: List[str] = []

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        self.requests.append(request)
        self.prompts.append(request.prompt)
        idx = min(self.call_count, len(self.replies) - 1)
        reply = self.replies[idx]
        self.call_count += 1
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        # 逐字符产出，顺便压测 StreamFilter
        for i, ch in enumerate(reply):
            yield GenerationChunk(text=ch, index=i)
        yield GenerationChunk(index=len(reply), finish_reason=FINISH_STOP,
                              stats=self.stats)


class ExplodingBackend(FakeBackend):
    """生成中途抛异常，用于验证错误传播。"""

    name = "exploding"

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text="partial")
        raise RuntimeError("boom")


# 注册一个可被 create_backend("fake") 取到的测试后端
register_backend("fake")(FakeBackend)
