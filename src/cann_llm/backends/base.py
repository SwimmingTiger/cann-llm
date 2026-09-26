"""推理后端协议与注册表。

设计意图
--------
上层（CLI / HTTP 服务）**只认** :class:`EngineBackend` 协议，不 import 任何具体后端。
所以将来加 vLLM / llama.cpp / 远端 OpenAI 后端时，只需要：

1. 新建 ``backends/xxx.py``，实现 ``EngineBackend``；
2. 用 ``@register_backend("xxx")`` 注册；
3. 在 ``backends/__init__.py`` 里 import 一次（触发注册）。

上层零改动。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional

from ..errors import BusyError, InvalidRequestError
from ..types import GenerationChunk, GenerationRequest, ModelInfo

__all__ = [
    "EngineBackend",
    "SerializedBackend",
    "register_backend",
    "create_backend",
    "available_backends",
    "backend_class",
]


class EngineBackend(ABC):
    """一次「模型加载 + 自回归生成」的抽象。

    实现方约定：

    * :meth:`load` 幂等，可在进程启动时调用一次；
    * :meth:`generate` 是**生成器**，逐块产出增量文本，最后一块带 ``finish_reason``；
    * 生成失败应当抛 :class:`~cann_llm.errors.GenerationError` 及其子类，
      而不是把异常吞掉后正常结束 —— 上层据此决定是否报 500；
    * :meth:`close` 必须可重入（进程退出时可能被调多次）。
    """

    #: 后端标识，用于配置里的 ``backend = "xxx"``
    name: str = "base"

    @abstractmethod
    def load(self) -> ModelInfo:
        """加载模型，返回元信息。失败抛 ModelLoadError。"""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        """按请求流式生成。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""

    # ---- 可选能力（默认实现给出保守行为）----

    def count_prompt_tokens(self, text: str) -> int:
        """估算输入 token 数。默认返回 0（未知）。"""
        return 0

    @property
    def info(self) -> ModelInfo:
        """模型元信息，``load()`` 之后有效。"""
        return getattr(self, "_info", ModelInfo(id="unknown", backend=self.name))

    @property
    def supports_streaming(self) -> bool:
        """是否支持真正的逐 token 回调（否则上层会以整块形式收到）。"""
        return False


class SerializedBackend(EngineBackend):
    """把只支持单路推理的后端串行化。

    CANN LLM Engine 一个 Executor 同时只能跑一路 Generate，因此 HTTP 服务必须
    串行化。把这件事放在后端层而不是服务层的好处是：将来接上支持并发的后端
    （vLLM 等）时，直接不用这层包装即可，上层无感。
    """

    def __init__(self, inner: EngineBackend, max_queue: int = 8, timeout_s: float = 120.0):
        self._inner = inner
        self._max_queue = max(1, int(max_queue))
        self._timeout = float(timeout_s)
        self._lock = threading.Lock()
        self._waiting = 0
        self._guard = threading.Lock()

    # ---- 透传 ----
    @property
    def name(self) -> str:            # type: ignore[override]
        return self._inner.name

    def load(self) -> ModelInfo:
        return self._inner.load()

    def close(self) -> None:
        self._inner.close()

    def count_prompt_tokens(self, text: str) -> int:
        return self._inner.count_prompt_tokens(text)

    @property
    def info(self) -> ModelInfo:
        return self._inner.info

    @property
    def supports_streaming(self) -> bool:
        return self._inner.supports_streaming

    # ---- 串行化 ----
    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        with self._guard:
            if self._waiting >= self._max_queue:
                raise BusyError(f"队列已满（max_queue={self._max_queue}），请稍后重试")
            self._waiting += 1
        acquired = self._lock.acquire(timeout=self._timeout)
        with self._guard:
            self._waiting -= 1
        if not acquired:
            raise BusyError(f"排队超过 {self._timeout:.0f}s，请稍后重试")
        try:
            yield from self._inner.generate(request)
        finally:
            self._lock.release()


# ------------------------------------------------------------------ 注册表

_REGISTRY: Dict[str, type] = {}


def register_backend(name: str) -> Callable[[type], type]:
    """类装饰器：把后端实现登记到注册表。"""

    def deco(cls: type) -> type:
        if not isinstance(cls, type):
            raise TypeError("register_backend 只能用于类")
        _REGISTRY[name] = cls
        cls.name = name            # type: ignore[attr-defined]
        return cls

    return deco


def backend_class(name: str) -> type:
    if name not in _REGISTRY:
        raise InvalidRequestError(
            f"未知后端 {name!r}；可用：{', '.join(available_backends()) or '(无)'}")
    return _REGISTRY[name]


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, **kwargs) -> EngineBackend:
    """按名字实例化后端。"""
    return backend_class(name)(**kwargs)  # type: ignore[call-arg]
