"""CANN LLM Engine（鸿蒙 NPU）后端。

通过 ``ctypes`` 直接调用 NDK 动态库 ``/system/lib64/ndk/libcann_llm_engine.so``，
不需要 HAP，也不需要 root。

.. warning::

   本模块里的若干"怪癖处理"是实测踩出来的，**改动前请务必读** ``docs/cann-engine-notes.md``：

   1. ``HMS_LLMEngineExecutor_Generate`` 的第 3 个参数是 **prompt 文本**，
      不是 ``Prompt*``。传 ``Prompt*`` 会让包装函数把对象的原始字节当 C 字符串读
      （首字符是 libc++ string 标识位），prompt 塌缩成 1 个 token。
   2. ``HMS_LLMEngine_Context_Destroy`` **会崩溃**（指针语义问题），一律不调用。
      只新建、不释放；Context 很小，用上限为 :data:`CONTEXT_CACHE_MAX` 的缓存约束增长。
   3. ``GetAllTokenGeneration`` / ``GetAllTokenGenerationLen`` **会引起堆破坏**
      （同进程多次 Generate 后 core dump），一律不使用。
      逐字输出改用 one-token 回调 + ``GetAllGeneration``。
   4. Context 必须**复用**：引擎会复用公共前缀的 KV 缓存，多轮对话第二轮明显更快；
      prompt 前缀不匹配时引擎会自行重新 prefill。
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
from typing import Dict, Iterator, Optional, Tuple

from ..errors import (
    BackendUnavailableError,
    GenerationError,
    InvalidRequestError,
    ModelLoadError,
)
from ..types import (
    FINISH_LENGTH,
    FINISH_STOP,
    GenerationChunk,
    GenerationParams,
    GenerationRequest,
    GenerationStats,
    ModelInfo,
)
from ..version import CANN_NDK_LIB
from .base import EngineBackend, register_backend

#: Qwen 系列结束符（由设备 tokenizer.json 的 added_tokens 确认 = 151645）
IM_END = "<|im_end|>"

#: Context 缓存上限。引擎不让我们销毁 Context，所以只能限制新建数量：
#: 每个「采样参数组合」对应一个常驻 Context，超出上限时新的一律新建但不再缓存。
CONTEXT_CACHE_MAX = 8

#: 用于把回调里的异常带回生成线程
_QUEUE_TIMEOUT_S = 0.5


class _NdkBindings:
    """``libcann_llm_engine.so`` 的 ctypes 绑定。

    单独一层，方便测试时替换/打桩。
    """

    def __init__(self, lib_path: str = CANN_NDK_LIB):
        if not os.path.exists(lib_path):
            raise BackendUnavailableError(
                f"找不到 CANN NDK 库 {lib_path}；本后端需要鸿蒙设备（NPU）环境")
        try:
            self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        except OSError as e:
            raise BackendUnavailableError(f"加载 {lib_path} 失败: {e}") from e

        self.context_create = self._bind(
            "HMS_LLMEngineContext_CreateFromContextJson",
            ctypes.c_void_p, [ctypes.c_char_p])
        self.executor_create = self._bind(
            "HMS_LLMEngineExecutor_CreateFromExecutorJson",
            ctypes.c_void_p, [ctypes.c_char_p])
        self.generate = self._bind(
            "HMS_LLMEngineExecutor_Generate",
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p])

        self.get_generation_len = self._bind(
            "HMS_LLMEngineContext_GetAllGenerationLen",
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)])
        self.get_generation = self._bind(
            "HMS_LLMEngineContext_GetAllGeneration",
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint])
        self.get_input_token_count = self._bind(
            "HMS_LLMEngineContext_GetInputTokenCount",
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)])
        self.get_output_token_count = self._bind(
            "HMS_LLMEngineContext_GetOutputTokenCount",
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)])

        self._time_funcs = {}
        for name in ("Prefill", "Decode", "Total"):
            self._time_funcs[name.lower()] = self._bind(
                f"HMS_LLMEngineContext_Get{name}TimeMs",
                ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)])

        # one-token 回调：void (*)(const HIAI_LLMEngine_Context*)
        self.callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self.set_on_token = self._bind(
            "HMS_LLMEngineContext_SetOnOneTokenGenerateDoneFunc",
            ctypes.c_int, [ctypes.c_void_p, self.callback_type])
        # 注：不绑定 HMS_LLMEngine_Context_Destroy（会崩，见模块 docstring）

    def _bind(self, name: str, restype, argtypes):
        try:
            fn = getattr(self.lib, name)
        except AttributeError as e:
            raise BackendUnavailableError(f"库中缺少符号 {name}") from e
        fn.restype = restype
        fn.argtypes = argtypes
        return fn


@register_backend("cann")
class CannNdkBackend(EngineBackend):
    """华为 CANN LLM Engine 的 NDK 后端。

    :param model_dir: 模型目录，需含 ``executor.json`` / ``context.json`` /
        ``tokenizer.json`` / ``*.omc`` / ``SubGraph_0.weight`` / embedding 文件。
    :param model_id: 对外暴露的模型 id（OpenAI ``model`` 字段）。
    :param context_length: 上下文长度，仅用于元信息展示。
    """

    def __init__(
        self,
        model_dir: str,
        model_id: str = "qwen2.5-1.5b",
        lib_path: str = CANN_NDK_LIB,
        context_length: int = 2048,
        default_params: Optional[GenerationParams] = None,
    ):
        self.model_dir = os.path.abspath(model_dir) if model_dir else ""
        self.model_id = model_id
        self.context_length = context_length
        self.default_params = default_params or GenerationParams()

        self._ndk: Optional[_NdkBindings] = None
        self._lib_path = lib_path
        self._executor = None
        self._contexts: Dict[Tuple, int] = {}      # 参数指纹 -> ctx 指针
        self._ctx_lock = threading.Lock()
        self._callback_ref = None                  # 必须持有引用，否则被 GC 后崩溃
        self._sink = None                          # 当前请求的增量消费者
        self._emitted = ""
        self._info = ModelInfo(id=model_id, backend="cann", path=self.model_dir,
                               context_length=context_length, chat_template="chatml")
        self._loaded = False

    # ------------------------------------------------------------ 生命周期

    def load(self) -> ModelInfo:
        if self._loaded:
            return self._info
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise ModelLoadError(f"模型目录不存在: {self.model_dir!r}")
        for need in ("executor.json", "context.json", "tokenizer.json"):
            if not os.path.exists(os.path.join(self.model_dir, need)):
                raise ModelLoadError(f"模型目录缺少 {need}：{self.model_dir}")

        self._ndk = _NdkBindings(self._lib_path)

        # 引擎按相对路径解析模型文件，切换到模型目录
        os.chdir(self.model_dir)

        self._executor = self._ndk.executor_create(b"executor.json")
        if not self._executor:
            raise ModelLoadError("Executor 创建失败（检查 executor.json 与模型文件）")

        self._callback_ref = self._ndk.callback_type(self._on_token)
        # 预建一个默认参数的 Context，尽早暴露配置问题
        self._context_for(self.default_params)

        self._loaded = True
        return self._info

    def close(self) -> None:
        # 故意不释放：Context_Destroy 会崩溃，Executor_Destroy 在退出阶段也无必要
        self._contexts.clear()
        self._loaded = False

    @property
    def supports_streaming(self) -> bool:
        return True

    # ------------------------------------------------------------ Context

    def _context_key(self, p: GenerationParams) -> Tuple:
        return (p.max_tokens, round(p.temperature, 4), p.top_k, round(p.top_p, 4),
                round(p.repetition_penalty, 4), p.seed, tuple(p.stop))

    def _context_for(self, p: GenerationParams) -> int:
        """取（或建）与该采样参数对应的 Context。"""
        key = self._context_key(p)
        with self._ctx_lock:
            ctx = self._contexts.get(key)
            if ctx:
                return ctx
            ctx = self._create_context(p)
            if len(self._contexts) < CONTEXT_CACHE_MAX:
                self._contexts[key] = ctx
            return ctx

    def _create_context(self, p: GenerationParams) -> int:
        assert self._ndk is not None
        with open(os.path.join(self.model_dir, "context.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        gen = cfg.setdefault("generate_options", {})
        gen["max_gen_tokens"] = int(p.max_tokens)
        # 停止符：<|im_end|> 之外再叠加调用方给的 stop
        gen["stop_sequence"] = [IM_END, *p.stop]
        # 每个 token 回调一次 → 逐字流式
        gen["callback_freq"] = 1

        sampler = cfg.setdefault("sampler", {})
        sampler.update({
            "do_sample": not p.greedy,
            "temperature": max(float(p.temperature), 1e-2) if p.greedy else float(p.temperature),
            "top-k": int(p.top_k),
            "top-p": float(p.top_p),
            "repetition_penalty": float(p.repetition_penalty),
        })
        if p.seed is not None:
            sampler["seed"] = int(p.seed)

        tmp = os.path.join(self.model_dir, ".context.live.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        ctx = self._ndk.context_create(tmp.encode())
        if not ctx:
            raise ModelLoadError("Context 创建失败（检查 context.json）")
        self._ndk.set_on_token(ctypes.c_void_p(ctx), self._callback_ref)
        return ctx

    # ------------------------------------------------------------ 回调

    def _read_partial(self, ctx_ptr) -> str:
        assert self._ndk is not None
        n = ctypes.c_uint(0)
        if self._ndk.get_generation_len(ctypes.c_void_p(ctx_ptr), ctypes.byref(n)) != 0:
            return ""
        if not n.value:
            return ""
        buf = ctypes.create_string_buffer(n.value + 1)
        if self._ndk.get_generation(ctx_ptr, buf, n.value) != 0:
            return ""
        return buf.value.decode("utf-8", "replace").split(IM_END)[0]

    def _on_token(self, ctx_ptr) -> None:
        """引擎每生成一个 token 回调一次（运行在引擎的工作线程里）。

        只做「读部分文本 + 把增量推入队列」；异常绝不能穿回 C 栈，
        否则会直接崩掉进程。
        """
        sink = self._sink
        if sink is None:
            return
        try:
            text = self._read_partial(ctx_ptr)
            if len(text) > len(self._emitted):
                delta = text[len(self._emitted):]
                self._emitted = text
                sink(delta)
        except Exception:      # noqa: BLE001 - 见 docstring
            pass

    # ------------------------------------------------------------ 生成

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        """流式生成。

        ``Generate`` 是阻塞的 C 调用，而回调不能 ``yield``，所以把
        ``Generate`` 放到工作线程，回调把增量推进 :class:`queue.Queue`，
        本生成器从队列里取出来 yield。这样 CLI / HTTP 两条上层都只面对
        一个普通生成器。
        """
        if not self._loaded:
            self.load()

        # 采样参数原样使用，不做任何「哨兵值」替换。
        # 曾经用「等于 dataclass 默认值就视为未指定」来合并后端默认值 ——
        # 那会吃掉调用方的明确意图：后端默认 temperature=0.1 时，
        # 调用方显式要求 0.7（恰好等于 dataclass 默认）会拿到 0.1。
        # 默认值的填充属于上层（CLI / HTTP 在构造请求前完成）。
        params = request.params
        self._check_prompt(request.prompt, params)

        events: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._emitted = ""

        def sink(delta: str) -> None:
            events.put(("delta", delta))

        result: Dict[str, object] = {}

        def worker() -> None:
            try:
                result.update(self._run(request.prompt, params, sink))
                events.put(("done", None))
            except BaseException as e:      # noqa: BLE001 - 转交给生成器抛出
                events.put(("error", e))

        self._sink = sink
        t = threading.Thread(target=worker, name="cann-generate", daemon=True)
        t.start()

        index = 0
        try:
            while True:
                try:
                    kind, payload = events.get(timeout=_QUEUE_TIMEOUT_S)
                except queue.Empty:
                    if not t.is_alive():
                        break
                    continue
                if kind == "delta":
                    yield GenerationChunk(text=str(payload), index=index)
                    index += 1
                elif kind == "error":
                    raise payload              # type: ignore[misc]
                else:
                    break
        finally:
            self._sink = None

        stats: GenerationStats = result.get("stats") or GenerationStats()
        reason = self._finish_reason(stats, params)
        yield GenerationChunk(index=index, finish_reason=reason, stats=stats)

    # ------------------------------------------------------------ 内部

    def _check_prompt(self, prompt: str, params: GenerationParams) -> None:
        """只拦真正的输入错误，**不限制长度**。

        本后端刻意不设上下文上限：引擎的 KV 缓存是 2048，超出后它不会报错，
        而是静默产出垃圾（实测 in_tokens≈2086 时开始出现 '-' 之类的重复）。
        这种"看起来成功但结果是错的"无法在客户端用任何启发式可靠预判
        （没有 tokenize 接口，按字节估算误差可达 2.3 倍），所以交给调用方
        自己观察输出、自己决定怎么控制长度。

        空 prompt 引擎自己也会拒绝（实测 Generate 返回 1），这里只是提前给出
        更清楚的报错，不改变引擎的行为。详见 docs/cann-engine-notes.md 第 9 节。
        """
        if not prompt:
            raise InvalidRequestError("prompt 不能为空")

    def _run(self, prompt: str, params: GenerationParams,
             sink) -> Dict[str, object]:
        assert self._ndk is not None
        ctx = self._context_for(params)
        ctx_p = ctypes.c_void_p(ctx)
        t0 = time.time()
        status = self._ndk.generate(ctypes.c_void_p(self._executor), ctx_p,
                                    prompt.encode("utf-8"))
        wall = time.time() - t0
        if status != 0:
            # 如实报告引擎的返回码，不替它断言原因 —— 我们无法区分到底是
            # 输入超长、含无法分词的字符，还是引擎内部错误。列出可能性即可。
            raise GenerationError(
                f"引擎 Generate 返回 {status}。无法从返回码判断具体原因，"
                f"常见可能：输入超出 KV 缓存（本模型 {self.context_length} token，"
                f"含输出）、含无法分词的字符、引擎内部错误。")

        in_tok = ctypes.c_ulong(0)
        out_tok = ctypes.c_ulong(0)
        self._ndk.get_input_token_count(ctx_p, ctypes.byref(in_tok))
        self._ndk.get_output_token_count(ctx_p, ctypes.byref(out_tok))

        times = {}
        for k, fn in self._ndk._time_funcs.items():
            v = ctypes.c_double(0)
            fn(ctx_p, ctypes.byref(v))
            times[k] = v.value

        return {
            "text": self._read_partial(ctx),
            "stats": GenerationStats(
                prompt_tokens=int(in_tok.value),
                completion_tokens=int(out_tok.value),
                prefill_ms=times.get("prefill", 0.0),
                decode_ms=times.get("decode", 0.0),
                total_ms=times.get("total", 0.0),
                wall_s=wall,
            ),
        }

    def _finish_reason(self, stats: GenerationStats, params: GenerationParams) -> str:
        """推断结束原因（OpenAI 协议要求这个字段）。

        引擎**没有**任何暴露「因何而停」的接口 —— 逐个试过
        GetStopReason / GetFinishReason / GetEndReason / GetGenerateState /
        IsFinished 等，都不存在；它还把命中的 stop_sequence 从输出里剥掉了
        （实测原始文本里不含 <|im_end|>）。唯一可用的证据是输出 token 数。

        因此判断分两种：
        * ``out_tokens < max_tokens`` —— **确定**不是被上限截断的
          （引擎只会在「够到 max_gen_tokens」或「命中 stop_sequence」时停），
          故为 stop。
        * ``out_tokens == max_tokens`` —— 说不准：既可能是被上限截断，
          也可能是模型恰好在第 N 个 token 自然结束。这里判为 length，
          因为「该继续却被截断」比「刚好说完了」常见得多，而且把截断误报成
          stop 会让调用方把半句话当成完整回答（更糟）。

        边界情况确实存在：实测 max_tokens=3 时模型恰好说了 3 个 token
        「谢谢！」，会被判成 length 并带 finish_reason=length 返回 ——
        而它其实说完了。这个歧义无法消除，只能如实记录在这里。
        """
        if stats.completion_tokens and stats.completion_tokens >= params.max_tokens:
            return FINISH_LENGTH
        return FINISH_STOP

    def count_prompt_tokens(self, text: str) -> int:
        """粗略估算 token 数，**仅供日志与展示，不作为任何限制的依据**。

        引擎没有只分词不生成的接口。系数按实测标定：原先用「字节 // 2」，
        实测对英文偏高约 2.3 倍（实际 1316 token 被估成 3011）；改用
        「字节 // 4」后与实测基本吻合，中文也大致成立。

        需要准确值时请用生成结果里的 ``GenerationStats.prompt_tokens``
        （直接来自引擎的 GetInputTokenCount），不要用这个估算。
        """
        return len(text.encode("utf-8")) // 4
