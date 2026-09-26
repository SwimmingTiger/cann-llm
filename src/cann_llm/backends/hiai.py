# -*- coding: utf-8 -*-
"""hiai 后端：驱动**系统内部**引擎 ``libhiai_llm_engine.so``（``HIAI_LLMEngine_*``）。

调用序列**照抄系统服务**（``libhm_model_engine_service.z.so`` 的
``AIMM::HIAI::HiaiSession``，文件名 ``hiai_session.cpp``）—— **全部使用导出符号**，
不碰任何内部函数：

    初始化（一次）:  Executor_CreateFromJson(executor_json)
    每次推理:        ctx = Context_Create()
                    Context_SetPrefixPrompt(ctx, prompt【文本】)        // hiai_session.cpp:1429
                    Context_SetInitTokenLen(ctx, initTokenLen)          // :1431（两个参数）
                    Context_SetMaxGenTokens(ctx, n)
                    SetOnAllTokensGenerateDoneFunc / SetOnSomeTokenGenerateDoneFunc
                    / SetOnGenerateAsyncFailed                          // :786/:793/:800
                    Executor_GenerateAsync(exec, ctx, prompt【文本】)    // :811
    取输出:          Context_GetAllGenerationLen / GetAllGeneration → 【明文】

三条容易踩错的点（都是实测/反编译确认的）：

1. ``GenerateAsync`` 的**第 3 个参数是 prompt 文本**（``std::string::c_str()``），
   不是 ``Prompt*``。传 ``Prompt*`` 会让流水线看到空的 tokenids，
   表现为 ``CheckPromptType`` 把 ``ctx+920`` 写成 3（"两者都空"）→ 生成失败。

2. **输入走 ``Context_SetPrefixPrompt(ctx, 文本)``**，不是 ``Prompt_SetTokenIds``。
   分词由引擎自己做（tokenizer 在模型配置里，``InitOption_SetTokenizer`` 告诉它）。

3. 输出用 ``Get*Generation`` 取**明文**；``GetAllTokenGeneration`` 给的是 int32
   token id（要自己解码，本项目改走明文后不再使用）。
   另外：**生成期间不要从外部轮询 Context**（与引擎工作线程竞态 →
   ``libc++abi: Pure virtual function called!`` abort）；等回调即可。

历史：本项目曾误判第 1 点，于是绕道内部函数（``0xFDA28`` / ``0x118704``）并自己
分词/解码。该做法已**全部删除** —— 它依赖特定构建的偏移，系统升级即失效。
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional

from ..modelpkg import detect_layout
from ..errors import BackendUnavailableError, GenerationError, ModelLoadError
from ..types import (
    GenerationChunk, GenerationParams, GenerationRequest, GenerationStats,
    ModelInfo,
)
from .base import EngineBackend, register_backend

#: 系统内部引擎（非 NDK 公开接口；仅用于本机自运行，不涉及分发）
HIAI_LIB = "/system/lib64/libhiai_llm_engine.so"

#: 官方目录结构的特征文件


def _find_official_files(model_dir: str) -> Dict[str, str]:
    """在官方结构目录里认出各角色的文件。"""
    found: Dict[str, Any] = {"omc": "", "model_json": "", "api": "", "tokenizer": ""}
    for n in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, n)
        if not os.path.isfile(p):
            continue
        low = n.lower()
        if low.endswith(".omc") and not found["omc"]:
            found["omc"] = n
        elif n == "api_config.json":
            found["api"] = n
        elif low == "tokenizer.json":
            found["tokenizer"] = n
        elif low.endswith(".json") and not found["model_json"]:
            # 只作候选：官方命名是 <omc 同名>.json，最后会优先选它
            found.setdefault("_json_candidates", [])
            found["_json_candidates"].append(n)      # type: ignore[attr-defined]
    # ★ 官方命名约定：模型配置与 .omc **同名**（qwen7b.omc <-> qwen7b.json）。
    # 目录里可能还堆着别的 json（比如转换脚本的产物），必须按这个约定挑。
    cands = found.pop("_json_candidates", [])          # type: ignore[arg-type]
    if found["omc"]:
        want = os.path.splitext(found["omc"])[0] + ".json"
        if want in cands:
            found["model_json"] = want
            return found
    found["model_json"] = cands[0] if cands else ""
    return found


def build_configs(model_dir: str) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """把官方目录里的两个文件合成引擎要的 executor / context（**超集**）。

    返回 ``(executor_dict, context_dict)``；调用方负责 ``json.dumps`` 后传给引擎。
    """
    f = _find_official_files(model_dir)
    layout = detect_layout(model_dir)

    if f["api"]:
        # packaged：<model>.json 提供 llm_config，api_config.json 提供运行参数
        with open(os.path.join(model_dir, f["model_json"]), encoding="utf-8") as fh:
            model_cfg: Dict[str, Any] = json.load(fh)
        with open(os.path.join(model_dir, f["api"]), encoding="utf-8") as fh:
            api: Dict[str, Any] = json.load(fh)
    else:
        # 缺 api_config.json：不再在后端里"猜"一份 —— 用 modelpkg 把它补齐即可
        # （转换脚本现在会自动产出，历史目录可手工补一次）。
        raise ModelLoadError(
            f"{model_dir} 里缺少 api_config.json —— 用下面任一方式补齐即可：\n"
            f"    python -m cann_llm.modelpkg {model_dir}\n"
            f"    scripts/model-conversion/build_model.py …（重新转换时自动生成）")

    # ---- executor：官方模型字段 + 官方路径字段，全部塞进 llm_config（超集）----
    llm = dict(model_cfg)
    llm.update({
        # 引擎按这些名字找文件（SetOptionModelPath / SetOptionTokenizer 读它们）
        "model_path": api.get("modelPath"),
        "weightDir": api.get("weightDir"),
        "weight_path": api.get("weightDir"),
        "tokenizerPath": api.get("tokenizerPath"),
        "tokenizerType": api.get("tokenizerType"),
        # 运行期字段也一并带上（引擎读不到就忽略）
        "inferType": api.get("inferType"),
        "modelType": api.get("modelType"),
        "isAsync": api.get("isAsync"),
        "pfxInitTokenLen": api.get("pfxInitTokenLen"),
        "pmtCacheOperation": api.get("pmtCacheOperation"),
        "prefixPrompt": api.get("prefixPrompt"),
        "callback_freq": api.get("callbackFreq"),
        "init_token_len": api.get("initTokenLen"),
        "max_gen_tokens": api.get("maxGenTokens"),
        "stop_sequence": api.get("stopSeq"),
    })
    executor = {
        "version": 1,
        "engine_type": "autoregressive",
        "llm_config": llm,
        "tokenizer": {"type": "qwen", "path": api.get("tokenizerPath") or f["tokenizer"]},
        "autoregressive": {
            "model_path": api.get("modelPath") or f["omc"],
            "weight_path": api.get("weightDir") or "./",
        },
    }

    # ---- context：结构化四件套 + 官方原始字段（引擎只读前四个，其余免费）----
    gen = {
        "callback_freq": api.get("callbackFreq", 1),
        "max_gen_tokens": api.get("maxGenTokens", 128),
        "stop_sequence": api.get("stopSeq") or [],
        "init_token_len": api.get("initTokenLen", 0),
    }
    sampler = {
        "do_sample": bool(api.get("sampleFlag", True)),
        "seed": api.get("seed", 99),
        "top-k": api.get("topK", 20),
        "top-p": api.get("topP", 0.95),
        "temperature": api.get("temperature", 0.7),
        "repetition_penalty": api.get("repetitionPenalty", 1.1),
    }
    context: Dict[str, Any] = {
        "version": 1,
        "engine_type": "autoregressive",
        "generate_options": gen,
        "sampler": sampler,
    }
    for k, v in api.items():
        context.setdefault(k, v)          # 官方字段原样带上，不丢
    return executor, context


class _HiaiBindings:
    """``libhiai_llm_engine.so`` 的 ctypes 绑定（符号名已按实测映射）。"""

    #: 名 -> (restype, argtypes)
    SIGS = {
        "HIAI_LLMEngine_Context_Destroy": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        "HIAI_LLMEngine_Context_GetAllGenerationLen": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        "HIAI_LLMEngine_Context_GetAllGeneration": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]),
        "HIAI_LLMEngine_Context_GetInputTokenCount": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        "HIAI_LLMEngine_Context_GetOutputTokenCount": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        # ★ 耗时/计数：反编译实锤都是「(ctx, T* out)」形式，返回 0 表示 ok。
        #   注意时间是 **double（8 字节）** —— 曾经用 c_float 去读，只拿到低 4 字节，
        #   显示成 0.000（小端下小值低位近 0），误以为签名不对。
        "HIAI_LLMEngine_Context_GetTotalTimeMs": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        "HIAI_LLMEngine_Context_GetPrefillTimeMs": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        "HIAI_LLMEngine_Context_GetDecodeTimeMs": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        "HIAI_LLMEngine_Executor_CreateFromJson": (ctypes.c_void_p, [ctypes.c_char_p]),
        # 服务每个请求都会设这两个（见 libhm_model_engine_service 的符号引用）
        "HIAI_LLMEngine_Context_SetInitTokenLen": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        # ★ 停止序列：签名反编译实锤 (ctx, const char** seqs, unsigned n)，n 有效范围 1..9
        #   官方模型的 api_config.json 里有 stopSeq = ["<|im_end|>", "<|endoftext|>"]
        "HIAI_LLMEngine_Context_SetStopSeq": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint]),
        "HIAI_LLMEngine_Context_SetMaxGenTokens": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        # ★ 输入通道：服务用 Context_SetPrefixPrompt(context_, param.prefixPrompt.c_str())
        #   （hiai_session.cpp:1429，反编译实锤）
        # ★★★ 第 3 参是 prompt【文本】—— 必须声明成 c_char_p：
        #     ctypes 会准备一份 NUL 结尾的副本；若声明成 c_void_p 或干脆不声明，
        #     传 Python bytes 得到的是无终止符的裸缓冲区 / 指针被当 int 截断
        #     → 引擎读越界 → segfault（实测）。
        "HIAI_LLMEngine_Executor_GenerateAsync": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]),
        "HIAI_LLMEngine_Context_SetPrefixPrompt": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
        # 流式所需（签名照已验证脚本 scripts/streaming_reference.py）
        "HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_Context_SetMaxGenTokens": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    }

    def __init__(self, lib_path: Optional[str] = None):
        lib_path = lib_path or os.environ.get("CANN_LLM_HIAI_LIB") or HIAI_LIB
        if not os.path.exists(lib_path):
            raise BackendUnavailableError(
                f"找不到内部引擎 {lib_path}；这个后端需要鸿蒙设备上的系统库")
        try:
            self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        except OSError as e:
            raise BackendUnavailableError(f"加载 {lib_path} 失败: {e}") from e
        for name, (res, args) in self.SIGS.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as e:
                raise BackendUnavailableError(f"{lib_path} 里没有符号 {name}") from e
            fn.restype = res
            fn.argtypes = args


def _gen_failure_msg(rc, context_length: int) -> str:
    """生成失败时的文案。

    ★ 注意：我们【无法】探测"当前终端有没有 NPU 权限" —— 实测 /dev/npu* 的
      stat/open 在 *有权限* 与 *没权限* 的终端上都失败（errno 13），没有区分度。
      所以这里只能把权限列为**可能原因之一**，不断言。
    """
    from ..enginelog import format_engine_log, recent_engine_log
    # ★ 措辞说明：
    #   · 不说"引擎报告生成失败" —— "report"（通报）直译成"报告"读不通；
    #   · 不提 GenerateAsync 的返回码 —— 实测它成功失败都是 0，没有信息量；
    #   · 不说"生成失败" —— CLI 的前缀（"生成失败: " / "[失败] "）已经说了，
    #     异常消息里只保留【引擎侧】的事实。
    head = "引擎出错"
    # ★ 拿到引擎日志就【只用它】—— 那四条"可能原因"是我没依据时的兜底，
    #   有引擎原话时再列出来只会分散注意力。读不到才退回兜底文案。
    if recent_engine_log():
        return f"{head}。" + format_engine_log()
    return (f"{head}。\n"
            f"    · 输入超出 KV 缓存（{_ctx_desc(context_length)}）\n"
            f"    · 当前终端没有访问 NPU 的权限（换一个系统终端试试）\n"
            f"    · 含无法分词的字符\n"
            f"    · 引擎内部错误"
            + format_engine_log())


def _ctx_desc(n: int) -> str:
    """把 KV 上限渲染成人读的说法；0 表示未知，别假装知道。"""
    return (f"本模型 KV 缓存 {n} token" if n else
            "本模型的上限未知 —— 模型目录里没找到 kv_cache_max_len")






@register_backend("hiai")
class HiaiBackend(EngineBackend):
    """驱动系统内部引擎，认**官方模型目录结构**。"""

    name = "hiai"

    def __init__(self, model_dir: str, lib_path: Optional[str] = None,
                 model_id: Optional[str] = None,
                 context_length: Optional[int] = None,
                 default_params: Optional[GenerationParams] = None):
        self.model_dir = os.path.abspath(model_dir) if model_dir else ""
        self._lib_path = lib_path
        self.model_id = model_id or (os.path.basename(self.model_dir) or "cann-llm")
        self.default_params = default_params or GenerationParams()
        self._bind: Optional[_HiaiBindings] = None
        self._ctx: Optional[int] = None          # 仅代表"最近一次"的 Context
        self._exec: Optional[int] = None
        self._ctx_json: bytes = b""               # 每请求用它新建 Context
        self._cb_done = None                       # 回调需长期持有，勿被 GC
        self._cb_fail = None
        self._cb_some = None
        self._ids = []
        self._bos: int = -1                        # 引擎期望的 BOS（缺它会 Generate 失败）
        self._init_token_len: int = 0              # 模型配置里的 initTokenLen（prefill 长度）
        self._stop_seq: list = []                  # 停止序列（模型配置里的 stopSeq）
        # 显式给的优先；没给则 load() 时从合成的 executor JSON 里读
        self._context_length = context_length or 0
        self._info: Optional[ModelInfo] = None

    # ---------------------------------------------------------------- 生命周期

    def load(self) -> ModelInfo:
        if self._exec:
            assert self._info is not None
            return self._info

        layout = detect_layout(self.model_dir)
        if layout != "packaged":
            # 不再在后端里"猜"一份配置 —— 用 modelpkg 补齐即可（转换脚本会自动产出）
            raise ModelLoadError(
                f"{self.model_dir} 里缺少 api_config.json（或 <model>.json）。\n"
                f"    补齐：python -m cann_llm.modelpkg {self.model_dir}\n"
                f"    （转换脚本 scripts/model-conversion/build_model.py 会自动生成）")

        executor_cfg, context_cfg = build_configs(self.model_dir)
        self._bind = _HiaiBindings(self._lib_path)

        # 引擎按相对路径解析模型文件
        os.chdir(self.model_dir)

        # ★ 传 JSON 内容（不是文件名）
        # ★★ 关键：Context 携带对话状态，**不能跨请求复用** ——
        #     复用时第二次 Generate 就会产出垃圾（实测：in 变成 1，输出固定胡话）。
        #     这里只保存 context JSON，每次 generate 新建一个 Context，
        #     与已验证可用的 C 程序做法一致。
        # ★ 只建 Executor —— 验证过的配方里【不】预先建 Context（预建会 SIGTRAP）。
        #   Context 每请求用 Context_Create()（无参）新建。
        self._ctx_json = json.dumps(context_cfg).encode()
        self._ctx = None
        self._exec = self._bind.lib.HIAI_LLMEngine_Executor_CreateFromJson(
            json.dumps(executor_cfg).encode())
        if not self._exec:
            raise ModelLoadError(
                "Executor 创建失败。常见可能：\n"
                "    · 当前终端没有访问 NPU 的权限（换一个系统终端试试）\n"
                "    · 模型目录不完整 / executor JSON 有问题")

        # 注：本后端【不需要】自己分词 —— 输入/输出都是明文，
        # 引擎用模型配置里的 tokenizer 自己处理。
        llm = executor_cfg["llm_config"]
        # 优先模型自带的扁平 <model>.json；我们生成的 executor.json 只作兜底。
        # 读不到就保持 0（= 未知）—— **不猜默认值**，否则提示里会写一个
        # 属于别个模型的上限。该值随模型而变（实测 7B=4096 / 1.5B=2048）。
        from ..modelcfg import read_kv_cache_max_len
        real, _src = read_kv_cache_max_len(self.model_dir)
        self._context_length = real or self._context_length or 0
        # 引擎期望 prompt 以 BOS 开头（实测：不加则 Generate 返回 1；
        # 且引擎对短 prompt 报的 in=7 比纯文本分词结果多 1，正是这个 BOS）
        try:
            self._bos = int(llm.get("bos_token_id", 2))
        except (TypeError, ValueError):
            self._bos = 2
        # 停止序列：优先用模型自带的（api_config.json 的 stopSeq），否则用 Qwen 的默认
        _ss = llm.get("stopSeq") or llm.get("stop_seq") or ["<|im_end|>", "<|endoftext|>"]
        if isinstance(_ss, str):
            _ss = [_ss]
        self._stop_seq = [x for x in _ss if isinstance(x, str) and x][:9] or ["<|im_end|>"]

        # 服务在 SetPrefixPrompt 之后调 SetInitTokenLen(ctx, param.initTokenLen)
        # （hiai_session.cpp:1431）—— 取模型配置里的 init_token_len / initTokenLen
        for key in ("init_token_len", "initTokenLen"):
            try:
                v = int(llm.get(key) or 0)
            except (TypeError, ValueError):
                v = 0
            if v > 0:
                self._init_token_len = v
                break

        self._info = ModelInfo(id=self.model_id, backend="hiai", path=self.model_dir,
                               context_length=self._context_length, chat_template="chatml")
        return self._info

    def close(self) -> None:
        # 与 cann 后端同样的顾虑：引擎的 Destroy 在退出阶段不稳，保守起见不主动调用
        self._ctx = None
        self._exec = None

    @property
    def supports_streaming(self) -> bool:
        # ★ 流式：SetOnSomeTokenGenerateDoneFunc 每生成一个 token 回调一次。
        #   回调里只用轻量的 GetOneTokenGeneration 取单 token 并 append；
        #   增量由主线程轮询列表后解码（已被 scripts/streaming_reference.py 验证）。
        return True

    # ------------------------------------------------------------------ 生成

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        if not self._exec:                      # Context 每请求新建，不参与判断
            raise GenerationError("引擎尚未 load()")
        assert self._bind is not None
        p = request.params if request.params is not None else self.default_params

        # ★ 每次请求新建 Context（不复用 —— 见 load() 里的说明）
        ctx = self._bind.lib.HIAI_LLMEngine_Context_Create()
        if not ctx:
            raise GenerationError("Context 创建失败")
        self._ctx = ctx
        try:
            yield from self._generate_with(ctx, request, p)
        finally:
            # Context 每请求一个，用完即销毁（引擎的 Destroy 收指针的指针）
            self._bind.lib.HIAI_LLMEngine_Context_Destroy(
                ctypes.byref(ctypes.c_void_p(ctx)))
            self._ctx = None

    def _generate_with(self, ctx, request, p) -> Iterator[GenerationChunk]:
        """把一个请求跑完（Context 的生命周期由 generate() 管）。"""
        assert self._bind is not None

        # ★ 服务每个请求都会显式设 initTokenLen / maxGenTokens
        maxgen = int(getattr(p, "max_tokens", 0) or 128)
        self._bind.lib.HIAI_LLMEngine_Context_SetMaxGenTokens(ctx, maxgen)

        # ★★ 必须注册回调：流水线在完成时通过 std::function 回调，
        #    未注册时引擎调用空的 std::function → libc++abi "Pure virtual function called!" 直接 abort。
        # 回调驱动完成（★ 不能在生成期间轮询 —— 与工作线程竞态会触发
        # libc++abi "Pure virtual function called!" 而 abort）
        import threading
        _ev_done = threading.Event()
        _ev_fail = threading.Event()
        _CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self._cb_done = _CB(lambda _p: _ev_done.set())
        self._cb_fail = _CB(lambda _p: _ev_fail.set())
        _lib = self._bind.lib
        _ctx = ctx
        _ctypes = ctypes

        def _read_text() -> str:
            """主线程读明文（等价 _peek_text，但闭包局部名，避免回 self）。"""
            n = _ctypes.c_int(0)
            r1 = _lib.HIAI_LLMEngine_Context_GetAllGenerationLen(_ctx, _ctypes.byref(n))
            if n.value <= 0:
                return ""
            buf = _ctypes.create_string_buffer(n.value + 64)
            # ★★ 这一族 getter 的返回值【不是成败标志】：实测 GetAllGeneration 返回 1
            #    而文本完全正确（GetOneGeneration 同样返回 1 带内容）。
            #    因此只按长度与缓冲区内容判断，不看返回值。
            _lib.HIAI_LLMEngine_Context_GetAllGeneration(_ctx, buf, n.value + 63)
            return buf.value.decode("utf-8", "replace")

        # 兜底：模型可能越过停止符继续生成，截断到第一个停止符
        import re as _re
        _STRIP = _re.compile(
            r"<\|im_end\|>.*|<\|endoftext\|>.*|<\|im_start\|>.*", _re.S)
        import queue as _queue
        _q: "_queue.Queue[str]" = _queue.Queue()
        _st = {"emitted": ""}

        def _on_some(peer_ctx: object) -> None:
            # ★★★ 服务的做法：回调在引擎线程上执行，**在这条线程里调 getter 是安全的**
            #     （AIMM::HIAI::HiaiSession::OnSomeTokensGenerated 就是这么干的）。
            #     从【外部】主线程读才会与工作线程竞态。
            try:
                _n = _ctypes.c_int(0)
                _lib.HIAI_LLMEngine_Context_GetAllGenerationLen(peer_ctx, _ctypes.byref(_n))
                if _n.value <= 0:
                    return
                _b = _ctypes.create_string_buffer(_n.value + 64)
                _lib.HIAI_LLMEngine_Context_GetAllGeneration(peer_ctx, _b, _n.value + 63)
                _t = _STRIP.sub("", _b.value.decode("utf-8", "replace"))
                if len(_t) > len(_st["emitted"]):
                    _q.put(_t[len(_st["emitted"]):])
                    _st["emitted"] = _t
            except Exception:      # noqa: BLE001 - 回调里不能抛
                pass
        self._cb_some = _CB(_on_some)
        self._bind.lib.HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
            ctx, ctypes.cast(self._cb_done, ctypes.c_void_p))
        self._bind.lib.HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(
            ctx, ctypes.cast(self._cb_fail, ctypes.c_void_p))
        self._bind.lib.HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
            ctx, ctypes.cast(self._cb_some, ctypes.c_void_p))

        # ★★★ 输入 = prompt【文本】，走两条服务实锤的通道（hiai_session.cpp 反编译）：
        #     Context_SetPrefixPrompt(ctx, text)   —— 1429
        #     Executor_GenerateAsync(exec, ctx, text) —— 811（第 3 参就是 std::string::c_str()）
        #   引擎自己分词（tokenizer 由 InitOption_SetTokenizer 给它），
        #   因此**不需要** Prompt_SetTokenIds，也不必自己解码。
        text = request.prompt.encode("utf-8")
        if self._bind.lib.HIAI_LLMEngine_Context_SetPrefixPrompt(ctx, text) != 0:
            raise GenerationError("Context_SetPrefixPrompt 失败")
        # 服务在此设 initTokenLen（= 模型配置里的 initTokenLen）；两参，调用点实锤
        self._bind.lib.HIAI_LLMEngine_Context_SetInitTokenLen(ctx, self._init_token_len)
        # ★ 不设它，引擎不会在 <|im_end|> 处停 —— 会继续编出 "<|im_end|>…Human: …" 这种假对话
        if self._stop_seq:
            _arr = (ctypes.c_char_p * len(self._stop_seq))(
                *[x.encode("utf-8") for x in self._stop_seq])
            self._bind.lib.HIAI_LLMEngine_Context_SetStopSeq(ctx, _arr, len(self._stop_seq))

        rc = self._bind.lib.HIAI_LLMEngine_Executor_GenerateAsync(self._exec, ctx, text)
        _idx = 0
        _emitted = ""
        if rc == 0:
            # ★★ 生成期间【只等】，绝不读 Context —— 实测：从外部读会与引擎
            #    工作线程竞态，轻则读到空、重则 libc++abi abort。
            #    （流式的读要放在 OnSomeToken 回调里，同线程才安全 —— 见下一步。）
            while not _ev_done.wait(timeout=0.05):
                while not _q.empty():
                    yield GenerationChunk(text=_q.get(), index=_idx)
                    _idx += 1
                if _ev_fail.is_set():
                    raise GenerationError(_gen_failure_msg(None, self._context_length))
            while not _q.empty():
                yield GenerationChunk(text=_q.get(), index=_idx)
                _idx += 1
            if _ev_fail.is_set():
                raise GenerationError(_gen_failure_msg(None, self._context_length))
            _emitted = _st["emitted"]
        else:
            raise GenerationError(_gen_failure_msg(rc, self._context_length))

        stats = self._read_stats(len(_emitted))
        yield GenerationChunk(index=_idx, finish_reason="stop", stats=stats)



    def _read_generation(self) -> str:
        assert self._bind is not None and self._ctx
        n = ctypes.c_int(0)
        if self._bind.lib.HIAI_LLMEngine_Context_GetAllGenerationLen(
                self._ctx, ctypes.byref(n)) != 0 or n.value <= 0:
            return ""
        buf = ctypes.create_string_buffer(n.value + 1)
        # 返回值不是成败标志（实测返回 1 而内容正确），只看内容
        self._bind.lib.HIAI_LLMEngine_Context_GetAllGeneration(self._ctx, buf, n.value)
        return buf.value.decode("utf-8", "replace")

    def _read_stats(self, out_len: int) -> GenerationStats:
        assert self._bind is not None and self._ctx
        n = ctypes.c_int(0)
        pin = pout = None
        if self._bind.lib.HIAI_LLMEngine_Context_GetInputTokenCount(
                self._ctx, ctypes.byref(n)) == 0:
            pin = n.value
        if self._bind.lib.HIAI_LLMEngine_Context_GetOutputTokenCount(
                self._ctx, ctypes.byref(n)) == 0:
            pout = n.value
        # tokens_per_second 是导出属性（由 decode_ms 算），不能传
        def _d(fn):
            v = ctypes.c_double(0.0)
            try:
                fn(self._ctx, ctypes.byref(v))
            except Exception:      # noqa: BLE001
                pass
            return float(v.value)

        lib = self._bind.lib
        pre = _d(lib.HIAI_LLMEngine_Context_GetPrefillTimeMs)
        dec = _d(lib.HIAI_LLMEngine_Context_GetDecodeTimeMs)
        tot = _d(lib.HIAI_LLMEngine_Context_GetTotalTimeMs)
        return GenerationStats(prompt_tokens=pin or 0,
                               completion_tokens=pout or 0,
                               prefill_ms=pre, decode_ms=dec, total_ms=tot)

    def count_prompt_tokens(self, text: str) -> int:
        # 内部引擎没有单独的分词接口；交给引擎在 Generate 时统计
        return -1


# ---------------------------------------------------------------------------
# 参考资料
#   调用序列的来源（可复核）：
#     x570: ~/re/svc_call.txt  —— libhm_model_engine_service.z.so 里
#           AIMM::HIAI::HiaiSession::HiaiSessionRun 的反编译（hiai_session.cpp:1419-1434）
#     x570: ~/re/svc_gen.txt   —— LLMEngineGenerateAsync / LLMEngineRun
#                                 （:805-811 显示第 3 参 = std::string::c_str()）
#  完整调查记录与已排除的假设见 docs/hiai-backend-handoff.md。
# ---------------------------------------------------------------------------
