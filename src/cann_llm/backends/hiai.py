# -*- coding: utf-8 -*-
"""hiai 后端：直接驱动**系统内部**的 LLM 引擎 ``libhiai_llm_engine.so``。

与 cann 后端（NDK 栈 ``libcann_llm_engine.so`` + ``HMS_LLMEngine*``）的区别，
都是这次逆向 + 实测确认的，共四处：

  1. Context / Executor 收的是 **JSON 内容**，不是文件路径
     （传路径会得到 nlohmann 的 parse_error: last read: 'c' —— 'context.json' 的首字母）
  2. 创建 Executor 的符号名是 ``Executor_CreateFromJson``
     （NDK 是 ``Executor_CreateFromExecutorJson``）
  3. **没有** ``Prompt_SetTokenId``（NDK 有）
  4. ★ ``Executor_Generate`` 的第 3 个参数是 **``Prompt`` 对象指针**，不是文本
     （NDK 收 ``const char*``）—— 传错会卡死，这是踩过的坑
  5. ★★ **不要用 ``Prompt_SetText``** —— 它内部走 ``std::string::assign()``，
     其堆分配路径在本机环境不可用：实测边界精确落在 **22→23 字节**
     （libc++ 的 SSO 容量），超过就只给引擎留下 1 个 token，输出恒为胡话。
     系统服务因此自己在外面分词，再用 ``Prompt_SetTokenIds`` 传整数数组 ——
     本后端照做（``hiai_tokenizer.QwenTokenizer``）。

另外这个后端自己认**官方模型目录结构**（``<model>.json`` + ``api_config.json``），
在内存里合成引擎要的配置，**不改动模型目录**：官方把配置拆在两个文件里，
而引擎的 ``CreateFromJson`` 要一个带 ``llm_config`` 的 JSON，合并必须在内存做。

引擎不认识的键会被忽略（nlohmann 按名取键），所以合成时给**超集**即可 ——
这比"挑字段"安全，因为挑字段会漏。
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional

from ..errors import BackendUnavailableError, GenerationError, ModelLoadError
from ..types import (
    GenerationChunk, GenerationParams, GenerationRequest, GenerationStats,
    ModelInfo,
)
from .base import EngineBackend, register_backend
from .hiai_tokenizer import QwenTokenizer

#: 系统内部引擎（非 NDK 公开接口；仅用于本机自运行，不涉及分发）
HIAI_LIB = "/system/lib64/libhiai_llm_engine.so"

#: 官方目录结构的特征文件
OFFICIAL_MARKERS = ("api_config.json",)
#: 我们自己的目录结构特征文件
OURS_MARKERS = ("executor.json", "context.json")


def detect_layout(model_dir: str) -> str:
    """判断模型目录是哪种结构。

    :return: ``"official"``（官方包）/ ``"ours"``（我们的 executor.json+context.json）
             / ``"unknown"``
    """
    if not model_dir or not os.path.isdir(model_dir):
        return "unknown"
    names = set(os.listdir(model_dir))
    # 官方标记优先：官方包里可能**同时**存在我们格式的文件（转换脚本留下的），
    # 但只要有 api_config.json，它本质上就是官方结构。
    if any(m in names for m in OFFICIAL_MARKERS):
        return "official"
    if all(m in names for m in OURS_MARKERS):
        return "ours"
    return "unknown"


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
    if not f["model_json"] or not f["api"]:
        raise ModelLoadError(
            f"官方结构目录里缺配置文件：model_json={f['model_json']!r} api={f['api']!r}"
        )
    with open(os.path.join(model_dir, f["model_json"]), encoding="utf-8") as fh:
        model_cfg: Dict[str, Any] = json.load(fh)
    with open(os.path.join(model_dir, f["api"]), encoding="utf-8") as fh:
        api: Dict[str, Any] = json.load(fh)

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
        "HIAI_LLMEngine_Context_CreateFromContextJson": (ctypes.c_void_p, [ctypes.c_char_p]),
        "HIAI_LLMEngine_Context_Destroy": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        "HIAI_LLMEngine_Context_GetAllGenerationLen": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        "HIAI_LLMEngine_Context_GetAllGeneration": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]),
        "HIAI_LLMEngine_Context_GetInputTokenCount": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        "HIAI_LLMEngine_Context_GetOutputTokenCount": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        "HIAI_LLMEngine_Executor_CreateFromJson": (ctypes.c_void_p, [ctypes.c_char_p]),
        # 服务每个请求都会设这两个（见 libhm_model_engine_service 的符号引用）
        "HIAI_LLMEngine_Context_SetInitTokenLen": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "HIAI_LLMEngine_Context_SetMaxGenTokens": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "HIAI_LLMEngine_Executor_Destroy": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        # ★ 第 3 参是 Prompt 对象，不是文本
        "HIAI_LLMEngine_Executor_Generate": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Prompt_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_Prompt_SetText": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
        # 流式所需（签名照已验证脚本 scripts/streaming_reference.py）
        "HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_GetOneTokenGeneration": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]),
        "HIAI_LLMEngine_Context_GetAllTokenGenerationLen": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]),
        # ★ 注意：token 是 int32 数组，不是文本（实测 GetOneTokenGeneration 只给一个 token）
        "HIAI_LLMEngine_Context_GetAllTokenGeneration": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]),
        "HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "HIAI_LLMEngine_Context_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_Prompt_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_Context_SetMaxGenTokens": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        # ★ 避开 Prompt_SetText 的 std::string 堆分配缺陷（见模块 docstring）
        "HIAI_LLMEngine_Prompt_SetTokenIds": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint]),
        "HIAI_LLMEngine_Prompt_Destroy": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
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


def _module_base() -> int:
    """libhiai_llm_engine.so 的加载基址（内部函数地址 = 基址 + 静态偏移）。"""
    lo = None
    with open("/proc/self/maps") as fh:
        for line in fh:
            if HIAI_LIB.rsplit("/", 1)[-1] in line:
                a = int(line.split("-", 1)[0], 16)
                lo = a if lo is None else min(lo, a)
    if lo is None:
        raise BackendUnavailableError(f"{HIAI_LIB} 未加载")
    return lo


#: 引擎内部函数静态偏移（已反编译确认，见模块 docstring 与 docs/hiai-backend-handoff.md）
OFF_PUSH_PROMPT = 0xFDA28   # vector<Prompt336>::push_back —— 原样搬 336 字节，保住 tokenids
OFF_RUN_GENERATE = 0x118704  # 真正的执行入口（GenerateAsync 的内部调用）


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
        self._tok: Optional[QwenTokenizer] = None
        self._cb_done = None                       # 回调需长期持有，勿被 GC
        self._cb_fail = None
        self._cb_some = None
        self._ids = []
        self._bos: int = -1                        # 引擎期望的 BOS（缺它会 Generate 失败）
        # 显式给的优先；没给则 load() 时从合成的 executor JSON 里读
        self._context_length = context_length or 0
        self._info: Optional[ModelInfo] = None

    # ---------------------------------------------------------------- 生命周期

    def load(self) -> ModelInfo:
        if self._exec:
            assert self._info is not None
            return self._info

        layout = detect_layout(self.model_dir)
        if layout != "official":
            raise ModelLoadError(
                f"{self.model_dir} 不是官方结构（需要 <model>.json + api_config.json）；"
                f"检测到 {layout!r} —— 用 cann 后端试试")

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
            raise ModelLoadError("Executor 创建失败（检查合成的 executor JSON）")

        # 分词器：tokenizer.json 在模型目录里
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise ModelLoadError(f"缺少 tokenizer.json: {tok_path}")
        self._tok = QwenTokenizer(tok_path)
        print(f"  [hiai] 分词器就绪: vocab={len(self._tok.vocab)}", flush=True)

        llm = executor_cfg["llm_config"]
        real = int(llm.get("kv_cache_max_len") or 0)
        # 模型自带的优先：外部传来的可能只是配置默认值（2048），会误导人
        self._context_length = real or self._context_length or 2048
        # 引擎期望 prompt 以 BOS 开头（实测：不加则 Generate 返回 1；
        # 且引擎对短 prompt 报的 in=7 比纯文本分词结果多 1，正是这个 BOS）
        try:
            self._bos = int(llm.get("bos_token_id", 2))
        except (TypeError, ValueError):
            self._bos = 2

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
        _acc: list = []
        _one_fn = self._bind.lib.HIAI_LLMEngine_Context_GetOneTokenGeneration
        _byref, _cast, _i32, _cp = (ctypes.byref, ctypes.cast,
                                    ctypes.c_int32, ctypes.c_char_p)

        def _on_some(p: object) -> None:
            # ★ 只用局部名，绝不触碰 self（引擎工作线程上访问 Python 对象属性会 segfault）
            try:
                _v = _i32(0)
                if _one_fn(p, _cast(_byref(_v), _cp), 4) == 0:
                    _acc.append(int(_v.value))
            except Exception:      # noqa: BLE001
                pass
        self._cb_some = _CB(_on_some)
        self._ids = _acc
        self._bind.lib.HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
            ctx, ctypes.cast(self._cb_done, ctypes.c_void_p))
        self._bind.lib.HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(
            ctx, ctypes.cast(self._cb_fail, ctypes.c_void_p))
        self._bind.lib.HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
            ctx, ctypes.cast(self._cb_some, ctypes.c_void_p))

        prompt = self._bind.lib.HIAI_LLMEngine_Prompt_Create()
        if not prompt:
            raise GenerationError("Prompt 创建失败")
        try:
            # ★ 走 token ids：Prompt_SetText 的 std::string 堆路径在本机不可用
            #   （实测边界精确在 22 字节 = libc++ SSO 容量，超过就只剩 1 个 token）
            assert self._tok is not None
            ids = self._tok.encode(request.prompt)
            if not ids:
                raise GenerationError("分词结果为空")
            if self._bos >= 0 and ids[0] != self._bos:
                ids = [self._bos] + ids          # ★ 补 BOS
            # ★ 告知引擎本prompt要 prefill 多少 token（服务在 SetTokenIds 前设它）
            # 注：不要自己调 SetInitTokenLen（流水线内部会设 base.cpp:401）
            arr = (ctypes.c_int32 * len(ids))(*ids)
            if self._bind.lib.HIAI_LLMEngine_Prompt_SetTokenIds(
                    prompt, arr, len(ids)) != 0:
                raise GenerationError("Prompt_SetTokenIds 失败")
            # ★★ 不走 Executor_Generate / GenerateAsync —— 二者都会丢 tokenids：
            #    实测 GenerateAsync 内部的 Prompt→params 转换会让流水线看到空 tokenids
            #    （证据：CheckPromptType 写入的 ctx+920 恒为 3 = "两者都空"）。
            #    这里自己组 params 向量，用引擎的 push_back 原样搬 336 字节的 Prompt。
            base = _module_base()
            push = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
                base + OFF_PUSH_PROMPT)
            run = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p)(base + OFF_RUN_GENERATE)
            vec = (ctypes.c_uint64 * 3)(0, 0, 0)
            push(ctypes.byref(vec), prompt)
            rc = run(self._exec, ctx, ctypes.byref(vec))
            _idx = 0
            _emitted = ""
            _seen = 0
            if rc == 0:
                # ★ 流式：主线程轮询回调累积的 token 列表（纯 Python，不碰引擎）
                while not _ev_done.is_set():
                    if len(_acc) > _seen:
                        _seen = len(_acc)
                        _txt = self._decode_ids(list(_acc))
                        if len(_txt) > len(_emitted):
                            yield GenerationChunk(text=_txt[len(_emitted):], index=_idx)
                            _emitted = _txt
                            _idx += 1
                    if _ev_fail.is_set():
                        raise GenerationError("引擎报告生成失败（OnGenerateAsyncFailed）")
                    _ev_done.wait(timeout=0.02)
                if _ev_fail.is_set():
                    raise GenerationError("引擎报告生成失败（OnGenerateAsyncFailed）")
                _txt = self._decode_ids(list(_acc))
                if len(_txt) > len(_emitted):
                    yield GenerationChunk(text=_txt[len(_emitted):], index=_idx)
                    _emitted = _txt
                    _idx += 1
            if rc != 0:
                raise GenerationError(
                    f"引擎 Generate 返回 {rc}（本模型 KV 缓存 {self._context_length} token）")

            _full = self._decode_ids(list(_acc))
            stats = self._read_stats(len(_full))
            yield GenerationChunk(index=_idx, finish_reason="stop", stats=stats)
        finally:
            self._bind.lib.HIAI_LLMEngine_Prompt_Destroy(
                ctypes.byref(ctypes.c_void_p(prompt)))
            # Context 每请求一个，用完即销毁（内部引擎的 Destroy 收指针的指针）
            self._bind.lib.HIAI_LLMEngine_Context_Destroy(
                ctypes.byref(ctypes.c_void_p(ctx)))

    def _decode_ids(self, ids: list) -> str:
        """token id 列表 → 文本（去掉尾部终止 token）。"""
        assert self._tok is not None
        for stop in (151645, 151643):      # <|im_end|> / <|endoftext|>
            while ids and ids[-1] == stop:
                ids.pop()
        return self._tok.decode(ids)

    def _read_tokens(self) -> str:
        """取生成结果：引擎给的是 **token id 数组**（不是文本），用自带分词器解码。"""
        assert self._bind is not None and self._ctx and self._tok is not None
        n = ctypes.c_int(0)
        if self._bind.lib.HIAI_LLMEngine_Context_GetAllTokenGenerationLen(
                self._ctx, ctypes.byref(n)) != 0 or n.value <= 0:
            return ""
        buf = (ctypes.c_int32 * (n.value + 8))()
        if self._bind.lib.HIAI_LLMEngine_Context_GetAllTokenGeneration(
                self._ctx, buf, n.value + 8) != 0:
            return ""
        ids = [int(x) for x in buf[:n.value]]
        # 去掉终止 token
        for stop in (151645, 151643):          # <|im_end|> / <|endoftext|>
            while ids and ids[-1] == stop:
                ids.pop()
        return self._tok.decode(ids)

    def _read_generation(self) -> str:
        assert self._bind is not None and self._ctx
        n = ctypes.c_int(0)
        if self._bind.lib.HIAI_LLMEngine_Context_GetAllGenerationLen(
                self._ctx, ctypes.byref(n)) != 0 or n.value <= 0:
            return ""
        buf = ctypes.create_string_buffer(n.value + 1)
        if self._bind.lib.HIAI_LLMEngine_Context_GetAllGeneration(
                self._ctx, buf, n.value) != 0:
            return ""
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
        return GenerationStats(prompt_tokens=pin or 0,
                               completion_tokens=pout or 0)

    def count_prompt_tokens(self, text: str) -> int:
        # 内部引擎没有单独的分词接口；交给引擎在 Generate 时统计
        return -1


# ============================================================================
# 待实现：服务所用的 async 路径（全部签名已反编译确认）
# ----------------------------------------------------------------------------
# 服务（libhm_model_engine_service.z.so）不用 CreateFromJson + 同步 Generate，
# 而是走下面这条。实测：同步 Generate 对长 prompt 返回 1，需按此重写。
#
#   opt  = InitOption_Create()                                  # 无参
#   InitOption_SetModel(opt, modelType, path)                   # (opt, int, char*)
#   InitOption_SetTokenizer(opt, tokType, path)                 # (opt, int, char*)
#   InitOption_SetInferType(opt, inferType)                     # (opt, int)
#   exec = Executor_Create()                                    # 无参
#   Executor_Init_Use_Option(exec, opt)                         # 要求 modelInfo->weightDir 非空
#   ctx  = Context_Create()                                     # 无参
#   Context_SetTemperature/TopP/TopK/Seed/RepetitionPenalty/DoSampleFlag
#   Context_SetStopSeq / SetInitTokenLen / SetMaxGenTokens / SetPrefixPrompt
#   Context_SetOnPrefillGenerateDoneFunc / SetOnSomeTokenGenerateDoneFunc
#   Context_SetOnAllTokensGenerateDoneFunc / SetOnGenerateAsyncFailed   # 流式靠前两个
#   prompt = Prompt_Create()
#   Prompt_SetTokenIds(prompt, int32* ids, uint32 n)
#   rc = Executor_GenerateAsync(exec, ctx, prompt)              # 与同步 Generate 同形
#   Context_GetGenerateStatus / GetOneTokenGeneration           # 轮询逐 token
#   Context_TerminateOnce(ctx) ; Prompt_Destroy ; Context_Destroy
#
# 关键地址（libhiai_llm_engine.so）：
#   InitOption_Create 0xfa468   SetInferType 0xfc0c4   SetTokenizer 0xfc118
#   SetModel 0xfc1a8            Executor_Create 0xfc2b8 Init_Use_Option 0xfc750
#   Prompt_Create 0xfa82c       Prompt_SetText 0xfaa00 Prompt_SetTokenIds 0xfbedc
#   Executor_Generate 0xfd884   GenerateAsync 0xfdbec   Context_Create 0x1083c8
#   Context_CreateFromContextJson 0x10d148              SetMaxGenTokens 0x108f30
#
# 两个环境事实（都已实测）：
#   * 服务是 dlopen/dlsym 按名字取这些符号的 —— 该 .so 里没有静态调用点，
#     所以 xref 追不到调用序列；要找它真正的参数值得定位 dlsym 调用点。
#   * hilog 在本沙箱不可读（抓到 0 行），引擎自己的报错原文看不到。
# ============================================================================


# ----------------------------------------------------------------------------
# 追加：InitOption 路径的实测进展（卡点定位）
# ----------------------------------------------------------------------------
# 已实测通过：
#   InitOption_Create() → opt                       ✓
#   InitOption_SetModel(opt, modelType, path)       ✓ rc=0
#   InitOption_SetTokenizer(opt, tokType, path)     ✓ rc=0
#   InitOption_SetInferType(opt, inferType)         ✓ rc=0
#   Executor_Create() → exec                        ✓（无参）
#   Executor_Init_Use_Option(exec, opt)             ✗ rc=1  ← 卡在这里
#
# Init_Use_Option 的失败条件是它自己的断言（已反编译）：
#     "initOptionImpl->modelInfo->weightDir.size() > 0"  "false, return FAIL."
#   即：opt + 40 处的 modelInfo->weightDir 为空 → 所以问题 = “谁给 modelInfo 填 weightDir”
#
# 已排除的候选：
#   * SetModel(opt, modelType, path) 的 path = **.omc 模型文件**（不是目录！）
#       传 "qwen7b.omc" → 干净返回 rc=1；传 "./" 或 "." → **段错误**
#   * SetModelComponent(opt, modelInfo) = 设**组件模型**（modelInfo 首个 int 是 modelType，
#       合法范围 [3, 3+0x76)，例如 5 = lmhead）—— **不涉及 weightDir**
#
# 下一步（新会话从这里接）：
#   反编译 SetModel(0xfc1a8) 本体 —— 它正是填 opt+40 那个 modelInfo 的函数，
#   看它对 path 做了什么、以及 weightDir 是自推导还是要另设。
#   x570 上结果在 ~/re/setmodel.txt（脚本 ~/re/run4.sh）。
#
# 环境事实（已实测，别再试）：
#   * hilog 在本沙箱读不到（抓 0 行）→ 引擎自己的报错原文看不到
#   * 服务用 dlopen/dlsym 按名字取引擎符号 → 该 .so 无静态调用点，xref 追不到调用序列
#   * Init_Use_Option 对入参敏感：错误取值会**段错误**，因此实验必须**一变体一进程**
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ★★★ 根因确定：InitOption_SetModel 的第 3 参**不是字符串，是 modelInfo* 结构体**
# ----------------------------------------------------------------------------
# 反编译 SetModel(0xfc1a8)：
#     HIAI_LLMEngine_InitOption_SetModel(initOption /*a1*/, int modelType /*a2*/, void* a3)
#     {
#         *(_DWORD *)(initOption + 32) = modelType;   // opt+32 = modelType
#         *(_QWORD *)(initOption + 40) = a3;          // opt+40 = ★ 原样存指针
#         return 0;
#     }
#
# 而 Init_Use_Option(0xfc750) 这样消费 opt+40：
#     v3 = *((_QWORD *)opt + 5);            // 取 opt+40
#     v4 = *(unsigned __int8 *)(v3 + 40);   // 当作结构体
#     v5 = *(_QWORD *)(v3 + 48);
#     "initOptionImpl->modelInfo->weightDir.size() > 0"  "false, return FAIL."
#
# 结论：opt+40 期望的是 **modelInfo 结构体指针**（其 +40 处是 std::string weightDir）。
# 我们传 Python 字符串指针 → 引擎把字符串内容当结构体读 → weightDir 视为空 → rc=1。
# 这同时解释了实测的三种表现：
#     传 "qwen7b.omc" → 垃圾恰似空串 → 干净 rc=1
#     传 "./" 或 "."   → 垃圾被当成巨大长度 → **段错误**
#
# 因此修法有两条：
#   A) 在 Python 侧按布局构造 modelInfo（+40 放 libc++ std::string weightDir）——需精确匹配 ABI，易错
#   B) ★ 写一个极小的 C 辅助库，用真正的 C++ std::string 组装 modelInfo 并代调
#      InitOption_SetModel / Executor_Init_Use_Option —— 绕开 ABI 猜测，最稳
#      设备上已有 aarch64-unknown-linux-ohos-clang，且 lldb_test/llm_run_hiai 就是这么编的
#
# 相关文件：
#   x570 ~/re/setmodel.txt   = SetModel 反编译结果
#   x570 ~/re/smc.txt        = SetModelComponent 反编译（组件模型，非 weightDir）
#   设备 probe2.py           = 单变体探针（一变体一进程，避免段错误带崩整轮）
# ----------------------------------------------------------------------------
