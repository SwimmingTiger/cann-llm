# -*- coding: utf-8 -*-
"""hiai 后端：驱动**系统内部**引擎 ``libhiai_llm_engine.so``（``HIAI_LLMEngine_*``）。

调用序列**照抄系统服务**（``libhm_model_engine_service.z.so`` 的
``AIMM::HIAI::HiaiSession``，文件名 ``hiai_session.cpp``）—— **全部使用导出符号**，
不碰任何内部函数：

    初始化（一次）:  opt = InitOption_Create()
                    InitOption_SetInferType(opt, inferType)
                    InitOption_SetTokenizer(opt, tokenizerType, tokenizerPath)
                    mi = LMEngine_ModelInfo_Create()
                    LMEngine_ModelInfo_SetModelPath(mi, modelPath)
                    LMEngine_ModelInfo_SetWeightDir(mi, weightDir)
                    LMEngine_ModelInfo_SetModelType(mi, 0)
                    InitOption_SetModel(opt, 0, mi)
                    exec = Executor_Create()
                    Executor_Init_Use_Option(exec, opt)
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
import random
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

    ★ **只有 context 那一份会真的交给引擎**（走 ``CreateFromContextJson``，与服务一致）。
      ``executor`` 那份现在【不再】送进引擎 —— 建 Executor 改走官方服务的
      ``InitOption`` 那条路，模型的结构超参由引擎自己按 ``modelPath`` 去读
      ``<omc 同名>.json``。这里仍然合成它、``load()`` 仍然读它，是因为
      ``bos_token_id`` / ``stopSeq`` / ``initTokenLen`` 这几个标量还要用
      （见 :func:`engine_options` / :func:`config_file_for_model`）。
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


def engine_options(model_dir: str) -> Dict[str, Any]:
    """``api_config.json`` 里喂给 ``InitOption`` / ``ModelInfo`` 的那几项。

    这是官方服务的做法：它只拿这几项去建 Executor，**模型的结构超参
    （num_hidden_layers / hidden_size / kv_cache_max_len / embedding 权重文件名 …）
    不在这里传** —— 引擎自己按 ``modelPath`` 推出配置文件名去读
    （见 :func:`config_file_for_model`），也就是模型目录里那份与 ``.omc``
    同名的扁平 ``<model>.json``。
    """
    path = os.path.join(model_dir, "api_config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            api = json.load(fh)
    except (OSError, ValueError) as e:
        raise ModelLoadError(f"读不了 {path}: {e}") from e
    if not isinstance(api, dict):
        raise ModelLoadError(f"{path} 不是 JSON 对象")
    return {
        "inferType": int(api.get("inferType") or 0),
        "tokenizerType": int(api.get("tokenizerType") or 0),
        "tokenizerPath": str(api.get("tokenizerPath") or "tokenizer.json"),
        "modelPath": str(api.get("modelPath") or ""),
        "weightDir": str(api.get("weightDir") if api.get("weightDir") is not None else "./"),
    }


def config_file_for_model(model_path: str) -> str:
    """复刻引擎的 ``InitOptionPacker::GetConfigFilePath``（@0x130088，反编译）。

    引擎内部就是：**取最后一个 ``.`` 之前的部分 + ``".json"``**；没有 ``.`` 就报
    ``model config path error``。所以 ``qwen3_8b_ceval_g256.omc`` →
    ``qwen3_8b_ceval_g256.json`` —— 这正是官方包里模型配置与 ``.omc`` 同名的原因，
    也是 llm_config 那条路真正读的文件。
    """
    i = model_path.rfind(".")
    return (model_path[:i] + ".json") if i >= 0 else ""


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
        # ---- 建 Executor：照官方服务（InitOption 那条路），【不用】CreateFromJson ----
        # 反编译来源：libai_large_model_enginesvr.z.so 的
        #   OHOS::AI::LargeModelEngineBase::LoadEngine @0x1deb04
        #       ModelInfo_Create → SetEngineModelBuffer(→ SetModelPath/SetWeightDir)
        #       → InitOption_SetModel(option, 0, modelInfo)
        #   OHOS::AI::LargeModelEngineBase::SetInferTypeAndTokenizer @0x1df2b8
        #       InitOption_SetInferType(option, 0)
        #       InitOption_SetTokenizer(option, tokenizerType, path)
        # 参数类型全部来自引擎侧的反编译（不是猜的）：
        #   InitOption_SetInferType(opt, int)                    @0xfc0c4
        #   InitOption_SetTokenizer(opt, int, const char*)       @0xfc118
        #       → 引擎内部 std::string::assign(opt+8, ptr)，第 3 参是【C 字符串】
        #   InitOption_SetModel(opt, int, ModelInfo*)            @0xfc1a8
        #   ModelInfo_SetModelPath / SetWeightDir (mi, const char*) @0x2cba9c / 0x2cbb20
        #   ModelInfo_SetModelType(mi, int)                      @0x2cbc28
        #   Executor_Create(void)                                @0xfc2b8
        #   Executor_Init_Use_Option(exec, opt)                  @0xfc750
        "HIAI_LLMEngine_InitOption_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_InitOption_SetInferType": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        # ★ 第 3 参必须声明成 c_char_p（引擎自己 assign 出一份 std::string）——
        #   这正是本后端【不需要】那个 C++ shim 的原因（旧记录里的 shim 是为
        #   「自己拼 std::string 塞进 ModelInfo」用的，官方 API 直接收 char*）。
        "HIAI_LLMEngine_InitOption_SetTokenizer": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]),
        "HIAI_LLMEngine_InitOption_SetModel": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]),
        "HIAI_LMEngine_ModelInfo_Create": (ctypes.c_void_p, []),
        "HIAI_LMEngine_ModelInfo_SetModelPath": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
        "HIAI_LMEngine_ModelInfo_SetWeightDir": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
        "HIAI_LMEngine_ModelInfo_SetModelType": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "HIAI_LLMEngine_Executor_Create": (ctypes.c_void_p, []),
        "HIAI_LLMEngine_Executor_Init_Use_Option": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        # ★ 这里**故意不绑定** `HIAI_LLMEngine_Executor_CreateFromJson`（旧的 JSON 入口）：
        #   Qwen3-8B 走它加载即 abort，而官方服务压根不用它。不声明就调不出去，
        #   顺便让本项目那条自查法（called == declared ⊆ exported）保持成立。
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
        # ---- 采样参数：**按请求**下发（引擎为每个 Context 导出了这些 setter）----
        # ★ 签名不是猜的：probe 用「设进去再读回来」逐条验证过（见 _probe_sampler*.py）。
        #   · 浮点参数是 **float(32 位)**，不是 double —— 传 c_double 会被读成 0.0
        #     （aarch64 上 c_double 走 d0、c_float 走 s0，即 d0 的低半；
        #      0.5 的 double 位模式低 32 位恰好是 0x00000000）
        #   · 整数/布尔参数走通用寄存器
        #   · 这些 Get*/Set* 都是 (ctx, T*) / (ctx, T) 且返回 0 表示成功
        "HIAI_LLMEngine_Context_SetTemperature": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_float]),
        "HIAI_LLMEngine_Context_SetTopP": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_float]),
        "HIAI_LLMEngine_Context_SetTopK": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "HIAI_LLMEngine_Context_SetSeed": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        # ★ 实测默认值：do_sample=0、greedy=0、seed=99、topK=100
        #   —— 采样【默认是关的】，所以同一提示每次都得到完全相同的输出。
        #   每次请求都要显式打开采样并换一个种子。
        "HIAI_LLMEngine_Context_SetDoSampleFlag": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_ubyte]),
        "HIAI_LLMEngine_Context_SetSampleGreedy": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_ubyte]),
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
    # ★ 不做翻译：直接给引擎的返回码与回调名，具体原因看下面的原始日志
    head = (f"GenerateAsync 返回 {rc}" if rc is not None
            else "GenerateAsync 失败回调被触发")
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


def _engine_log_suffix() -> str:
    """加载失败时附在异常消息后面的原始日志（与生成失败用的是同一套）。

    加载阶段是最需要日志的地方（缺文件、参数不匹配都在这里暴露）；引擎自己的
    原话不翻译、不加工，读不到才退回那几条"可能原因"。
    """
    from ..enginelog import format_engine_log, recent_engine_log
    extra = format_engine_log()
    if not recent_engine_log():
        extra = ("\n常见可能：\n"
                 "    · 当前终端没有访问 NPU 的权限（换一个系统终端试试）\n"
                 "    · 模型目录不完整（缺 <model>.json / 权重 / tokenizer）" + extra)
    return extra






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
        from ..modelpkg import derive_model_name
        self.model_id = model_id or derive_model_name(self.model_dir)
        self.default_params = default_params or GenerationParams()
        self._bind: Optional[_HiaiBindings] = None
        self._ctx: Optional[int] = None          # 仅代表"最近一次"的 Context
        self._exec: Optional[int] = None
        # InitOption / ModelInfo：引擎把这两个指针原样存进 executor，必须活到进程结束
        self._opt: Optional[int] = None
        self._model_info: Optional[int] = None
        self._ctx_json: bytes = b""               # 每请求用它新建 Context
        self._cb_done = None                       # 回调需长期持有，勿被 GC
        self._cb_fail = None
        self._cb_some = None
        self._ids = []
        self._bos: int = -1                        # 引擎期望的 BOS（缺它会 Generate 失败）
        self._init_token_len: int = 0              # 模型配置里的 initTokenLen（prefill 长度）
        self._stop_seq: list = []                  # 停止序列（模型配置里的 stopSeq）
        # 模型自带的采样配置（api_config.json），load() 时读入 —— 权威来源
        self._model_sampler: Dict[str, Any] = {}
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
        # ★ 模型自带的采样配置（api_config.json）—— 权威来源。
        #   上层会用它来填"用户没显式指定"的采样项，避免我们用写死的值覆盖模型真值。
        #   （实测踩过：本项目的默认 top_p=0.95 盖掉了官方模型的 0.8。）
        from ..modelcfg import read_sampler
        self._model_sampler = read_sampler(self.model_dir)
        self._bind = _HiaiBindings(self._lib_path)

        # 引擎按相对路径解析模型文件
        os.chdir(self.model_dir)

        # ★ 只建 Executor —— 验证过的配方里【不】预先建 Context（预建会 SIGTRAP）。
        #   Context 每请求用 Context_Create()（无参）新建。
        # ★ 传 JSON 内容（不是文件名）—— 那是 context 那一路，它仍走 CreateFromContextJson
        #   （服务也是这么做的，见 hiai_session.cpp）。**只有 Executor 换了入口。**
        self._ctx_json = json.dumps(context_cfg).encode()
        self._ctx = None
        # 失败时 _create_executor 自己抛 ModelLoadError（附原始日志），不返回 None
        self._exec = self._create_executor()

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

    # ------------------------------------------------------------- 建 Executor

    def _create_executor(self):
        """按**官方服务**的序列建 Executor（``InitOption`` 那条路）。

        为什么不用 ``Executor_CreateFromJson(整份 JSON)``
        ------------------------------------------------
        那是**次要入口**：流量最大的那条路（``libai_large_model_enginesvr`` /
        ``libhm_model_engine_service``）用的是下面这套逐项 setter，两个服务的导入表里
        **根本没有** ``Executor_CreateFromJson``。实测差别是决定性的：

        * 走 JSON 入口时，引擎要用我们**合成**的 executor JSON 去填 ``llmConfig_``；
          Qwen3-8B 的 ``<model>.json`` 里 ``architectures`` 是数组，引擎在 LoRA
          相关代码里按字符串取它 → ``nlohmann::json type_error.302`` 直接 abort。
        * 走这条路时引擎**不读我们拼的 JSON**，而是自己按 ``modelPath`` 去掉扩展名
          ＋ ``.json`` 去找模型自带的那份扁平 ``<model>.json``
          （``InitOptionPacker::GetConfigFilePath`` @0x130088，反编译见
          :func:`config_file_for_model`）。7B / 8B 两种情况实测都 rc=0。

        参数取值（都来自反编译，不是猜的）
        ----------------------------------
        * ``InitOption_SetModel(opt, 0, mi)`` —— 第 2 参官方传 **0**
          （``LargeModelEngineBase::LoadEngine`` @0x1deb04 的调用点实锤）
        * ``ModelInfo->modelType`` 官方**不设**（保持 ``ModelInfo_Create`` 的清零值 0）——
          官方的 ``SetEngineModelBuffer``（@0x1dee08，反编译）只做三件事：
          ``SetModelBuffer`` → ``SetWeightDir`` → ``SetModelPath``，**没有** ``SetModelType``；
          那个 ``SetModelType(..., 3)`` 是给 **lmhead 组件** 的（同一个 LoadEngine 里，
          设置完紧接着 ``SetModelComponent``）。实测基座给 0 与给 3 都能 rc=0，
          这里按官方取证取 0。（``[3,121)`` 那条断言是 ``SetModelComponent`` 自己的。）
        * ``weightDir`` **不能为空** —— ``Init_Use_Option`` 自己会断言
          ``initOptionImpl->modelInfo->weightDir.size() > 0``，空了直接返回失败
        """
        assert self._bind is not None
        lib = self._bind.lib

        opts = engine_options(self.model_dir)

        # 引擎按 modelPath 推配置文件，推不出来 / 文件不在 → 到引擎里只会得到一句
        # "readConfigBuffer null"，在这里先说清楚是哪个文件。
        cfg_name = config_file_for_model(opts["modelPath"])
        if not cfg_name:
            raise ModelLoadError(
                f"api_config.json 的 modelPath={opts['modelPath']!r} 里没有 '.'，"
                f"引擎推不出模型配置文件（GetConfigFilePath 会报 model config path error）")
        if not os.path.isfile(cfg_name):
            raise ModelLoadError(
                f"引擎要读的模型配置文件不存在：{cfg_name}\n"
                f"    （它由 api_config.json 的 modelPath={opts['modelPath']!r} "
                f"去掉扩展名 + .json 得来）\n"
                f"    补齐：python -m cann_llm.modelpkg {self.model_dir}")
        if not opts["weightDir"]:
            raise ModelLoadError(
                f"api_config.json 的 weightDir 是空的；引擎的 Init_Use_Option 会直接失败")

        # ★ option / modelInfo 的生命周期必须覆盖整个进程：引擎把这两个指针原样存进
        #   executor（Init 里把 option 传给了流水线），提前 Destroy 就是 use-after-free。
        #   一次 load 漏 96 + 112 字节，无所谓；这里挂在 self 上防止被 GC 回收。
        opt = lib.HIAI_LLMEngine_InitOption_Create()
        if not opt:
            raise ModelLoadError("InitOption_Create 返回空。")
        mi = lib.HIAI_LMEngine_ModelInfo_Create()
        if not mi:
            raise ModelLoadError("LMEngine_ModelInfo_Create 返回空。")

        steps = (
            ("InitOption_SetInferType",
             lambda: lib.HIAI_LLMEngine_InitOption_SetInferType(opt, opts["inferType"])),
            ("InitOption_SetTokenizer",
             lambda: lib.HIAI_LLMEngine_InitOption_SetTokenizer(
                 opt, opts["tokenizerType"], opts["tokenizerPath"].encode())),
            # 官方 SetEngineModelBuffer 的次序：weightDir 在 modelPath 之前
            ("ModelInfo_SetWeightDir",
             lambda: lib.HIAI_LMEngine_ModelInfo_SetWeightDir(
                 mi, opts["weightDir"].encode())),
            ("ModelInfo_SetModelPath",
             lambda: lib.HIAI_LMEngine_ModelInfo_SetModelPath(
                 mi, opts["modelPath"].encode())),
            # 显式写 0 == ModelInfo_Create 的清零值 == 官方基座的实际值。
            # 写出来是为了让「代码里调用的符号」与 SIGS 声明的一一对应
            # （项目自查法要求 called == declared；见 handoff 文档末尾那节）。
            ("ModelInfo_SetModelType",
             lambda: lib.HIAI_LMEngine_ModelInfo_SetModelType(mi, 0)),
            ("InitOption_SetModel",
             lambda: lib.HIAI_LLMEngine_InitOption_SetModel(opt, 0, mi)),
        )
        for name, call in steps:
            rc = call()
            if rc != 0:
                self._opt, self._model_info = opt, mi
                raise ModelLoadError(f"{name} 返回 {rc}（0 才是成功）"
                                     + _engine_log_suffix())

        self._opt, self._model_info = opt, mi

        ex = lib.HIAI_LLMEngine_Executor_Create()
        if not ex:
            raise ModelLoadError("Executor_Create 返回空。" + _engine_log_suffix())
        rc = lib.HIAI_LLMEngine_Executor_Init_Use_Option(ex, opt)
        if rc != 0:
            raise ModelLoadError(
                f"Executor_Init_Use_Option 返回 {rc}（0 才是成功）。" + _engine_log_suffix())
        return ex

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

    def sampler_defaults(self) -> Dict[str, Any]:
        """模型自带的采样配置（``api_config.json``），load() 时读入。

        **不含 seed** —— 那个由"每次请求换一个随机种子"的策略决定，
        沿用模型里写死的 99 会让每次输出完全相同。
        """
        return {k: v for k, v in self._model_sampler.items() if k != "seed"}

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

        # ---- 采样参数：**必须每个请求显式下发** ----
        # 每个请求都是全新的 Context，其采样默认值来自 executor，实测是：
        #     do_sample = 0（★ 采样关着）、greedy = 0、seed = 99、topK = 100
        # 于是同一提示每次都会得到【完全一样】的输出。这里逐项下发，
        # 让 temperature / top_k / top_p / seed 真正生效。
        # （签名见 _SIGS 处的说明：浮点是 float32，不是 double。）
        lib = self._bind.lib
        greedy = bool(getattr(p, "greedy", False))
        lib.HIAI_LLMEngine_Context_SetDoSampleFlag(ctx, 0 if greedy else 1)
        lib.HIAI_LLMEngine_Context_SetSampleGreedy(ctx, 1 if greedy else 0)
        if not greedy:
            lib.HIAI_LLMEngine_Context_SetTemperature(
                ctx, float(getattr(p, "temperature", 0.7) or 0.7))
            topk = int(getattr(p, "top_k", 0) or 0)
            if topk > 0:
                lib.HIAI_LLMEngine_Context_SetTopK(ctx, topk)
            topp = float(getattr(p, "top_p", 0.0) or 0.0)
            if 0.0 < topp <= 1.0:
                lib.HIAI_LLMEngine_Context_SetTopP(ctx, topp)
            # 种子：调用方显式给了就用它（可复现）；没给就【每次换一个】——
            # 这正是"每次返回相同结果"的解药。固定 99 是引擎的默认值，别沿用。
            seed = getattr(p, "seed", None)
            if seed is None:
                seed = random.randrange(1, 2 ** 31 - 1)
            lib.HIAI_LLMEngine_Context_SetSeed(ctx, int(seed) & 0x7FFFFFFF)

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
