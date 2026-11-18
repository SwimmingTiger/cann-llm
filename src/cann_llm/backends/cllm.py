"""cann-local-lm 引擎后端（``libcann_lm_engine.so``）。

与 :mod:`cann_llm.backends.cann` 的区别（**名字很像，别混**）：

======================  ====================================  ==============================
                        cann 后端（本文件之外）                 cllm 后端（本文件）
======================  ====================================  ==============================
加载的库                 ``/system/lib64/ndk/libcann_llm_engine.so``（系统）  ``libcann_lm_engine.so``（**自己编的**）
C API 前缀               ``HMS_LLMEngine*``                     ``CANN_LMEngine*``
模型形态                 官方 LLM 引擎包（api_config.json 等）    **prefill.omc + decode.omc + 外置权重**
算子执行                 系统 LLM 引擎                          设备 HiAI/NNCore（跑我们编译出的 .omc）
======================  ====================================  ==============================

模型目录（或 ``testcase.json``）需要包含::

    <dir>/prefill/prefill.omc      <dir>/prefill/weight/     （含 meta.json + *.weight）
    <dir>/decode/decode.omc        <dir>/decode/weight/
    <dir>/model.json               引擎 meta（model_config / embedding_config / tokenizer_config）
    <dir>/tokenizer.json
    <dir>/embedding_weights.bin   外部 embedding 四段文件（模型带 embedding 工位时必需）

引擎库查找顺序：``CANN_LLM_CLLM_LIB`` → ``<模型目录>/{lib,py/cann_local_lm/lib}`` →
``/system/lib64/libcann_lm_engine.so``。

模型怎么转换、怎么部署（含全部踩坑）见 ``docs/cann-local-lm.md``。
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

from ..errors import BackendUnavailableError, GenerationError, ModelLoadError
from ..types import (
    FINISH_ERROR,
    FINISH_LENGTH,
    FINISH_STOP,
    GenerationChunk,
    GenerationParams,
    GenerationRequest,
    GenerationStats,
    ModelInfo,
)
from .base import EngineBackend, register_backend

__all__ = [
    "CllmBackend",
    "CLLM_LIB_ENV",
    "resolve_artifact",
    "find_engine_lib",
    "status_text",
    "split_stop",
    "special_stop_ids",
]

#: 显式指定引擎库的变量
CLLM_LIB_ENV = "CANN_LLM_CLLM_LIB"

#: 模型目录内的引擎库相对位置（部署包两种放法都认）
_LIB_IN_MODEL = (
    os.path.join("lib", "libcann_lm_engine.so"),
    os.path.join("py", "cann_local_lm", "lib", "libcann_lm_engine.so"),
    "libcann_lm_engine.so",
)

#: 系统里也可能有一份
_LIB_SYSTEM = "/system/lib64/libcann_lm_engine.so"

_STATUS_TEXT = {
    0: "SUCCESS",
    1: "FAILURE",
    2: "INVALID_ARGUMENT",
    3: "INVALID_STATE",
    4: "NOT_FOUND",
    5: "CANCELED",
}

#: C 枚举 CANN_LMEngine_FinishReason → 本仓库的 finish_reason 字符串
_FINISH_MAP = {
    0: FINISH_LENGTH,   # MAX_TOKENS
    1: FINISH_STOP,     # STOP_TOKEN
    2: FINISH_STOP,     # STOP_STRING
    3: FINISH_STOP,     # CANCELED
    4: FINISH_ERROR,    # ERROR
}

#: testcase.json 字段 → 统一的 artifact 键（与 cann-local-lm 的 config.py 同名）
_TESTCASE_KEYS = {
    "tokenizer_path": "tokenizer_path",
    "model_path_prefill": "prefill_omc",
    "model_path_decode": "decode_omc",
    "model_meta_config": "meta_json",
    "model_weight_dir_prefill": "prefill_weight",
    "model_weight_dir_decode": "decode_weight",
}


def _as_id_list(value) -> "List[int]":
    """``eos_token_id`` 可能是 int 或 list；统一成 int 列表。"""
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def special_stop_ids(meta: "dict", generation_config: "Optional[dict]" = None) -> "List[int]":
    """该模型的「结束」token id：**模型自报的 eos 列表** + meta 的结束型特殊 token。

    ★为什么不是"所有 special_tokens"★：gemma4 词表里既有开始标记 ``<|turn>``(105)
    也有结束标记 ``<turn|>``(106)；把开始标记也当停止符会提前截断 ✗。
    正确来源是模型自己的 ``generation_config.eos_token_id = [1, 106, 50]``
    （``<turn|>`` 本身就是 EOS ✓），meta 里的 ``special_tokens`` 只作补充，
    且只取"以 ``|>`` 结尾"的结束型标记。
    """
    out: "List[int]" = []
    for i in _as_id_list((generation_config or {}).get("eos_token_id")):
        if i not in out:
            out.append(i)
    tc = (meta or {}).get("tokenizer_config") or {}
    special = tc.get("special_tokens") or {}
    if isinstance(special, dict):
        for name, v in special.items():
            if not (isinstance(name, str) and name.endswith("|>")):
                continue          # <|turn> / <pad> / <bos> 等一律不当停止符 ✗
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if i not in out:
                out.append(i)
    return out


def status_text(code: int) -> str:
    """把 C API 的状态码翻成人读形式（未收录就报数字）。"""
    return _STATUS_TEXT.get(int(code), f"status {int(code)}")


# ---------------------------------------------------------------- 路径解析（纯函数，可单测）

def resolve_artifact(model_dir: str) -> Dict[str, str]:
    """解析模型目录里的 artifact 路径。

    ``<dir>/testcase.json`` 存在时以它为准（相对路径按 ``model_dir`` 展开），
    否则用部署包的固定布局。

    只校验**必需项**是否存在；缺失会一次性列全（不要一个报一次，排查很痛苦）。
    """
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    art: Dict[str, str] = {
        "prefill_omc": os.path.join(model_dir, "prefill", "prefill.omc"),
        "decode_omc": os.path.join(model_dir, "decode", "decode.omc"),
        "prefill_weight": os.path.join(model_dir, "prefill", "weight"),
        "decode_weight": os.path.join(model_dir, "decode", "weight"),
        "meta_json": os.path.join(model_dir, "model.json"),
        "tokenizer_path": os.path.join(model_dir, "tokenizer.json"),
    }
    case = os.path.join(model_dir, "testcase.json")
    if os.path.isfile(case):
        try:
            with open(case, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"读不了 {case}: {exc}") from exc
        for key, target in _TESTCASE_KEYS.items():
            val = raw.get(key)
            if not isinstance(val, str) or not val:
                continue
            path = val if os.path.isabs(val) else os.path.join(model_dir, val)
            art[target] = path

    missing: List[str] = []
    for key in ("prefill_omc", "decode_omc", "meta_json"):
        if not os.path.isfile(art[key]):
            missing.append(f"{key}: {art[key]}")
    for key in ("prefill_weight", "decode_weight"):
        if not os.path.isdir(art[key]):
            missing.append(f"{key}: {art[key]}（应为目录）")
    if missing:
        raise ModelLoadError(
            "cllm 模型目录不完整，缺少：\n    "
            + "\n    ".join(missing)
            + "\n  转换/部署步骤见 docs/cann-local-lm.md（转换 → 编译 → 部署 三段都要做完）。"
        )
    return art


def find_engine_lib(model_dir: Optional[str] = None) -> str:
    """按优先级找引擎库；都没有就抛 :class:`BackendUnavailableError` 并列出试过的路径。

    顺序：``$CANN_LLM_CLLM_LIB`` → ``<cann-llm>/lib/``（自动构建的安装位置）→
    模型目录的 ``lib/`` / ``py/cann_local_lm/lib/`` → ``/system/lib64/``。
    """
    tried: List[str] = []
    env = os.environ.get(CLLM_LIB_ENV)
    if env:
        tried.append(f"{CLLM_LIB_ENV}={env}")
        if os.path.isfile(env):
            return env
    from ..engine_build import repo_root
    repo_lib = os.path.join(repo_root(), "lib", "libcann_lm_engine.so")
    tried.append(repo_lib)
    if os.path.isfile(repo_lib):
        return repo_lib
    if model_dir:
        for rel in _LIB_IN_MODEL:
            cand = os.path.join(model_dir, rel)
            tried.append(cand)
            if os.path.isfile(cand):
                return cand
    tried.append(_LIB_SYSTEM)
    if os.path.isfile(_LIB_SYSTEM):
        return _LIB_SYSTEM
    raise BackendUnavailableError(
        "找不到 cann-local-lm 引擎库 libcann_lm_engine.so，试过：\n    "
        + "\n    ".join(tried)
        + f"\n  用 {CLLM_LIB_ENV}=/path/to/libcann_lm_engine.so 指定，"
        "或把库放到模型目录的 lib/ 下。它是从 cann-local-lm 源码交叉编译并**签名**的，"
        "见 docs/cann-local-lm.md。"
    )


# ---------------------------------------------------------------- ctypes 绑定（与 c_api.h 逐字段对应）

class _Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _ModelArtifact(ctypes.Structure):
    _fields_ = [
        ("prefill_omc_path", ctypes.c_char_p),
        ("prefill_omc_buf", _Buffer),
        ("decode_omc_path", ctypes.c_char_p),
        ("decode_omc_buf", _Buffer),
        ("prefill_weight", ctypes.c_char_p),
        ("decode_weight", ctypes.c_char_p),
        ("meta_json_path", ctypes.c_char_p),
        ("meta_json_buf", _Buffer),
        ("isEnableSharedWeight", ctypes.c_bool),
    ]


class _EngineOptions(ctypes.Structure):
    _fields_ = [
        ("default_chunk_size", ctypes.c_int),
        ("max_concurrent_prefill", ctypes.c_int),
        ("max_batch_size_override", ctypes.c_int),
        ("tokenizer_type", ctypes.c_char_p),
        ("tokenizer_path", ctypes.c_char_p),
    ]


class _SamplingParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("repetition_penalty", ctypes.c_float),
    ]


class _RequestOptions(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int),
        ("stop_token_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("stop_token_ids_count", ctypes.c_size_t),
        ("stop_strings", ctypes.POINTER(ctypes.c_char_p)),
        ("stop_strings_count", ctypes.c_size_t),
    ]


class _Result(ctypes.Structure):
    _fields_ = [
        ("token_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("token_ids_count", ctypes.c_size_t),
        ("text", ctypes.c_char_p),
        ("finish_reason", ctypes.c_int),
    ]


_StreamCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p)


class _Lib:
    """把 C API 绑成 Python 方法（一次绑定，多次调用）。"""

    def __init__(self, path: str) -> None:
        # libc++_shared.so 是引擎的 NEEDED，但进程启动后改 LD_LIBRARY_PATH 没用 ✗
        # ⇒ 先按绝对路径把它 RTLD_GLOBAL 加载进来 ✓（引擎随后就能解析到它）
        cxx = os.path.join(os.path.dirname(os.path.abspath(path)), "libc++_shared.so")
        if os.path.isfile(cxx):
            try:
                ctypes.CDLL(cxx, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        try:
            self.lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            raise BackendUnavailableError(f"加载 {path} 失败：{exc}") from exc
        self.path = path
        self._bind("CANN_LMEngine_ExecutorCreate", ctypes.c_int,
                   [ctypes.POINTER(_ModelArtifact), ctypes.POINTER(_EngineOptions),
                    ctypes.POINTER(ctypes.c_void_p)])
        self._bind("CANN_LMEngine_ExecutorDestroy", None, [ctypes.c_void_p])
        self._bind("CANN_LMEngine_ContextCreate", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)])
        self._bind("CANN_LMEngine_ContextDestroy", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.c_uint64])
        self._bind("CANN_LMEngine_ExecutorSetContext", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.c_uint64])
        self._bind("CANN_LMEngine_ContextCancel", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.c_uint64])
        self._bind("CANN_LMEngine_ExecutorGenerate", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_SamplingParams),
                    ctypes.POINTER(_RequestOptions), ctypes.POINTER(_Result)])
        self._bind("CANN_LMEngine_ExecutorGenerateAsync", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_SamplingParams),
                    ctypes.POINTER(_RequestOptions), ctypes.POINTER(ctypes.c_void_p),
                    _StreamCallback, ctypes.c_void_p])
        self._bind("CANN_LMEngine_RequestIsDone", ctypes.c_int, [ctypes.c_void_p])
        self._bind("CANN_LMEngine_RequestWait", ctypes.c_int,
                   [ctypes.c_void_p, ctypes.POINTER(_Result)])
        self._bind("CANN_LMEngine_RequestCancel", ctypes.c_int, [ctypes.c_void_p])
        self._bind("CANN_LMEngine_RequestDestroy", None, [ctypes.c_void_p])
        self._bind("CANN_LMEngine_ResultDestroy", None, [ctypes.POINTER(_Result)])

    def _bind(self, name: str, restype, argtypes) -> None:
        fn = getattr(self.lib, name, None)
        if fn is None:
            raise BackendUnavailableError(
                f"{self.path} 里没有符号 {name} —— 这个 .so 不是 cann-local-lm 引擎"
                "（系统里那份 libcann_llm_engine.so 用的是 HMS_LLMEngine* 接口，"
                "请用 -b cann）。"
            )
        fn.restype = restype
        fn.argtypes = argtypes


# ---------------------------------------------------------------- 后端实现

@register_backend("cllm")
class CllmBackend(EngineBackend):
    """cann-local-lm（prefill/decode 双图 + 外置权重）后端。

    模型是 **两段**：``prefill``（一次喂 N 个 token）与 ``decode``（一次一个），
    每段都是**整网**（gemma4-E2B 是 35 层全在一个图里），不是按层切段。
    KV 上限由 ``model.json`` 的 ``kv_cache_max_len`` 决定（gemma4 部署时
    = ``sliding_window + prefill_len``）。
    """

    name = "cllm"

    def __init__(self, model_dir: str = "", model_id: Optional[str] = None,
                 context_length: int = 0, lib_path: Optional[str] = None,
                 tokenizer_type: str = "",
                 default_params: Optional[GenerationParams] = None,
                 **_: object) -> None:
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else ""
        self.model_id = model_id or ""
        self.context_length = int(context_length or 0)
        self.lib_path = lib_path or ""
        self.tokenizer_type = tokenizer_type or ""
        self._lib: Optional[_Lib] = None
        self._executor: Optional[int] = None
        self._ctx: Optional[int] = None
        #: CLI/服务在 load() 之后会读它（没显式指定采样参数时用）
        self.default_params = default_params or GenerationParams()
        self._info = ModelInfo(id="unknown", backend=self.name)
        self._stop_ids: "List[int]" = []
        self._lock = threading.Lock()
        #: ★引擎同一时刻只接受一个在飞请求★（第二个会 INVALID_STATE）。
        #: 客户端断连后的"遗留请求"由后台线程继续抽干；这期间新请求必须排队 ✓
        self._gen_lock = threading.Lock()
        self._closed = False

    # ---- 生命周期

    def load(self) -> ModelInfo:
        if self._executor:
            return self._info
        if not self.model_dir:
            raise ModelLoadError("cllm 后端需要模型目录（-d 或 models/<名字>/）")
        art = resolve_artifact(self.model_dir)
        if self._lib is None:
            lib_path = self.lib_path or os.environ.get(CLLM_LIB_ENV) or ""
            if not lib_path or not os.path.isfile(lib_path):
                try:
                    lib_path = find_engine_lib(self.model_dir)
                except BackendUnavailableError:
                    # ★缺库就编★（需要源码树 + OHOS SDK；条件不足时这里会抛出带指引的错）
                    from ..engine_build import ensure_engine_lib
                    lib_path = ensure_engine_lib(self.model_dir)
            self._lib = _Lib(lib_path)
        lib = self._lib

        meta = _read_json(art["meta_json"])
        gen_cfg = None
        gen_path = os.path.join(self.model_dir, "generation_config.json")
        if os.path.isfile(gen_path):
            try:
                gen_cfg = _read_json(gen_path)
            except (OSError, ValueError):
                gen_cfg = None
        self._stop_ids = special_stop_ids(meta, gen_cfg)
        mc = meta.get("model_config") or {}
        kv_max = int(mc.get("kv_cache_max_len") or 0)
        vocab = int(mc.get("vocab_size") or 0)
        tok_type = (self.tokenizer_type
                    or os.environ.get("CANN_LLM_CLLM_TOKENIZER", "")
                    or _default_tokenizer_type(self.model_dir))

        self._bytes: Dict[str, bytes] = {
            k: os.fsencode(v) for k, v in art.items()
        }
        artifact = _ModelArtifact(
            prefill_omc_path=self._bytes["prefill_omc"], prefill_omc_buf=_Buffer(),
            decode_omc_path=self._bytes["decode_omc"], decode_omc_buf=_Buffer(),
            prefill_weight=self._bytes["prefill_weight"],
            decode_weight=self._bytes["decode_weight"],
            meta_json_path=self._bytes["meta_json"], meta_json_buf=_Buffer(),
            isEnableSharedWeight=False,
        )
        options = _EngineOptions(
            default_chunk_size=1024, max_concurrent_prefill=1, max_batch_size_override=0,
            tokenizer_type=tok_type.encode() if tok_type else None,
            tokenizer_path=self._bytes.get("tokenizer_path"),
        )
        out = ctypes.c_void_p()
        st = lib.lib.CANN_LMEngine_ExecutorCreate(
            ctypes.byref(artifact), ctypes.byref(options), ctypes.byref(out))
        if st != 0 or not out.value:
            raise ModelLoadError(
                f"引擎创建失败（{status_text(st)}）：模型路径存在但引擎拒绝了。\n"
                f"  常见原因：kv_cache_max_len 与图维度不一致、embedding 配置与图工位不符、\n"
                f"  weight/meta.json 里仍是编译机的绝对路径。排错表见 docs/cann-local-lm.md。"
            )
        self._executor = out.value

        ctx = ctypes.c_uint64(0)
        st = lib.lib.CANN_LMEngine_ContextCreate(self._executor, ctypes.byref(ctx))
        if st != 0:
            self.close()
            raise ModelLoadError(f"ContextCreate 失败（{status_text(st)}）")
        self._ctx = ctx.value
        st = lib.lib.CANN_LMEngine_ExecutorSetContext(self._executor, self._ctx)
        if st != 0:
            self.close()
            raise ModelLoadError(f"ExecutorSetContext 失败（{status_text(st)}）")

        self._info = ModelInfo(
            id=self.model_id or os.path.basename(self.model_dir.rstrip(os.sep)) or "cllm",
            backend=self.name,
            path=self.model_dir,
            context_length=self.context_length or kv_max,
            max_tokens=kv_max,
            chat_template=_chat_template(mc, self.model_dir),
            vocab_size=vocab,
            extra={"engine_lib": lib.path, "tokenizer_type": tok_type},
        )
        return self._info

    def close(self) -> None:
        # ★先等在飞的（含被移交给后台抽干的）请求结束★：ContextDestroy 有在飞请求
        #   会 INVALID_STATE，且销毁期间引擎还在回调就是段错误 ✗
        self._gen_lock.acquire()
        try:
            self._close_locked()
        finally:
            self._gen_lock.release()

    def _close_locked(self) -> None:
        with self._lock:
            lib = self._lib
            if lib is not None:
                if self._executor and self._ctx:
                    try:
                        lib.lib.CANN_LMEngine_ContextDestroy(self._executor, self._ctx)
                    except Exception:      # noqa: BLE001 —— 退出路径不因清理报错
                        pass
                if self._executor:
                    try:
                        lib.lib.CANN_LMEngine_ExecutorDestroy(self._executor)
                    except Exception:      # noqa: BLE001
                        pass
            self._executor = None
            self._ctx = None
            self._closed = True

    # ---- 生成

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        if not self._executor or self._lib is None:
            raise GenerationError("cllm 后端尚未 load()")
        if self._closed:
            raise GenerationError("cllm 后端已 close()")

        self._gen_lock.acquire()
        handoff = False
        params = request.params
        sampling = _SamplingParams(
            temperature=float(params.temperature),
            top_k=int(params.top_k),
            top_p=float(params.top_p),
            seed=int(params.seed or 0),
            repetition_penalty=float(params.repetition_penalty),
        )
        stop_keep = [s.encode() for s in (params.stop or ()) if s]
        stop_arr = (ctypes.c_char_p * len(stop_keep))(*stop_keep) if stop_keep else None
        stop_ids = list(getattr(self, "_stop_ids", []) or [])
        ids_arr = (ctypes.c_uint32 * len(stop_ids))(*stop_ids) if stop_ids else None
        options = _RequestOptions(
            max_tokens=int(params.max_tokens),
            stop_token_ids=ctypes.cast(ids_arr, ctypes.POINTER(ctypes.c_uint32)) if ids_arr else None,
            stop_token_ids_count=len(stop_ids),
            stop_strings=ctypes.cast(stop_arr, ctypes.POINTER(ctypes.c_char_p)) if stop_arr else None,
            stop_strings_count=len(stop_keep),
        )

        lib = self._lib.lib
        chunks: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()

        def _on_token(token, text, text_len, _ud):  # noqa: ANN001 —— ctypes 回调
            try:
                chunks.put((int(token), bytes(text[:text_len]) if text else b""))
            except Exception:              # noqa: BLE001 —— 回调里不能抛
                pass

        cb = _StreamCallback(_on_token)
        req = ctypes.c_void_p()
        prompt = request.prompt.encode()
        st = lib.CANN_LMEngine_ExecutorGenerateAsync(
            self._executor, prompt, ctypes.byref(sampling), ctypes.byref(options),
            ctypes.byref(req), cb, None)
        if st != 0 or not req.value:
            raise GenerationError(
                f"GenerateAsync 失败（{status_text(st)}）—— 检查 EngineOptions.tokenizer_path "
                "是否指向可加载的 tokenizer.json（路径不可加载会让 tokenizer 置空）。")

        stop_text = [s.decode("utf-8", "replace") for s in stop_keep]
        t0 = time.monotonic()
        index = 0
        buf = ""
        stopped = False
        done = False          # 引擎侧是否已正常完成（决定 finally 要不要取消）
        try:
            while True:
                drained = False
                while True:
                    try:
                        tok, raw = chunks.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    buf += raw.decode("utf-8", "replace")
                    out, buf, stopped = split_stop(buf, stop_text)
                    if out:
                        yield GenerationChunk(text=out, index=index, token_id=tok,
                                              finish_reason=None, stats=None)
                        index += 1
                    if stopped:
                        break
                if stopped or lib.CANN_LMEngine_RequestIsDone(req):
                    break
                if not drained:
                    time.sleep(0.01)
            if buf and not stopped:
                out, _, _ = split_stop(buf, stop_text)
                if out:
                    yield GenerationChunk(text=out, index=index, finish_reason=None, stats=None)
                    index += 1

            result = _Result()
            st = lib.CANN_LMEngine_RequestWait(req, ctypes.byref(result))
            if st != 0:
                raise GenerationError(f"RequestWait 失败（{status_text(st)}）")
            try:
                n_new = int(result.token_ids_count)
                finish = _FINISH_MAP.get(int(result.finish_reason), FINISH_ERROR)
                # ★只在"一个 token 都没流出来"时才用 result.text 兜底★——
                #   GenerateAsync 的 result.text 是**全文**，流式路径下再吐一次就重复了 ✗
                tail = ""
                if result.text and index == 0:
                    tail = bytes(result.text).decode("utf-8", "replace")
                    for stop in stop_keep:
                        tail = tail.replace(stop.decode("utf-8", "replace"), "")
                    if tail:
                        index += 1
                wall = time.monotonic() - t0
                # 引擎不回传 prefill/decode 分项 ⇒ 用总耗时填 decode（tok/s 才不是 0 ✗）
                stats = GenerationStats(
                    prompt_tokens=0,
                    completion_tokens=n_new,
                    prefill_ms=0.0,
                    decode_ms=wall * 1000.0,
                    total_ms=wall * 1000.0,
                    wall_s=wall,
                )
                done = True                  # RequestWait 已完成 ⇒ 引擎无在飞请求
                yield GenerationChunk(text=tail, index=index, finish_reason=finish, stats=stats)
            finally:
                lib.CANN_LMEngine_ResultDestroy(ctypes.byref(result))
        finally:
            if done:
                lib.CANN_LMEngine_RequestDestroy(req)
            else:
                # ★提前退出（客户端断连/异常）★：不能直接 Cancel/Destroy ✗ ——
                #   RequestCancel 在这个异步实现上会段错误（实测）；
                #   直接 Destroy 则引擎回调打进已释放闭包 → 段错误 ✗；
                #   不销毁就走下一个请求 ⇒ GenerateAsync INVALID_STATE ✗。
                #   唯一安全做法：移交后台线程继续抽干到自然结束，期间持有
                #   _gen_lock 让新请求排队（引擎只支持单在飞请求 ✓）
                handoff = True
                threading.Thread(target=self._abort_request_bg,
                                 args=(req, chunks, cb), daemon=True).start()
            if not handoff:
                self._gen_lock.release()

    def _abort_request_bg(self, req, chunks, keepalive) -> None:
        """把"客户端已断连"的请求继续抽干到自然结束，再销毁（释放串行锁）。

        实测（KirinX90 上的 libcann_lm_engine.so）：
          · ``RequestCancel`` ⇒ 段错误 ✗
          · 直接 ``RequestDestroy`` ⇒ 引擎回调进已释放闭包 ⇒ 段错误 ✗
          · 不销毁就走下一个请求 ⇒ ``GenerateAsync INVALID_STATE`` ✗
        所以只能"等它自己跑完"：引擎同一时刻只接受一个在飞请求 ✓
        """
        # 引擎回调指针就是它，整个后台期间必须持有强引用（否则段错误 ✗）
        self._abort_keepalive = keepalive
        lib = self._lib.lib if self._lib else None
        if lib is None:
            self._gen_lock.release()
            return
        try:
            deadline = time.monotonic() + 900.0      # 抽干上限（由 max_tokens 决定）
            while time.monotonic() < deadline:
                try:
                    if lib.CANN_LMEngine_RequestIsDone(req):
                        break
                except Exception:          # noqa: BLE001
                    break
                while True:                # 回调还在往队列里塞 ⇒ 持续排空
                    try:
                        chunks.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:      # noqa: BLE001
                        break
                time.sleep(0.01)
            try:                           # 先取走 Result 再销毁（与正常路径一致）
                res = _Result()
                if lib.CANN_LMEngine_RequestWait(req, ctypes.byref(res)) == 0:
                    lib.CANN_LMEngine_ResultDestroy(ctypes.byref(res))
            except Exception:              # noqa: BLE001
                pass
            try:
                lib.CANN_LMEngine_RequestDestroy(req)
            except Exception:              # noqa: BLE001
                pass
        finally:
            self._abort_keepalive = None
            self._gen_lock.release()

    # ---- 可选能力

    @property
    def supports_streaming(self) -> bool:
        return True

    def sampler_defaults(self) -> "Dict[str, object]":
        return {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0}


def split_stop(buf: str, stops: "Sequence[str]") -> "Tuple[str, str, bool]":
    """流式输出的 stop 串裁剪（纯函数，便于单测）。

    引擎收到 stop 串后会**把它一起留在文本里**（实测：``<end_of_turn>`` 会粘在尾巴上 ✗）。
    流式场景没法"收回"已经吐出去的字符，所以这里做**保留后缀**：

    * 命中 stop ⇒ 返回 ``(stop 之前的文本, "", True)``
    * 否则返回 ``(可安全输出的前缀, 需留到下一轮的后缀, False)`` —— 后缀是
      "某个 stop 串的前缀"且恰好是当前缓冲的结尾（可能下一 token 就凑成 stop ✓）
    """
    if not stops:
        return buf, "", False
    hit = -1
    for s in stops:
        if not s:
            continue
        i = buf.find(s)
        if i >= 0 and (hit < 0 or i < hit):
            hit = i
    if hit >= 0:
        return buf[:hit], "", True
    hold = 0
    for s in stops:
        if not s:
            continue
        for k in range(min(len(s) - 1, len(buf)), 0, -1):
            if buf.endswith(s[:k]):
                hold = max(hold, k)
                break
    if hold:
        return buf[:-hold], buf[-hold:], False
    return buf, "", False


# ---------------------------------------------------------------- 小工具

def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ModelLoadError(f"读不了 meta：{path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise ModelLoadError(f"meta 不是 JSON 对象：{path}")
    return data


def _default_tokenizer_type(model_dir: str) -> str:
    """按 tokenizer.json 的 ``model.type`` 选算法：BPE / Unigram(SPM) / Qwen。"""
    path = os.path.join(model_dir, "tokenizer.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            tok = json.load(f)
        kind = str(((tok.get("model") or {}).get("type") or "")).lower()
    except (OSError, ValueError, AttributeError):
        return "BPE"
    if kind in ("bpe",):
        return "BPE"
    if kind in ("unigram",):
        return "SPM"
    return "BPE"


def _chat_template(model_config: dict, model_dir: "Optional[str]" = None) -> str:
    """推断对话模板：优先 meta 的 model_type，其次看 tokenizer 里的 gemma 标记。

    gemma4 的 tokenizer 里有 ``<|turn>``/``<turn|>``（gemma3 是 ``<start_of_turn>``）；
    只看 meta 会漏判（我们的 model.json 没有 model_type 字段 ✗ ⇒ 被套上 ChatML）。
    """
    kind = str(model_config.get("model_type") or "").lower()
    if kind.startswith("gemma"):
        return "gemma"
    if model_dir:
        try:
            tok = _read_json(os.path.join(model_dir, "tokenizer.json"))
        except (OSError, ValueError):
            tok = {}
        names = " ".join(str(t.get("content", "")) for t in (tok.get("added_tokens") or ()))
        names += " " + " ".join((tok.get("model") or {}).get("vocab") or {})
        if "<|turn>" in names or "<start_of_turn>" in names or "<turn|>" in names:
            return "gemma"
    return "chatml"
