"""NNRt 离线模型后端。

通过 ``ctypes`` 调用系统自带的 ``/system/lib64/ndk/libmindspore_lite_ndk.so``
（MindSpore Lite 的 C API），把**第三方离线模型**（``.ms``）交给 **NNRt** 后端执行。

它和 ``cann`` / ``hiai`` 后端的根本区别
----------------------------------------
那两个后端跑的是华为 LLM 引擎，模型必须是「逐层一对 K/V + 每层 ``past_key_value``
进出」的结构 —— 像 Gemma 4 那类新架构（每层不同配置、K=V、K/V 跨层共享）**装不进去**。

而**离线模型对 MindSpore Lite 是个黑盒**：输入输出张量由转换时的扩展配置声明，
模型内部结构上层完全不解析，真正算它的是硬件厂商的实现。
所以这条路**不受任何结构约束**，接口由我们自己设计。

.. note::

    这个后端目前**不是** LLM 后端：它跑一次前向、把输出以文本返回，
    用于验证「模型 → OMG → ``.ms`` → NNRt」这条链路。
    要跑真正的 LLM，需要把带 KV 接口的图交给 OMG 编成 ``.om``（见
    ``docs/offline-model-nnrt.md``）。

.. warning::

    ``OH_AI_ContextDestroy`` / ``OH_AI_ModelDestroy`` 在进程退出阶段会让进程
    **core dump**（``exit code 139``）—— 推理结果已经正确，崩在析构/动态库卸载阶段。
    与 ``cann`` 后端不调用 ``Context_Destroy`` 是同一类处理：**只新建、不释放**，
    靠进程退出回收。

输入约定（``generate`` 的 prompt）
----------------------------------
prompt 里给**第一个浮点输入**的数据，逗号分隔，例如 ``"1.0,2.0,3.0"``；
其余浮点输入填 0、整型输入填 0。输出按同样的逗号分隔文本返回。
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..errors import GenerationError, InvalidRequestError, ModelLoadError
from ..types import (
    FINISH_STOP,
    GenerationChunk,
    GenerationParams,
    GenerationRequest,
    ModelInfo,
)
from .base import EngineBackend, register_backend

__all__ = ["NnrtBackend"]

#: 设备上 MindSpore Lite NDK 库的默认位置
DEFAULT_LIB = "/system/lib64/ndk/libmindspore_lite_ndk.so"

# ---- 头文件里的枚举（native/sysroot/usr/include/mindspore/）----
OH_AI_STATUS_SUCCESS = 0
OH_AI_MODELTYPE_MINDIR = 0
OH_AI_DEVICETYPE_CPU = 0
OH_AI_DEVICETYPE_NNRT = 60

# 数据类型枚举（OH_AI_DataType）—— 取自 native/sysroot/usr/include/mindspore/data_type.h。
# 注意这些值是 MindSpore 的 TypeId，不是 0,1,2… 的紧凑编号（踩过：FLOAT32 是 43）。
_DTYPE_NAMES = {
    30: "BOOL", 32: "INT8", 33: "INT16", 34: "INT32", 35: "INT64",
    37: "UINT8", 38: "UINT16", 39: "UINT32", 40: "UINT64",
    42: "FLOAT16", 43: "FLOAT32", 44: "FLOAT64",
}
_DTYPE_SIZES = {
    30: 1, 32: 1, 33: 2, 34: 4, 35: 8, 37: 1, 38: 2, 39: 4, 40: 8, 42: 2, 43: 4, 44: 8,
}
_DTYPE_FLOAT32 = 43
_DTYPE_FLOAT16 = 42


class _TensorHandleArray(ctypes.Structure):
    """``OH_AI_TensorHandleArray``：数量 + 句柄数组。"""

    _fields_ = [("handle_num", ctypes.c_size_t),
                ("handle_list", ctypes.POINTER(ctypes.c_void_p))]


@register_backend("nnrt")
class NnrtBackend(EngineBackend):
    """用 MindSpore Lite NDK + NNRt 后端执行一个离线模型（``.ms``）。"""

    name = "nnrt"

    def __init__(
        self,
        model_dir: str,
        model_id: Optional[str] = None,
        context_length: int = 0,
        lib_path: Optional[str] = None,
        device: str = "nnrt",
        **_: object,
    ) -> None:
        """:param model_dir: 模型目录，里面放一个 ``.ms`` 文件（多个时报错，要求明确）。
        :param device: ``nnrt``（默认，走 NPU）或 ``cpu``（对照用）。
        """
        self.model_dir = os.path.abspath(model_dir) if model_dir else ""
        self.model_id = model_id or os.path.basename(self.model_dir) or "nnrt-model"
        self.context_length = int(context_length or 0)
        self._lib_path = lib_path or os.environ.get("CANN_LLM_MSLITE_LIB", DEFAULT_LIB)
        # 允许用环境变量切到 cpu 做对照（CPU 侧没有 custom 算子实现，输出通常是 0）
        device = os.environ.get("CANN_LLM_MSLITE_DEVICE", device)
        self._device = device if device in ("nnrt", "cpu") else "nnrt"

        self._lib = None
        self._model = None
        self._ctx = None
        self._outs = None
        self._ms_path = ""
        self._inputs: List[Tuple[str, int, Tuple[int, ...], int]] = []   # (name, dtype, shape, elems)
        self._outputs: List[Tuple[str, int, Tuple[int, ...], int]] = []
        self._default_params = GenerationParams()
        self._llm = None
        #: 上一轮推理的统计（诊断用）
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ 属性
    @property
    def default_params(self) -> GenerationParams:  # type: ignore[override]
        return self._default_params

    @default_params.setter
    def default_params(self, value: GenerationParams) -> None:  # type: ignore[override]
        self._default_params = value

    @property
    def info(self) -> ModelInfo:
        return getattr(self, "_info", ModelInfo(id=self.model_id, backend=self.name))

    def sampler_defaults(self) -> Dict[str, object]:
        """离线模型不带采样配置，返回空让上层用自己的兜底值。"""
        return {}

    # ------------------------------------------------------------------ 加载
    def _bind(self) -> ctypes.CDLL:
        """加载 NDK 库并声明用到的符号（延迟到 load()，import 阶段不碰 .so）。"""
        if self._lib is not None:
            return self._lib
        if not os.path.exists(self._lib_path):
            raise ModelLoadError(
                f"找不到 MindSpore Lite NDK 库: {self._lib_path!r}；"
                f"用环境变量 CANN_LLM_MSLITE_LIB 指定")
        lib = ctypes.CDLL(self._lib_path)
        H = ctypes.c_void_p
        lib.OH_AI_ContextCreate.restype = H
        lib.OH_AI_ContextAddDeviceInfo.argtypes = [H, H]
        lib.OH_AI_DeviceInfoCreate.restype = H
        lib.OH_AI_DeviceInfoCreate.argtypes = [ctypes.c_int]
        lib.OH_AI_ModelCreate.restype = H
        lib.OH_AI_ModelBuildFromFile.restype = ctypes.c_int
        lib.OH_AI_ModelBuildFromFile.argtypes = [H, ctypes.c_char_p, ctypes.c_int, H]
        lib.OH_AI_ModelGetInputs.restype = _TensorHandleArray
        lib.OH_AI_ModelGetInputs.argtypes = [H]
        lib.OH_AI_ModelGetOutputs.restype = _TensorHandleArray
        lib.OH_AI_ModelGetOutputs.argtypes = [H]
        lib.OH_AI_ModelPredict.restype = ctypes.c_int
        lib.OH_AI_ModelPredict.argtypes = [H, _TensorHandleArray,
                                          ctypes.POINTER(_TensorHandleArray), H, H]
        for fn in ("OH_AI_TensorGetName",):
            getattr(lib, fn).restype = ctypes.c_char_p
            getattr(lib, fn).argtypes = [H]
        lib.OH_AI_TensorGetDataType.restype = ctypes.c_int
        lib.OH_AI_TensorGetDataType.argtypes = [H]
        lib.OH_AI_TensorGetElementNum.restype = ctypes.c_int64
        lib.OH_AI_TensorGetElementNum.argtypes = [H]
        lib.OH_AI_TensorGetDataSize.restype = ctypes.c_size_t
        lib.OH_AI_TensorGetDataSize.argtypes = [H]
        lib.OH_AI_TensorGetData.restype = ctypes.c_void_p
        lib.OH_AI_TensorGetData.argtypes = [H]
        lib.OH_AI_TensorGetMutableData.restype = ctypes.c_void_p
        lib.OH_AI_TensorGetMutableData.argtypes = [H]
        lib.OH_AI_TensorGetShape.restype = ctypes.POINTER(ctypes.c_int64)
        lib.OH_AI_TensorGetShape.argtypes = [H, ctypes.POINTER(ctypes.c_size_t)]
        self._lib = lib
        return lib

    def _describe(self, arr: _TensorHandleArray) -> List[Tuple[str, int, Tuple[int, ...], int]]:
        """把句柄数组读成 ``(名字, dtype, 形状, 元素数)`` 列表。"""
        out: List[Tuple[str, int, Tuple[int, ...], int]] = []
        lib = self._lib
        assert lib is not None
        for i in range(int(arr.handle_num)):
            h = ctypes.c_void_p(arr.handle_list[i])
            raw = lib.OH_AI_TensorGetName(h)
            name = raw.decode() if raw else f"tensor_{i}"
            nd = ctypes.c_size_t(0)
            sp = lib.OH_AI_TensorGetShape(h, ctypes.byref(nd))
            shape = tuple(int(sp[j]) for j in range(int(nd.value))) if sp else ()
            dt = int(lib.OH_AI_TensorGetDataType(h))
            n = int(lib.OH_AI_TensorGetElementNum(h))
            out.append((name, dt, shape, n))
        return out

    # ---- 张量访问辅助（供 nnrt_llm 循环使用）----
    def _handles(self, which: str):
        lib = self._lib
        assert lib is not None
        arr = (lib.OH_AI_ModelGetInputs(self._model) if which == "in"
               else lib.OH_AI_ModelGetOutputs(self._model))
        return arr

    def _index_of_input(self, name: str) -> int:
        for i, t in enumerate(self._inputs):
            if t[0] == name:
                return i
        raise KeyError(name)

    def _index_of_output(self, name: str) -> int:
        for i, t in enumerate(self._outputs):
            if t[0] == name:
                return i
        raise KeyError(name)

    def _input_data_ptr(self, i: int):
        lib = self._lib
        arr = self._handles("in")
        return lib.OH_AI_TensorGetMutableData(ctypes.c_void_p(arr.handle_list[i]))

    def _output_data_ptr(self, i: int):
        """★ 必须用【传给 Predict 的那个数组】读输出。

        MindSpore Lite 的语义是：调用方准备一个 OH_AI_TensorHandleArray 传给
        OH_AI_ModelPredict，算完从**同一个数组**里取数据。重新调
        OH_AI_ModelGetOutputs 拿到的句柄其 data 指针是空的（我们踩过：
        "拿不到输出 past_key0 的数据指针"）。
        """
        lib = self._lib
        # 先试 Predict 时用的那个数组，再试重新查一次（NNRt 可能把结果写到新句柄）
        cands = []
        if self._outs is not None:
            cands.append(ctypes.c_void_p(self._outs.handle_list[i]))
        fresh = lib.OH_AI_ModelGetOutputs(self._model)
        cands.append(ctypes.c_void_p(fresh.handle_list[i]))
        for h in cands:
            for getter in (lib.OH_AI_TensorGetData, lib.OH_AI_TensorGetMutableData):
                p = getter(h)
                if p:
                    return p
        return None

    def _predict_checked(self) -> None:
        """跑一次 Predict 并**检查返回值**（失败时输出是空的，不检查会伪装成"能跑但输出垃圾"）。"""
        st = self._predict()
        if st != OH_AI_STATUS_SUCCESS:
            raise GenerationError(
                "NPU 执行失败：OH_AI_ModelPredict -> %d（0 才是成功）。模型是 %s"
                % (st, os.path.basename(self._ms_path or self.model_dir)))

    def _get_output_raw(self, name: str, nbytes: int) -> bytes:
        """按名字把某个输出的数据整块读出来（分段 LLM 用）。"""
        return self._output_bytes(self._index_of_output(name), nbytes)

    def _output_bytes(self, i: int, nbytes: int) -> bytes:
        p = self._output_data_ptr(i)
        if not p:
            raise GenerationError("拿不到输出张量 %d 的数据指针" % i)
        return ctypes.string_at(p, nbytes)

    def _predict(self) -> int:
        """跑一次 Predict，返回状态码（0 = 成功）。"""
        lib = self._lib
        assert lib is not None
        ins = lib.OH_AI_ModelGetInputs(self._model)
        self._outs = lib.OH_AI_ModelGetOutputs(self._model)          # ★ 保留这个数组
        return int(lib.OH_AI_ModelPredict(self._model, ins, ctypes.byref(self._outs), None, None))

    def _find_ms(self) -> str:
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise ModelLoadError(f"模型目录不存在: {self.model_dir!r}")
        found = sorted(f for f in os.listdir(self.model_dir) if f.endswith(".ms"))
        if not found:
            raise ModelLoadError(f"目录里没有 .ms 文件: {self.model_dir!r}")
        if len(found) > 1:
            raise ModelLoadError(
                f"目录里有多个 .ms（{', '.join(found)}），请只留一个，或用 CANN_LLM_MSLITE_MODEL 指定")
        return os.path.join(self.model_dir, found[0])

    def load(self) -> ModelInfo:
        """加载并构建离线模型（这一步会让 NPU 侧解析/编译它）。

        模型目录里若有 ``seg*`` 子目录，则进入**分段 LLM** 模式：
        外层只建分段 runner（不加载整模型的 ``.ms``，因为整模型上不了 NPU）。
        """
        # ★ Gemma 4 分段包：目录里有 graphP/ 与 lm/ 子目录（分段之外的两块图）✓
        if (os.path.isdir(self.model_dir)
                and os.path.isdir(os.path.join(self.model_dir, "graphP"))
                and os.path.isdir(os.path.join(self.model_dir, "lm"))):
            from .gemma4_runner import Gemma4ChatRunner, Gemma4KvRunner
            # ★ 有 decode/prefill 图就走 KV 路径（每 token 只算 1 个）；
            #   没有则回退到"每 token 重跑全上下文"的旧路径 ✓
            _kv = all(os.path.isdir(os.path.join(self.model_dir, d))
                      for d in ("dec0", "pre0"))
            runner = (Gemma4KvRunner if _kv else Gemma4ChatRunner)(self.model_dir)
            runner.load()
            self._llm = runner
            self._info = ModelInfo(
                id=self.model_id, backend=self.name, path=self.model_dir,
                extra={"device": self._device, "family": "gemma4",
                       "segments": len(runner.SEG_STARTS), "seq": runner.SEQ})
            return self._info          # ★ 必须 return：否则会继续落到下面 Qwen 的分支 ✗

        # ★ 分段 LLM：先判、先返回 —— 否则会在没有整模型 .ms 的目录上报错。
        if (os.path.isdir(self.model_dir)
                and os.path.isfile(os.path.join(self.model_dir, "tokenizer.json"))
                and any(d.startswith("seg") and os.path.isdir(os.path.join(self.model_dir, d))
                        for d in os.listdir(self.model_dir))):
            from .nnrt_seg import SegmentedLlmRunner
            runner = SegmentedLlmRunner(self.model_dir)
            runner.load()
            self._llm = runner
            self._info = ModelInfo(
                id=self.model_id, backend=self.name, path=self.model_dir,
                extra={"device": self._device, "segments": len(runner.segs),
                       "segments_dir": [os.path.basename(x.be.model_dir) for x in runner.segs]})
            return self._info

        self._ms_path = os.environ.get("CANN_LLM_MSLITE_MODEL") or self._find_ms()
        lib = self._bind()

        dev_type = OH_AI_DEVICETYPE_NNRT if self._device == "nnrt" else OH_AI_DEVICETYPE_CPU
        self._ctx = lib.OH_AI_ContextCreate()
        dev = lib.OH_AI_DeviceInfoCreate(dev_type)
        lib.OH_AI_ContextAddDeviceInfo(self._ctx, dev)
        self._model = lib.OH_AI_ModelCreate()

        # ★--large-mem：第一次 build 之前"报到—等放行"★（没设环境变量时零开销 ✓）
        #   顺序见 cann_llm.large_mem.rendezvous：先 dlopen 补丁目标的库，
        #   再等 lldb 打完补丁解锁 —— 这样 4 处补丁赶在这次 build 的权重拷贝之前 ✓
        from ..large_mem import rendezvous as _large_mem_rendezvous
        _large_mem_rendezvous()

        st = lib.OH_AI_ModelBuildFromFile(
            self._model, self._ms_path.encode(), OH_AI_MODELTYPE_MINDIR, self._ctx)
        if st != OH_AI_STATUS_SUCCESS:
            raise ModelLoadError(
                f"离线模型加载失败（后端={self._device}，rc={st}）: {self._ms_path}\n"
                f"  提示：`.ms` 与它依赖的权重文件必须放在同一目录；"
                f"离线模型只在 NNRt 后端有效，可用 CANN_LLM_MSLITE_DEVICE=cpu 对照"
                f"（CPU 侧没有 custom 算子实现，输出通常是 0）。")

        self._inputs = self._describe(lib.OH_AI_ModelGetInputs(self._model))
        self._outputs = self._describe(lib.OH_AI_ModelGetOutputs(self._model))

        # ★ LLM 模式：目录里有分词器与嵌入表，就启用逐 token 的聊天循环
        self._llm = None
        if (os.path.isfile(os.path.join(self.model_dir, "tokenizer.json"))
                and any("embedding_weights" in f for f in os.listdir(self.model_dir))
                and any(n[0] == "embed_scales" for n in self._inputs)):
            from .nnrt_llm import NnrtLlmRunner
            runner = NnrtLlmRunner(self, self.model_dir)
            runner.load()
            self._llm = runner
        self._info = ModelInfo(
            id=self.model_id, backend=self.name, path=self._ms_path,
            context_length=self.context_length,
            extra={
                "device": self._device,
                "inputs": [{"name": n, "dtype": _DTYPE_NAMES.get(d, str(d)),
                            "shape": list(s)} for n, d, s, _ in self._inputs],
                "outputs": [{"name": n, "dtype": _DTYPE_NAMES.get(d, str(d)),
                             "shape": list(s)} for n, d, s, _ in self._outputs],
            },
        )
        return self._info

    # ------------------------------------------------------------------ 推理
    def _parse_prompt(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        vals: List[float] = []
        for piece in text.replace("\n", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                vals.append(float(piece))
            except ValueError:
                raise InvalidRequestError(f"输入里有非数字: {piece!r}（用逗号分隔的浮点数）")
        return vals

    def _fill(self, idx: int, spec: Tuple[str, int, Tuple[int, ...], int],
              values: Sequence[float]) -> None:
        """把数据写进第 idx 个输入；浮点输入用 values，其余填 0。"""
        lib = self._lib
        assert lib is not None
        ins = lib.OH_AI_ModelGetInputs(self._model)
        h = ctypes.c_void_p(ins.handle_list[idx])
        name, dt, _shape, n = spec
        p = lib.OH_AI_TensorGetMutableData(h)
        if not p:
            raise GenerationError(f"拿不到输入 {name} 的数据指针")
        if dt == _DTYPE_FLOAT32:
            buf = (ctypes.c_float * n).from_address(p)
            for k in range(n):
                buf[k] = float(values[k]) if k < len(values) else 0.0
        else:   # 其它类型（int8/int32/...）一律先填 0，够跑通链路
            size = _DTYPE_SIZES.get(dt, 1) * n
            ctypes.memset(p, 0, size)

    def generate(self, request: GenerationRequest) -> Iterator[GenerationChunk]:
        """跑一次前向：prompt 当第一个浮点输入的数据，输出以逗号分隔文本返回。"""
        # ★ LLM 模式（含分段）没有单一的 _model —— 必须先走这条路，
        #   否则会被下面那句"模型尚未 load()"误伤（分段模式下 _model 恒为 None）。
        if self._llm is not None:
            # ★ 真正的流式：逐 token 把新增文本吐出去（见 runner.stream 的增量切法）
            params = request.params
            max_new = int(getattr(params, "max_tokens", 32) or 32)
            index, last_tok, n = 0, None, 0
            for piece, tok_id, _ in self._llm.stream(request.prompt, max_new=max_new,
                                                     stop_ids=self._stop_ids(),
                                                     params=params):
                last_tok, n = tok_id, n + 1
                yield GenerationChunk(text=piece, index=index, token_id=tok_id,
                                      finish_reason=None, stats=None)
                index += 1
            yield GenerationChunk(index=index, token_id=last_tok, finish_reason=FINISH_STOP,
                                  stats=None)
            self.last_stats = {"completion_tokens": float(n)}
            return

        if self._model is None:
            raise GenerationError("模型尚未 load()")
        lib = self._lib
        assert lib is not None

        if False:
            params = request.params
            text, toks, stats = self._llm.generate(
                request.prompt, max_new=int(getattr(params, "max_tokens", 32) or 32),
                stop_ids=self._stop_ids())
            yield GenerationChunk(text=text, index=0, token_id=(toks[-1] if toks else None),
                                  finish_reason=FINISH_STOP, stats=None)
            self.last_stats = dict(stats)
            return

        values = self._parse_prompt(request.prompt)
        t0 = time.time()
        # 第一个浮点输入吃 values，其余输入填 0
        first_float = None
        for i, spec in enumerate(self._inputs):
            if spec[1] == _DTYPE_FLOAT32:
                first_float = i
                break
        if first_float is None:
            raise GenerationError("模型没有 float32 输入，这个后端暂不支持")
        for i, spec in enumerate(self._inputs):
            self._fill(i, spec, values if i == first_float else ())

        ins = lib.OH_AI_ModelGetInputs(self._model)
        outs = lib.OH_AI_ModelGetOutputs(self._model)
        st = lib.OH_AI_ModelPredict(self._model, ins, ctypes.byref(outs), None, None)
        dt_ms = (time.time() - t0) * 1000.0
        if st != OH_AI_STATUS_SUCCESS:
            raise GenerationError(f"推理失败（rc={st}）")

        parts: List[str] = []
        for name, dt, _shape, n in self._outputs:
            o = lib.OH_AI_ModelGetOutputs(self._model)
            h = ctypes.c_void_p(o.handle_list[self._outputs.index((name, dt, _shape, n))])
            p = lib.OH_AI_TensorGetData(h)
            if not p:
                parts.append(f"{name}=<null>")
                continue
            if dt == _DTYPE_FLOAT32:
                buf = (ctypes.c_float * n).from_address(p)
                parts.append(f"{name}=" + ",".join(f"{buf[k]:.6g}" for k in range(n)))
            elif dt == _DTYPE_FLOAT16:
                parts.append(f"{name}=<float16 x{n}>")
            else:
                parts.append(f"{name}=<{_DTYPE_NAMES.get(dt, dt)} x{n}>")
        self.last_stats = {"wall_ms": dt_ms, "inputs": float(len(self._inputs))}
        yield GenerationChunk(text="; ".join(parts), index=0,
                              finish_reason=FINISH_STOP,
                              stats=None)

    def _stop_ids(self):
        """停止 token：Qwen 的 <|im_end|> / <|endoftext|>。"""
        if self._llm is None or self._llm.tok is None:
            return ()
        sp = self._llm.tok.special_ids
        base = tuple(v for k, v in sp.items() if k in ("<|im_end|>", "<|endoftext|>"))
        # ★ Gemma 4：它的停用符在 generation_config.eos_token_id = [1,106,50]，
        #   名字不是 Qwen 那套 ⇒ 直接用 runner 声明的 STOP_IDS ✓
        extra = getattr(self._llm, "STOP_IDS", None)
        if extra:
            return tuple(sorted(set(base) | set(extra)))
        return base

    @property
    def supports_streaming(self) -> bool:
        """LLM 模式（含分段）支持真正的逐 token 流式；纯前向模式一次性返回。"""
        return self._llm is not None

    @property
    def is_llm(self) -> bool:
        """是否启用了 LLM 聊天模式。"""
        return self._llm is not None

    # ------------------------------------------------------------------ 释放
    def close(self) -> None:
        """**故意不做任何释放**：销毁会在退出阶段 core dump（见模块 docstring）。

        只把 Python 侧的引用丢掉；Context/Model 交给进程退出回收。
        """
        self._model = None
        self._ctx = None
