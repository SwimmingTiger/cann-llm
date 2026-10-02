"""``nnrt`` 后端的 LLM 模式：在 NPU 上跑我们自己转出来的 LLM 离线模型。

它接在 :class:`~cann_llm.backends.nnrt.NnrtBackend` 上：模型目录里同时有
``tokenizer.json`` 与 ``embedding_weights`` 时自动启用（见 ``nnrt.py``）。

模型接口（由 converter_lite 的 ``[third_party_model]`` 扩展配置声明，
这里是 Qwen2.5-0.5B W4 实测的形状，全部按**名字**匹配，换模型只需改元数据）：

======================  ==========  =====================  ==========================
输入                      dtype       形状                   来源
======================  ==========  =====================  ==========================
``input_embed``          INT8        [1,1,896]              嵌入表第 id 行
``attention_mask``       FP32        [1,1,1,2048]           0 有效 / -3.4e38 屏蔽
``position_ids``         INT32       [1,1]                  当前 token 的位置
``past_key_in{i}``       FP32        [2048,2,1,64]          自己的 KV 缓存
``past_value_in{i}``     FP32        [2048,2,1,64]          同上
``new_kv_cache_pos``     INT32       [1]                    这一轮往哪个位置写
``embed_scales``         FP32        [1,1,896]              嵌入的反量化 scale
======================  ==========  =====================  ==========================

输出：``lm_logits`` FP32 [1,1,151936] + ``past_key{i}`` / ``past_value{i}``（新缓存）。

因为导出时用的是 ``seq_len: 1``（见 ``docs/offline-model-nnrt.md`` 第 9 节），
**prefill 与 decode 都是"一次一个 token"**，循环体完全一样，实现最简单。
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import GenerationError, ModelLoadError
from ..tokenizer_bpe import QwenTokenizer

__all__ = ["NnrtLlmRunner"]

_MASK_NEG = -3.4e38          # 加性掩码的"屏蔽"值（与引擎用的一致）


def _read_embedding(path: str, rows: int, cols: int) -> bytes:
    """读嵌入表（裸 int8 数据；有的导出带一个小头部，这里按大小自适应）。"""
    size = os.path.getsize(path)
    want = rows * cols
    with open(path, "rb") as f:
        if size == want:
            return f.read()
        for off in range(0, min(4096, size - want + 1)):
            if size - off == want:
                f.seek(off)
                return f.read()
        raise ModelLoadError(
            f"嵌入表大小对不上：{path} 是 {size} 字节，期望 {want}（{rows}x{cols}）")
    # pragma: no cover


def _read_floats(path: str, count: int) -> List[float]:
    """读一段 float32。"""
    size = os.path.getsize(path)
    want = count * 4
    off = size - want if size >= want else 0
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(want)
    n = len(raw) // 4
    return list(struct.unpack("<%df" % n, raw[:n * 4]))


class NnrtLlmRunner:
    """把 ``NnrtBackend`` 的模型句柄包成"分词 → 逐 token 前向 → 采样"的循环。"""

    def __init__(self, backend, model_dir: str, meta: Optional[Dict] = None) -> None:
        self.be = backend
        self.dir = model_dir
        self.meta = meta or {}
        self.tok: Optional[QwenTokenizer] = None
        self.ids_in: Dict[str, int] = {}
        self.ids_out: Dict[str, int] = {}
        self.n_layers = 0
        self.kv_len = 0
        self.hidden = 0
        self.kv_elems = 0
        self.vocab = 0
        self.embed: Optional[bytes] = None
        self.embed_scales: List[float] = []
        self.scale_per_token = False
        self._kv: List[bytearray] = []

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        ins = {t[0]: t for t in self.be._inputs}      # name -> (name, dtype, shape, elems)
        outs = {t[0]: t for t in self.be._outputs}

        # 1) 分词器
        tpath = os.path.join(self.dir, "tokenizer.json")
        if not os.path.isfile(tpath):
            raise ModelLoadError(f"缺少 tokenizer.json: {self.dir}")
        self.tok = QwenTokenizer.from_file(tpath)

        # 2) 结构（从名字推）
        self.n_layers = sum(1 for n in ins if n.startswith("past_key_in"))
        if self.n_layers == 0:
            raise ModelLoadError("看不出层数：模型里没有 past_key_in*")
        _n, _d, kv_shape, self.kv_elems = ins["past_key_in0"]   # (name, dtype, shape, elems)
        self.kv_len = int(kv_shape[0])                          # [2048,2,1,64] ⇒ 2048
        self.hidden = ins["input_embed"][3]                     # [1,1,896] ⇒ 896
        self.vocab = outs["lm_logits"][3]                       # [1,1,V] ⇒ V
        for n in ins:
            self.ids_in[n] = self.be._index_of_input(n)
        for n in outs:
            self.ids_out[n] = self.be._index_of_output(n)

        # 3) 嵌入表 + 反量化 scale
        ew = self._find_file("embedding_weights")
        es = self._find_file("embedding_dequant_scale") if self._find_file_opt("embedding_dequant_scale") else None
        # ★ 行数用【模型的 vocab】（lm_logits 宽度，151936），不是分词器的 vocab（151643）：
        #   嵌入表是按完整词表存的，最后几百行是补齐位。
        if ew:
            self.embed = _read_embedding(ew, self.vocab, self.hidden)
            got = len(self.embed) // self.hidden
            if got < len(self.tok.vocab):
                raise ModelLoadError(f"嵌入表行数不足：{got} < {len(self.tok.vocab)}")
        if es:
            # ★ scale 有两种可能：每个 token 一个标量（实测 594KB/4 ≈ vocab）
            #   或每个元素一个（vocab*hidden）。写错长度会把张量写爆 ⇒ 崩。
            n = os.path.getsize(es) // 4
            per_elem = n >= self.vocab * self.hidden
            self.scale_per_token = not per_elem          # True = 每 token 一个标量
            self.embed_scales = _read_floats(es, self.vocab * self.hidden if per_elem else self.vocab)

        # 4) KV 缓冲（自己持有，逐轮喂进去、把输出拷回来）
        self._kv = [bytearray(self.kv_elems * 4) for _ in range(self.n_layers * 2)]

    def _find_file(self, kind: str) -> str:
        p = self._find_file_opt(kind)
        if not p:
            raise ModelLoadError(f"模型目录里找不到 {kind} 文件: {self.dir}")
        return p

    def _find_file_opt(self, kind: str) -> str:
        hits = [f for f in sorted(os.listdir(self.dir)) if kind in f]
        return os.path.join(self.dir, hits[0]) if hits else ""

    # ------------------------------------------------------------------ 前向
    def _set_input(self, name: str, data: bytes) -> None:
        """把字节写进某个输入张量（按名字）。"""
        p = self.be._input_data_ptr(self.ids_in[name])
        if not p:
            raise GenerationError(f"拿不到输入 {name} 的数据指针")
        ctypes.memmove(p, data, len(data))

    def _get_output(self, name: str, nbytes: int) -> bytes:
        p = self.be._output_data_ptr(self.ids_out[name])
        if not p:
            raise GenerationError(f"拿不到输出 {name} 的数据指针")
        return ctypes.string_at(p, nbytes)

    def step(self, token_id: int, pos: int) -> List[float]:
        """跑一步（一个 token），返回 lm_logits（真值表）。"""
        lib = self.be._lib
        assert lib is not None and self.embed is not None

        # ① input_embed：嵌入表整行（int8）
        row = self.embed[token_id * self.hidden:(token_id + 1) * self.hidden]
        self._set_input("input_embed", row)

        # ② embed_scales：按 token 取，或整表一份
        if self.embed_scales:
            if self.scale_per_token:
                # 每个 token 一个标量 ⇒ 广播到整行（张量是 [1,1,hidden]）
                s = [self.embed_scales[token_id]] * self.hidden
            else:
                s = self.embed_scales[token_id * self.hidden:(token_id + 1) * self.hidden]
            if len(s) != self.hidden:      # 兜底：长度不对就别写，免得写爆张量
                s = (s + [0.0] * self.hidden)[:self.hidden]
            self._set_input("embed_scales", struct.pack("<%df" % self.hidden, *s))

        # ③ position_ids / new_kv_cache_pos
        self._set_input("position_ids", struct.pack("<i", pos))
        self._set_input("new_kv_cache_pos", struct.pack("<i", pos))

        # ④ attention_mask：0..pos 有效，其余屏蔽
        mask = [_MASK_NEG] * self.kv_len
        for i in range(min(pos + 1, self.kv_len)):
            mask[i] = 0.0
        self._set_input("attention_mask", struct.pack("<%df" % self.kv_len, *mask))

        # ⑤ KV 输入
        for i in range(self.n_layers):
            self._set_input(f"past_key_in{i}", bytes(self._kv[2 * i]))
            self._set_input(f"past_value_in{i}", bytes(self._kv[2 * i + 1]))

        # ⑥ 跑
        self.be._predict()
        for i in range(self.n_layers):
            self._kv[2 * i][:] = self._get_output(f"past_key{i}", self.kv_elems * 4)
            self._kv[2 * i + 1][:] = self._get_output(f"past_value{i}", self.kv_elems * 4)
        raw = self._get_output("lm_logits", self.vocab * 4)
        return list(struct.unpack("<%df" % self.vocab, raw))

    # ------------------------------------------------------------------ 生成
    def generate(self, prompt: str, max_new: int = 32,
                 stop_ids: Sequence[int] = ()) -> Tuple[str, List[int], Dict[str, float]]:
        """贪心解码（PoC 阶段：确定性、可复现）。返回 (文本, 新 token, 统计)。"""
        assert self.tok is not None
        ids = self.tok.encode(prompt)
        if not ids:
            return "", [], {}
        t0 = time.time()
        # prefill：逐个 token 过一遍，最后一个 token 的 logits 用来出第一个新 token
        logits: List[float] = []
        pos = 0
        for tid in ids:
            logits = self.step(tid, pos)
            pos += 1
        prefill_ms = (time.time() - t0) * 1000.0

        t1 = time.time()
        out: List[int] = []
        stopset = set(stop_ids)
        for _ in range(max(1, max_new)):
            nxt = max(range(len(logits)), key=logits.__getitem__)
            out.append(nxt)
            if nxt in stopset:
                break
            logits = self.step(nxt, pos)
            pos += 1
        decode_ms = (time.time() - t1) * 1000.0
        text = self.tok.decode(out)
        return text, out, {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
                           "prompt_tokens": len(ids), "completion_tokens": len(out)}
