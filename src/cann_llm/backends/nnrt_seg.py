"""``nnrt`` 后端的**分段 LLM** 模式：把整模型按层切成若干段、逐段在 NPU 上跑。

为什么要分段
------------
实测（见 ``docs/offline-model-nnrt.md`` §11）：这条离线模型路径对规模有两条硬限制 ——
设备的 KV 张量不能太大、OMG 对「层数 × KV」另有限制。两者叠加使 24 层无法整图上 NPU。
而 **4 层 + KV 1024 实测能 Predict 成功**（含中间段），所以把模型按 4 层切段即可。

模型目录布局
------------
每段的权重文件名都叫 ``SubGraph_0.weight``，只能放在**各自的子目录**里：

    <model_dir>/
        seg0/   seg0.ms   SubGraph_0.weight     ← 首段：吃 token id，吐 hidden
        seg4/   seg4.ms   SubGraph_0.weight     ← 中间段：吃 hidden，吐 hidden
        ...
        seg20/  seg20.ms  SubGraph_0.weight     ← 末段：吃 hidden，吐 lm_logits
        tokenizer.json

段的顺序按目录名里的数字排（也可用 ``segments.json`` 显式给出）。
每段有**自己的一套 KV 缓冲**，逐 token 前向时各段依次跑一遍。
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
from typing import Dict, List, Optional, Tuple

from ..errors import GenerationError, ModelLoadError
from ..tokenizer_bpe import QwenTokenizer

__all__ = ["SegmentedLlmRunner"]

_MASK_NEG = -3.4e38
_F32, _I32, _I8 = 43, 34, 32


def _seg_key(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


class _Seg:
    """一段：自己的 backend + 自己的 KV 缓冲 + 张量索引。"""

    def __init__(self, be, is_first: bool, is_last: bool) -> None:
        self.be = be
        self.is_first = is_first
        self.is_last = is_last
        ins = {t[0]: t for t in be._inputs}
        self.ins = ins
        self.kv_names: List[Tuple[str, str]] = []          # [(key_in, key_out), ...]
        self.kv_elems = 0
        for i in range(64):
            ki = "past_key_in%d" % i
            if ki not in ins:
                break
            self.kv_names.append((ki, "past_key%d" % i))
            self.kv_names.append(("past_value_in%d" % i, "past_value%d" % i))
        if self.kv_names:
            self.kv_elems = ins[self.kv_names[0][0]][3]
        self.kv = [bytearray(self.kv_elems * 4) for _ in self.kv_names]
        # 首段输入名 / 其余段输入名
        self.hidden_name = "input_ids" if is_first else "input_embed"
        if self.hidden_name not in ins:
            raise ModelLoadError(
                "段 %s 里没有输入 %s（首段应为 input_ids、其余段应为 input_embed）"
                % (be.model_dir, self.hidden_name))
        # 末段输出 lm_logits，其余段输出 hidden（导出时名字沿用了 lm_logits）
        self.logits_name = "lm_logits" if "lm_logits" in {t[0] for t in be._outputs} else None
        self.hidden_elems = ins[self.hidden_name][3]
        self.mask_elems = ins.get("attention_mask", (None, None, None, 0))[3]
        self.hidden_dtype = ins[self.hidden_name][1]

    # ---- 写输入 ----
    def _put(self, name: str, data: bytes) -> None:
        i = self.be._index_of_input(name)
        p = self.be._input_data_ptr(i)
        if not p:
            raise GenerationError("拿不到输入 %s 的数据指针" % name)
        ctypes.memmove(p, data, len(data))

    def run(self, hidden: bytes, pos: int, kv_len: int) -> bytes:
        """跑一段：hidden 是 token id（首段）或上一段的 hidden；返回本段输出字节。"""
        self._put(self.hidden_name, hidden)
        self._put("position_ids", struct.pack("<i", pos))
        if "new_kv_cache_pos" in self.ins:
            self._put("new_kv_cache_pos", struct.pack("<i", pos))
        if self.mask_elems:
            mask = [_MASK_NEG] * self.mask_elems
            for i in range(min(pos + 1, self.mask_elems)):
                mask[i] = 0.0
            self._put("attention_mask", struct.pack("<%df" % self.mask_elems, *mask))
        if "embed_scales" in self.ins:                     # 中间段带它，但我们的 forward 不用
            n = self.ins["embed_scales"][3]
            self._put("embed_scales", struct.pack("<%df" % n, *([1.0] * n)))
        for (kin, _), buf in zip(self.kv_names, self.kv):
            self._put(kin, bytes(buf))
        self.be._predict_checked()
        for (_, kout), buf in zip(self.kv_names, self.kv):
            buf[:] = self.be._get_output_raw(kout, self.kv_elems * 4)
        out = self.logits_name if self.is_last else self.hidden_name
        # 末段取 lm_logits；其余段取图的第一个输出（就是 hidden）
        oname = out if (self.is_last and out) else self.be._outputs[0][0]
        n = self.be._index_of_output(oname)
        elems = self.be._outputs[n][3]
        return self.be._get_output_raw(oname, elems * 4)


class SegmentedLlmRunner:
    """多段串起来的 LLM 跑法（接口与 :class:`NnrtLlmRunner` 一致）。"""

    def __init__(self, model_dir: str) -> None:
        self.dir = model_dir
        self.tok: Optional[QwenTokenizer] = None
        self.segs: List[_Seg] = []
        self.vocab = 0

    def load(self) -> None:
        from .nnrt import NnrtBackend            # 延迟导入，避免循环

        tpath = os.path.join(self.dir, "tokenizer.json")
        if not os.path.isfile(tpath):
            raise ModelLoadError("缺少 tokenizer.json: %s" % self.dir)
        self.tok = QwenTokenizer.from_file(tpath)

        subs = [d for d in os.listdir(self.dir)
                if os.path.isdir(os.path.join(self.dir, d)) and d.startswith("seg")]
        if not subs:
            raise ModelLoadError("没有 seg* 子目录: %s" % self.dir)
        subs.sort(key=_seg_key)

        bes = []
        for name in subs:
            be = NnrtBackend(model_dir=os.path.join(self.dir, name))
            be.load()
            bes.append(be)
        for i, be in enumerate(bes):
            seg = _Seg(be, is_first=(i == 0), is_last=(i == len(bes) - 1))
            if seg.is_last:
                self.vocab = be._outputs[be._index_of_output("lm_logits")][3]
            self.segs.append(seg)

    def step(self, token_id: int, pos: int) -> List[float]:
        first = self.segs[0]
        hidden = struct.pack("<i", token_id) if first.hidden_dtype == _I32 \
            else struct.pack("<f", float(token_id))
        logits_bytes = b""
        for si, seg in enumerate(self.segs):
            out = seg.run(hidden, pos, 0)
            hidden = out
            if seg.is_last:
                logits_bytes = out
        n = len(logits_bytes) // 4
        return list(struct.unpack("<%df" % n, logits_bytes[:n * 4]))

    def stream(self, prompt: str, max_new: int = 32, stop_ids=()):
        """真正的流式：每生成一个 token 就 yield 一段**新增文本**。

        ★ 增量怎么切：**累积解码、只吐新增后缀**。
        字节级 BPE 里一个汉字常跨多个 token，逐 token 单独 decode 会得到半个字符
        （替换符）。所以每步都拿【到目前为止的全部 token】解码出整串，
        再和上一次的整串比，只把多出来的部分交出去 —— 半个字符自然被留在里面等补齐。
        """
        assert self.tok is not None
        ids = self.tok.encode(prompt)
        if not ids:
            return
        pos = 0
        logits: List[float] = []
        for tid in ids:                       # prefill
            logits = self.step(tid, pos)
            pos += 1
        out: List[int] = []
        prev = ""
        stop = set(stop_ids)
        for _ in range(max(1, max_new)):
            nxt = max(range(len(logits)), key=logits.__getitem__)
            out.append(nxt)
            cur = self.tok.decode(out)
            if len(cur) > len(prev):
                yield cur[len(prev):], nxt, None
                prev = cur
            if nxt in stop:
                return
            logits = self.step(nxt, pos)
            pos += 1

    def generate(self, prompt: str, max_new: int = 32,
                 stop_ids=()) -> Tuple[str, List[int], Dict[str, float]]:
        import time
        assert self.tok is not None
        ids = self.tok.encode(prompt)
        if not ids:
            return "", [], {}
        t0 = time.time()
        logits: List[float] = []
        pos = 0
        for tid in ids:
            logits = self.step(tid, pos)
            pos += 1
        prefill_ms = (time.time() - t0) * 1000.0
        t1 = time.time()
        out: List[int] = []
        stop = set(stop_ids)
        for _ in range(max(1, max_new)):
            nxt = max(range(len(logits)), key=logits.__getitem__)
            out.append(nxt)
            if nxt in stop:
                break
            logits = self.step(nxt, pos)
            pos += 1
        decode_ms = (time.time() - t1) * 1000.0
        return (self.tok.decode(out), out,
                {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
                 "prompt_tokens": len(ids), "completion_tokens": len(out)})
