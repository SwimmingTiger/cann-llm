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
        # ★每个槽按【各自的】元素数处理★✓（§159.3 ✓）：
        #   qwen3_5 线性层的 key / value 槽【不等长】✗（实测 past_key_in0=18432 vs
        #   past_value_in0=262144 ✓）—— 原来统一用 key 槽的 elems ✗ ⇒ value 槽只写/读了一小部分 ✗
        #   （gemma4 的 key/value 等长 ✓ 所以一直没暴露 ✓）
        self.kv_elems_list = [ins[k][3] for k, _ in self.kv_names]
        self.kv = [bytearray(n * 4) for n in self.kv_elems_list]
        # ★首段输入名★（§153 适配 ✓）：
        #   · gemma4 式：首段在【设备上】做 embedding ⇒ 输入名 input_ids，喂 token id ✓
        #   · qwen3_5 式：段图从 embedding 开始（主机侧算 embedding ✓ 与 nnrt_llm 同思路 ✓）
        #     ⇒ 首段输入名是 input_embed ✓，喂【embedding 行】✓（dtype 按图声明 ✓）
        if is_first:
            if "input_ids" in ins:
                self.hidden_name = "input_ids"
                self.first_is_token = True
            elif "input_embed" in ins:
                self.hidden_name = "input_embed"
                self.first_is_token = False
            else:
                raise ModelLoadError(
                    "首段 %s 里既没有 input_ids 也没有 input_embed ✗" % be.model_dir)
        else:
            self.hidden_name = "input_embed"
            self.first_is_token = False
        if self.hidden_name not in ins:
            raise ModelLoadError(
                "段 %s 里没有输入 %s（首段应为 input_ids/input_embed、其余段应为 input_embed）"
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

    def _put_ints(self, name: str, first: int, n: int) -> None:
        """把整型张量【整宽】写满 ✓：第 0 位给 first ✓ 其余补 0 ✓。

        ★为什么必须整宽★：图输入是 `[1, seq]`（如 [1,64] = 256 字节 ✓），
        原来只 `struct.pack("<i", pos)` 写 4 字节 ✗ ⇒ 后面 63 个位置保留上次的旧值 ✗
        ⇒ 行为不确定（同一输入可能给出不同结果 ✗）。§162 修 ✓
        """
        dt = self.ins[name][1]
        if dt == 7:                                          # INT64
            self._put(name, struct.pack("<%dq" % n, first, *([0] * (n - 1))))
        else:                                                # INT32（我们的导出 ✓）
            self._put(name, struct.pack("<%di" % n, first, *([0] * (n - 1))))

    def run(self, hidden: bytes, pos: int, kv_len: int) -> bytes:
        """跑一段：hidden 是 token id（首段）或上一段的 hidden；返回本段输出字节。"""
        self._put(self.hidden_name, hidden)
        # ★按需写★：纯线性注意力层组成的段【没有】position_ids / attention_mask ✓
        #   （线性层的状态走 conv/KV 槽 ✓ 不需要 RoPE 位置与掩码 ✓ —— §153 实测 seg0 就是这样 ✓）
        if "position_ids" in self.ins:
            self._put_ints("position_ids", pos, self.ins["position_ids"][3])
        if "new_kv_cache_pos" in self.ins:
            self._put_ints("new_kv_cache_pos", pos, self.ins["new_kv_cache_pos"][3])
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
        for ((_, kout), buf, n) in zip(self.kv_names, self.kv, self.kv_elems_list):
            buf[:] = self.be._get_output_raw(kout, n * 4)     # ★按各自的元素数读回✓★（§162 ✓）
        out = self.logits_name if self.is_last else self.hidden_name
        # 末段取 lm_logits；其余段取图的第一个输出（就是 hidden）
        oname = out if (self.is_last and out) else self.be._outputs[0][0]
        n = self.be._index_of_output(oname)
        elems = self.be._outputs[n][3]
        return self.be._get_output_raw(oname, elems * 4)


class _HostTable:
    """主机侧 fp16 权重表（mmap ✓ 按行取 / 分块 matmul ✓）。

    ★为什么需要★（§155）：qwen3_5 的段图【从 embedding 开始、到 hidden 结束】——
    首段吃 `input_embed`（主机侧算 embedding ✓ 与 ``nnrt_llm`` 同思路 ✓），
    末段只吐 `hidden_states`（词表投影在主机侧 ✓）。Qwen 系 lm_head 与 embedding
    ★tied★ ⇒ 这两侧可以复用同一张表 ✓（gemma4 也是这么做的 ✓ 见 gemma4_runner.py ✓）。
    """

    def __init__(self, path: str, rows: int, cols: int) -> None:
        import numpy as _np                                  # 延迟导入（模块本身不强依赖 numpy ✓）
        self._np = _np
        self.rows, self.cols = int(rows), int(cols)
        self.path = path
        self.mm = _np.memmap(path, dtype=_np.float16, mode="r", shape=(self.rows, self.cols))

    def row(self, token_id: int):
        return self.mm[token_id % self.rows].astype(self._np.float32)

    def matmul(self, h) -> List[float]:
        """logits = W @ h ✓（W: [vocab, hidden] fp16 ✓ · h: [hidden] fp32 ✓）分块算 ✓ 省内存 ✓。"""
        np = self._np
        x = np.asarray(h, dtype=np.float32).reshape(-1)[: self.cols]
        out = np.empty(self.rows, dtype=np.float32)
        step = 8192
        for s in range(0, self.rows, step):
            blk = self.mm[s:s + step].astype(np.float32)
            out[s:s + step] = blk @ x
        return [float(v) for v in out]


def _load_host_table(d: str):
    """读 ``emb_manifest.json`` + 它的权重文件 ✓；没有就返回 None ✓（不报错 ✓）。"""
    man = os.path.join(d, "emb_manifest.json")
    if not os.path.isfile(man):
        return None
    try:
        with open(man, encoding="utf-8") as fh:
            info = json.load(fh)
        path = os.path.join(d, info.get("file", "emb_f16.bin"))
        rows, cols = info["shape"]
        if not os.path.isfile(path):
            return None
        return _HostTable(path, rows, cols)
    except Exception:                                        # noqa: BLE001
        return None


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
            self.segs.append(seg)
        # ★末段输出 lm_logits 时才直接用；否则（我们的 qwen3_5 段只吐 hidden_states ✓）
        #   词表投影要在【主机侧】做 ✓（与 gemma4 同思路 ✓ 模板 gemma4_runner.py ✓）
        self.last_has_logits = self.segs[-1].logits_name is not None
        self.host_tab = None
        if not self.last_has_logits or not self.segs[0].first_is_token:
            self.host_tab = _load_host_table(self.dir)      # ★fp16 表 ✓ 供 embedding/lm_head 两侧复用✓★
        if self.last_has_logits:
            self.vocab = self.segs[-1].be._outputs[
                self.segs[-1].be._index_of_output("lm_logits")][3]
        else:
            self.vocab = self.host_tab.rows if self.host_tab is not None else 0

    def _embed_bytes(self, token_id: int) -> bytes:
        """首段要 embedding 时：把该 token 的 embedding 行做成图输入字节（fp32 ✓）。

        ★注意★：图输入的宽度是【整段 seq × hidden】✓（S=1 时正好一行 ✓；S=64 时 64 行 ✗）
        ⇒ 把行放在【第 0 个位置】、其余补零 ✓（S=1 语义完全正确 ✓；S=64 只用于管道验证 ✓，
          真正的 prefill 应走 chunked 喂入 ✓ 见 §155/§156 ✓）。
        """
        seg = self.segs[0]
        n = seg.hidden_elems
        if self.host_tab is None:
            raise GenerationError("首段要 input_embed，但没找到主机侧嵌入表 ✗")
        row = self.host_tab.row(token_id)
        cols = min(len(row), n)
        buf = [0.0] * n
        buf[:cols] = [float(x) for x in row[:cols]]
        return struct.pack("<%df" % n, *buf)

    def _head_logits(self, hidden_bytes: bytes) -> List[float]:
        """末段只吐 hidden 时：主机侧做 lm_head（与 embedding 【tied】⇒ 复用同一张表 ✓）。

        ★取【最后一个位置】的 hidden★✓：S=1 时就一行 ✓；S=64（整 chunk prefill ✓）时
        取第 64 个 token 的 hidden ✓ —— 正是下一步 logits 该用的那个 ✓。
        """
        if self.host_tab is None:
            raise GenerationError("末段没有 lm_logits，但没找到主机侧词表 ✓")
        cols = self.host_tab.cols
        h = struct.unpack("<%df" % (len(hidden_bytes) // 4),
                          hidden_bytes[:(len(hidden_bytes) // 4) * 4])
        return self.host_tab.matmul(list(h[-cols:]))

    def step(self, token_id: int, pos: int) -> List[float]:
        first = self.segs[0]
        if first.first_is_token:
            hidden = struct.pack("<i", token_id) if first.hidden_dtype == _I32 \
                else struct.pack("<f", float(token_id))
        else:
            hidden = self._embed_bytes(token_id)             # ★主机侧 embedding ✓（§155）★
        logits_bytes = b""
        last_hidden = b""
        for si, seg in enumerate(self.segs):
            out = seg.run(hidden, pos, 0)
            hidden = out
            if seg.is_last:
                last_hidden = out
                logits_bytes = out
        if self.last_has_logits:
            n = len(logits_bytes) // 4
            return list(struct.unpack("<%df" % n, logits_bytes[:n * 4]))
        return self._head_logits(last_hidden)                # ★主机侧 lm_head ✓（tied ✓）★

    def stream(self, prompt: str, max_new: int = 32, stop_ids=(), params=None):
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
        # ★ 采样：给了 params 就按它来（温度/top-k/top-p/重复惩罚/seed），
        #   没给则保持贪心。history 传 prompt+已生成，供重复惩罚使用。
        import random as _random
        from ..sampling import sample_token
        rng = _random.Random(getattr(params, "seed", None)) if params is not None else None
        history: List[int] = list(ids)
        for _ in range(max(1, max_new)):
            nxt = (sample_token(logits, params, history, rng) if params is not None
                   else max(range(len(logits)), key=logits.__getitem__))
            history.append(nxt)
            out.append(nxt)
            cur = self.tok.decode(out)
            if len(cur) > len(prev):
                yield cur[len(prev):], nxt, None
                prev = cur
            if nxt in stop:
                return
            # 额外停止串（params.stop）：解码后一旦出现就收尾
            stops = tuple(getattr(params, "stop", ()) or ()) if params is not None else ()
            if stops and any(x in prev for x in stops):
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
