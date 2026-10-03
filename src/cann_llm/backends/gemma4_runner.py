"""Gemma 4 E2B 在 NNRt 上的分段 runner（纯 Python + ctypes，跑在设备上）。

为什么是现在这个样子（每一条都有实测依据）：
  · 9 段（4×8+3）经 OMG/converter 编成 .ms，段间只传 hidden 与 2 组共享 KV
    （embed_tokens_per_layer 2.35e9 元素 > OMG 的 INT_MAX ⇒ 必须按层切开）
  · 图 P 负责 per_layer 的投影（[8960,1536] 矩阵乘 ⇒ 纯 Python 太慢，必须上 NPU）
    主机侧只做 mmap 按行查表（几十微秒）✓
  · lm_head 与 embed_tokens 绑定（tied）⇒ 切成 4 块各 [.,65536]，否则整词表输出
    [1,S,262144] = 1MB 正好顶到设备"单张量 <1MB"上限 ✗
  · seq 必须固定：段图按 SEQ=128 导（受图 P 的 [1,S,35,256] 上限约束），
    图 P/lm 按 SEQ=1 导（逐 token）✓
  · 目前是 prefill-only（图没有 KV 输入）⇒ 每生成一个 token 重跑全上下文：
    正确但慢；KV 缓存是后续优化 ✓
"""
from __future__ import annotations
import ctypes as C, json, math, os, struct, sys
from typing import Iterator, List, Optional

_NDK = "/system/lib64/ndk/libmindspore_lite_ndk.so"
_DEV_NNRT = 60
_MINDIR = 0


class _TA(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]


class _Mslite:
    """最小 ctypes 封装（照已跑通的最小写法：按序号喂输入、不主动 destroy）"""
    def __init__(self, lib_path: str = _NDK):
        self.lib = C.CDLL(lib_path)
        L = self.lib
        for fn, rt, at in [
            ("OH_AI_ModelCreate", C.c_void_p, []),
            ("OH_AI_ContextCreate", C.c_void_p, []),
            ("OH_AI_DeviceInfoCreate", C.c_void_p, [C.c_int]),
            ("OH_AI_ContextAddDeviceInfo", None, [C.c_void_p, C.c_void_p]),
            ("OH_AI_ModelBuildFromFile", C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p]),
            ("OH_AI_ModelGetInputs", _TA, [C.c_void_p]),
            ("OH_AI_ModelGetOutputs", _TA, [C.c_void_p]),
            ("OH_AI_TensorGetMutableData", C.c_void_p, [C.c_void_p]),
            ("OH_AI_TensorGetElementNum", C.c_size_t, [C.c_void_p]),
            ("OH_AI_TensorGetName", C.c_char_p, [C.c_void_p]),
            ("OH_AI_ModelPredict", C.c_int, [C.c_void_p, _TA, C.POINTER(_TA), C.c_void_p, C.c_void_p]),
        ]:
            f = getattr(L, fn)
            if rt:
                f.restype = rt
            if at:
                f.argtypes = at

    def run(self, ms: str, feeds: dict) -> dict:
        L = self.lib
        ctx = L.OH_AI_ContextCreate()
        L.OH_AI_ContextAddDeviceInfo(ctx, L.OH_AI_DeviceInfoCreate(_DEV_NNRT))
        m = L.OH_AI_ModelCreate()
        if L.OH_AI_ModelBuildFromFile(m, ms.encode(), _MINDIR, ctx) != 0:
            raise RuntimeError("Build 失败: %s" % ms)
        ins = L.OH_AI_ModelGetInputs(m)
        for i in range(ins.handle_num):
            t = ins.handle_list[i]
            nm = L.OH_AI_TensorGetName(t).decode()
            n = L.OH_AI_TensorGetElementNum(t)
            d = feeds[nm]
            if len(d) != n * 4:
                raise ValueError("输入 %s 大小不符: %d vs %d" % (nm, len(d), n * 4))
            C.memmove(L.OH_AI_TensorGetMutableData(t), d, len(d))
        outs = _TA()
        if L.OH_AI_ModelPredict(m, ins, C.byref(outs), None, None) != 0:
            raise RuntimeError("Predict 失败: %s" % ms)
        res = {}
        for i in range(outs.handle_num):
            t = outs.handle_list[i]
            nm = L.OH_AI_TensorGetName(t).decode()
            res[nm] = C.string_at(L.OH_AI_TensorGetMutableData(t),
                                  L.OH_AI_TensorGetElementNum(t) * 4)
        return res


def _f32(b: bytes) -> List[float]:
    return list(struct.unpack("<%df" % (len(b) // 4), b))


class Gemma4SegRunner:
    """model_dir 布局（本会话产出的目录）：
         graphP/graphP.ms · seg{0,4,...,32}/seg.ms · lm/lm{0..3}.ms
         weights/{manifest.json,embed_tokens.f16,embed_tokens_per_layer.f16}
         tokenizer.json · io/{cos,sin}_{sl,fu}.bin
    """
    SEQ = 128
    SEG_STARTS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    N_LAYERS, PLE = 35, 256

    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.ms = _Mslite()
        self.wdir = os.path.join(model_dir, "weights")
        man = json.load(open(os.path.join(self.wdir, "manifest.json")))
        self.e_dim = man["embed_tokens"]["shape"][1]
        self.pl_dim = man["embed_tokens_per_layer"]["shape"][1]
        self._f_emb = open(os.path.join(self.wdir, "embed_tokens.f16"), "rb")
        self._f_pl = open(os.path.join(self.wdir, "embed_tokens_per_layer.f16"), "rb")
        iod = os.path.join(model_dir, "io")
        self.cos_sl = open(os.path.join(iod, "cos_sl.bin"), "rb").read()
        self.sin_sl = open(os.path.join(iod, "sin_sl.bin"), "rb").read()
        self.cos_fu = open(os.path.join(iod, "cos_fu.bin"), "rb").read()
        self.sin_fu = open(os.path.join(iod, "sin_fu.bin"), "rb").read()
        self._mask = self._causal_mask(self.SEQ)
        self._kv = {}

    # ---------- 主机侧查表 ----------
    def emb(self, i: int) -> bytes:
        self._f_emb.seek(i * self.e_dim * 2)
        return self._f_emb.read(self.e_dim * 2)

    def pl_row_scaled(self, i: int) -> bytes:
        """token-identity：表行 × sqrt(ple_dim)（= HF 的 get_per_layer_inputs ✓）"""
        self._f_pl.seek(i * self.pl_dim * 2)
        sc = math.sqrt(self.PLE)
        return struct.pack("<%df" % self.pl_dim,
                           *[v * sc for v in struct.unpack("<%de" % self.pl_dim,
                                                           self._f_pl.read(self.pl_dim * 2))])

    @staticmethod
    def _causal_mask(seq: int) -> bytes:
        neg = -1.0e9
        out = []
        for i in range(seq):
            out.extend([0.0 if j <= i else neg for j in range(seq)])
        return struct.pack("<%df" % (seq * seq), *out)

    # ---------- 前向 ----------
    def forward(self, ids: List[int]) -> List[float]:
        n = len(ids)
        if n > self.SEQ:
            raise ValueError("上下文 %d 超过图的 %d（受图 P 张量上限约束）" % (n, self.SEQ))
        pad = list(ids) + [0] * (self.SEQ - n)
        # 真实 token 放在【最前面】，padding 在尾部；因果掩码保证真实 token 互不影响 ✓
        ident = b"".join(self.pl_row_scaled(i) for i in pad).__class__  # 占位（见下）
        ident = b"".join(self.pl_row_scaled(i) for i in pad)
        ident = (struct.pack("<%df" % (self.SEQ * self.N_LAYERS * self.PLE),
                             *struct.unpack("<%df" % (self.SEQ * self.pl_dim), ident)))
        per_layer = self.ms.run(os.path.join(self.dir, "graphP", "graphP.ms"),
                                {"input_ids": struct.pack("<%di" % self.SEQ, *pad),
                                 "identity": ident})["per_layer"]
        hidden = b"".join(self.emb(i) for i in pad)
        kv = {}
        for st in self.SEG_STARTS:
            no = 4 if st < 32 else 3
            feeds = {"hidden": hidden, "mask3": self._mask,
                     "cos_sl": self.cos_sl[:self.SEQ * 256 * 4],
                     "sin_sl": self.sin_sl[:self.SEQ * 256 * 4],
                     "cos_fu": self.cos_fu[:self.SEQ * 512 * 4],
                     "sin_fu": self.sin_fu[:self.SEQ * 512 * 4]}
            if st >= 16:                       # 段 16~32 接收共享 KV ✓
                feeds.update(kv)
            for i in range(no):
                base = (st + i) * self.PLE * 4
                feeds["per_layer_%d" % i] = b"".join(
                    per_layer[(t * self.N_LAYERS + st + i) * self.PLE * 4:
                              (t * self.N_LAYERS + st + i + 1) * self.PLE * 4]
                    for t in range(self.SEQ))
            r = self.ms.run(os.path.join(self.dir, "seg%d" % st, "seg.ms"), feeds)
            hidden = r["hidden_out"]
            if "sk_out" in r:                  # 段 12 产出共享 KV ✓
                kv = {"sk": r["sk_out"], "sv": r["sv_out"],
                      "fk": r["fk_out"], "fv": r["fv_out"]}
        # 末段之后：只对【最后一个真实 token】做 lm_head（S=1，否则输出超 1MB ✗）
        off = (n - 1) * self.e_dim * 4
        h_last = hidden[off:off + self.e_dim * 4]
        logits = []
        for J in range(4):
            r = self.ms.run(os.path.join(self.dir, "lm", "lm%d.ms" % J), {"hidden": h_last})
            logits.extend(_f32(r["logits"]))
        cap = 30.0                              # final_logit_softcapping ✓
        return [math.tanh(v / cap) * cap for v in logits]

    # ---------- 生成 ----------
    def generate(self, ids: List[int], max_new: int = 32, eos_ids=(1, 106, 50),
                 temperature: float = 0.0) -> Iterator[int]:
        import random
        rng = random.Random(0)
        for _ in range(max_new):
            logits = self.forward(ids)
            if temperature and temperature > 0:
                m = max(logits)
                ex = [math.exp((v - m) / temperature) for v in logits]
                s = sum(ex)
                r, acc = rng.random(), 0.0
                nxt = len(logits) - 1
                for i, p in enumerate(ex):
                    acc += p / s
                    if acc >= r:
                        nxt = i
                        break
            else:
                nxt = max(range(len(logits)), key=logits.__getitem__)
            ids.append(nxt)
            yield nxt
            if nxt in eos_ids:
                return
