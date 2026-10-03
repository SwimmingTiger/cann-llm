"""Gemma 4 E2B 在 NNRt 上的分段 runner（纯 Python + ctypes，跑在设备上）。

为什么长这样（每条都有实测依据，详见 docs 与提交记录）：
  · 9 段（4×8+3）经 OMG/converter 编成 .ms，段间只传 hidden 与 2 组共享 KV
    （embed_tokens_per_layer 2.35e9 元素 > OMG 的 INT_MAX ⇒ 必须按层切开）
  · 图 P 负责 per_layer 的投影（[8960,1536] 矩阵乘 ⇒ 纯 Python 太慢，必须上 NPU），
    主机侧只做 mmap 按行查表；★图 P 按 S=1 导★（它的输出 [1,S,35,256] 是最紧的约束）
  · lm_head 与 embed_tokens 绑定（tied）⇒ 切 4 块各 65536（整词表输出 1MB 顶到上限）；
    且只对最后一个真实 token 做 lm（S=1）
  · 段图按 SEQ=128 导（其它中间张量都在"单张量 <1MB"内）
  · 目前 prefill-only（段图没有 KV 输入）⇒ 每生成一个 token 重跑全上下文：正确但慢
  · ★已 build 的模型要缓存★：build 是慢的根源；★故意不 destroy★（销毁会 core dump）
"""
from __future__ import annotations
import ctypes as C, json, math, os, struct
from typing import Iterator, List

_NDK = "/system/lib64/ndk/libmindspore_lite_ndk.so"
_DEV_NNRT = 60
_MINDIR = 0


class _TA(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]


class _Mslite:
    """最小 ctypes 封装（照已跑通的最小写法：按序号喂输入、不 destroy）"""

    def __init__(self, lib_path: str = _NDK):
        self.lib = C.CDLL(lib_path)
        self._cache = {}
        self._bind()

    def _bind(self):
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
        m = self._cache.get(ms)
        if m is None:
            ctx = L.OH_AI_ContextCreate()
            L.OH_AI_ContextAddDeviceInfo(ctx, L.OH_AI_DeviceInfoCreate(_DEV_NNRT))
            m = L.OH_AI_ModelCreate()
            if L.OH_AI_ModelBuildFromFile(m, ms.encode(), _MINDIR, ctx) != 0:
                raise RuntimeError("Build 失败: %s" % ms)
            self._cache[ms] = m
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
    """model_dir 布局：
         graphP/graphP.ms · seg{0,4,...,32}/seg.ms · lm/lm{0..3}.ms
         weights/{manifest.json,embed_tokens.f16,embed_tokens_per_layer.f16}
         tokenizer.json · io/{cos,sin}_{sl,fu}.bin
    """
    SEQ = 128
    SEG_STARTS = (0, 4, 8, 12, 16, 20, 24, 28, 32)
    N_LAYERS, PLE = 35, 256
    LOGIT_CAP = 30.0

    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.ms = _Mslite()
        self.wdir = os.path.join(model_dir, "weights")
        man = json.load(open(os.path.join(self.wdir, "manifest.json")))
        self.e_dim = man["embed_tokens"]["shape"][1]
        self.pl_dim = man["embed_tokens_per_layer"]["shape"][1]
        self._f_emb = open(os.path.join(self.wdir, "embed_tokens.f16"), "rb")
        self._f_pl = open(os.path.join(self.wdir, "embed_tokens_per_layer.f16"), "rb")
        # ★ 段图里【故意去掉了最终的 self.norm】（否则每段都会归一一次 ✗）
        #   ⇒ 这里必须在 lm_head 之前补上 ✓（曾经漏掉 ⇒ lm_head 拿到的 hidden 是原始尺度 ⇒ 乱码 ✗）
        self._f_norm = open(os.path.join(self.wdir, "final_norm.f16"), "rb")
        self.norm_w = struct.unpack("<%de" % self.e_dim, self._f_norm.read(self.e_dim * 2))
        iod = os.path.join(model_dir, "io")
        self.cos_sl = open(os.path.join(iod, "cos_sl.bin"), "rb").read()
        self.sin_sl = open(os.path.join(iod, "sin_sl.bin"), "rb").read()
        self.cos_fu = open(os.path.join(iod, "cos_fu.bin"), "rb").read()
        self.sin_fu = open(os.path.join(iod, "sin_fu.bin"), "rb").read()
        self._mask = self._causal_mask(self.SEQ)

    # ---------- 主机侧 mmap 查表 ----------
    # ★ embed_tokens 是 Gemma4TextScaledWordEmbedding ★
    # 它 forward 时会乘 embed_scale = sqrt(hidden_size)（实测 39.191835884530846 ✓）
    # 曾经漏掉这个缩放 ⇒ 段的输入小了 39 倍 ⇒ 输出全是乱码 ✗
    EMB_SCALE = 39.191835884530846

    def emb(self, i: int) -> bytes:
        """裸表行 × embed_scale，并转成 fp32（权重存 fp16，图要 fp32 ✗）"""
        self._f_emb.seek(i * self.e_dim * 2)
        raw = self._f_emb.read(self.e_dim * 2)
        return struct.pack("<%df" % self.e_dim,
                           *[v * self.EMB_SCALE
                             for v in struct.unpack("<%de" % self.e_dim, raw)])

    def pl_row_scaled(self, i: int) -> bytes:
        """token-identity：表行 × sqrt(ple_dim)（= HF get_per_layer_inputs ✓）"""
        self._f_pl.seek(i * self.pl_dim * 2)
        raw = self._f_pl.read(self.pl_dim * 2)
        sc = math.sqrt(self.PLE)
        return struct.pack("<%df" % self.pl_dim,
                           *[v * sc for v in struct.unpack("<%de" % self.pl_dim, raw)])

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
            raise ValueError("上下文 %d 超过图的 %d" % (n, self.SEQ))
        pad = list(ids) + [0] * (self.SEQ - n)
        # 图 P 按 S=1 导 ⇒ 逐 token 跑再拼（有缓存，每次只剩一次 Predict）✓
        per_layer = b"".join(
            self.ms.run(os.path.join(self.dir, "graphP", "graphP.ms"),
                        {"input_ids": struct.pack("<i", i),
                         "identity": self.pl_row_scaled(i)})["per_layer"]
            for i in pad)
        hidden = b"".join(self.emb(i) for i in pad)
        kv = {}
        for st in self.SEG_STARTS:
            no = 4 if st < 32 else 3
            feeds = {"hidden": hidden, "mask3": self._mask,
                     "cos_sl": self.cos_sl[:self.SEQ * 256 * 4],
                     "sin_sl": self.sin_sl[:self.SEQ * 256 * 4],
                     "cos_fu": self.cos_fu[:self.SEQ * 512 * 4],
                     "sin_fu": self.sin_fu[:self.SEQ * 512 * 4]}
            if st >= 16:
                feeds.update(kv)
            for i in range(no):
                feeds["per_layer_%d" % i] = b"".join(
                    per_layer[(t * self.N_LAYERS + st + i) * self.PLE * 4:
                              (t * self.N_LAYERS + st + i + 1) * self.PLE * 4]
                    for t in range(self.SEQ))
            r = self.ms.run(os.path.join(self.dir, "seg%d" % st, "seg.ms"), feeds)
            hidden = r["hidden_out"]
            if "sk_out" in r:
                kv = {"sk": r["sk_out"], "sv": r["sv_out"],
                      "fk": r["fk_out"], "fv": r["fv_out"]}
        off = (n - 1) * self.e_dim * 4
        h_last = hidden[off:off + self.e_dim * 4]
        # ★ 最终 RMSNorm（Gemma4RMSNorm：用 pow 而非 rsqrt；eps=1e-6）★
        hv = struct.unpack("<%df" % self.e_dim, h_last)
        ms = sum(v * v for v in hv) / self.e_dim + 1e-6
        sc = ms ** -0.5
        h_last = struct.pack("<%df" % self.e_dim,
                             *[v * sc * w for v, w in zip(hv, self.norm_w)])
        logits: List[float] = []
        for J in range(4):
            logits.extend(_f32(self.ms.run(os.path.join(self.dir, "lm", "lm%d.ms" % J),
                                           {"hidden": h_last})["logits"]))
        return [math.tanh(v / self.LOGIT_CAP) * self.LOGIT_CAP for v in logits]

    # ---------- 采样 ----------
    def _pick(self, logits: List[float], params, rng) -> int:
        if params is not None:
            from ..sampling import sample_token
            return sample_token(logits, params, [], rng)
        return max(range(len(logits)), key=logits.__getitem__)


# --------------------------------------------------------------------------
# 接入 cann-llm 的适配层：与 SegmentedLlmRunner 同名同签名
# --------------------------------------------------------------------------
class Gemma4ChatRunner(Gemma4SegRunner):
    """对话模板（chat_template.jinja + added_tokens 实测）：
        <bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n
    停止符 eos_token_id = [1, 106, 50]（106 即 <turn|> ✓）
    """
    TURN_OPEN, TURN_CLOSE, BOS = 105, 106, 2
    STOP_IDS = (1, 106, 50)

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        from ..tokenizer_gemma import GemmaTokenizer
        self.tok = GemmaTokenizer(os.path.join(model_dir, "tokenizer.json"))

    def load(self) -> None:
        assert self.tok is not None

    def _prompt_ids(self, prompt: str, raw: bool = False) -> List[int]:
        if raw:
            return self.tok.encode(prompt, add_bos=True)
        ids = [self.BOS, self.TURN_OPEN] + self.tok.encode("user\n")
        ids += self.tok.encode(prompt)
        ids += [self.TURN_CLOSE] + self.tok.encode("\n")
        ids += [self.TURN_OPEN] + self.tok.encode("model\n")
        return ids

    def stream(self, prompt: str, max_new: int = 32, stop_ids=(), params=None, raw: bool = False):
        """逐 token 产出【新增文本】（累积解码、只吐后缀 —— 与 Qwen 版同一手法）"""
        import random as _random
        ids = self._prompt_ids(prompt, raw)
        if not ids:
            return
        stop = set(stop_ids) or set(self.STOP_IDS)
        rng = _random.Random(getattr(params, "seed", None)) if params is not None else None
        out: List[int] = []
        prev = ""
        for _ in range(max(1, max_new)):
            logits = self.forward(ids + out)
            nxt = self._pick(logits, params, rng)
            out.append(nxt)
            cur = self.tok.decode(out)
            if len(cur) > len(prev):
                yield cur[len(prev):], nxt, None
                prev = cur
            if nxt in stop:
                return
            stops = tuple(getattr(params, "stop", ()) or ()) if params is not None else ()
            if stops and any(x in prev for x in stops):
                return

    def generate(self, prompt: str, max_new: int = 32, stop_ids=(), raw: bool = False):
        import time
        ids = self._prompt_ids(prompt, raw)
        t0 = time.time()
        out: List[int] = []
        stop = set(stop_ids) or set(self.STOP_IDS)
        for _ in range(max(1, max_new)):
            logits = self.forward(ids + out)
            nxt = max(range(len(logits)), key=logits.__getitem__)
            out.append(nxt)
            if nxt in stop:
                break
        return (self.tok.decode(out), out,
                {"total_ms": (time.time() - t0) * 1000.0,
                 "prompt_tokens": len(ids), "completion_tokens": len(out)})
