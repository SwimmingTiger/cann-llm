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
        self._cache = {}          # ★ build 很贵：同一个 .ms 只建一次（不 destroy ⇒ 不会悬垂 ✓）
        self._bind()              # ★ 必须调用：argtypes/restype 都在里面设置

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
        # 表行已是 float32 小端 ⇒ 直接拼即可（不需要再解包重打一遍 ✗）
        # ★ 图P 是按 S=1 导的（[1,S,35,256] 的输出决定了 S 不能大）⇒ 逐 token 跑再拼 ✓
        #   有了模型缓存，每次只剩一次 Predict ✓
        pl_parts = []
        for i in pad:
            r = self.ms.run(os.path.join(self.dir, "graphP", "graphP.ms"),
                            {"input_ids": struct.pack("<i", i),
                             "identity": self.pl_row_scaled(i)})
            pl_parts.append(r["per_layer"])
        per_layer = b"".join(pl_parts)
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


# --------------------------------------------------------------------------
# 接入 cann-llm 的适配层：与 SegmentedLlmRunner 同名同签名 ⇒ NnrtBackend 无需区分
# --------------------------------------------------------------------------
class Gemma4ChatRunner(Gemma4SegRunner):
    """在 Gemma4SegRunner 上补 load/stream/generate（接口对齐 nnrt_seg.SegmentedLlmRunner）。

    对话模板（来自 chat_template.jinja 与 added_tokens 实测）：
        <bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n
    停止符：eos_token_id = [1, 106, 50]，其中 106 就是 <turn|> ✓
    """
    TURN_OPEN, TURN_CLOSE, BOS = 105, 106, 2
    STOP_IDS = (1, 106, 50)

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        from ..tokenizer_gemma import GemmaTokenizer
        self.tok = GemmaTokenizer(os.path.join(model_dir, "tokenizer.json"))

    def load(self) -> None:
        """图在 forward 里按需 build（每个 .ms 一次），这里只做一次热身校验。"""
        assert self.tok is not None

    # 拼对话 prompt（不进 chat 模板时可用 .raw = True 走纯文本续写）
    def _prompt_ids(self, prompt: str, raw: bool) -> List[int]:
        if raw:
            return self.tok.encode(prompt, add_bos=True)
        ids = [self.BOS, self.TURN_OPEN]
        ids += self.tok.encode("user\n", add_bos=False)
        ids += self.tok.encode(prompt, add_bos=False)
        ids += [self.TURN_CLOSE]
        ids += self.tok.encode("\n", add_bos=False)
        ids += [self.TURN_OPEN]
        ids += self.tok.encode("model\n", add_bos=False)
        return ids

    def stream(self, prompt: str, max_new: int = 32, stop_ids=(), params=None,
               raw: bool = False):
        """逐 token 产出【新增文本】（累积解码、只吐后缀 —— 与 Qwen 版同一套做法 ✓）"""
        import random as _random
        from ..sampling import sample_token
        ids = self._prompt_ids(prompt, raw)
        if not ids:
            return
        stop = set(stop_ids) or set(self.STOP_IDS)
        rng = _random.Random(getattr(params, "seed", None)) if params is not None else None
        out: List[int] = []
        prev = ""
        for _ in range(max(1, max_new)):
            logits = self.forward(ids)                 # ★ 每步重跑全上下文（prefill-only ✓）
            hist = ids + out
            nxt = (sample_token(logits, params, hist, rng) if params is not None
                   else max(range(len(logits)), key=logits.__getitem__))
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
        prev = ""
        stop = set(stop_ids) or set(self.STOP_IDS)
        for _ in range(max(1, max_new)):
            logits = self.forward(ids)
            nxt = max(range(len(logits)), key=logits.__getitem__)
            out.append(nxt)
            if nxt in stop:
                break
            prev = self.tok.decode(out)
        return (self.tok.decode(out), out,
                {"total_ms": (time.time() - t0) * 1000.0,
                 "prompt_tokens": len(ids), "completion_tokens": len(out)})
