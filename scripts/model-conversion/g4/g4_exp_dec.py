"""★可移植版：模型用 MODEL_DIR / 输出用 OUTDIR 指定 ✓（原脚本写死本机路径 ✗）★
decode 图导出器（seq=1 + KV）—— 支持三种形态。

为什么需要 decode 图：现在的段图是 prefill-only ⇒ 每生成一个 token 都要重跑整个上下文
（实测 60~100 秒 / 8 token）。decode 图每步只算 1 个 token ⇒ 这是数量级的提速。

接口（KVMAX 默认 128；掩码长度 KVMAX+1，因为图内会把新 token 接在缓存尾部）：
  own 模式（段 0/4/8/12，含层 0~15）：
      入 : hidden[1,1,1536] · cos_sl/sin_sl[1,256] · cos_fu/sin_fu[1,512] · kv_mask[1,1,129]
           · per_layer_0..n-1[1,1,256] · 每个【非共享层】的 k/v 缓存[1,128,D]
      出 : hidden_out · 每个非共享层的【新】k/v[1,1,D]
      ★ 段 12 特殊：层 13/14 还会写入段内共享槽，层 15 直接读它 ⇒ 段内自洽 ✓（图内用 Python 变量跟踪）
      ★ 段 12 只有 3 个非共享层（12/13/14）⇒ KV 入出各 3 组，D 依次 256/256/512 ✓
  shared 模式（段 16/20/24/28/32，层 16~34 全是共享层）：
      入 : hidden · cos/sin · kv_mask · per_layer · ★2 个共享槽 [1,129,D]★（已含本步新 token ✓）
      出 : hidden_out（槽不变 ⇒ 不需输出 ✓）

★ 图内不做 scatter ★：新 K/V 单独输出，主机写回缓存第 pos 个位置（微秒级 ✓）；
  而共享层的槽由主机在段 12 之后维持（槽里第 pos 个位置 = 段 12 输出的新 K/V ✓）。
"""
import os, io, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 可移植 ✓
import g4_seg3d as G

START = int(os.environ.get("START", "0")); NO = int(os.environ.get("NO", "4"))
KVMAX = int(os.environ.get("KVMAX", "128"))
MODE = os.environ.get("MODE") or ("shared" if START >= 16 else "own")
# ★ KV 的 I/O dtype：fp32（默认）/ fp16 / ★int8★ —— 用来测设备对不同 dtype 的 KV 张量上限 ✓
# ★ KVND=4：把 KV 槽张量改成 4 维 [KV,2,1,D/2]（元素数不变）—— 用来判定"限制是否来自 4 维" ✓
KVND = int(os.environ.get("KVND", "3"))
KVDT = os.environ.get("KVDT", "fp32")
_TDT = {"fp32": torch.float32, "fp16": torch.float16, "int8": torch.int8}[KVDT]
OUT = os.environ.get("OUTDIR") or (os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "g4seg") % START)
os.makedirs(OUT, exist_ok=True)

m, tm = G.load()
seg = G.Seg3D(tm, START, NO).eval()
DIMS = seg.dims
LTYPES = list(seg.layer_types)
OWN = [i for i in range(NO)
       if not getattr(seg.layers[i].self_attn, "is_kv_shared_layer", False)]
print("  MODE=%s · 段 [%d:%d] · head_dim=%s · 非共享层(段内序号)=%s" %
      (MODE, START, START + NO, DIMS, OWN), flush=True)


class Dec(nn.Module):
    def __init__(self, seg, mode):
        super().__init__()
        self.seg = seg; self.mode = mode
        self.layers = seg.layers
        self.dims = seg.dims; self.ltypes = seg.layer_types
        # ★ rot_P_* 是 buffer（张量）不是 Module ⇒ 不能用 nn.ModuleList ✗
        self.rot = [getattr(seg, "rot_P_%d" % d) for d in seg.dims]

    def forward(self, hidden, cos_sl, sin_sl, cos_fu, sin_fu, kv_mask, *rest):
        n = len(self.layers)
        ples = rest[:n]
        if self.mode == "shared":
            nk = 0
            ssl_k, ssl_v, sfu_k, sfu_v = rest[n:n + 4]
            kvs = [ssl_k if self.ltypes[i] == "sliding_attention" else sfu_k for i in range(n)]
            vvs = [ssl_v if self.ltypes[i] == "sliding_attention" else sfu_v for i in range(n)]
        else:
            nk = len(OWN)
            kvs = rest[n:n + nk]
            vvs = rest[n + nk:n + 2 * nk]
            ownpos = {gi: p for p, gi in enumerate(OWN)}
        slot = {}
        # ★ 输入侧的 KV 若是低精度，先转 fp32 再算（图内只做类型转换，不改数值语义）
        if self.mode == "shared":
            kvs = [t.float() if t.dtype != torch.float32 else t for t in kvs]
            vvs = [t.float() if t.dtype != torch.float32 else t for t in vvs]
        else:
            kvs = [t.float() if t.dtype != torch.float32 else t for t in kvs]
            vvs = [t.float() if t.dtype != torch.float32 else t for t in vvs]
        x = hidden
        outs_k, outs_v = [], []
        for i, L in enumerate(self.layers):
            A = L.self_attn
            D = self.dims[i]; lt = self.ltypes[i]; P = self.rot[i]
            cos, sin = (cos_sl, sin_sl) if lt == "sliding_attention" else (cos_fu, sin_fu)
            residual = x
            xn = L.input_layernorm(x)
            heads = A.q_proj.out_features // D
            q = A.q_proj(xn).view(1, heads, D).transpose(0, 1)
            q = G.rope3(G.rmsnorm(q, A.q_norm), cos, sin, P)
            if getattr(A, "is_kv_shared_layer", False):
                # ★ 共享层没有 k_proj/v_proj（是"没有属性"而非 None ✓）
                kk, vv = slot[lt] if lt in slot else (kvs[i], vvs[i])
            else:
                k = A.k_proj(xn).view(1, 1, D).transpose(0, 1)
                k = G.rope3(G.rmsnorm(k, A.k_norm), cos, sin, P)
                v = A.v_proj(xn).view(1, 1, D).transpose(0, 1)
                v = G.rmsnorm(v, A.v_norm)
                oi = ownpos.get(i, 0) if self.mode != "shared" else 0
                base_k = kvs[oi] if self.mode != "shared" else kvs[i]
                base_v = vvs[oi] if self.mode != "shared" else vvs[i]
                kk = torch.cat([base_k, k], dim=1)          # [1,KVMAX+1,D]
                vv = torch.cat([base_v, v], dim=1)
                if getattr(A, "store_full_length_kv", False):
                    slot[lt] = (kk, vv)                     # ★ 层 13/14 写段内共享槽
                outs_k.append(k); outs_v.append(v)   # 先按 fp32 收集，最后统一转换
            sc = getattr(A, "scaling", None) or (D ** -0.5)
            att = F.softmax(torch.matmul(q, kk.transpose(-1, -2)) * sc + kv_mask, dim=-1)
            o = torch.matmul(att, vv).transpose(0, 1).reshape(1, 1, heads * D)
            xn = A.o_proj(o)
            x = residual + L.post_attention_layernorm(xn)
            residual = x
            xn = L.mlp(L.pre_feedforward_layernorm(x))
            x = residual + L.post_feedforward_layernorm(xn)
            if getattr(L, "hidden_size_per_layer_input", 0):
                residual = x
                xn = L.act_fn(L.per_layer_input_gate(x)) * ples[i]
                x = residual + L.post_per_layer_input_norm(L.per_layer_projection(xn))
            x = x * L.layer_scalar
        if self.mode == "shared":
            return (x,)
        # ★ 输出侧也按目标 dtype：新 K/V 用 .to(int8) 截断（数值无意义，只测设备接受度与大小）✓
        if _TDT is not torch.float32:
            outs_k = [t.to(_TDT) for t in outs_k]
            outs_v = [t.to(_TDT) for t in outs_v]
        return (x, *outs_k, *outs_v)


w = Dec(seg, MODE).eval()
H = tm.config.hidden_size; PLE = tm.config.hidden_size_per_layer_input
args = [torch.zeros([1, 1, H]), torch.zeros([1, 256]), torch.zeros([1, 256]),
        torch.zeros([1, 512]), torch.zeros([1, 512]), torch.zeros([1, 1, KVMAX + 1])]
names = ["hidden", "cos_sl", "sin_sl", "cos_fu", "sin_fu", "kv_mask"]
for i in range(NO):
    args.append(torch.zeros([1, 1, PLE])); names.append("per_layer_%d" % i)
if MODE == "shared":
    d_sl = next((DIMS[i] for i in range(NO) if LTYPES[i] == "sliding_attention"), 256)
    d_fu = next((DIMS[i] for i in range(NO) if LTYPES[i] == "full_attention"), 512)
    for nm, d in (("slot_sl_k", d_sl), ("slot_sl_v", d_sl), ("slot_fu_k", d_fu), ("slot_fu_v", d_fu)):
        if KVND == 4:
            args.append(torch.zeros([KVMAX + 1, 2, 1, d // 2])); names.append(nm)
        else:
            args.append(torch.zeros([1, KVMAX + 1, d])); names.append(nm)
    onam = ["hidden_out"]
else:
    for i in OWN:
        args.append(torch.zeros([1, KVMAX, DIMS[i]])); names.append("k_%d" % i)
    for i in OWN:
        args.append(torch.zeros([1, KVMAX, DIMS[i]])); names.append("v_%d" % i)
    onam = ["hidden_out"] + ["k_%d_out" % i for i in OWN] + ["v_%d_out" % i for i in OWN]

_KVIDX = None
if KVDT != "fp32":
    # KV 相关的输入（shared: 4 个槽；own: k_*/v_* 各 len(OWN) 个）全部换成目标 dtype
    if MODE == "shared":
        for j in range(len(args) - 4, len(args)):
            args[j] = args[j].to(_TDT)
    else:
        nk = len(OWN)
        for j in range(len(args) - 2 * nk, len(args)):
            args[j] = args[j].to(_TDT)
with torch.no_grad():
    o = w(*args)
print("  前向 ✓ 输出 %s" % str([tuple(t.shape) for t in o]), flush=True)
p = os.path.join(OUT, "dec.onnx")
torch.onnx.export(w, tuple(args), p, input_names=names, output_names=onam,
                  opset_version=14, do_constant_folding=True, dynamo=False)
import onnx
from onnx import shape_inference
mm = shape_inference.infer_shapes(onnx.load(p, load_external_data=False), strict_mode=False)
g = mm.graph
shp = {}
def rec(t):
    tt = t.type.tensor_type
    if tt.HasField("shape"):
        shp[t.name] = [(d.dim_value if d.HasField("dim_value") else "?") for d in tt.shape.dim]
for t in list(g.input) + list(g.output) + list(g.value_info): rec(t)
for nn_ in g.initializer: shp[nn_.name] = list(nn_.dims)
n4 = sum(1 for k, v in shp.items() if len(v) >= 4)
ops = {}
for nn_ in g.node: ops[nn_.op_type] = ops.get(nn_.op_type, 0) + 1
print("  ★ 导出 ✓ 节点 %d · ≥4维 %d · Slice %d · %.2f GB" %
      (len(g.node), n4, ops.get("Slice", 0) + ops.get("StridedSlice", 0),
       sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)
           if os.path.isfile(os.path.join(OUT, f))) / 1e9), flush=True)
T = {1: "FP32", 6: "INT32", 7: "INT64", 10: "FP16"}; T2 = {1: "float32", 6: "int32", 7: "int64", 10: "float16"}
def sh(t): return ",".join(str(x) for x in shp[t.name])
ins = [(i.name, T.get(i.type.tensor_type.elem_type, "FP32"), sh(i)) for i in g.input]
outs = [(o_.name, T.get(o_.type.tensor_type.elem_type, "FP32"), sh(o_)) for o_ in g.output]
io.open(os.path.join(OUT, "omg.txt"), "w").write(
    ";".join("%s:%s" % (n_, s) for n_, _, s in ins) + "\n"
    + ";".join("%s:%s" % (n_, d) for n_, d, _ in ins) + "\n"
    + ";".join("%s:%s" % (n_, d) for n_, d, _ in outs) + "\n")
io.open(os.path.join(OUT, "c.cfg"), "w").write("\n".join(["[third_party_model]",
    "input_names=" + ";".join(n_ for n_, _, _ in ins),
    "input_dtypes=" + ";".join(T2.get(d, "float32") for _, d, _ in ins),
    "input_shapes=" + ";".join(s for _, _, s in ins),
    "output_names=" + ";".join(n_ for n_, _, _ in outs),
    "output_dtypes=" + ";".join(T2.get(d, "float32") for _, d, _ in outs),
    "output_shapes=" + ";".join(s for _, _, s in outs)]) + "\n")
print("  ✓ 输入 %d · 输出 %d · omg.txt/c.cfg 已写" % (len(ins), len(outs)), flush=True)
