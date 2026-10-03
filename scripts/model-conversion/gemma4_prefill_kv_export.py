"""prefill + KV 输出的导出器（seq=S，一次算完整段）。

为什么需要它：decode 图是 seq=1 的 ⇒ 若拿它逐 token 跑 prefill，15 个 token × 9 段 ≈ 200 秒 ✗。
所以 prefill 必须保留"一次算完 S 个 token"的图，但★必须把 KV 暴露出来★给 decode 用 ✓。

接口（S 由 env 给，KV 形状与 decode 的缓存一致 ⇒ 主机直接拷进缓存 ✓）：
  own 模式（段 0/4/8/12）：
      入 : hidden[1,S,1536] · mask3[1,S,S]（因果）· cos_sl/sin_sl[S,256] · cos_fu/sin_fu[S,512]
           · per_layer_0..n-1[1,S,256]
      出 : hidden_out · 每个【非共享层】的 k/v[1,S,D]
      ★ 段 12：层 13/14 的 KV 同时就是 2 个共享槽 ⇒ 主机从这 3 组里取 ✓（不用额外输出 ✓）
  shared 模式（段 16~32）：
      入 : hidden · mask3 · cos/sin · per_layer · ★2 个槽 [1,S,D]★（来自段 12 的输出 ✓）
      出 : hidden_out
"""
import os, io, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
import g4_seg3d as G

START = int(os.environ.get("START", "0")); NO = int(os.environ.get("NO", "4"))
S = int(os.environ.get("SEQ", "32"))
MODE = os.environ.get("MODE") or ("shared" if START >= 16 else "own")
OUT = os.environ.get("OUTDIR") or ("/home/hu60/work/llm/ddk-llm/llm-poc/g4pre%d" % START)
os.makedirs(OUT, exist_ok=True)

m, tm = G.load()
seg = G.Seg3D(tm, START, NO).eval()
DIMS, LTYPES = seg.dims, list(seg.layer_types)
OWN = [i for i in range(NO) if not getattr(seg.layers[i].self_attn, "is_kv_shared_layer", False)]
print("  PREFILL MODE=%s · 段 [%d:%d] · S=%d · head_dim=%s · 非共享层=%s" %
      (MODE, START, START + NO, S, DIMS, OWN), flush=True)


class Pre(nn.Module):
    def __init__(self, seg, mode):
        super().__init__()
        self.layers = seg.layers; self.mode = mode
        self.dims = seg.dims; self.ltypes = seg.layer_types
        self.rot = [getattr(seg, "rot_P_%d" % d) for d in seg.dims]

    def forward(self, hidden, mask3, cos_sl, sin_sl, cos_fu, sin_fu, *rest):
        n = len(self.layers)
        ples = rest[:n]
        if self.mode == "shared":
            ssl_k, ssl_v, sfu_k, sfu_v = rest[n:n + 4]
            kvs = [ssl_k if self.ltypes[i] == "sliding_attention" else sfu_k for i in range(n)]
            vvs = [ssl_v if self.ltypes[i] == "sliding_attention" else sfu_v for i in range(n)]
        slot = {}
        x = hidden
        outs_k, outs_v = [], []
        for i, L in enumerate(self.layers):
            A = L.self_attn
            D = self.dims[i]; lt = self.ltypes[i]; P = self.rot[i]
            cos, sin = (cos_sl, sin_sl) if lt == "sliding_attention" else (cos_fu, sin_fu)
            residual = x
            xn = L.input_layernorm(x)
            B, Sq, _ = xn.shape
            heads = A.q_proj.out_features // D
            q = A.q_proj(xn).view(B, Sq, heads, D).transpose(1, 2)     # [1,heads,S,D]
            q = G.rope3(q, cos, sin, P)
            # q_norm 要在分头之后做（与 HF 一致 ✓）
            q = G.rmsnorm(q, A.q_norm) if False else q
            if getattr(A, "is_kv_shared_layer", False):
                kk, vv = slot[lt] if lt in slot else (kvs[i], vvs[i])
            else:
                k = A.k_proj(xn).view(B, Sq, 1, D).transpose(1, 2)
                k = G.rope3(G.rmsnorm(k, A.k_norm), cos, sin, P)
                v = A.v_proj(xn).view(B, Sq, 1, D).transpose(1, 2)
                v = G.rmsnorm(v, A.v_norm)
                kk, vv = k, v                                  # ★ 整段 S 个 token 的 K/V
                if getattr(A, "store_full_length_kv", False):
                    slot[lt] = (kk, vv)
                outs_k.append(k); outs_v.append(v)
            sc = getattr(A, "scaling", None) or (D ** -0.5)
            att = F.softmax(torch.matmul(q, kk.transpose(-1, -2)) * sc + mask3, dim=-1)
            o = torch.matmul(att, vv).transpose(1, 2).reshape(B, Sq, heads * D)
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
        return (x, *outs_k, *outs_v)


w = Pre(seg, MODE).eval()
H = tm.config.hidden_size; PLE = tm.config.hidden_size_per_layer_input
args = [torch.zeros([1, S, H]), torch.zeros([1, S, S]),
        torch.zeros([S, 256]), torch.zeros([S, 256]), torch.zeros([S, 512]), torch.zeros([S, 512])]
names = ["hidden", "mask3", "cos_sl", "sin_sl", "cos_fu", "sin_fu"]
for i in range(NO):
    args.append(torch.zeros([1, S, PLE])); names.append("per_layer_%d" % i)
if MODE == "shared":
    d_sl = next((DIMS[i] for i in range(NO) if LTYPES[i] == "sliding_attention"), 256)
    d_fu = next((DIMS[i] for i in range(NO) if LTYPES[i] == "full_attention"), 512)
    for nm, d in (("slot_sl_k", d_sl), ("slot_sl_v", d_sl), ("slot_fu_k", d_fu), ("slot_fu_v", d_fu)):
        args.append(torch.zeros([1, S, d])); names.append(nm)
    onam = ["hidden_out"]
else:
    for i in OWN:
        args.append(torch.zeros([1, S, DIMS[i]])); names.append("k_%d" % i)
    for i in OWN:
        args.append(torch.zeros([1, S, DIMS[i]])); names.append("v_%d" % i)
    onam = ["hidden_out"] + ["k_%d_out" % i for i in OWN] + ["v_%d_out" % i for i in OWN]

with torch.no_grad():
    o = w(*args)
print("  前向 ✓ 输出 %s" % str([tuple(t.shape) for t in o]), flush=True)
p = os.path.join(OUT, "pre.onnx")
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
