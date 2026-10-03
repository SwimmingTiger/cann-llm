"""导出 decode 图（seq=1 + KV 输入/输出）—— 先导段 0，作为整个 decode 路线的验证。

接口（每层一个 KV：非共享层各有自己的；共享层从共享槽取 —— 段 0 全是非共享层 ✓）：
  输入 : hidden[1,1,1536] · cos_sl[1,256] · sin_sl[1,256] · cos_fu[1,512] · sin_fu[1,512]
         · per_layer_0..n-1[1,1,256]
         · 本段每层的 k/v 缓存 [1,KVMAX,D]（D 按该层 head_dim ✓）
         · kv_mask[1,1,KVMAX+1]
  输出 : hidden_out · ★只有"新 token"的 k/v [1,1,D]★
  ★设计要点★：图【不做 scatter】—— 只把新 K/V 单独输出，由主机写回缓存的第 pos 个位置 ✓
              （NPU 对 scatter/动态写不友好 ✗；主机侧改几十字节是几微秒 ✓）
              图内 attention 用 Concat(缓存, 新K) ⇒ 掩码长度 KVMAX+1 ✓
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
import g4_seg3d as G

START = int(os.environ.get("START", "0")); NO = int(os.environ.get("NO", "4"))
KVMAX = int(os.environ.get("KVMAX", "128"))
OUT = os.environ.get("OUTDIR") or ("/home/hu60/work/llm/ddk-llm/llm-poc/g4dec%d" % START)
os.makedirs(OUT, exist_ok=True)

m, tm = G.load()
w0 = G.Seg3D(tm, START, NO)          # 借它的 buffer/layer_types/dims 定义
DIMS = w0.dims
LTYPES = list(w0.layer_types)

class Dec(nn.Module):
    def __init__(self, tm, seg):
        super().__init__()
        self.tm = tm; self.seg = seg
        self.layers = seg.layers
        self.dims = seg.dims; self.ltypes = seg.layer_types
        self.rot = nn.ModuleList([getattr(seg, "rot_P_%d" % d) for d in seg.dims])
    def forward(self, hidden, cos_sl, sin_sl, cos_fu, sin_fu, kv_mask, *rest):
        n = len(self.layers)
        ples = rest[:n]
        kvs = rest[n:2 * n]                     # 每层的 (k_cache) 拼接：k0..k(n-1)
        vvs = rest[2 * n:3 * n]
        x = hidden
        outs_k, outs_v = [], []
        for i, L in enumerate(self.layers):
            A = L.self_attn
            D = self.dims[i]; lt = self.ltypes[i]; P = self.rot[i]
            cos, sin = (cos_sl, sin_sl) if lt == "sliding_attention" else (cos_fu, sin_fu)
            residual = x
            xn = L.input_layernorm(x)
            heads = A.q_proj.out_features // D
            q = A.q_proj(xn).view(1, heads, D).transpose(0, 1)          # [heads,1,D]
            q = G.rope3(G.rmsnorm(q, A.q_norm), cos, sin, P)
            k = A.k_proj(xn).view(1, 1, D).transpose(0, 1)
            k = G.rope3(G.rmsnorm(k, A.k_norm), cos, sin, P)            # [1,1,D]
            v = A.v_proj(xn).view(1, 1, D).transpose(0, 1)
            v = G.rmsnorm(v, A.v_norm)
            # ★ attention：历史缓存 + 新 token（Concat 在末尾）✓
            kk = torch.cat([kvs[i], k], dim=1)                          # [1,KVMAX+1,D]
            vv = torch.cat([vvs[i], v], dim=1)
            sc = getattr(A, "scaling", None) or (D ** -0.5)
            att = torch.matmul(q, kk.transpose(-1, -2)) * sc + kv_mask  # 无需 causal（seq=1 ✓）
            att = F.softmax(att, dim=-1)
            o = torch.matmul(att, vv).transpose(0, 1).reshape(1, 1, heads * D)
            xn = A.o_proj(o)
            x = residual + L.post_attention_layernorm(xn)
            residual = x
            xn = L.mlp(L.pre_ffeedforward_layernorm(x)) if False else L.mlp(L.pre_feedforward_layernorm(x))
            x = residual + L.post_feedforward_layernorm(xn)
            if getattr(L, "hidden_size_per_layer_input", 0):
                residual = x
                xn = L.act_fn(L.per_layer_input_gate(x)) * ples[i]
                x = residual + L.post_per_layer_input_norm(L.per_layer_projection(xn))
            x = x * L.layer_scalar
            outs_k.append(k); outs_v.append(v)
        return (x, *outs_k, *outs_v)

seg = G.Seg3D(tm, START, NO).eval()
w = Dec(tm, seg).eval()
H = tm.config.hidden_size; PLE = tm.config.hidden_size_per_layer_input
args = [torch.zeros([1, 1, H]), torch.zeros([1, 256]), torch.zeros([1, 256]),
        torch.zeros([1, 512]), torch.zeros([1, 512]), torch.zeros([1, 1, KVMAX + 1])]
for i in range(NO):
    args.append(torch.zeros([1, 1, PLE]))
for i in range(NO):
    args.append(torch.zeros([1, KVMAX, DIMS[i]]))       # k 缓存
for i in range(NO):
    args.append(torch.zeros([1, KVMAX, DIMS[i]]))       # v 缓存
names = (["hidden", "cos_sl", "sin_sl", "cos_fu", "sin_fu", "kv_mask"]
         + ["per_layer_%d" % i for i in range(NO)]
         + ["k_%d" % i for i in range(NO)] + ["v_%d" % i for i in range(NO)])
onam = ["hidden_out"] + ["k_%d_out" % i for i in range(NO)] + ["v_%d_out" % i for i in range(NO)]
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
for t in list(g.input)+list(g.output)+list(g.value_info): rec(t)
for n in g.initializer: shp[n.name] = list(n.dims)
n4 = sum(1 for k, v in shp.items() if len(v) >= 4)
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("  ★ 导出 ✓ 节点 %d · ≥4维 %d · Slice %d · %.2f GB" %
      (len(g.node), n4, ops.get("Slice",0)+ops.get("StridedSlice",0),
       sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT,f)))/1e9), flush=True)
print("  算子:", ", ".join("%s×%d" % kv for kv in sorted(ops.items(), key=lambda x:-x[1])[:10]), flush=True)
T={1:"FP32",6:"INT32",7:"INT64",10:"FP16"}; T2={1:"float32",6:"int32",7:"int64",10:"float16"}
def sh(t): return ",".join(str(x) for x in shp[t.name])
ins=[(i.name, T.get(i.type.tensor_type.elem_type,"FP32"), sh(i)) for i in g.input]
outs=[(o_.name, T.get(o_.type.tensor_type.elem_type,"FP32"), sh(o_)) for o_ in g.output]
import io
io.open(os.path.join(OUT,"omg.txt"),"w").write(
  ";".join("%s:%s"%(n,s) for n,_,s in ins)+"\n"+";".join("%s:%s"%(n,d) for n,d,_ in ins)+"\n"
  +";".join("%s:%s"%(n,d) for n,d,_ in outs)+"\n")
io.open(os.path.join(OUT,"c.cfg"),"w").write("\n".join(["[third_party_model]",
  "input_names="+";".join(n for n,_,_ in ins),"input_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in ins),
  "input_shapes="+";".join(s for _,_,s in ins),"output_names="+";".join(n for n,_,_ in outs),
  "output_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in outs),
  "output_shapes="+";".join(s for _,_,s in outs)])+"\n")
print("  输入数 %d · 输出数 %d · omg.txt/c.cfg ✓" % (len(ins), len(outs)), flush=True)
