"""Gemma 4 分段导出（统一接口，含 KV 共享槽）。

图接口（所有段一致 ✓）：
  输入 : hidden[1,seq,1536] · mask3[1,seq,seq] · cos_sl/sin_sl[seq,256] · cos_fu/sin_fu[seq,512]
         · sk/sv[1,seq,256]（sliding 共享槽）· fk/fv[1,seq,512]（full 共享槽）
         · per_layer_0..n-1[1,seq,256]
  输出 : hidden_out · sk/sv/fk/fv（更新后的共享槽）
说明：
  · cos/sin 只依赖 position_ids 与 layer_type（与 hidden 值无关 ✓）⇒ 全链只需 2 组 ✓
  · 不写槽的段把输入槽原样传出（pass-through ✓）⇒ 9 段可以统一串起来 ✓
  · 末尾的 norm / lm_head 不做（词表 262144×4B = 1MB 顶到设备上限 ✗ ⇒ 主机侧做 ✓）
"""
import os, sys, torch, torch.nn as nn
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
import g4_seg3d as G

START = int(os.environ.get("START", "0"))
NO    = int(os.environ.get("NO", "4"))
SEQ   = int(os.environ.get("SEQ", "4"))
OUT   = "/home/hu60/work/llm/ddk-llm/llm-poc/g4seg%d" % START
os.makedirs(OUT, exist_ok=True)

tm = G.tm if hasattr(G, "tm") else None
if tm is None:
    _, tm = G.load()
NTOT = len(tm.layers)
no = min(NO, NTOT - START)
w = G.Seg3D(tm, START, no).eval()
print("  段 [%d:%d] · head_dim=%s" % (START, START + no, w.dims), flush=True)

class W(nn.Module):
    def __init__(self, w, n): super().__init__(); self.w = w; self.n = n
    def forward(self, hidden, mask3, cos_sl, sin_sl, cos_fu, sin_fu, sk, sv, fk, fv, *ples):
        ln = list(self.w.layer_types)
        args = []
        for i in range(self.n):
            lt = ln[i]
            args += [cos_sl, sin_sl] if lt == "sliding_attention" else [cos_fu, sin_fu]
        args += list(ples)
        out = self.w(hidden, mask3, sk, sv, fk, fv, *args)
        return out            # (x, sk, sv, fk, fv)

H, PLE = tm.config.hidden_size, tm.config.hidden_size_per_layer_input
ww = W(w, no).eval()
hidden = torch.zeros([1, SEQ, H]); mask3 = torch.zeros([1, SEQ, SEQ])
cos_sl = torch.zeros([SEQ, 256]); sin_sl = torch.zeros([SEQ, 256])
cos_fu = torch.zeros([SEQ, 512]); sin_fu = torch.zeros([SEQ, 512])
sk = torch.zeros([1, SEQ, 256]); sv = torch.zeros([1, SEQ, 256])
fk = torch.zeros([1, SEQ, 512]); fv = torch.zeros([1, SEQ, 512])
ples = [torch.zeros([1, SEQ, PLE]) for _ in range(no)]
names = ["hidden", "mask3", "cos_sl", "sin_sl", "cos_fu", "sin_fu", "sk", "sv", "fk", "fv"]
onam  = ["hidden_out", "sk_out", "sv_out", "fk_out", "fv_out"]
with torch.no_grad():
    o = ww(hidden, mask3, cos_sl, sin_sl, cos_fu, sin_fu, sk, sv, fk, fv, *ples)
print("  前向 ✓ 输出 %s" % str([tuple(t.shape) for t in o]), flush=True)

p = os.path.join(OUT, "seg.onnx")
torch.onnx.export(ww, (hidden, mask3, cos_sl, sin_sl, cos_fu, sin_fu, sk, sv, fk, fv, *ples), p,
                  input_names=names + ["per_layer_%d" % i for i in range(no)],
                  output_names=onam, opset_version=14, do_constant_folding=True, dynamo=False)
import onnx
from onnx import shape_inference
mm = shape_inference.infer_shapes(onnx.load(p, load_external_data=False), strict_mode=False)
g = mm.graph
shp = {}
def rec(t):
    tt = t.type.tensor_type
    if tt.HasField("shape"):
        shp[t.name] = [(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")) for d in tt.shape.dim]
for t in list(g.input)+list(g.output)+list(g.value_info): rec(t)
for n in g.initializer: shp[n.name] = list(n.dims)
n4 = sum(1 for k, v in shp.items() if len(v) >= 4)
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("  ★ 导出 ✓ 节点 %d · ≥4维 %d · Slice %d · %.2f GB" %
      (len(g.node), n4, ops.get("Slice",0)+ops.get("StridedSlice",0),
       sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT))/1e9), flush=True)
print("  算子:", ", ".join("%s×%d" % kv for kv in sorted(ops.items(), key=lambda x:-x[1])[:12]), flush=True)
T = {1:"FP32",6:"INT32",7:"INT64",9:"BOOL",10:"FP16"}; T2 = {1:"float32",6:"int32",7:"int64",9:"bool",10:"float16"}
def sh(t): return ",".join(str(x) for x in shp[t.name])
ins = [(i.name, T.get(i.type.tensor_type.elem_type,"FP32"), sh(i)) for i in g.input]
outs = [(o.name, T.get(o.type.tensor_type.elem_type,"FP32"), sh(o)) for o in g.output]
import io
io.open(os.path.join(OUT,"omg.txt"),"w").write(
  ";".join("%s:%s"%(n,s) for n,_,s in ins)+"\n"+";".join("%s:%s"%(n,d) for n,d,_ in ins)+"\n"
  +";".join("%s:%s"%(n,d) for n,d,_ in outs)+"\n")
io.open(os.path.join(OUT,"c.cfg"),"w").write("\n".join(["[third_party_model]",
  "input_names="+";".join(n for n,_,_ in ins),"input_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in ins),
  "input_shapes="+";".join(s for _,_,s in ins),"output_names="+";".join(n for n,_,_ in outs),
  "output_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in outs),
  "output_shapes="+";".join(s for _,_,s in outs)])+"\n")
print("  omg.txt / c.cfg 已写好 ✓", flush=True)
