"""把 3 维版 Gemma 段导出为 ONNX（prefill、无 KV），并立刻跑 OMG。"""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
from transformers import AutoModelForCausalLM
import importlib

M   = "/home/hu60/work/llm/ddk-llm/models/gemma-4-E2B-it"
OUT = "/home/hu60/work/llm/ddk-llm/llm-poc/g4seg3d"
START, NO, SEQ = 0, 4, 4
os.makedirs(OUT, exist_ok=True)

# 直接复用 g4_seg3d.py 里的 Seg3D / rmsnorm / rope3（导入会顺带跑一遍验证，可接受）
import g4_seg3d as G
tm, Seg3D = G.tm, G.Seg3D

w = Seg3D(tm, START, NO).eval()
class W(nn.Module):
    def __init__(self, w): super().__init__(); self.w = w
    def forward(self, hidden, cos, sin, mask3, *ples):
        return self.w(hidden, cos, sin, mask3, *ples)
ww = W(w).eval()
H = tm.config.hidden_size; D = tm.layers[0].self_attn.head_dim
hidden = torch.zeros([1, SEQ, H]); cos = torch.zeros([SEQ, D]); sin = torch.zeros([SEQ, D])
mask3 = torch.zeros([1, SEQ, SEQ])
ples = [torch.zeros([1, SEQ, 256]) for _ in range(NO)]
with torch.no_grad():
    print("  前向 ✓", tuple(ww(hidden, cos, sin, mask3, *ples).shape), flush=True)

p = os.path.join(OUT, "seg3d.onnx")
torch.onnx.export(ww, (hidden, cos, sin, mask3, *ples), p,
                  input_names=["hidden", "cos", "sin", "mask3"] + ["per_layer_%d" % i for i in range(NO)],
                  output_names=["hidden_out"], opset_version=14,
                  do_constant_folding=True, dynamo=False)
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
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
n4 = sum(1 for k, v in shp.items() if len(v) >= 4)
print("  ★ 导出 ✓ 目录 %.2f GB · 节点 %d · ★≥4维张量 %d 个★" %
      (sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))/1e9, len(g.node), n4), flush=True)
print("  算子:", ", ".join("%s×%d" % kv for kv in sorted(ops.items(), key=lambda x: -x[1])[:14]), flush=True)
T = {1:"FP32",6:"INT32",7:"INT64",9:"BOOL",10:"FP16"}
T2 = {1:"float32",6:"int32",7:"int64",9:"bool",10:"float16"}
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
print("  输入:", "; ".join("%s[%s]" % (n, s) for n, _, s in ins), flush=True)
