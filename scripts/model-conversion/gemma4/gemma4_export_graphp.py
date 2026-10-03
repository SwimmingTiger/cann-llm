"""★可移植版：模型用 MODEL_DIR / 输出用 OUTDIR 指定 ✓（原脚本写死本机路径 ✗）★
图 P：per_layer 前处理（把 HF 的 project_per_layer_inputs 原样搬进图）。

输入：input_ids[1,seq] int32 · identity[1,seq,8960]（主机侧 mmap 查表得到的 token-identity ✓）
输出：per_layer[1,seq,35,256]（每段再切自己那几块 ✓）
理由：per_layer_model_projection 是 [8960,1536] 的矩阵乘 ⇒ 纯 Python 单 token 要几十秒 ✗
      ⇒ 必须放 NPU；而查表（mmap 取行）在 Python 里只要几十微秒 ⇒ 留在主机 ✓
"""
import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 可移植 ✓
import gemma4_model as G

DTYPE = os.environ.get("DTYPE", "fp32")          # ★ fp32 / fp16 ✓
_TD = __import__("torch").float16 if DTYPE == "fp16" else __import__("torch").float32
SEQ = int(os.environ.get("SEQ", "4"))
OUT = os.environ.get("OUTDIR") or os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "g4seg"); os.makedirs(OUT, exist_ok=True)
m, tm = G.load()

class P(nn.Module):
    def __init__(self, tm): super().__init__(); self.tm = tm
    def forward(self, input_ids, identity):
        e = self.tm.embed_tokens(input_ids)                       # 已含 embed_scale ✓
        # ★ identity 是 [1,SEQ,8960] = 35 层 × 256 维的拼接 ⇒ 必须 reshape 再喂 HF ✓
        #   （HF 内部期望 [batch, seq, num_layers, ple_dim]；少了这步会报 256 vs 8960 ✗）
        n_layers = self.tm.config.num_hidden_layers
        ple_dim = identity.shape[-1] // n_layers
        pl = self.tm.project_per_layer_inputs(e, identity.view(1, identity.shape[1], n_layers, ple_dim))
        return pl

w = P(tm).eval()
if DTYPE == "fp16":
    w = w.half()          # ★ 显式转，别再用正则 ✗
    print("  图P 权重已转 fp16", flush=True)
ids = torch.tensor([[1, 42, 777, 9000][:SEQ]], dtype=torch.int32)
ident = torch.zeros([1, SEQ, 8960], dtype=_TD)   # ★ I/O 跟着 DTYPE ✓
with torch.no_grad():
    o = w(ids, ident)
print("  前向 ✓ 输出", tuple(o.shape), flush=True)
p = os.path.join(OUT, "graphP.onnx")
torch.onnx.export(w, (ids, ident), p, input_names=["input_ids", "identity"],
                  output_names=["per_layer"], opset_version=14,
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
n4 = sum(1 for k,v in shp.items() if len(v) >= 4)
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("  ★ 导出 ✓ 节点 %d · ≥4维 %d · 算子: %s" %
      (len(g.node), n4, ", ".join("%s×%d" % kv for kv in sorted(ops.items(), key=lambda x:-x[1])[:8])), flush=True)
print("  目录 %.2f GB" % (sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT,f)))/1e9), flush=True)
T={1:"FP32",6:"INT32",7:"INT64",10:"FP16"}; T2={"FP32":"float32","FP16":"float16","INT32":"int32","INT64":"int64","FP64":"float64","BOOL":"bool"}  # ★按【名字】索引，与 T 一致 ✓（原来按数值索引 ⇒ .get("FP16") 永远回落 float32 ✗）
def sh(t): return ",".join(str(x) for x in shp[t.name])
ins=[(i.name, T.get(i.type.tensor_type.elem_type,"FP32"), sh(i)) for i in g.input]
outs=[(o.name, T.get(o.type.tensor_type.elem_type,"FP32"), sh(o)) for o in g.output]
import io
io.open(os.path.join(OUT,"omg.txt"),"w").write(
  ";".join("%s:%s"%(n,s) for n,_,s in ins)+"\n"+";".join("%s:%s"%(n,d) for n,d,_ in ins)+"\n"
  +";".join("%s:%s"%(n,d) for n,d,_ in outs)+"\n")
io.open(os.path.join(OUT,"c.cfg"),"w").write("\n".join(["[third_party_model]",
  "input_names="+";".join(n for n,_,_ in ins),"input_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in ins),
  "input_shapes="+";".join(s for _,_,s in ins),"output_names="+";".join(n for n,_,_ in outs),
  "output_dtypes="+";".join(T2.get(d,"float32") for _,d,_ in outs),
  "output_shapes="+";".join(s for _,_,s in outs)])+"\n")
print("  omg.txt / c.cfg ✓", flush=True)
