"""★可移植版：模型用 MODEL_DIR / 输出用 OUTDIR 指定 ✓（原脚本写死本机路径 ✗）★
lm_head 分块图：hidden[1,seq,1536] → logits_j[1,seq,CHUNK]。
理由：整词表输出 [1,seq,262144] = 1 MB 正好顶到设备"单张量 <1MB"上限 ✗ ⇒ 切 4 块 ✓
lm_head 与 embed_tokens 是 tied 的 ✓ ⇒ 直接用 embed_tokens^T 的分块 ✓
"""
import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 可移植 ✓
import g4_seg3d as G

DTYPE = os.environ.get("DTYPE", "fp32")
_TD = __import__("torch").float16 if DTYPE == "fp16" else __import__("torch").float32
SEQ = int(os.environ.get("SEQ", "4")); J = int(os.environ.get("J", "0")); CH = 65536
m, tm = G.load()
W = tm.embed_tokens.weight.detach()          # [262144,1536] ✓ tied
WJ = W[J*CH:(J+1)*CH].t().contiguous()       # [1536, CH] ✓
OUT = os.environ.get("OUTDIR") or (os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "g4seg") % J)
os.makedirs(OUT, exist_ok=True)

class L(nn.Module):
    def __init__(self, W): super().__init__(); self.W = nn.Parameter(W, requires_grad=False)
    def forward(self, h): return torch.matmul(h, self.W)
w = L(WJ).eval()
if DTYPE == "fp16":
    w = w.half()                    # ★ 权重转 fp16 ✓
    print("  lm 权重已转 fp16", flush=True)
h = torch.zeros([1, SEQ, 1536], dtype=_TD)   # ★ I/O 跟着 DTYPE ✓
with torch.no_grad(): print("  前向 ✓", tuple(w(h).shape), flush=True)
p = os.path.join(OUT, "lm.onnx")
torch.onnx.export(w, (h,), p, input_names=["hidden"], output_names=["logits"],
                  opset_version=14, do_constant_folding=True, dynamo=False)
import onnx, io
from onnx import shape_inference
mm = shape_inference.infer_shapes(onnx.load(p, load_external_data=False), strict_mode=False)
g = mm.graph
shp = {}
def rec(t):
    tt = t.type.tensor_type
    if tt.HasField("shape"):
        shp[t.name] = [(d.dim_value if d.HasField("dim_value") else "?") for d in tt.shape.dim]
for t in list(g.input)+list(g.output)+list(g.value_info): rec(t)
def sh(t): return ",".join(str(x) for x in shp[t.name])
print("  ★ 块 %d 导出 ✓ 节点 %d · 输出 %s" % (J, len(g.node), sh(g.output[0])), flush=True)
# ★ 从 ONNX 【实读】dtype 与形状 ✓ —— 写死 FP32 会让 fp16 的图在 OMG 报
#   "InputOutputGraphComplete fail" ✗（实测：图是 FP16、omg.txt 写 FP32 ⇒ 直接失败 ✓）
T = {1:"FP32", 6:"INT32", 7:"INT64", 10:"FP16", 11:"FP64", 9:"BOOL"}
T2={"FP32":"float32","FP16":"float16","INT32":"int32","INT64":"int64","FP64":"float64","BOOL":"bool"}  # ★按【名字】索引，与 T 一致 ✓（原来按数值索引 ⇒ .get("FP16") 永远回落 float32 ✗）
ins  = [(i.name, T.get(i.type.tensor_type.elem_type, "FP32"), sh(i)) for i in g.input]
outs = [(o.name, T.get(o.type.tensor_type.elem_type, "FP32"), sh(o)) for o in g.output]
io.open(os.path.join(OUT,"omg.txt"),"w").write(
    ";".join("%s:%s" % (n, s2) for n, _, s2 in ins) + "\n"
    + ";".join("%s:%s" % (n, d) for n, d, _ in ins) + "\n"
    + ";".join("%s:%s" % (n, d) for n, d, _ in outs) + "\n")
io.open(os.path.join(OUT,"c.cfg"),"w").write("\n".join(["[third_party_model]",
  "input_names=" + ";".join(n for n, _, _ in ins),
  "input_dtypes=" + ";".join(T2.get(d, "float32") for _, d, _ in ins),
  "input_shapes=" + ";".join(s2 for _, _, s2 in ins),
  "output_names=" + ";".join(n for n, _, _ in outs),
  "output_dtypes=" + ";".join(T2.get(d, "float32") for _, d, _ in outs),
  "output_shapes=" + ";".join(s2 for _, _, s2 in outs)]) + "\n")
print("  目录 %.0f MB" % (sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT,f)))/1e6), flush=True)
