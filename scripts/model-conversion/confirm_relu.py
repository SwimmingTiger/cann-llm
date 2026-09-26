import os
import sys

import numpy as np, onnx
from onnx import numpy_helper as NH
from safetensors.torch import safe_open as st_open

ONNX = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONNX_MODEL", "model.onnx")
HF = os.environ.get("HF_MODEL", "/path/to/Qwen2.5-1.5B-Instruct")
m = onnx.load(ONNX, load_external_data=False)
inits = {i.name: i for i in m.graph.initializer}
matmuls = {n.name: n.input[1] for n in m.graph.node if n.op_type == "MatMul" and len(n.input) >= 2}
rng = np.random.default_rng(1)

def corr(a, b, n=1_500_000):
    i = rng.choice(a.size, size=min(n, a.size), replace=False)
    return float(np.corrcoef(a[i], b[i])[0, 1])

with st_open(HF + "/model.safetensors", framework="pt") as f:
    for p in ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj",
              "model.layers.13.mlp.up_proj", "model.layers.27.self_attn.o_proj"]:
        W_onnx = NH.to_array(inits[matmuls[p]]).astype(np.float32).T   # [out, in]
        W_hf = f.get_tensor(p + ".weight").float().numpy()
        W_onnx = W_onnx.ravel(); W_hf = W_hf.ravel()
        neg_frac = float((W_hf < 0).mean())
        print(f"{p}")
        print(f"   HF 中负值占比: {neg_frac*100:.1f}%    ONNX 中负值占比: {(W_onnx<0).mean()*100:.2f}%")
        print(f"   corr(ONNX, HF)          = {corr(W_onnx, W_hf):+.5f}")
        print(f"   corr(ONNX, ReLU(HF))    = {corr(W_onnx, np.maximum(W_hf,0)):+.5f}   <== 若接近 1 则确认被 ReLU")
        print(f"   corr(ONNX, |HF|)        = {corr(W_onnx, np.abs(W_hf)):+.5f}")
        print()
