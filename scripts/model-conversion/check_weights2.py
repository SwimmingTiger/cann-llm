"""Check that the exported ONNX weights match the original HF checkpoint.

MatMul stores its weight as [in_features, out_features]; HuggingFace stores
Linear weights as [out_features, in_features].  So we try both orientations and
report the best-matching one (a fake-quantized copy should score corr > 0.99).
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper as NH
from safetensors.torch import safe_open as st_open

ONNX = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONNX_MODEL", "qwen_28l_split.onnx")
HF = os.environ.get("HF_MODEL", "/path/to/Qwen2.5-1.5B-Instruct")

m = onnx.load(ONNX, load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}

matmuls = {}
for n in g.node:
    if n.op_type == "MatMul" and len(n.input) >= 2:
        matmuls[n.name] = (n.input[0], n.input[1])

probes = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.13.mlp.up_proj",
    "model.layers.27.mlp.gate_proj",
]

rng = np.random.default_rng(0)


def corr(a, b, n=1_000_000):
    idx = rng.choice(a.size, size=min(n, a.size), replace=False)
    aa, bb = a[idx], b[idx]
    return float(np.corrcoef(aa, bb)[0, 1]), float(np.abs(aa - bb).mean() / (np.abs(bb).mean() + 1e-9))


with st_open(HF + "/model.safetensors", framework="pt") as f:
    for p in probes:
        key = p + ".weight"
        if p not in matmuls:
            print(f"  {p:40s} ONNX 无此节点")
            continue
        if key not in f.keys():
            print(f"  {p:40s} HF 无 {key}")
            continue
        a_name, b_name = matmuls[p]
        W_onnx = NH.to_array(inits[b_name]).astype(np.float32)
        W_hf = f.get_tensor(key).float().numpy()
        print(f"  {p}")
        print(f"      ONNX 权重 {b_name} dims={W_onnx.shape}   (MatMul 左输入 {a_name})")
        print(f"      HF   shape={W_hf.shape}")
        best = None
        for label, cand in (("as-is", W_onnx), ("transposed", W_onnx.T)):
            if cand.shape != W_hf.shape:
                continue
            c, r = corr(cand.ravel(), W_hf.ravel())
            print(f"        {label:11s} corr={c:+.5f}  relerr={r:.5f}")
            if best is None or c > best[0]:
                best = (c, r, label)
        if best is None:
            print("        形状无法对齐！")
        else:
            flag = "OK" if best[0] > 0.99 else "★不匹配★"
            print(f"        -> 最佳 {best[2]} corr={best[0]:+.5f}  [{flag}]")
