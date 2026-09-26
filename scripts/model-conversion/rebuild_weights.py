"""Rebuild the exported ONNX's weights from the original HuggingFace checkpoint.

The dopt quantization step wrote weights whose negative half was clamped to zero
(ONNX weights have 0.00% negative values; corr(ONNX, ReLU(HF)) ~ 0.98).  That
destroys half the information and makes the model output garbage.

This script restores every weight from the HF checkpoint:
  * named initializers   model.model.layers.N.X        -> model.layers.N.X
  * MatMul node weights  node "model.layers.N.X"       -> model.layers.N.X.weight (transposed)
  * lm_head              node "lm_head"                -> lm_head.weight (transposed, tied)

Weights stay unquantized (the model is converted with --weight_data_type FP16
anyway), which removes the broken fake-quant step from the equation entirely.

usage: rebuild_weights.py <in.onnx> <out.onnx> [hf_model_dir]
       (hf_model_dir 也可用环境变量 HF_MODEL 指定)
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper as NH
from safetensors.torch import safe_open as st_open

if len(sys.argv) < 3:
    sys.exit(__doc__ + "\n用法: rebuild_weights.py <in.onnx> <out.onnx> [hf_model_dir]")

SRC = sys.argv[1]
DST = sys.argv[2]
HF = (sys.argv[3] if len(sys.argv) > 3
      else os.environ.get("HF_MODEL", "/path/to/Qwen2.5-1.5B-Instruct"))

m = onnx.load(SRC, load_external_data=False)
g = m.graph
init_by_name = {i.name: i for i in g.initializer}

print("loading HF checkpoint index ...", flush=True)
f = st_open(HF + "/model.safetensors", framework="pt")
hf_keys = set(f.keys())


def hf_get(name):
    return f.get_tensor(name).float().numpy()


def set_init(iname, arr, tag):
    """Replace initializer `iname` in place with `arr` (float32)."""
    new_t = NH.from_array(np.ascontiguousarray(arr.astype(np.float32)), iname)
    for idx, i in enumerate(g.initializer):
        if i.name == iname:
            g.initializer[idx].CopyFrom(new_t)
            return True
    print(f"   !! initializer {iname} not found for {tag}")
    return False


replaced = 0
skipped = []

# ---------- 1. named initializers ----------
for ini in list(g.initializer):
    n = ini.name
    if n.startswith("onnx::"):
        continue
    cand = None
    if n in hf_keys:
        cand = n
    elif n.startswith("model.model.") and n.replace("model.model.", "model.", 1) in hf_keys:
        cand = n.replace("model.model.", "model.", 1)
    if cand is None:
        skipped.append(n)
        continue
    arr = hf_get(cand)
    if list(arr.shape) != list(ini.dims):
        skipped.append(n + " (shape)")
        continue
    set_init(n, arr, cand)
    replaced += 1

print(f"named initializers replaced: {replaced}   skipped: {len(skipped)}")
if skipped[:5]:
    print("  e.g. skipped:", skipped[:5])

# ---------- 2. MatMul node weights ----------
mm_replaced = 0
mm_missing = []
for node in g.node:
    if node.op_type != "MatMul" or len(node.input) < 2:
        continue
    wname = node.input[1]
    nname = node.name
    key = None
    if nname == "lm_head":
        key = "lm_head.weight"
    elif nname.startswith("model.layers."):
        key = nname + ".weight"
    if key == "lm_head.weight" and key not in hf_keys:
        key = "model.embed_tokens.weight"      # tied embeddings
    # split chunks produced by split_downproj_fixed.py:  ...down_proj_k<N>
    chunk_idx = None
    if nname.startswith("model.layers.") and "_k" in nname.rsplit(".", 1)[-1]:
        base, _, tail = nname.rpartition("_k")
        if tail.isdigit():
            key = base + ".weight"
            chunk_idx = int(tail)
    if key is None or key not in hf_keys:
        if nname.startswith("model.layers.") or nname == "lm_head":
            mm_missing.append((nname, key))
        continue
    W = hf_get(key)                      # [out, in]
    Wt = np.ascontiguousarray(W.T)       # MatMul wants [in, out]
    if chunk_idx is not None:
        # split_downproj_fixed.py cut Wt along K (axis 0) into chunks of CHUNK
        CHUNK = 4480
        K = Wt.shape[0]
        parts = [(i, min(i + CHUNK, K)) for i in range(0, K, CHUNK)]
        if chunk_idx >= len(parts):
            mm_missing.append((nname, "chunk index out of range"))
            continue
        a, b = parts[chunk_idx]
        Wt = np.ascontiguousarray(Wt[a:b])
    old = NH.to_array(init_by_name[wname])
    if list(old.shape) != list(Wt.shape):
        mm_missing.append((nname, f"shape {old.shape} vs {Wt.shape}"))
        continue
    set_init(wname, Wt, key)
    mm_replaced += 1

print(f"MatMul weights replaced: {mm_replaced}   problems: {len(mm_missing)}")
for a, b in mm_missing[:6]:
    print("   ", a, b)

print("saving ...", flush=True)
onnx.save(m, DST, save_as_external_data=True, all_tensors_to_one_file=True,
          location=DST.split("/")[-1][:-5] + ".pb", size_threshold=1024)
print("saved", DST)
