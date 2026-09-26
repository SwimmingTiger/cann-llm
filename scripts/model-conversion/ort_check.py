"""Decisive numerical check: run the exported CANN LLM ONNX in ONNXRuntime and
see whether the model itself produces the right next token.

If ORT gives ' Paris' (token 12095) for "The capital of France is", the ONNX +
weights are numerically correct and any garbage on the NPU comes from the
OMG/engine side.  If ORT is also garbage, the exported graph itself is wrong.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ONNX_MODEL", "qwen_28l_split.onnx")
EMB = os.environ.get("EMB_WEIGHTS", "model_64_2048.embedding_weights")
SCL = os.environ.get("EMB_SCALES", "model_64_2048.embedding_dequant_scale")

S = 64          # prefill slots
HID = 1536
KV = 2048
TOKENS = [785, 6722, 315, 9625, 374]     # "The capital of France is"
NEG = -3.4028234663852886e38

emb = np.memmap(EMB, dtype=np.int8, mode="r").reshape(-1, HID)
scl = np.fromfile(SCL, dtype=np.float32)

input_embed = np.zeros((1, S, HID), dtype=np.int8)
embed_scales = np.zeros((1, S, 1), dtype=np.float32)
position_ids = np.zeros((1, S), dtype=np.int32)
new_kv = np.arange(S, dtype=np.int32)
mask = np.full((1, 1, S, KV), NEG, dtype=np.float32)

n = len(TOKENS)
for i, t in enumerate(TOKENS):
    input_embed[0, i, :] = emb[t]
    embed_scales[0, i, 0] = scl[t]
    position_ids[0, i] = i
    mask[0, 0, i, : i + 1] = 0.0
# padded query rows: let them attend to the real tokens so they don't poison the graph
for q in range(n, S):
    mask[0, 0, q, :n] = 0.0

feeds = {
    "input_embed": input_embed,
    "attention_mask": mask,
    "position_ids": position_ids,
    "new_kv_cache_pos": new_kv,
    "embed_scales": embed_scales,
}
for i in range(28):
    z = np.zeros((KV, 2, 1, 128), dtype=np.float32)
    feeds["past_key_in%d" % i] = z
    feeds["past_value_in%d" % i] = z.copy()

print("loading model:", MODEL, flush=True)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
print("running (this loads ~6 GB of weights, be patient) ...", flush=True)
out = sess.run(["lm_logits"], feeds)[0]
print("lm_logits shape:", out.shape, "dtype:", out.dtype)

row = out[0, n - 1, :]          # logits of the LAST real prompt token
top = np.argsort(row)[::-1][:8]
print()
print("last-prompt-token top-8:")
for t in top:
    print(f"   id={int(t):>7}  logit={row[t]:.4f}")
print()
print("argmax =", int(top[0]), "  (expected 12095 = ' Paris' for a correct model)")

# also show the first padded row for reference
row0 = out[0, 0, :]
print("row0 argmax =", int(np.argmax(row0)))
