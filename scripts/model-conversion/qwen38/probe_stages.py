#!/usr/bin/env python3
"""分级链：从 proj 开始逐件加，导出 5 级 ⇒ 后面统一 OMG + Init。"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
import npu_layers                                                    # noqa: E402
from transformers import AutoModelForCausalLM                        # noqa: E402

B, S = 1, 64
model = AutoModelForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32,
                                             trust_remote_code=False).eval()
model.requires_grad_(False)
cm = model.config
layer = model.model.layers[0]; la = layer.linear_attn
heads = getattr(cm, "num_attention_heads", 16); kv_heads = getattr(cm, "num_key_value_heads", 2)
hd = getattr(cm, "head_dim", 256)
n_k = getattr(cm, "linear_num_key_heads", 16); n_v = getattr(cm, "linear_num_value_heads", 16)
d_k = getattr(cm, "linear_key_head_dim", 128); d_v = getattr(cm, "linear_value_head_dim", 128)
ksize = la.conv1d.weight.shape[-1]; conv_dim = 2 * n_k * d_k + n_v * d_v

STAGES = [
    ("s0_proj",        ("conv", "delta", "norm", "out")),
    ("s1_conv",        ("delta", "norm", "out")),
    ("s2_delta",       ("norm", "out")),
    ("s3_norm",        ("out",)),
    ("s4_full",        ()),
]

class Stage(nn.Module):
    def __init__(s, skip):
        super().__init__(); s.skip = skip
    def forward(s, hidden, conv_state, rec_state):
        out, nc, nr = npu_layers.linear_attention_layer(
            layer, hidden, conv_state, rec_state, heads, kv_heads, hd,
            seq=S, batch=B, skip=s.skip)
        return out

for tag, skip in STAGES:
    try:
        torch.onnx.export(Stage(skip).eval(),
                          (torch.randn(B, S, cm.hidden_size),
                           torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
                          "st_%s.onnx" % tag, input_names=["hidden", "conv_state", "rec_state"],
                          output_names=["out"], opset_version=14, dynamo=False)
        print("  导出 %-10s ✓ skip=%s" % (tag, list(skip)))
    except Exception as e:
        print("  导出 %-10s ✗ %s" % (tag, str(e)[:80]))
