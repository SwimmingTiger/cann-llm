#!/usr/bin/env python3
"""两层堆叠探针：直接串联两层 linear_attention_layer（★不用整模型包装✓★）⇒ 定位堆叠 vs 包装。"""
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
L = model.model.layers
heads = getattr(cm, "num_attention_heads", 16); kv_heads = getattr(cm, "num_key_value_heads", 2)
hd = getattr(cm, "head_dim", 256)
n_k = getattr(cm, "linear_num_key_heads", 16); n_v = getattr(cm, "linear_num_value_heads", 16)
d_k = getattr(cm, "linear_key_head_dim", 128); d_v = getattr(cm, "linear_value_head_dim", 128)
ksize = L[0].linear_attn.conv1d.weight.shape[-1]; conv_dim = 2 * n_k * d_k + n_v * d_v
H = cm.hidden_size
print("  层 0/1 类型：%s / %s" % (type(L[0].linear_attn).__name__, type(L[1].linear_attn).__name__))

class Two(nn.Module):
    def forward(s, h, c0, r0, c1, r1):
        o0, nc0, nr0 = npu_layers.linear_attention_layer(L[0], h, c0, r0, heads, kv_heads, hd, seq=S, batch=B)
        o1, nc1, nr1 = npu_layers.linear_attention_layer(L[1], o0, c1, r1, heads, kv_heads, hd, seq=S, batch=B)
        return o1, nc0, nr0, nc1, nr1

try:
    torch.onnx.export(Two().eval(),
                      (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v),
                       torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
                      "two.onnx", input_names=["hidden", "c0", "r0", "c1", "r1"],
                      output_names=["out", "nc0", "nr0", "nc1", "nr1"], opset_version=14, dynamo=False)
    print("  导出 two ✓")
except Exception as e:
    print("  导出 two ✗ %s" % str(e)[:100])
