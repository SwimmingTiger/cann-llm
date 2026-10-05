#!/usr/bin/env python3
"""串联的微变体 ⇒ 试出"串联为何让 Init 失败"。
  ident : o0 → (o0*1.0) → layer1       （中间加恒等 ✓）
  extra : 串联，但 o0 也作为输出        （多一个输出引用 ✓）
  liveh : 串联，但 layer1 的输入 = o0 + 0*h （保持对 h 的引用 ✓）
"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
import npu_layers                                                    # noqa: E402
from transformers import AutoModelForCausalLM                        # noqa: E402
B, S = 1, 64
model = AutoModelForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32,
                                             trust_remote_code=False).eval()
model.requires_grad_(False)
cm = model.config; L = model.model.layers
heads = getattr(cm, "num_attention_heads", 16); kv_heads = getattr(cm, "num_key_value_heads", 2)
hd = getattr(cm, "head_dim", 256)
n_k = getattr(cm, "linear_num_key_heads", 16); n_v = getattr(cm, "linear_num_value_heads", 16)
d_k = getattr(cm, "linear_key_head_dim", 128); d_v = getattr(cm, "linear_value_head_dim", 128)
ksize = L[0].linear_attn.conv1d.weight.shape[-1]; conv_dim = 2 * n_k * d_k + n_v * d_v
H = cm.hidden_size
def call(layer, h, c, r):
    return npu_layers.linear_attention_layer(layer, h, c, r, heads, kv_heads, hd, seq=S, batch=B)

class Ident(nn.Module):
    def forward(s, h, c0, r0, c1, r1):
        o0, nc0, nr0 = call(L[0], h, c0, r0)
        o1, nc1, nr1 = call(L[1], o0 * 1.0, c1, r1)
        return o1, nc0, nr0, nc1, nr1

class Extra(nn.Module):
    def forward(s, h, c0, r0, c1, r1):
        o0, nc0, nr0 = call(L[0], h, c0, r0)
        o1, nc1, nr1 = call(L[1], o0, c1, r1)
        return o1, o0, nc0, nr0, nc1, nr1

class LiveH(nn.Module):
    def forward(s, h, c0, r0, c1, r1):
        o0, nc0, nr0 = call(L[0], h, c0, r0)
        o1, nc1, nr1 = call(L[1], o0 + h * 0.0, c1, r1)
        return o1, nc0, nr0, nc1, nr1

INS = ["hidden", "c0", "r0", "c1", "r1"]
CASES = [("ident", Ident(), ["out", "nc0", "nr0", "nc1", "nr1"]),
         ("extra", Extra(), ["out", "o0", "nc0", "nr0", "nc1", "nr1"]),
         ("liveh", LiveH(), ["out", "nc0", "nr0", "nc1", "nr1"])]
for tag, mod, outs in CASES:
    try:
        torch.onnx.export(mod.eval(),
                          (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v),
                           torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
                          "c_%s.onnx" % tag, input_names=INS, output_names=outs,
                          opset_version=14, dynamo=False)
        print("  导出 %-6s ✓" % tag)
    except Exception as e:
        print("  导出 %-6s ✗ %s" % (tag, str(e)[:70]))
