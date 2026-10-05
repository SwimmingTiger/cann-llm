#!/usr/bin/env python3
"""把 gated RMSNorm 换成恒等 ⇒ 用【真函数】linear_attention_layer 导出 ⇒ 看 OMG 是否通过。"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
import npu_layers                                                    # noqa: E402
from transformers import AutoModelForCausalLM                        # noqa: E402

B, S = 1, 64          # ★与真实 prefill 图同形✓★
model = AutoModelForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32,
                                             trust_remote_code=False).eval()
model.requires_grad_(False)
cm = model.config
layer = model.model.layers[0]
la = layer.linear_attn
heads = getattr(cm, "num_attention_heads", 16); kv_heads = getattr(cm, "num_key_value_heads", 2)
hd = getattr(cm, "head_dim", 256)
n_k = getattr(cm, "linear_num_key_heads", 16); n_v = getattr(cm, "linear_num_value_heads", 16)
d_k = getattr(cm, "linear_key_head_dim", 128); d_v = getattr(cm, "linear_value_head_dim", 128)
ksize = la.conv1d.weight.shape[-1]; conv_dim = 2 * n_k * d_k + n_v * d_v

class Identity(nn.Module):
    """吃掉 gate 参数 ✓ 只做恒等（形状不变 ✓）。"""
    def forward(self, x, *a, **kw):
        return x

class LayerProbe(nn.Module):
    def __init__(s, l):
        super().__init__(); s.l = l
    def forward(s, hidden, conv_state, rec_state):
        out, nc, nr = npu_layers.linear_attention_layer(
            s.l, hidden, conv_state, rec_state, heads, kv_heads, hd, seq=S, batch=B)
        return out

orig_norm = la.norm
for tag, nrm in (("withnorm", orig_norm), ("nonorm", Identity())):
    la.norm = nrm
    try:
        torch.onnx.export(LayerProbe(layer).eval(),
                          (torch.randn(B, S, cm.hidden_size),
                           torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
                          "nl_%s.onnx" % tag, input_names=["hidden", "conv_state", "rec_state"],
                          output_names=["out"], opset_version=14, dynamo=False)
        print("  导出 nl_%-9s ✓" % tag)
    except Exception as e:
        print("  导出 nl_%-9s ✗ %s" % (tag, str(e)[:90]))
la.norm = orig_norm
