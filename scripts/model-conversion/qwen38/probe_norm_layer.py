#!/usr/bin/env python3
"""① gated RMSNorm 单独探针 ② 整层（linear_attention_layer）探针 —— 都用真实权重 ✓ 小 S ✓"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
import npu_layers                                        # noqa: E402
from transformers import AutoModelForCausalLM            # noqa: E402

B, S = 1, 8
model = AutoModelForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32,
                                             trust_remote_code=False).eval()
layer = model.model.layers[0]
la = layer.linear_attn
heads = getattr(model.config, "num_attention_heads", 16)
kv_heads = getattr(model.config, "num_key_value_heads", 2)
hd = getattr(model.config, "head_dim", 256)
n_k = getattr(model.config, "linear_num_key_heads", heads)
n_v = getattr(model.config, "linear_num_value_heads", heads)
d_k = getattr(model.config, "linear_key_head_dim", 128)
d_v = getattr(model.config, "linear_value_head_dim", 128)
ksize = la.conv1d.weight.shape[-1]
conv_dim = 2 * n_k * d_k + n_v * d_v
print("  配置：n_k=%d n_v=%d d_k=%d d_v=%d ksize=%d conv_dim=%d" % (n_k, n_v, d_k, d_v, ksize, conv_dim))
print("  norm 类型：", type(la.norm).__name__)

class NormProbe(nn.Module):
    def __init__(s, nrm):
        super().__init__(); s.nrm = nrm
    def forward(s, core, z):
        return s.nrm(core, z) if "rmsnormgated" in type(s.nrm).__name__.lower() else s.nrm(core)

try:
    torch.onnx.export(NormProbe(la.norm).eval(),
                      (torch.randn(B, S, n_v, d_v), torch.randn(B, S, n_v * d_v)),
                      "norm.onnx", input_names=["core", "z"], output_names=["out"],
                      opset_version=14, dynamo=False)
    print("  导出 norm ✓")
except Exception as e:
    print("  导出 norm ✗", str(e)[:90])
    try:
        torch.onnx.export(NormProbe(la.norm).eval(), (torch.randn(B, S, n_v, d_v),),
                          "norm.onnx", input_names=["core"], output_names=["out"],
                          opset_version=14, dynamo=False)
        print("  导出 norm（单输入版 ✓）")
    except Exception as e2:
        print("  导出 norm 单输入也 ✗", str(e2)[:90])

class LayerProbe(nn.Module):
    def __init__(s, l):
        super().__init__(); s.l = l
    def forward(s, hidden, conv_state, rec_state):
        out, nc, nr = npu_layers.linear_attention_layer(
            s.l, hidden, conv_state, rec_state, heads, kv_heads, hd, seq=S, batch=B)
        return out, nc, nr

try:
    torch.onnx.export(LayerProbe(layer).eval(),
                      (torch.randn(B, S, model.config.hidden_size),
                       torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
                      "layer.onnx", input_names=["hidden", "conv_state", "rec_state"],
                      output_names=["out", "conv_out", "rec_out"], opset_version=14, dynamo=False)
    print("  导出 layer ✓")
    import onnx
    from collections import Counter
    g = onnx.load("layer.onnx", load_external_data=False).graph
    print("  layer 节点数 %d · 算子 %s" % (len(g.node), dict(Counter(n.node.op_type for n in g.node).most_common(10))))
except Exception as e:
    print("  导出 layer ✗", str(e)[:110])
