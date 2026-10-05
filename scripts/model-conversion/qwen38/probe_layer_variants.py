#!/usr/bin/env python3
"""整层探针的【逐件拆除】变体 ⇒ 找出引入不支持的 Sub / 激活的那一件。"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
from npu_gated_delta import npu_chunk_gated_delta_rule            # noqa: E402
from transformers import AutoModelForCausalLM                     # noqa: E402

B, S = 1, 8
model = AutoModelForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32,
                                             trust_remote_code=False).eval()
model.requires_grad_(False)          # ★避免 "Cannot insert a Tensor that requires grad" ✗★
la = model.model.layers[0].linear_attn
cm = model.config
n_k = getattr(cm, "linear_num_key_heads", 16); n_v = getattr(cm, "linear_num_value_heads", 16)
d_k = getattr(cm, "linear_key_head_dim", 128); d_v = getattr(cm, "linear_value_head_dim", 128)
ksize = la.conv1d.weight.shape[-1]; conv_dim = 2 * n_k * d_k + n_v * d_v
print("  norm 类型:", type(la.norm).__name__, "| 激活:", type(getattr(la, "act", None)).__name__)

class V(nn.Module):
    """mode: proj / conv / delta / norm / full（逐步加件 ✓）"""
    def __init__(s, mode):
        super().__init__(); s.mode = mode
    def forward(s, hidden, conv_state, rec_state):
        b, sq = hidden.shape[0], hidden.shape[1]
        mixed = la.in_proj_qkv(hidden).transpose(1, 2)                  # [B,conv_dim,S]
        if s.mode == "proj":
            return mixed.transpose(1, 2)
        z = la.in_proj_z(hidden)                                        # [B,S,v_dim]
        conv_in = torch.cat([conv_state, mixed], dim=2)                 # [B,conv_dim,K-1+S]
        conv_out = la.conv1d(conv_in).transpose(1, 2)                   # [B,S,conv_dim] ✓
        conv_out = torch.nn.functional.silu(conv_out)
        if s.mode == "conv":
            return conv_out
        q, k, v = torch.split(conv_out, [n_k * d_k, n_k * d_k, n_v * d_v], dim=2)
        q = q.reshape(b, sq, n_k, d_k); k = k.reshape(b, sq, n_k, d_k); v = v.reshape(b, sq, n_v, d_v)
        g = la.in_proj_a(hidden); beta = la.in_proj_b(hidden).sigmoid()
        g = g.reshape(b, sq, n_v) if g.shape[-1] == n_v else g
        beta = beta.reshape(b, sq, n_v) if beta.shape[-1] == n_v else beta
        if s.mode == "delta_in":
            return q, k, v, g, beta
        if s.mode == "norm_in":
            return v, z
        core, new_rec = npu_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, initial_state=rec_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True, seq_len=sq, batch=b)
        if s.mode == "delta":
            return core
        if "rmsnormgated" in type(la.norm).__name__.lower():
            core = la.norm(core, z)
        else:
            core = la.norm(core)
        core = core.reshape(b, sq, n_v * d_v)
        if s.mode == "full":
            out = la.out_proj(core)
            return out, new_rec
        return core

def go(mode, ins, inshape):
    try:
        torch.onnx.export(V(mode).eval(), ins, "v_%s.onnx" % mode, input_names=list(inshape.keys()),
                          output_names=["out"] if mode in ("proj", "conv", "delta", "full") else None,
                          opset_version=14, dynamo=False)
        print("  导出 %-9s ✓" % mode); return True
    except Exception as e:
        print("  导出 %-9s ✗ %s" % (mode, str(e)[:80])); return False

H = cm.hidden_size
go("proj", (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
   {"hidden": (B, S, H), "conv_state": (B, conv_dim, ksize - 1), "rec_state": (B, n_v, d_k, d_v)})
go("conv", (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
   {"hidden": (B, S, H), "conv_state": (B, conv_dim, ksize - 1), "rec_state": (B, n_v, d_k, d_v)})
go("delta", (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
   {"hidden": (B, S, H), "conv_state": (B, conv_dim, ksize - 1), "rec_state": (B, n_v, d_k, d_v)})
go("full", (torch.randn(B, S, H), torch.zeros(B, conv_dim, ksize - 1), torch.zeros(B, n_v, d_k, d_v)),
   {"hidden": (B, S, H), "conv_state": (B, conv_dim, ksize - 1), "rec_state": (B, n_v, d_k, d_v)})
