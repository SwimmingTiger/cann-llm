#!/usr/bin/env python3
"""把线性注意力层拆成几块分别做探针 ⇒ 走同一条流水线找病灶。

块：
  conv     —— 深度可分离 Conv1d(kernel=4) + silu（合成张量 ✓ 不需要模型 ✓）
  delta    —— 我们的 npu_chunk_gated_delta_rule（合成小张量 ✓）
  l2norm   —— q/k 的 l2norm + rsqrt 组合 ✓
"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
from npu_gated_delta import npu_chunk_gated_delta_rule   # noqa: E402

B, H, S, Dk, Dv = 1, 2, 8, 8, 8
CD = 2 * H * Dk + H * Dv          # conv_dim（本例 48 ✓）

class ConvProbe(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Conv1d(CD, CD, 4, groups=CD)      # ★深度可分离✓★
    def forward(s, mixed):
        y = s.conv(mixed)                             # [B, CD, S-3] ✓
        return torch.nn.functional.silu(y)

class DeltaProbe(nn.Module):
    def forward(s, q, k, v, g, beta, st):
        o, st2 = npu_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, initial_state=st,
            output_final_state=True, use_qk_l2norm_in_kernel=True, seq_len=S, batch=B)
        return o, st2

class L2Probe(nn.Module):
    def forward(s, q, k):
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        return qn, kn

def export(name, model, args, ins, outs):
    try:
        torch.onnx.export(model.eval(), tuple(args), "%s.onnx" % name, input_names=ins,
                          output_names=outs, opset_version=14, dynamo=False)
        print("  导出 %-8s ✓ 输入 %d 输出 %d" % (name, len(ins), len(outs)))
        return True
    except Exception as e:
        print("  导出 %-8s ✗ %s" % (name, str(e)[:80]))
        return False

export("conv", ConvProbe(), [torch.randn(B, CD, S)], ["mixed"], ["out"])
export("delta", DeltaProbe(),
       [torch.randn(B, H, S, Dk), torch.randn(B, H, S, Dk), torch.randn(B, H, S, Dv),
        torch.randn(B, H, S), torch.randn(B, H, S), torch.randn(B, H, Dk, Dv)],
       ["q", "k", "v", "g", "beta", "st"], ["o", "st_out"])
export("l2", L2Probe(), [torch.randn(B, H, S, Dk), torch.randn(B, H, S, Dk)], ["q", "k"], ["qn", "kn"])
