#!/usr/bin/env python3
"""最干净的鉴别：普通 MatMul 的【串联】vs【并联】（不含 delta rule 等任何复杂结构 ✓）"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
H, S = 64, 64

class Chain(nn.Module):
    def __init__(s):
        super().__init__(); s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        return (x @ s.w1) @ s.w2                      # ★串联✓★

class Parallel(nn.Module):
    def __init__(s):
        super().__init__(); s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        return x @ s.w1, x @ s.w2                     # ★并联✓★

for tag, mod, outs in (("tchain", Chain(), ["out"]), ("tpar", Parallel(), ["o1", "o2"])):
    torch.onnx.export(mod.eval(), (torch.randn(1, S, H),), "%s.onnx" % tag,
                      input_names=["x"], output_names=outs, opset_version=14, dynamo=False)
    print("  导出 %s ✓" % tag)
