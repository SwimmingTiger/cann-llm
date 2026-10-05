#!/usr/bin/env python3
"""细分 b4_math 那一跳：Exp / Pow / ReduceSum / Sigmoid 各自贡献多少子图 ✓"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
B, S, H = 1, 64, 64
class P(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind
        s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        y = x @ s.w1
        k = s.kind
        if k == "m0_base":
            pass
        elif k == "m1_exp":
            y = torch.exp(y * 0.001)
        elif k == "m2_pow":
            y = y.pow(1.0001)
        elif k == "m3_reducesum":
            y = y / y.sum(-1, keepdim=True) * 1.0
        elif k == "m4_sigmoid":
            y = y * torch.sigmoid(y)
        elif k == "m5_exp_pow":
            y = torch.exp(y * 0.001).pow(1.0001)
        return y @ s.w2
for k in ["m0_base", "m1_exp", "m2_pow", "m3_reducesum", "m4_sigmoid", "m5_exp_pow"]:
    torch.onnx.export(P(k).eval(), (torch.randn(B, S, H),), "m_%s.onnx" % k,
                      input_names=["x"], output_names=["out"], opset_version=14, dynamo=False)
    print("  导出 %-12s ✓" % k)
