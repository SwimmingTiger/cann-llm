#!/usr/bin/env python3
"""★算子子图成本表★：每个算子单独加在干净 MatMul 底座上 ⇒ 测 model 计数增量 ✓。"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
B, S, H = 1, 64, 64

class P(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind
        s.w1 = nn.Parameter(torch.randn(H, H) * 0.05)
        s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
        s.conv = nn.Conv1d(H, H, 4, groups=H)          # 深度可分离 ✓
    def forward(s, x):
        y = x @ s.w1
        k = s.kind
        if k == "base":            pass
        elif k == "add":           y = y + 1.0
        elif k == "mul":           y = y * 1.0001
        elif k == "sub":           y = y - (y * 0.0)
        elif k == "div":           y = y / 2.0
        elif k == "where":         y = torch.where(y > 0, y, y * 1.0)
        elif k == "cast":          y = y.to(torch.float16).to(torch.float32)
        elif k == "exp":           y = torch.exp(y * 0.001)
        elif k == "pow":           y = y.pow(1.0001)
        elif k == "sqrt":          y = torch.sqrt(y * y + 1.0)
        elif k == "reducesum":     y = y / y.sum(-1, keepdim=True) * 1.0
        elif k == "reducemean":    y = y / y.mean(-1, keepdim=True) * 1.0
        elif k == "rsqrt":         y = y * torch.rsqrt(y * y + 1.0)
        elif k == "sigmoid":       y = y * torch.sigmoid(y)
        elif k == "softplus":      y = torch.nn.functional.softplus(y)
        elif k == "reshape":       y = y.reshape(B, S, H)
        elif k == "transpose":     y = y.transpose(1, 2).transpose(1, 2)
        elif k == "slice":         y = torch.cat([y[:, :, : H // 2], y[:, :, H // 2:]], dim=2)
        elif k == "gather":        y = y.index_select(2, torch.arange(H - 1, -1, -1))
        elif k == "split":         a, b = torch.split(y, [H // 2, H // 2], dim=2); y = torch.cat([a, b], dim=2)
        elif k == "conv":          y = s.conv(y.transpose(1, 2)).transpose(1, 2)
        elif k == "matmul":        y = y @ torch.ones(H, H, dtype=y.dtype) * 0.01
        return y @ s.w2

KINDS = ["base", "add", "mul", "sub", "div", "where", "cast", "exp", "pow", "sqrt",
         "reducesum", "reducemean", "rsqrt", "sigmoid", "softplus", "reshape", "transpose",
         "slice", "gather", "split", "conv", "matmul"]
for k in KINDS:
    try:
        torch.onnx.export(P(k).eval(), (torch.randn(B, S, H),), "c_%s.onnx" % k,
                          input_names=["x"], output_names=["out"], opset_version=14, dynamo=False)
    except Exception as e:
        print("  导出 %-11s ✗ %s" % (k, str(e)[:50]))
print("  （导出阶段结束 ✓）")
