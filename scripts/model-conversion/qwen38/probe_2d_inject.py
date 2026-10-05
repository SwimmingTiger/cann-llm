#!/usr/bin/env python3
"""确认实验：往干净的 MatMul 串联里【注入 2 维中间量】⇒ 看 weight 文件数（子图代理）是否跳 ✓"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
H, S = 64, 64
class Clean(nn.Module):
    def __init__(s):
        super().__init__(); s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        return (x @ s.w1) @ s.w2                       # 全 3 维 ✓
class With2D(nn.Module):
    def __init__(s):
        super().__init__(); s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        y = x @ s.w1                                   # [1,64,64] ✓
        flat = y.reshape(H, H)                         # ★2 维中间量✗★
        g = torch.exp(flat * 0.001)                    # ★2 维上的 Exp✗★
        y2 = g.reshape(1, H, H)                        # 回去 3 维 ✓
        return y2 @ s.w2
class With2DSelect(nn.Module):
    def __init__(s):
        super().__init__(); s.w1 = nn.Parameter(torch.randn(H, H) * 0.05); s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        y = x @ s.w1
        flat = y.reshape(H, H)                         # 2 维 ✗
        sel = torch.where(flat > 0, flat, flat * (-1.0))   # ★2 维上的 Select + Sub✗★
        return sel.reshape(1, H, H) @ s.w2
for tag, mod in (("clean", Clean()), ("w2d", With2D()), ("w2dsel", With2DSelect())):
    torch.onnx.export(mod.eval(), (torch.randn(1, S, H),), "%s.onnx" % tag,
                      input_names=["x"], output_names=["out"], opset_version=14, dynamo=False)
    print("  导出 %s ✓" % tag)
