#!/usr/bin/env python3
"""路线甲：以"纯 MatMul 串联"为底座，逐批加入 delta rule 用到的算子 ⇒ 看 model 计数何时跳 ✓。

拓扑保持【串联+有状态进出】的形状（与 k2_delta 同类 ✓），便于与 9/13 的门槛比较 ✓。
批次：
  b0_base  纯 MatMul 串联
  b1_elem  + Add / Mul（elementwise ✓）
  b2_shape + Reshape / Transpose（3 维 ✓）
  b3_slice + Slice / Concat（3 维 ✓）
  b4_math  + Exp / Pow / ReduceSum / Sigmoid
  b5_where + Where（Select ✓）
  b6_sub   + Sub
  b7_cast  + Cast（INT32→FP32 ✓）
"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
B, S, H = 1, 64, 64

class P(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind
        s.w1 = nn.Parameter(torch.randn(H, H) * 0.05)
        s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
        s.w3 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        y = x @ s.w1
        k = s.kind
        if k in ("b1_elem", "b2_shape", "b3_slice", "b4_math", "b5_where", "b6_sub", "b7_cast"):
            y = y + 1.0
            y = y * 1.0001
        if k in ("b2_shape", "b3_slice", "b4_math", "b5_where", "b6_sub", "b7_cast"):
            y = y.reshape(B, S, H).transpose(1, 2).transpose(1, 2)
        if k in ("b3_slice", "b4_math", "b5_where", "b6_sub", "b7_cast"):
            a = y[:, :, : H // 2]
            b = y[:, :, H // 2:]
            y = torch.cat([a, b], dim=2)
        if k in ("b4_math", "b5_where", "b6_sub", "b7_cast"):
            y = torch.exp(y * 0.001)
            y = y.pow(1.0001)
            y = y / y.sum(-1, keepdim=True) * 1.0
            y = y * torch.sigmoid(y)
        if k in ("b5_where", "b6_sub", "b7_cast"):
            y = torch.where(y > 0, y, y * 1.0)
        if k in ("b6_sub", "b7_cast"):
            y = y - (y * 0.0)
        if k == "b7_cast":
            z = (y * 0.0).to(torch.int32).to(torch.float32)     # ★Cast 往返✓★（不引入 int 输入 ✓）
            y = y + z
        return y @ s.w2

KINDS = ["b0_base", "b1_elem", "b2_shape", "b3_slice", "b4_math", "b5_where", "b6_sub", "b7_cast"]
for k in KINDS:
    try:
        torch.onnx.export(P(k).eval(), (torch.randn(B, S, H),),
                          "b_%s.onnx" % k, input_names=["x"], output_names=["out"],
                          opset_version=14, dynamo=False)
        print("  导出 %-9s ✓" % k)
    except Exception as e:
        print("  导出 %-9s ✗ %s" % (k, str(e)[:60]))
