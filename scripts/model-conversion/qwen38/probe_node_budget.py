#!/usr/bin/env python3
"""★节点预算实验★：在干净底座上重复叠加【免费算子】(add ✓) N 次 ⇒ 看 model 计数何时跳 ✓。

若在某个 N 处跳 ⇒ 说明 OMG 是按【子图的节点/规模上限】切分的 ✗
（那就能解释"gateonly 只加一个乘法也跳到 13" ✗ 这个矛盾 ✓）
"""
import torch, torch.nn as nn
torch.set_grad_enabled(False)
B, S, H = 1, 64, 64

class P(nn.Module):
    def __init__(s, n):
        super().__init__(); s.n = n
        s.w1 = nn.Parameter(torch.randn(H, H) * 0.05)
        s.w2 = nn.Parameter(torch.randn(H, H) * 0.05)
    def forward(s, x):
        y = x @ s.w1
        for i in range(s.n):
            y = y + 0.001                      # ★同一个免费算子重复 N 次✓★
        return y @ s.w2

for n in [1, 4, 8, 16, 32, 64, 128]:
    torch.onnx.export(P(n).eval(), (torch.randn(B, S, H),), "nb_%03d.onnx" % n,
                      input_names=["x"], output_names=["out"], opset_version=14, dynamo=False)
    print("  导出 nb_%03d ✓（%d 个 Add）" % (n, n))
