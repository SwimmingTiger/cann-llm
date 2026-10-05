"""用 ★torch 导出★ 造单算子小模型（保证 converter 能编译出真内容 ✓）。

手写 ONNX 那批 `.ms` 只有 ~1 KB ✗（转换器把内容丢了 ✗），所以改用 torch 导出 ✓ ——
和 qwen3_5/真实模型走同一条路 ✓，结果可比 ✓。

跑法：~/q38env/bin/python torch_toys.py     # 生成 toys2/<name>.onnx
"""
from __future__ import annotations

import os

import torch

OUT = os.path.expanduser("~/q38/toys2")
S = 64
D = 256


class Toy(torch.nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.w = torch.nn.Parameter(torch.randn(D, D) * 0.1)
        self.gate = torch.nn.Parameter(torch.randn(D) * 0.1)

    def forward(self, x):
        k = self.kind
        if k == "matmul_ctrl":                    # 对照组：gemma4 用过 ✓
            return x @ self.w
        if k == "softplus":
            return torch.nn.functional.softplus(x)
        if k == "cossin":
            return torch.cos(x) + torch.sin(x)
        if k == "sqrt_recip":
            return torch.rsqrt(x * x + 1.0)
        if k == "div_exp_neg":
            return -(torch.exp(x) - x) / 2.0
        if k == "greater_where":
            return torch.where(x > 0, x, -x)
        if k == "equal_not":
            return torch.where(x == 0, torch.ones_like(x), x)
        if k == "and_mask":
            return torch.where((x > 0) & (x < 1), x, torch.zeros_like(x))
        if k == "expand_bcast":
            return x + self.gate                     # 广播 ✓（会引入 Expand ✗）
        if k == "split_concat":
            a, b = torch.split(x, D // 2, dim=-1)
            return torch.cat([b, a], dim=-1)
        if k == "squeeze_unsq":
            return x.unsqueeze(0).squeeze(0)
        if k == "reducesum":
            return x - x.sum(-1, keepdim=True) / D
        if k == "sigmoid_softmax":
            return torch.softmax(torch.sigmoid(x), dim=-1)
        if k == "cumsum_equiv":                  # 我们的常量矩阵前缀和 ✓
            tl = torch.triu(torch.ones(S, S))
            return x @ tl
        if k == "gather_rows":
            return x[:, :D // 2]
        raise ValueError(k)


KINDS = ["matmul_ctrl", "softplus", "cossin", "sqrt_recip", "div_exp_neg", "greater_where",
         "equal_not", "and_mask", "expand_bcast", "split_concat", "squeeze_unsq",
         "reducesum", "sigmoid_softmax", "cumsum_equiv", "gather_rows"]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    x = torch.randn(1, S, D)
    for k in KINDS:
        m = Toy(k).eval()
        p = os.path.join(OUT, k + ".onnx")
        try:
            torch.onnx.export(m, (x,), p, input_names=["x"], output_names=["y"],
                              dynamo=True, opset_version=18, do_constant_folding=True)
            sz = os.path.getsize(p) + (os.path.getsize(p + ".data") if os.path.exists(p + ".data") else 0)
            print("  %-16s ✓ %.1f MB" % (k, sz / 1e6))
        except Exception as e:
            print("  %-16s ✗ %s: %s" % (k, type(e).__name__, str(e)[:70]))
    print("⇒ %s" % OUT)


if __name__ == "__main__":
    main()
