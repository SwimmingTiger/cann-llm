"""逐算子二分：哪个算子会被 OMG 路由到 ascendc 内核（该内核此 DDK 版本不存在 ✗）。

做法：每个算子造一个 tiny torch 图 → 旧导出器（opset 14 ✓）→ 安全 lowering ✓
      → OMG（--target=omc）→ 记录 rc / 是否出现 ascendc 报错 / 是否产出 omc

跑法（hu60tx，在 ~/q38 下）：
    ~/q38env/bin/python op_bisect2.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

OUT = os.path.expanduser("~/q38/ops2")
DDK = os.path.expanduser("~/ddk")
S = 64
D = 256


class Toy(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.w = torch.nn.Parameter(torch.randn(D, D) * 0.1)
        self.gate = torch.nn.Parameter(torch.randn(D) * 0.1)

    def forward(self, x):
        k = self.kind
        if k == "matmul_ctrl":
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
            return x + self.gate
        if k == "split_concat":
            a, b = torch.split(x, D // 2, dim=-1)
            return torch.cat([b, a], dim=-1)
        if k == "squeeze_unsq":
            return x.unsqueeze(0).squeeze(0)
        if k == "reducesum":
            return x - x.sum(-1, keepdim=True) / D
        if k == "sigmoid_softmax":
            return torch.softmax(torch.sigmoid(x), dim=-1)
        if k == "constant_matmul":                 # 我们的常量矩阵前缀和 ✓（2 维 @ 2 维 ✓）
            tl = torch.triu(torch.ones(S, S))
            return (x[:, :, 0] @ tl).unsqueeze(-1).expand(-1, -1, D)
        if k == "gather_slice":
            return x[:, : D // 2]
        if k == "conv_shift":                      # 移位切片版因果卷积 ✓
            xp = torch.nn.functional.pad(x, (3, 0))
            return xp[:, :, 0:S] + xp[:, :, 1:S + 1]
        raise ValueError(k)


KINDS = ["matmul_ctrl", "softplus", "cossin", "sqrt_recip", "div_exp_neg", "greater_where",
         "equal_not", "and_mask", "expand_bcast", "split_concat", "squeeze_unsq",
         "reducesum", "sigmoid_softmax", "constant_matmul", "gather_slice", "conv_shift"]


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, os.path.expanduser("~/q38"))
    import onnx
    import onnx_lower

    torch.set_grad_enabled(False)
    x = torch.randn(1, S, D)
    env = dict(os.environ)
    env["SOC_VERSION"] = "kirinx90"
    env["LD_LIBRARY_PATH"] = os.path.join(DDK, "tools/tools_omg/master/lib64") + ":" + \
                             os.path.join(DDK, "tools/platform/kirinx90/lib64")
    omg = os.path.join(DDK, "tools/tools_omg/omg")
    print("%-16s %-6s %-9s %s" % ("算子", "OMG", "ascendc", "备注"))
    for k in KINDS:
        p = os.path.join(OUT, k + ".onnx")
        m = Toy(k).eval()
        torch.onnx.export(m, (x,), p, input_names=["x"], output_names=["y"],
                          opset_version=14, dynamo=False)
        mm = onnx.load(p)
        onnx_lower.lower_model(mm)
        onnx_lower.fix_static_shapes(mm, verbose=False)
        onnx.save(mm, p)
        odir = os.path.join(OUT, "o_" + k)
        os.makedirs(odir, exist_ok=True)
        tag = os.path.join(odir, "seg")
        log = os.path.join(odir, "omg.log")
        with open(log, "w") as fh:
            rc = subprocess.call([omg, "--model", p, "--framework", "5", "--output", tag,
                                  "--input_shape=x:1,%d,%d" % (S, D), "--input_type=x:FP32",
                                  "--output_type=y:FP32", "--weight_data_type", "FP16",
                                  "--platform=kirinx90", "--target=omc"],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=os.path.expanduser("~/q38"))
        txt = open(log, errors="ignore").read()
        asc = "ascendc" if ("ascendc_adaptee" in txt or "ascendc_kernel" in txt) else "-"
        omc = os.path.exists(tag + ".omc")
        note = "★产出 omc ✓★" if omc else ("ascendc 路由 ✗" if asc != "-" else "其它失败")
        print("%-16s %-6s %-9s %s" % (k, rc, asc, note), flush=True)
    print("BISECT-DONE")


if __name__ == "__main__":
    main()
