#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查量化产物有没有被「钳位」—— dopt 的 quant_param_2 写错时会出这个问题。

症状：所有权重的负半轴被钳成 0（等价于对权重做了一次 ReLU），
模型在 NPU 上能跑但输出恒定重复垃圾。详见 docs/model-conversion.md 附录 B。

用法：
    python3 check_quant_clamp.py <fake_quant_weight.pth> [...]

正常应输出：负值占比 ≈ 43%~44%，全非负的权重张量 0 个
被钳位则是：负值占比 ≈ 13%，几乎全部权重张量都没有负值
"""
import sys

import torch

for p in sys.argv[1:]:
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "module"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    tot = neg = n = allpos = 0
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.numel() < 10000:
            continue
        if not k.endswith(".weight"):       # 只看真正的权重，排除 scale/offset
            continue
        if v.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        t = v.numel()
        g = int((v < 0).sum().item())
        tot += t
        neg += g
        n += 1
        allpos += (g == 0)

    pct = neg / max(tot, 1) * 100
    print(f"{p}")
    print(f"  权重张量 {n} 个，负值占比 {pct:.2f}%，全非负 {allpos} 个 "
          f"({allpos / max(n, 1) * 100:.0f}%)")
    if n and allpos >= n * 0.9:
        print("  ✗ 疑似被钳位 —— 检查 config.yaml 里的 quant_param_2"
              "（kirinx90 应为 False）")
    else:
        print("  ✓ 正常")
