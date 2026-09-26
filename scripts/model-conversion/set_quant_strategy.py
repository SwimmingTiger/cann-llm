#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 dopt 自动生成的 dopt_config.json 设置量化策略。

`opt_main.py` 首次运行只会生成一份**全是 float（即不量化）**的 dopt_config.json，
提示 "generate plugin quang config please set quant strategy firstly" 然后退出。
必须人工把量化策略填进去，再重跑。

这个脚本按下面的规则自动填：

  * `model.layers.N.*` 的 Linear 层  → Quant_act_weight_eco（W4 / group 128 / act 16）
  * `lm_head`                        → 保持 float（官方示例就是这么配的）
  * embedding / norm 等非 Linear     → 保持 float

用法：
    python3 set_quant_strategy.py <dopt_config.json>              # 原地改写
    python3 set_quant_strategy.py <dopt_config.json> --dry-run    # 只看结果不改
    python3 set_quant_strategy.py <dopt_config.json> -o out.json  # 另存

策略参数可用 --w-bits / --group-size / --act-bits 覆盖，默认与官方 run.sh 一致
（`--group-size 128 --w-bits 4 --act-bits 16`）。
"""
import argparse
import collections
import json
import sys

LINEAR = "torch.nn.modules.linear.Linear"
# 这些层即便类型是 Linear 也不量化
KEEP_FLOAT = {"lm_head"}


def is_linear(entry) -> bool:
    return isinstance(entry, dict) and LINEAR in str(entry.get("type", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="设置 dopt 量化策略")
    ap.add_argument("config", help="dopt_config.json 路径")
    ap.add_argument("-o", "--out", help="输出路径（默认原地改写）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--strategy", default="Quant_act_weight_eco")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--act-bits", type=int, default=16)
    args = ap.parse_args()

    raw = json.load(open(args.config))
    body = raw.get("layer_strategy", raw)
    if not isinstance(body, dict):
        print("错误：找不到层配置字典", file=sys.stderr)
        return 1

    changed = kept = other = 0
    for key, entry in body.items():
        if entry is None:
            other += 1
            continue
        if not is_linear(entry):
            other += 1
            continue
        if key in KEEP_FLOAT or key.startswith("lm_head"):
            kept += 1
            continue
        entry["quant_strategy"] = args.strategy
        entry["weight"] = {"bit": args.w_bits, "group_size": args.group_size}
        entry["input"] = {"bit": args.act_bits}
        changed += 1

    strat = collections.Counter(
        v.get("quant_strategy") for v in body.values() if isinstance(v, dict)
    )
    print(f"  条目总数      : {len(body)}")
    print(f"  设为量化      : {changed}")
    print(f"  保持 float    : {kept}  (lm_head 等)")
    print(f"  非 Linear     : {other}  (embedding / null)")
    print("  策略分布:")
    for s, n in strat.most_common():
        print(f"      {str(s):26} x{n}")

    if changed == 0:
        print("警告：没有任何层被改动 —— 检查输入文件是不是已经设置过了", file=sys.stderr)

    if args.dry_run:
        print("  (--dry-run，未写文件)")
        return 0

    out = args.out or args.config
    with open(out, "w") as f:
        json.dump(raw, f, indent=4, ensure_ascii=False)
    print(f"  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
