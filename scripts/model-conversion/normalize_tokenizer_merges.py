#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把新版 HuggingFace tokenizer.json 的 merges 规范成 NPU 引擎认的旧格式。

背景（实测发现）：
  新版 HF（Qwen3 等）的 tokenizer.json 把 BPE merges 存成「数组的数组」：

      "merges": [ ["Ġ", "t"], ["Ġ", "a"], ... ]

  而旧版（Qwen2.5 等）存成「每项是空格分隔的字符串」：

      "merges": [ "Ġ t", "Ġ a", ... ]

  CANN 的 LLM 引擎解析 tokenizer.json 时用的是旧格式，遇到数组会抛：

      nlohmann::json type_error.302: type must be string, but is array

  引擎随即 abort（core dumped），模型根本加载不起来。

  另外新版还多一个 `model.ignore_merges` 字段，旧版没有，一并去掉以贴合旧 schema。

用法:
    python3 normalize_tokenizer_merges.py <tokenizer.json> [-o out.json]
    python3 normalize_tokenizer_merges.py <tokenizer.json> --check   # 只看是否需要转换
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="规范化 tokenizer.json 的 merges 格式")
    ap.add_argument("tokenizer")
    ap.add_argument("-o", "--out", help="输出路径（默认原地改写）")
    ap.add_argument("--check", action="store_true", help="只检查，不写文件")
    args = ap.parse_args()

    raw = json.load(open(args.tokenizer))
    model = raw.get("model", {})
    merges = model.get("merges")
    if merges is None:
        print("错误：找不到 model.merges", file=sys.stderr)
        return 1

    n = len(merges)
    arrays = sum(1 for m in merges if isinstance(m, list))
    strs = sum(1 for m in merges if isinstance(m, str))
    print(f"  merges 条目: {n}  (数组 {arrays} / 字符串 {strs})")
    print(f"  model.ignore_merges 存在: {'是' if 'ignore_merges' in model else '否'}")

    if arrays == 0:
        print("  ✓ 已是旧格式，无需转换")
        return 0

    if strs:
        print(f"  警告：格式混用（{strs} 项已是字符串）—— 只转数组项", file=sys.stderr)

    fixed = []
    for m in merges:
        if isinstance(m, list):
            fixed.append(" ".join(str(x) for x in m))
        else:
            fixed.append(m)
    print(f"  转换示例: {merges[0]!r}  ->  {fixed[0]!r}")

    if args.check:
        print("  (--check，未写文件)")
        return 0

    model["merges"] = fixed
    model.pop("ignore_merges", None)
    raw["model"] = model

    out = args.out or args.tokenizer
    with open(out, "w") as f:
        json.dump(raw, f, ensure_ascii=False)
    print(f"  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
