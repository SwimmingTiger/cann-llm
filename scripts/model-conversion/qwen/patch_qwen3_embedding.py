#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复官方示例 export_model_single_qwen3.py 里被整段注释掉的「独立 embedding 导出」。

背景（实测发现）：
  qwen2 的导出脚本（export_model_single_qwen2.py:77-84）会调用
      from do_opt import process_embedding_weights
      process_embedding_weights(ckpt, mul_twice, onnx_output_dir,
                                f"{onnx_output_model_name}_{seq_len}_{kv_cache_max_len}")
  把 embedding 表单独导出成 <name>.embedding_weights / .embedding_dequant_scale，
  供 NPU 引擎在图外算 input_embed（模型目录的 executor.json 要引用这两个文件）。

  但 qwen3 的导出脚本把这段整体注释掉了，只留下
      model_wrapper.model.load_state_dict(torch.load(quant_pth, ...), strict=False)
  结果：导出的 ONNX 图输入是 input_embed，却没有任何 embedding 文件，模型目录装配不起来。

本脚本把那段恢复成与 qwen2 一致的写法（并把 do_opt 的 import 一并修正，
qwen3 脚本原来写的是不存在的 dopt.do_opt）。

用法:
    python3 patch_qwen3_embedding.py <export_model_single_qwen3.py> [--dry-run]
"""
import argparse
import re
import shutil
import sys

BLOCK = '''    if quant_pth is not None:
        ckpt = torch.load(quant_pth, map_location="cpu")
        from do_opt import process_embedding_weights
        process_embedding_weights(ckpt, mul_twice, onnx_output_dir, f"{onnx_output_model_name}_{seq_len}_{kv_cache_max_len}")
        from do_opt import load_state_dict; load_state_dict(model_wrapper.model, ckpt)

        model_wrapper.model.load_state_dict(torch.load(quant_pth, map_location="cpu"), strict=False)
'''

# 匹配「if quant_pth is not None:」起到「load_state_dict(torch.load(quant_pth」那行为止的整段
PATTERN = re.compile(
    r"( *)if quant_pth is not None:\n"
    r"(?:.*\n)*?"
    r"\1    model_wrapper\.model\.load_state_dict\(torch\.load\(quant_pth[^\n]*\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = open(args.script).read()
    m = PATTERN.search(src)
    if not m:
        print("错误：没找到目标代码块 —— 脚本可能已经打过补丁或版本不同", file=sys.stderr)
        return 1

    old = m.group(0)
    active = [
        ln for ln in old.split("\n")
        if "process_embedding_weights" in ln and not ln.lstrip().startswith("#")
    ]
    if active:
        print("提示：目标块里已有未注释的 process_embedding_weights，看起来已打过补丁")
        return 1

    print("=== 将替换的原始代码块 ===")
    for line in old.rstrip("\n").split("\n"):
        print("  | " + line)
    print("=== 替换为 ===")
    for line in BLOCK.rstrip("\n").split("\n"):
        print("  | " + line)

    if args.dry_run:
        print("(--dry-run，未写文件)")
        return 0

    shutil.copy(args.script, args.script + ".before-embed-patch")
    open(args.script, "w").write(src[:m.start()] + BLOCK + src[m.end():])
    print(f"已打补丁: {args.script}  (备份: {args.script}.before-embed-patch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
