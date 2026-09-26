#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给官方示例打两个补丁，让 Qwen3-4B 的导出在这台 31 GB 内存的机器上能跑完。

补丁 1（必须）——恢复 FP32 导出
  export_model_single_qwen3.py 里 `hf_model_dtype` 是硬编码的 torch.float32。
  实测：**改成 float16 虽然能让导出不 OOM，但会破坏 RoPE 的模式匹配**
  （OMG 日志里出现 36×4=144 条
   `rope_llm_fusion_pass.cc CheckMul0: mul0 weight size invalid 0 != 1`），
  最终引擎能加载模型但 Generate 恒返回 1。所以必须保持 FP32。

补丁 2（内存）——让 onnxsim.simplify 可跳过
  FP32 导出时 onnx_utils.process_onnx 里的 simplify() 是内存峰值所在
  （16 GB 模型 + onnxsim 的工作副本 > 31 GB，进程被 OOM kill）。
  改成受环境变量 CANN_SKIP_ONNX_SIMPLIFY=1 控制，跳过它仍能产出 OMG 可用的图。

用法:
    python3 patch_qwen3_export_mem.py <npu_tuned_export 目录> [--revert]
"""
import argparse
import os
import re
import shutil
import sys

SIMPLIFY_ORIG = '''    print("sim onnx model")
    tensor_size_threshold = f"{size_th_kb}KB"
    skipped_optimizers = ["fuse_matmul_add_bias_into_gemm", "fuse_qkv", "eliminate_duplicate_initializer"]
    onnx_model, check = simplify(onnx_model, tensor_size_threshold=tensor_size_threshold,
                                 skipped_optimizers=skipped_optimizers)
    print(f"sim check stats: {check}")
'''

SIMPLIFY_NEW = '''    print("sim onnx model")
    tensor_size_threshold = f"{size_th_kb}KB"
    skipped_optimizers = ["fuse_matmul_add_bias_into_gemm", "fuse_qkv", "eliminate_duplicate_initializer"]
    if os.environ.get("CANN_SKIP_ONNX_SIMPLIFY") == "1":
        print("  [CANN_SKIP_ONNX_SIMPLIFY=1] 跳过 onnxsim.simplify（内存峰值所在）")
        check = True
    else:
        onnx_model, check = simplify(onnx_model, tensor_size_threshold=tensor_size_threshold,
                                     skipped_optimizers=skipped_optimizers)
    print(f"sim check stats: {check}")
'''


# compress / uncompress 是**一对**：它们的存在只是为了给 simplify 省内存。
# 跳过 simplify 时这一对毫无意义，却各自持一份完整模型副本（4B 约 16 GB ×2）。
# 一起跳过。
COMPRESS_ORIG = '''    size_th_kb = 1024
    size_th_bytes = size_th_kb * 1024
    onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
    print("compress model success")
'''

COMPRESS_NEW = '''    size_th_kb = 1024
    size_th_bytes = size_th_kb * 1024
    if os.environ.get("CANN_SKIP_ONNX_SIMPLIFY") == "1":
        print("  [CANN_SKIP_ONNX_SIMPLIFY=1] 跳过 compress（省下一份完整模型副本）")
        removed_inits = []
    else:
        onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
    print("compress model success")
'''

UNCOMPRESS_ORIG = '''    onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print("uncompress model success")
'''

UNCOMPRESS_NEW = '''    if os.environ.get("CANN_SKIP_ONNX_SIMPLIFY") == "1":
        print("  [CANN_SKIP_ONNX_SIMPLIFY=1] 跳过 uncompress（与 compress 是一对）")
    else:
        onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print("uncompress model success")
'''


def patch_file(path, old, new, tag, revert):
    src = open(path).read()
    if revert:
        if new not in src:
            print(f"  {tag}: 没找到已打补丁的痕迹，跳过")
            return False
        open(path, "w").write(src.replace(new, old, 1))
        print(f"  {tag}: 已还原")
        return True
    if new in src:
        print(f"  {tag}: 已经打过补丁了")
        return False
    if old not in src:
        print(f"  {tag}: ✗ 没找到目标代码 —— 版本可能不同，请手工处理", file=sys.stderr)
        return False
    if not os.path.exists(path + ".premem-patch"):
        shutil.copy(path, path + ".premem-patch")
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {tag}: 已打补丁")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    d = args.export_dir

    # 1) 恢复 FP32
    q3 = os.path.join(d, "export_model_single_qwen3.py")
    src = open(q3).read()
    if args.revert:
        pass
    if "hf_model_dtype = torch.float16" in src:
        if not os.path.exists(q3 + ".premem-patch"):
            shutil.copy(q3, q3 + ".premem-patch")
        open(q3, "w").write(src.replace("hf_model_dtype = torch.float16",
                                        "hf_model_dtype = torch.float32", 1))
        print("  hf_model_dtype: float16 -> float32（恢复官方值）")
    else:
        print("  hf_model_dtype: 已是 float32")

    # 2) simplify 可跳过
    u = os.path.join(d, "onnx_utils.py")
    patch_file(u, SIMPLIFY_ORIG, SIMPLIFY_NEW, "onnxsim.simplify", args.revert)
    patch_file(u, COMPRESS_ORIG, COMPRESS_NEW, "compress_onnx_model", args.revert)
    patch_file(u, UNCOMPRESS_ORIG, UNCOMPRESS_NEW, "uncompress_onnx_model", args.revert)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
