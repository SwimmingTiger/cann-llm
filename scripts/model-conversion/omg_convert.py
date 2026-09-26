#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成并执行华为 OMG 的转换命令（ONNX -> .omc）。

为什么要有这个脚本：官方示例里的 OMG 命令行把 28 层 / 2048 上下文 / 1536 隐藏维
全部写死成超长的 `--input_shape=...` 字符串，换成别的模型就没法用。这里按
「层数 + KV 长度 + 隐藏维」自动拼出来。

用法：
    # 只打印命令，不执行（先看看对不对）
    python3 omg_convert.py --onnx model.onnx --out ./om_out --dry-run

    # 实际执行（需要 OMG 工具与环境，见 docs/model-conversion.md）
    python3 omg_convert.py --onnx model.onnx --out ./om_out

    # 无压缩配置（rebuild_weights 走 FP16 路线时用这个）
    python3 omg_convert.py --onnx model.onnx --out ./om_out --weight-data-type FP16

环境变量：
    OMG_DIR   OMG 工具目录（含 master/omg 或 omg），默认 /path/to/ddk/tools/tools_omg
    ASC_DIR   AscendC 工具目录，默认 /path/to/ddk/tools/tools_ascendc
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import List


def build_input_shape(nlayers: int, kv_len: int, hidden: int) -> str:
    parts = [
        f"input_embed:1,-1,{hidden}",
        f"attention_mask:1,1,-1,{kv_len}",
        "position_ids:1,-1",
    ]
    for i in range(nlayers):
        parts.append(f"past_key_in{i}:{kv_len},2,1,128")
        parts.append(f"past_value_in{i}:{kv_len},2,1,128")
    parts += ["new_kv_cache_pos:-1", "embed_scales:1,-1,1"]
    return ";".join(parts)


def build_past_type(nlayers: int, dtype: str) -> str:
    parts: List[str] = []
    for i in range(nlayers):
        parts.append(f"past_key_in{i}:{dtype}")
        parts.append(f"past_value_in{i}:{dtype}")
    return ";".join(parts)


def build_output_type(nlayers: int, dtype: str) -> str:
    parts = ["lm_logits:FP32"]
    for i in range(nlayers):
        parts.append(f"past_key{i}:{dtype}")
        parts.append(f"past_value{i}:{dtype}")
    return ";".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/执行 OMG 转换命令")
    ap.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    ap.add_argument("--out", required=True, help="输出前缀（会生成 <out>.omc 等）")
    ap.add_argument("--layers", type=int, default=28, help="Transformer 层数（默认 28）")
    ap.add_argument("--kv-len", type=int, default=2048, help="KV 缓存长度（默认 2048）")
    ap.add_argument("--hidden", type=int, default=1536, help="hidden size（默认 1536）")
    ap.add_argument("--platform", default="kirinx90", help="平台（默认 kirinx90）")
    ap.add_argument("--compress-conf", default=None,
                    help="dopt 量化参数文件；给了就用量化，不给就走 FP16")
    ap.add_argument("--weight-data-type", default=None,
                    help="例如 FP16（与 --compress-conf 二选一，FP16 不量化）")
    ap.add_argument("--omg-dir", default=os.environ.get("OMG_DIR", "/path/to/ddk/tools/tools_omg"))
    ap.add_argument("--asc-dir", default=os.environ.get("ASC_DIR", "/path/to/ddk/tools/tools_ascendc"))
    ap.add_argument("--dynamic-dims", default="1,1,1,1,1;64,64,64,64,64",
                    help="prefill / decode 两个动态档位")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    omg_bin = os.path.join(args.omg_dir, "master", "omg")
    if not os.path.exists(omg_bin):
        omg_bin = os.path.join(args.omg_dir, "omg")

    cmd = [
        omg_bin,
        "--model", os.path.abspath(args.onnx),
        "--framework", "5",
        "--output", os.path.abspath(args.out),
        f"--input_shape={build_input_shape(args.layers, args.kv_len, args.hidden)}",
        f"--dynamic_dims={args.dynamic_dims}",
        f"--input_type={build_past_type(args.layers, 'FP16')}",
        f"--output_type={build_output_type(args.layers, 'FP16')}",
        "--save_weights_as_external_data=true",
        f"--platform={args.platform}",
        "--target=omc",
    ]
    if args.compress_conf:
        cmd += ["--compress_conf", os.path.abspath(args.compress_conf)]
    if args.weight_data_type:
        cmd += ["--weight_data_type", args.weight_data_type]
    if not (args.compress_conf or args.weight_data_type):
        print("提示：既没给 --compress-conf 也没给 --weight-data-type，"
              "默认按不量化（FP32）走。若要 FP16 请显式加 --weight-data-type FP16。",
              file=sys.stderr)
    del args.asc_dir

    env = dict(os.environ)
    env.setdefault("SOC_VERSION", args.platform)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(args.omg_dir, "..", "platform", args.platform, "ops", "impl"),
         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    print("### OMG 命令")
    print(" ".join(shlex.quote(c) for c in cmd))
    print("### 环境")
    print(f"  SOC_VERSION={env['SOC_VERSION']}")
    print(f"  PYTHONPATH={env['PYTHONPATH']}")
    if args.dry_run:
        return 0

    if not os.path.exists(omg_bin):
        print(f"错误：找不到 OMG 可执行文件 {omg_bin}\n"
              f"      用 --omg-dir 指定，或设环境变量 OMG_DIR。", file=sys.stderr)
        return 2
    print("### 执行…", flush=True)
    rc = subprocess.call(cmd, cwd=args.omg_dir, env=env)
    print(f"### OMG_EXIT={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
