#!/bin/bash
# dopt(pytorch) 三阶段 —— 用 CUDA venv（~/q38cuda ✓ torch 2.5.1+cu121 ✓）
set -u
uv pip install -q --python ~/q38cuda/bin/python transformers datasets accelerate pyyaml numpy \
    sentencepiece protobuf 2>&1 | tail -2
ROOT=~/q38/dopt_work
D=~/ddk/tools/tools_dopt/dopt_pytorch_py3
OUT=$ROOT/train_output
mkdir -p $OUT
cd "$D" || exit 9
for st in stage1 stage2 stage3; do
  echo "=== $st $(date +%H:%M:%S) ==="
  PYTHONPATH=$D timeout 7200 ~/q38cuda/bin/python dopt/dopt_lm/opt_main.py \
    --model-path /home/hu60/q38 \
    --dopt-config $ROOT/dopt_config.json \
    --optimize-config $ROOT/config.yaml \
    --quant-stage $st --block-size 128 --output-dir $OUT 2>&1 | tail -6
  ls -l $OUT 2>/dev/null | tail -4
done
echo "ALL-STAGES-DONE"
