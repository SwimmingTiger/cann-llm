#!/bin/bash
# dopt(pytorch 版) 三阶段 ⇒ fake_quant_weight.pth（§47 路线 ✓）
set -u
D=~/ddk/tools/tools_dopt/dopt_pytorch_py3
cd "$D" || exit 9
for st in stage1 stage2 stage3; do
  echo "=== $st $(date +%H:%M:%S) ==="
  PYTHONPATH=$D timeout 3600 ~/q38env/bin/python dopt/dopt_lm/opt_main.py \
    --model-path /home/hu60/q38 --quant-stage $st --output-dir ~/q38/dopt_out \
    --dopt-config ~/q38/dopt_out/dopt_config.json \
    --w-bits 4 --act-bits 16 --group-size 128 2>&1 | tail -6
  ls -l ~/q38/dopt_out/ 2>/dev/null | tail -4
done
echo "ALL-STAGES-DONE"
