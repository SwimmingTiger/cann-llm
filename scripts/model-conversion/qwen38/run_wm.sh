#!/bin/bash
# 用 --weight_merge false 试外置权重形态（官方 SubGraph_0.weight 形态 ✓）
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
for tag in fp32 fp16; do
  if [ "$tag" = "fp32" ]; then M=q35_hiai_low.onnx; else M=q35_hiai_fp16.onnx; fi
  [ -f "$M" ] || { echo "  ($M 不存在，跳过)"; continue; }
  rm -rf omg_wm_$tag && mkdir -p omg_wm_$tag
  echo "=== --weight_merge false · $tag $(date +%H:%M:%S) ==="
  timeout 1800 $D/tools/tools_omg/omg --model "$M" --framework 5 --output omg_wm_$tag/seg \
    --input_shape="input_embed:1,64,2048" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
    --weight_data_type FP16 --weight_merge false --platform=kirinx90 --target=omc > omg_wm_$tag/omg.log 2>&1
  echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_wm_$tag/omg.log)"
  ls -l omg_wm_$tag/ 2>/dev/null | awk '{printf "    %10.1f MB  %s\n", $5/1e6, $9}' | grep -vE "omg.log"
done
