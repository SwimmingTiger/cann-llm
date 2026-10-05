#!/bin/bash
# 用 --save_weights_as_external_data=true 产官方外置权重形态（omc + SubGraph_0.weight）
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
rm -rf omg_ext && mkdir -p omg_ext && rm -f check_result.json
echo "=== OMG（外置权重 ✓）$(date +%H:%M:%S) ==="
timeout 2400 $D/tools/tools_omg/omg --model q35_hiai_fp16.onnx --framework 5 --output omg_ext/seg \
  --input_shape="input_embed:1,64,2048" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
  --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_ext/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_ext/omg.log)"
ls -l omg_ext/ | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'
