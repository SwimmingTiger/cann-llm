#!/bin/bash
# fp32 图 + 外置权重（不转 fp16 ✗ —— 避开疑似元数据破坏 ✓）
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
rm -rf omg_f32 && mkdir -p omg_f32 && rm -f check_result.json
echo "=== OMG（fp32 + 外置 ✓）$(date +%H:%M:%S) ==="
timeout 3000 $D/tools/tools_omg/omg --model q35_hiai_low.onnx --framework 5 --output omg_f32/seg \
  --input_shape="input_embed:1,64,2048" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
  --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_f32/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_f32/omg.log)"
find omg_f32 -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'; done
grep -aE "^E/" omg_f32/omg.log 2>/dev/null | grep -avE "ascendc|TE_FUSION|ops/impl|TbeInitialize" | head -3 | sed 's/.*::"//' | cut -c1-120
