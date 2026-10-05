#!/bin/bash
# INT32 假设验证：导出(已完成) → lower → OMG(fp32+外置) ⇒ 拿到 omc 直接用 runner 测 Init
set -u
cd ~/q38 || exit 9
echo "=== lower $(date +%H:%M:%S) ==="
timeout 900 ~/q38env/bin/python lower_hiai.py q35_i32.onnx q35_i32_low.onnx 2>&1 | tail -2
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
rm -rf omg_i32 && mkdir -p omg_i32 && rm -f check_result.json
echo "=== OMG $(date +%H:%M:%S) ==="
timeout 3000 $D/tools/tools_omg/omg --model q35_i32_low.onnx --framework 5 --output omg_i32/seg \
  --input_shape="input_embed:1,64,2048;attention_mask:1,1,64,2048;position_ids:1,64;new_kv_cache_pos:64" \
  --input_type="input_embed:FP32;attention_mask:FP32;position_ids:INT32;new_kv_cache_pos:INT32" \
  --output_type="hidden_states:FP32" --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_i32/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_i32/omg.log)"
find omg_i32 -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'; done
echo "ALL-DONE"
