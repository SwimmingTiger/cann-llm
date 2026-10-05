#!/bin/bash
# 一轮完整流程：导出 → lower → fp16 → OMG ✓（§73）
set -u
cd ~/q38 || exit 9
rm -f onnx__* q35_hiai_full* q35_hiai_low* q35_hiai_fp16* check_result.json 2>/dev/null
echo "=== ① 导出 $(date +%H:%M:%S) ==="
timeout 700 ~/q38env/bin/python export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 \
    --no-embed-head --legacy --out q35_hiai_full.onnx 2>&1 | grep -E "产物|Error" | tail -2
echo "=== ② lower ==="
timeout 700 ~/q38env/bin/python lower_hiai.py q35_hiai_full.onnx q35_hiai_low.onnx 2>&1 | tail -2
echo "=== ③ fp16 ==="
timeout 700 ~/q38env/bin/python onnx_weights_to_fp16.py q35_hiai_low.onnx q35_hiai_fp16.onnx 2>&1 | tail -2
echo "=== ④ OMG $(date +%H:%M:%S) ==="
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
rm -rf omg_sp && mkdir -p omg_sp
timeout 1500 $D/tools/tools_omg/omg --model q35_hiai_fp16.onnx --framework 5 --output omg_sp/seg \
  --input_shape="input_embed:1,64,2048" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
  --weight_data_type FP16 --platform=kirinx90 --target=omc > omg_sp/omg.log 2>&1
echo "  OMG rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_sp/omg.log)"
ls -l omg_sp/seg.omc 2>/dev/null | awk '{printf "  ★ omc = %.1f MB\n", $5/1e6}'
