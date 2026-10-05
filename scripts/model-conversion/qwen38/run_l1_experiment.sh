#!/bin/bash
# 1 层实验：fp32 权重就能 < 4 GB ✓ ⇒ 不需要 fp16 Cast ✓
# 目的：判断"引擎 CPUCL 尺寸报错"是不是 fp16 Cast 结构造成的 ✓
set -u
cd ~/q38 || exit 9
rm -f l1* check_result.json 2>/dev/null
echo "=== ① 导出 1 层 $(date +%H:%M:%S) ==="
timeout 600 ~/q38env/bin/python export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 \
    --layers 1 --no-embed-head --legacy --out l1_full.onnx 2>&1 | grep -E "产物|Error" | tail -2
echo "=== ② lower ==="
timeout 600 ~/q38env/bin/python lower_hiai.py l1_full.onnx l1_low.onnx 2>&1 | tail -1
echo "=== ③ OMG（fp32 权重 ✓ 不转 fp16 ✗）$(date +%H:%M:%S) ==="
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
rm -rf omg_l1 && mkdir -p omg_l1
timeout 900 $D/tools/tools_omg/omg --model l1_low.onnx --framework 5 --output omg_l1/seg \
  --input_shape="input_embed:1,64,2048" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
  --weight_data_type FP16 --platform=kirinx90 --target=omc > omg_l1/omg.log 2>&1
echo "  OMG rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_l1/omg.log)"
ls -l omg_l1/seg.omc 2>/dev/null | awk '{printf "  ★ omc = %.1f MB\n", $5/1e6}'
