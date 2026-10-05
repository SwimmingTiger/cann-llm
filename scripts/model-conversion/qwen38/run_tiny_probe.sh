#!/bin/bash
# 决定性实验：极简模型（单层 MatMul）走同一条流水线 ⇒ 用 runner Init rc 判定
set -u
cd ~/q38 || exit 9
~/q38env/bin/python - <<'PY'
import torch
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.w = torch.nn.Parameter(torch.randn(2048, 2048) * 0.02)
    def forward(s, x):
        return x @ s.w
torch.set_grad_enabled(False)
torch.onnx.export(M().eval(), (torch.randn(1, 64, 2048),), "/home/hu60/q38/tiny.onnx",
                  input_names=["input_embed"], output_names=["hidden_states"],
                  opset_version=14, dynamo=False)
print("  造 tiny.onnx ✓")
PY
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
for tag in static dyn; do
  rm -rf omg_tiny_$tag && mkdir -p omg_tiny_$tag && rm -f check_result.json
  if [ "$tag" = "static" ]; then
    SH="input_embed:1,64,2048"; DD=""
  else
    SH="input_embed:1,-1,2048"; DD='--dynamic_dims=1,1;64,64'
  fi
  echo "=== tiny/$tag OMG ==="
  timeout 900 $D/tools/tools_omg/omg --model tiny.onnx --framework 5 --output omg_tiny_$tag/seg \
    --input_shape="$SH" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
    --weight_data_type FP16 --save_weights_as_external_data=true $DD \
    --platform=kirinx90 --target=omc > omg_tiny_$tag/omg.log 2>&1
  echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_tiny_$tag/omg.log)"
  find omg_tiny_$tag -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "    %10.1f MB  %s\n", $5/1e6, $9}'; done
done
