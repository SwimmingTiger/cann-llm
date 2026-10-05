#!/bin/bash
# 24 层 + dtype 修复后的 lower ⇒ OMG ⇒ 交回本机测 Init
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
rm -f q35_24f.onnx q35_24f_low.onnx q35_24f_low.onnx.weights 2>/dev/null
echo "=== 导出 24 层 $(date +%H:%M:%S) ==="
timeout 1500 ~/q38env/bin/python export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 \
   --no-embed-head --legacy --out q35_24f.onnx 2>&1 | grep -E "产物" | tail -1
echo "=== lower（dtype 修复 ✓）$(date +%H:%M:%S) ==="
timeout 1500 ~/q38env/bin/python lower_hiai.py q35_24f.onnx q35_24f_low.onnx 2>&1 | tail -2
INS=$(~/q38env/bin/python -c "
import onnx,os; m=onnx.load(os.path.expanduser(\"~/q38/q35_24f_low.onnx\"),load_external_data=False)
ps=[];ts=[]
for i in m.graph.input:
    t=i.type.tensor_type
    if not t.HasField(\"shape\"): continue
    d=[x.dim_value for x in t.shape.dim]
    if all(v>0 for v in d):
        ps.append(\"%s:%s\"%(i.name,\",\".join(map(str,d)))); ts.append(\"%s:%s\"%(i.name,\"INT32\" if t.elem_type==6 else \"FP32\"))
print(\";\".join(ps)); print(\";\".join(ts))")
rm -rf omg_24f; mkdir -p omg_24f; rm -f check_result.json
echo "=== OMG $(date +%H:%M:%S) ==="
timeout 3000 $D/tools/tools_omg/omg --model q35_24f_low.onnx --framework 5 --output omg_24f/seg \
  --input_shape="$(echo "$INS" | sed -n 1p)" --input_type="$(echo "$INS" | sed -n 2p)" \
  --output_type="hidden_states:FP32" --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_24f/omg.log 2>&1
echo "  OMG成功=$(grep -ac "OMG generate offline model success" omg_24f/omg.log)"
echo "  ★model 计数=$(strings omg_24f/seg/seg.omc 2>/dev/null | grep -c model)★（修前是 ~150 量级? 之前 24 层实测 1 ✗）"
find omg_24f -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "    %12.1f MB  %s\n", $5/1e6, $9}'; done
echo ALL-DONE
