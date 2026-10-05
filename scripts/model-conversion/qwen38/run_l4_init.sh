#!/bin/bash
# 4 层版：导出 → lower → OMG(fp32+外置) ⇒ 交回本机用 runner 测 Init
set -u
cd ~/q38 || exit 9
rm -f q35_l4i* 2>/dev/null
echo "=== 导出 4 层 $(date +%H:%M:%S) ==="
timeout 900 ~/q38env/bin/python export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 \
   --layers 4 --no-embed-head --legacy --out q35_l4i.onnx 2>&1 | grep -E "产物|Error" | tail -2
echo "=== lower ==="
timeout 900 ~/q38env/bin/python lower_hiai.py q35_l4i.onnx q35_l4i_low.onnx 2>&1 | tail -1
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
INS=$(~/q38env/bin/python - <<'PY'
import onnx, os
m = onnx.load(os.path.expanduser("~/q38/q35_l4i_low.onnx"), load_external_data=False)
ps, ts = [], []
for i in m.graph.input:
    t = i.type.tensor_type
    if not t.HasField("shape"): continue
    d = [x.dim_value for x in t.shape.dim]
    if not all(v > 0 for v in d): continue
    ps.append("%s:%s" % (i.name, ",".join(map(str, d))))
    ts.append("%s:%s" % (i.name, "INT32" if t.elem_type == 6 else "FP32"))
print(";".join(ps)); print(";".join(ts))
PY
)
rm -rf omg_l4i && mkdir -p omg_l4i && rm -f check_result.json
echo "=== OMG $(date +%H:%M:%S) ==="
timeout 1800 $D/tools/tools_omg/omg --model q35_l4i_low.onnx --framework 5 --output omg_l4i/seg \
  --input_shape="$(echo "$INS" | sed -n 1p)" --input_type="$(echo "$INS" | sed -n 2p)" \
  --output_type="hidden_states:FP32" --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_l4i/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_l4i/omg.log)"
find omg_l4i -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'; done
