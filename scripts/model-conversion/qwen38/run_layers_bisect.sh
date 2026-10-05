#!/bin/bash
# 层数二分：2 层 / 3 层 ⇒ OMG rc，然后回本机测 Init
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
for n in 2 3; do
  rm -f q35_L${n}* 2>/dev/null
  echo "=== ${n} 层：导出 $(date +%H:%M:%S) ==="
  timeout 900 ~/q38env/bin/python export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 \
     --layers $n --no-embed-head --legacy --out q35_L${n}.onnx 2>&1 | grep -E "产物|Error" | tail -1
  timeout 900 ~/q38env/bin/python lower_hiai.py q35_L${n}.onnx q35_L${n}_low.onnx 2>&1 | tail -1
  INS=$(~/q38env/bin/python - <<PY
import onnx, os
m = onnx.load(os.path.expanduser("~/q38/q35_L${n}_low.onnx"), load_external_data=False)
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
  rm -rf omg_L${n} && mkdir -p omg_L${n} && rm -f check_result.json
  echo "=== ${n} 层：OMG $(date +%H:%M:%S) ==="
  timeout 1800 $D/tools/tools_omg/omg --model q35_L${n}_low.onnx --framework 5 --output omg_L${n}/seg \
    --input_shape="$(echo "$INS" | sed -n 1p)" --input_type="$(echo "$INS" | sed -n 2p)" \
    --output_type="hidden_states:FP32" --weight_data_type FP16 --save_weights_as_external_data=true \
    --platform=kirinx90 --target=omc > omg_L${n}/omg.log 2>&1
  echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_L${n}/omg.log)"
  find omg_L${n} -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "    %10.1f MB  %s\n", $5/1e6, $9}'; done
done
