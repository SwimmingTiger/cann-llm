#!/bin/bash
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
INS=$(~/q38env/bin/python - <<'PY'
import onnx, os
m = onnx.load(os.path.expanduser("~/q38/q35_i32_low.onnx"), load_external_data=False)
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
SH=$(echo "$INS" | sed -n 1p); IT=$(echo "$INS" | sed -n 2p)
rm -rf omg_i32 && mkdir -p omg_i32 && rm -f check_result.json
echo "=== OMG（全量输入 ✓ INT32 ✓）$(date +%H:%M:%S) ==="
timeout 3000 $D/tools/tools_omg/omg --model q35_i32_low.onnx --framework 5 --output omg_i32/seg \
  --input_shape="$SH" --input_type="$IT" \
  --output_type="hidden_states:FP32" --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_i32/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_i32/omg.log)"
find omg_i32 -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'; done
