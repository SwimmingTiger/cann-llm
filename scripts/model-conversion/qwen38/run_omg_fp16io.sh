#!/bin/bash
# 用 ★FP16 输入类型★ 重编（对齐官方导出脚本用 bf16 例化输入的做法 ✓）
set -u
cd ~/q38 || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
echo "=== 生成 input_type 串（浮点→FP16 ✓ position 类→INT64 ✓）==="
INS=$(~/q38env/bin/python - <<'PY'
import onnx, os
m = onnx.load(os.path.expanduser("~/q38/q35_hiai_low.onnx"), load_external_data=False)
parts = []
for i in m.graph.input:
    t = i.type.tensor_type
    if not t.HasField("shape"):
        continue
    dims = [d.dim_value for d in t.shape.dim]
    if any(v <= 0 for v in dims):
        continue
    dt = "INT64" if t.elem_type == 7 else "FP16"
    parts.append("%s:%s" % (i.name, dt))
print(";".join(parts))
PY
)
echo "  $(echo "$INS" | tr ';' '\n' | head -3 | tr '\n' ' ') … 共 $(echo "$INS" | tr ';' '\n' | wc -l) 个"
rm -rf omg_f16io && mkdir -p omg_f16io && rm -f check_result.json
echo "=== OMG（FP16 IO ✓ 外置权重 ✓）$(date +%H:%M:%S) ==="
timeout 3000 $D/tools/tools_omg/omg --model q35_hiai_low.onnx --framework 5 --output omg_f16io/seg \
  --input_shape="$(~/q38env/bin/python -c "
import onnx, os
m = onnx.load(os.path.expanduser('~/q38/q35_hiai_low.onnx'), load_external_data=False)
ps=[]
for i in m.graph.input:
    t=i.type.tensor_type
    if not t.HasField('shape'): continue
    d=[x.dim_value for x in t.shape.dim]
    if all(v>0 for v in d): ps.append('%s:%s' % (i.name, ','.join(map(str,d))))
print(';'.join(ps))")" \
  --input_type="$INS" --output_type="hidden_states:FP16" \
  --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > omg_f16io/omg.log 2>&1
echo "  rc=$? · 成功标志=$(grep -ac 'OMG generate offline model success' omg_f16io/omg.log)"
find omg_f16io -type f 2>/dev/null | while read f; do ls -l "$f" | awk '{printf "  %12.1f MB  %s\n", $5/1e6, $9}'; done
