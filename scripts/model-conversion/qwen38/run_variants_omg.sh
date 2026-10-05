#!/bin/bash
set -u
cd ~/q38/parts || exit 9
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
~/q38env/bin/python ../probe_layer_variants.py 2>&1 | grep -E "导出|norm 类型" | tail -5
for k in proj conv delta full; do
  [ -f v_$k.onnx ] || { echo "  $k 未导出 ✗"; continue; }
  SH=$(~/q38env/bin/python -c "
import onnx; m=onnx.load('v_$k.onnx',load_external_data=False)
print(';'.join('%s:%s'%(i.name,','.join(str(x.dim_value) for x in i.type.tensor_type.shape.dim)) for i in m.graph.input))")
  IT=$(~/q38env/bin/python -c "
import onnx; m=onnx.load('v_$k.onnx',load_external_data=False)
print(';'.join('%s:FP32'%i.name for i in m.graph.input))")
  rm -rf ov_$k && mkdir -p ov_$k && rm -f check_result.json
  timeout 1200 $D/tools/tools_omg/omg --model v_$k.onnx --framework 5 --output ov_$k/seg \
    --input_shape="$SH" --input_type="$IT" --weight_data_type FP16 --save_weights_as_external_data=true \
    --platform=kirinx90 --target=omc > ov_$k/omg.log 2>&1
  echo "  ★$k OMG成功=$(grep -ac 'OMG generate offline model success' ov_$k/omg.log)★"
done
