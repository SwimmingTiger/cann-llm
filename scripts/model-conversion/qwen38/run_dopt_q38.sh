#!/bin/bash
# Qwen3.8 的 dopt 量化：造校准 bin → cal_conf → dopt ⇒ 量化 ONNX + compress_conf ✓
set -u
cd ~/q38 || exit 9
D=~/ddk
Q=~/q38            # 本工作目录
mkdir -p calib_bin && rm -f calib_bin/*.bin 2>/dev/null
echo "=== ① 造校准 bin（每个输入一个 ✓ 全 float32 ✓）$(date +%H:%M:%S) ==="
~/q38env/bin/python - <<'PY'
import os, sys
import numpy as np, onnx
sys.path.insert(0, os.path.expanduser("~/q38"))
from make_calib_bin import dump_tensor
m = onnx.load(os.path.expanduser("~/q38/q35_hiai_low.onnx"), load_external_data=False)
out = os.path.expanduser("~/q38/calib_bin")
rng = np.random.default_rng(0)
n = 0
for i in m.graph.input:
    t = i.type.tensor_type
    if not t.HasField("shape"):
        continue
    shape = [d.dim_value for d in t.shape.dim]
    if any(v <= 0 for v in shape):
        continue
    arr = rng.standard_normal(shape).astype(np.float32) * 0.1
    dump_tensor(arr, shape, os.path.join(out, i.name + ".bin"))
    n += 1
print("  ✓ 写了 %d 个 bin（形状取自图 ✓）" % n)
PY
echo "=== ② 生成 cal_conf.prototxt ==="
~/q38env/bin/python ~/q38/make_cal_conf.py q35_hiai_low.onnx calib_bin cal_conf.prototxt 2>&1 | tail -2
echo "=== ③ dopt $(date +%H:%M:%S) ==="
INS=$(~/q38env/bin/python - <<'PY'
import onnx, os
m = onnx.load(os.path.expanduser("~/q38/q35_hiai_low.onnx"), load_external_data=False)
parts = []
for i in m.graph.input:
    t = i.type.tensor_type
    if t.HasField("shape"):
        dims = [d.dim_value for d in t.shape.dim]
        if all(v > 0 for v in dims):
            parts.append("%s:%s" % (i.name, ",".join(str(d) for d in dims)))
print(";".join(parts))
PY
)
DOPT_PY=~/q38env/bin/python DOPT_DIR=$D/tools/tools_dopt/dopt_onnx_py3 \
# ★必须绝对路径✗★：run_dopt.sh 会 cd 到 dopt 目录 ✓
DOPT_PY=$DOPT_PY DOPT_DIR=$DOPT_DIR \
MODEL=$PWD/q35_hiai_low.onnx CAL_CONF=$PWD/cal_conf.prototxt OUT=$PWD/q38_w8.onnx CONF=$PWD/q38_compress.json \
  timeout 7200 ~/q38/run_dopt.sh "$INS" "hidden_states" 2>&1 | tail -6
echo "=== 产物 ==="
ls -l q38_w8.onnx* q38_compress.json 2>/dev/null | awk '{printf "  %10.1f MB  %s\n", $5/1e6, $9}'
