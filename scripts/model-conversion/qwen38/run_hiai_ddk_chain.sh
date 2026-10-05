#!/bin/bash
# Qwen3.8（qwen3_5）· DDK hiai 直跑路线 · 一键链路（导出 → lower → OMG → 组装）
#
# 用法：
#   ./run_hiai_ddk_chain.sh <HF模型目录> [层数] [kv_len] [输出目录]
# 例：
#   ./run_hiai_ddk_chain.sh ~/q38 24 2048 ~/q38/out24
#
# 前置：OMG 需要 DDK 的 LD_LIBRARY_PATH / SOC_VERSION（脚下会自己设 ✓）
# 说明：本脚本只做①②③④；⑤（DDK 侧 Init 判据）用 hiai_runner ✓
set -eu
HF=${1:?需要 HF 模型目录}
LAYERS=${2:-24}
KV=${3:-2048}
OUT=${4:-$PWD/out_qwen38}
SEQ=64
NAME=qwen38_2b
HERE=$(cd "$(dirname "$0")" && pwd)
D=${DDK_DIR:-$HOME/ddk}
PY=${PY:-$HOME/q38env/bin/python}

mkdir -p "$OUT"; cd "$OUT"
echo "=== ① 导出（layers=$LAYERS kv=$KV seq=$SEQ）==="
"$PY" "$HERE/export_hiai_q35.py" --hf "$HF" --seq $SEQ --kv-len $KV \
      --layers $LAYERS --no-embed-head --legacy --out "$OUT/q35.onnx" | tail -2

echo "=== ② lower（含 ★dtype 保留★ 与类型守卫 ✓）==="
"$PY" "$HERE/lower_hiai.py" "$OUT/q35.onnx" "$OUT/q35_low.onnx" | tail -2

echo "=== ③ OMG（★--save_weights_as_external_data=true★ ⇒ 官方形态）==="
export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
# 由 ONNX 生成 input_shape / input_type（★类型必须与图一致：INT32 就写 INT32★）
SH=$("$PY" - "$OUT/q35_low.onnx" <<'PY'
import sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
ps = []
for i in m.graph.input:
    t = i.type.tensor_type
    if not t.HasField("shape"):
        continue
    d = [x.dim_value for x in t.shape.dim]
    if all(v > 0 for v in d):
        ps.append("%s:%s" % (i.name, ",".join(map(str, d))))
print(";".join(ps))
PY
)
IT=$("$PY" - "$OUT/q35_low.onnx" <<'PY'
import sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
ts = []
for i in m.graph.input:
    t = i.type.tensor_type
    if not t.HasField("shape"):
        continue
    d = [x.dim_value for x in t.shape.dim]
    if all(v > 0 for v in d):
        ts.append("%s:%s" % (i.name, "INT32" if t.elem_type == 6 else "FP32"))
print(";".join(ts))
PY
)
rm -rf "$OUT/omg"; mkdir -p "$OUT/omg"; rm -f check_result.json
"$D/tools/tools_omg/omg" --model "$OUT/q35_low.onnx" --framework 5 --output "$OUT/omg/seg" \
  --input_shape="$SH" --input_type="$IT" --output_type="hidden_states:FP32" \
  --weight_data_type FP16 --save_weights_as_external_data=true \
  --platform=kirinx90 --target=omc > "$OUT/omg/omg.log" 2>&1
echo "  OMG 成功标志=$(grep -ac 'OMG generate offline model success' "$OUT/omg/omg.log")（1=成功 ✓）"
ls -l "$OUT/omg/seg" | awk '{printf "    %12.1f MB  %s\n", $5/1e6, $9}'

echo "=== ④ 组装模型包 ==="
"$PY" "$HERE/build_hiai_pkg.py" --hf "$HF" --name $NAME --seq $SEQ --kv $KV \
      --omc "$OUT/omg/seg/seg.omc" --out "$OUT/pkg" | tail -3
cp "$OUT/omg/seg/SubGraph_0.weight" "$OUT/pkg/" 2>/dev/null || true
# ★omg 可能产出多个 SubGraph_N.weight ⇒ 全部拷过去 ✓★
cp "$OUT"/omg/seg/SubGraph_*.weight "$OUT/pkg/" 2>/dev/null || true
ls -l "$OUT/pkg" | awk '{printf "    %12.1f MB  %s\n", $5/1e6, $9}'
echo "=== 完成 ✓ 产物在 $OUT/pkg ==="
echo "    下一步：cd $OUT/pkg && ~/path/to/hiai_runner <omc名>   # 判据看 Init rc ✓"
