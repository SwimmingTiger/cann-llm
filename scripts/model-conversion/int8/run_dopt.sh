#!/bin/sh
# dopt 量化：ONNX + 校准数据 ⇒ 量化 ONNX + ★compress_conf（给 OMG 用）★
#
# ★两个环境坑★：
#  ① dopt 的 .so 是 ★python3.10★ 编的 ⇒ 别的版本会报 `undefined symbol: _PyUnicode_Ready`
#  ② protobuf 太新会报 `Descriptors cannot be created directly`
#     ⇒ 设 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python（或降到 protobuf ≤3.20）
#
# 用法：
#   DOPT_PY=<python3.10> DOPT_DIR=<.../dopt_onnx_py3> \
#   MODEL=model.onnx CAL_CONF=config.prototxt OUT=quant.onnx CONF=compress.json \
#     ./run_dopt.sh "hidden:1,128,1536;mask3:1,128,128" "hidden_out"
set -e
: "${DOPT_PY:?需要 DOPT_PY（python3.10 解释器）}"
: "${DOPT_DIR:?需要 DOPT_DIR（dopt_onnx_py3 目录）}"
: "${MODEL:?}"; : "${CAL_CONF:?}"; : "${OUT:?}"; : "${CONF:?}"
INS="$1"; OUTS="$2"

cd "$DOPT_DIR"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$DOPT_PY" dopt_so.py \
  --mode 0 --framework 5 \
  --model "$MODEL" --cal_conf "$CAL_CONF" \
  --input_shape "$INS" --out_nodes "$OUTS" \
  --output "$OUT" --compress_conf "$CONF"
echo "  产出：$OUT · $CONF"
