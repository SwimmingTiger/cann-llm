#!/bin/sh
# 把 dopt 的量化结果交给 OMG（带 --compress_conf）⇒ .omc ⇒ converter ⇒ .ms
#
# 注意：第三方模型需要 [third_party_model] 配置（见 scripts/model-conversion/gemma4/README.md）✓
#
# 用法：
#   OMG=<.../tools_omg/omg> CONVERTER=<.../converter_lite> MODEL=quant.onnx \
#   COMPRESS=compress.json OMGTXT=omg.txt CFG=c.cfg OUT=seg ./build_with_omg.sh
set -e
: "${OMG:?}"; : "${CONVERTER:?}"; : "${MODEL:?}"; : "${COMPRESS:?}"; : "${OMGTXT:?}"; : "${CFG:?}"; : "${OUT:?}"
SH=$(sed -n 1p "$OMGTXT"); TY=$(sed -n 2p "$OMGTXT"); OT=$(sed -n 3p "$OMGTXT")
rm -rf q && mkdir -p q
"$OMG" --model "$MODEL" --framework 5 --output "q/$OUT" \
  --input_shape="$SH" --input_type="$TY" --output_type="$OT" \
  --weight_data_type FP16 --compress_conf "$COMPRESS" \
  --platform=kirinx90 --target=omc
# OMG 失败时先看 check_result.json（逐算子 pass/fail ✓）
[ -f check_result.json ] && echo "  check_result.json 已生成（失败时看它 ✓）"
OMC=$(ls q/*.omc 2>/dev/null | head -1)
[ -z "$OMC" ] && { echo "  ✗ 没产出 .omc"; exit 1; }
"$CONVERTER" --fmk=THIRDPARTY --modelFile="$OMC" --outputFile="$OUT" --configFile="$CFG"
echo "  产出：$OUT.ms"
