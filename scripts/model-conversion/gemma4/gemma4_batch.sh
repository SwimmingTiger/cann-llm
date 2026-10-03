#!/bin/bash
# 9 段（4×8 + 3）整条流水线：ONNX → OMG → converter_lite
DDK=$HOME/ddk
B=${MSLITE_BUILD:-/src/mindspore-src/source/output/tmp/mindspore-lite-2.7.0-linux-x64}  # 容器内路径，可用 MSLITE_BUILD 覆盖 ✓
POC=$HOME/work/llm/ddk-llm/llm-poc
TT=${TINY_TEST:-$HOME/tiny-test}
OUTROOT=$HOME/g4segs; mkdir -p "$OUTROOT"
export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$DDK/tools/tools_omg/master/lib64:$DDK/tools/platform/kirinx90/lib64
export PYTHONPATH=$DDK/tools/platform/kirinx90/ops/impl
for ST in 0 4 8 12 16 20 24 28 32; do
  D=$POC/g4seg$ST
  echo "=== 段 START=$ST $(date +%H:%M:%S) ==="
  rm -rf "$D"
  cd "${WORKDIR:-$PWD}" && START=$ST NO=4 ${PYTHON:-python3} gemma4_export_seg.py 2>&1 | grep -aE '导出|前向' | tail -2
  cd "$D" || { echo "  ✗ 无目录"; continue; }
  export TMPDIR="$D/tmp"; mkdir -p "$TMPDIR"
  SH=$(sed -n 1p omg.txt); TY=$(sed -n 2p omg.txt); OT=$(sed -n 3p omg.txt)
  rm -rf q && mkdir -p q
  timeout 500 "$DDK/tools/tools_omg/omg" --model seg.onnx --framework 5 --output q/seg \
    --input_shape="$SH" --input_type="$TY" --output_type="$OT" --weight_data_type FP16 \
    --platform=kirinx90 --target=omc > omg.log 2>&1
  OK=$(grep -ac 'OMG generate offline model success' omg.log)
  echo "  OMG: $OK"
  [ "$OK" = "0" ] && { echo "  ✗ OMG 失败，停"; grep -aE '^E/' omg.log | tail -3; break; }
  T=$TT/g4seg$ST; mkdir -p "$T"
  cp q/*.omc "$T/" 2>/dev/null; W=$(dirname $(ls q/*.omc|head -1)); [ -f "$W/SubGraph_0.weight" ] && cp "$W/SubGraph_0.weight" "$T/"
  cp c.cfg "$T/"
  chmod -R a+rX "$T"
  docker exec mslite-dev bash -c "
    export LD_LIBRARY_PATH=$B/tools/converter/lib:$B/runtime/lib:$B/runtime/third_party/glog
    cd /src/tiny-test/g4seg$ST; rm -f seg.ms; OMC=\$(ls *.omc|head -1)
    $B/tools/converter/converter/converter_lite --fmk=THIRDPARTY --modelFile=\$OMC --outputFile=seg --configFile=c.cfg 2>&1 | tail -1
    chmod 644 seg.ms 2>/dev/null" 2>&1 | tail -1 | sed 's/^/  /'
  if [ -f "$T/seg.ms" ]; then
    mkdir -p "$OUTROOT/seg$ST"; cp "$T/seg.ms" "$OUTROOT/seg$ST/"
    echo "  ✓ seg$ST/seg.ms $(du -h "$T/seg.ms"|cut -f1)"
  else
    echo "  ✗ 段 $ST 转换失败"
  fi
done
echo "=== 批处理结束 $(date +%H:%M:%S) ==="
du -sh "$OUTROOT" 2>/dev/null; ls "$OUTROOT"
