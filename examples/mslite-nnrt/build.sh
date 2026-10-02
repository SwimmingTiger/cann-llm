#!/bin/sh
# build.sh —— 在鸿蒙设备上编译 mslite_run。
#
# 设备自带 libmindspore_lite_ndk.so（/system/lib64/ndk/）与 SDK 头文件
# （native/sysroot/usr/include/mindspore/）。注意本机 /tmp 是只读的，
# 所以默认编译到本目录下的 bin/。
#
# 用法:
#   ./build.sh
#   SDK=/path/to/sysroot/usr/include OUT=/somewhere ./build.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SDK="${SDK:-}"

# 没显式给 SDK 时，从常见位置找一找
if [ -z "$SDK" ]; then
    for c in "$HOME"/ohos-sdk/*/native/sysroot/usr/include \
             "$HOME"/OpenHarmony/Sdk/*/native/sysroot/usr/include \
             "$HOME"/DevEco*/sdk/*/openharmony/native/sysroot/usr/include \
             /opt/ohos-sdk/*/native/sysroot/usr/include; do
        [ -d "$c/mindspore" ] && SDK="$c" && break
    done
fi
if [ -z "$SDK" ] || [ ! -d "$SDK/mindspore" ]; then
    echo "找不到 MindSpore NDK 头文件（<sysroot>/usr/include/mindspore）。" >&2
    echo "用 SDK=<sysroot>/usr/include 指定。" >&2
    exit 1
fi

OUT="${OUT:-$HERE/bin}"
mkdir -p "$OUT"
for src in mslite_run mlp_run; do
    echo "  cc $src.c  (SDK=$SDK)"
    cc -O1 -I"$SDK" "$HERE/$src.c" -o "$OUT/$src" \
       -L/system/lib64/ndk -lmindspore_lite_ndk -lm
done
echo "编译完成 -> $OUT/"
echo "运行示例: LD_LIBRARY_PATH=/system/lib64/ndk $OUT/mslite_run <model.ms> nnrt"
echo "          LD_LIBRARY_PATH=/system/lib64/ndk $OUT/mlp_run   <model.ms> nnrt"
