#!/bin/sh
# build.sh —— 在鸿蒙设备上编译本目录下的 NNRt 探针。
#
# 设备自带的 SDK 里有 NNRt 头文件；运行时链接 /system/lib64/ndk 下的
# libneural_network_runtime.so / libneural_network_core.so。
#
# 注意：本机 /tmp 是只读的，所以默认编译到本目录下的 bin/（别编到 /tmp）。
#
# 用法:
#   ./build.sh
#   SDK=/path/to/sysroot/usr/include OUT=/somewhere ./build.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SDK="${SDK:-$HOME/work/hmos/ohos-sdk/ohos/native/sysroot/usr/include}"
OUT="${OUT:-$HERE/bin}"

if [ ! -d "$SDK/neural_network_runtime" ]; then
    echo "找不到 NNRt 头文件: $SDK/neural_network_runtime" >&2
    echo "用 SDK=<sysroot>/usr/include 指定 SDK 头文件目录。" >&2
    exit 1
fi

mkdir -p "$OUT"
for s in add_test block_test ops_probe offline_probe diag_spec_rule diag_spec_variant diag_dtype; do
    printf '  cc %s.c\n' "$s"
    cc -O1 -I"$SDK" "$HERE/$s.c" -o "$OUT/$s" \
        -L/system/lib64/ndk -lneural_network_runtime -lneural_network_core -lm
done

echo "编译完成 -> $OUT"
echo "运行示例: LD_LIBRARY_PATH=/system/lib64/ndk $OUT/add_test"
