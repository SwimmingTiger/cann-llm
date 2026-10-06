#!/bin/sh
# 给**随包 python** 装 numpy（鸿蒙 / OHOS 平台专用一步 ✓）。
#
#   sh scripts/install_numpy.sh
#   PYTHON=/path/to/python3 sh scripts/install_numpy.sh      # 指定别的解释器
#
# 为什么需要这一步（§164 实测 ✓）：
#   随包解释器是 aarch64-linux-ohos 平台（★musl★ ✓，PT_INTERP=/lib/ld-musl-aarch64.so.1 ✓），
#   而 PyPI 上【没有】ohos 标签的轮子 ✗ ⇒ pip 会选 musllinux_1_2_aarch64 轮子 ✓，
#   其扩展模块名是 `_multiarray_umath.cpython-314-aarch64-linux-musl.so` ✗，
#   而这个解释器接受的后缀是 `['.cpython-314.so', '.abi3.so', '.so']` ✓
#   ⇒ 名字对不上 ⇒ 导入时报 `No module named 'numpy._core._multiarray_umath'` ✗
#   （libc 本身同为 musl ✓，二进制没问题 ✓ —— 纯粹是【文件名平台标签】问题 ✓）
#   ⇒ 把扩展名里的 `-aarch64-linux-musl` 一段去掉即可 ✓。
#
# 本脚本可重复执行 ✓（已装过则只补改名 ✓）。
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="${PYTHON:-$ROOT/python3/bin/python3}"

if [ ! -x "$PY" ]; then
    printf '找不到解释器：%s\n（纯源码树里没有 python3/ —— 用 PYTHON=/path/to/python3 指定 ✓）\n' "$PY" >&2
    exit 1
fi

printf '== 解释器：%s\n' "$PY"
"$PY" -c 'import sysconfig; print("   平台：%s · 扩展后缀：%s" % (sysconfig.get_platform(), __import__("_imp").extension_suffixes()))'

printf '== pip install numpy（只取二进制轮子 ✓ 不编译 ✓）\n'
"$PY" -m pip install --only-binary=:all: numpy

SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
BASE=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')
BASE=${BASE%.so}                       # 例如 .cpython-314

printf '== 把扩展名里的平台标签去掉（*%s-<platform>.so → *%s.so ✓）\n' "$BASE" "$BASE"
moved=0
for f in "$SP"/numpy/*.so "$SP"/numpy/*/*.so "$SP"/numpy/*/*/*.so; do
    [ -f "$f" ] || continue
    case "$f" in
        *"$BASE".so) ;;                                    # 已经没标签 ✓ 跳过
        *"$BASE"-*.so)
            nf="${f%"$BASE"-*.so}$BASE.so"
            mv -f "$f" "$nf"
            moved=$((moved + 1))
            ;;
    esac
done
printf '   改名 %d 个 ✓\n' "$moved"

printf '== 验证\n'
"$PY" -c '
import numpy as np
print("   numpy %s ✓" % np.__version__)
a = np.ones((2, 3), dtype=np.float16)
b = np.ones((3, 1), dtype=np.float32)
print("   矩阵乘自检：%s ✓" % (np.asarray(a, dtype=np.float32) @ b).ravel())
'
