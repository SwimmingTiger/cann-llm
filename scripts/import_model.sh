#!/bin/sh
# 导入官方 OMC 模型包 —— 把下载到的 zip 变成能直接启动的模型目录。
#
#   scripts/import_model.sh <包.zip>                    # 解压到同名目录并转换
#   scripts/import_model.sh <包.zip> -d models/qwen7b   # 指定目标目录
#   scripts/import_model.sh <已解压目录>                 # 目录也能直接转换
#   scripts/import_model.sh <包.zip> --dry-run           # 只看会生成什么，不写盘
#
# 下载官方模型包的网页点击步骤见 docs/get-models.md。
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

die()  { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!\033[0m %s\n' "$*" >&2; }

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

选项:
  -d, --dest DIR    解压/转换的目标目录（.zip 时默认与包同名的目录）
      --dry-run     只解析并打印，不写文件
  -h, --help        显示本帮助

环境变量:
  PYTHON            指定用哪个解释器（默认自动挑一个 ≥3.9 的）

说明:
  本脚本只是 scripts/import_omc_package.py 的包装：导入过程只用标准库
  （zipfile / json），不加载引擎，因此任何 Python ≥3.9 都行 —— 不要求
  设备上那个 musl 解释器（那是跑推理时才需要的）。
EOF
}

SRC=""; DEST=""; DRY=""
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--dest)   DEST="${2:?--dest 需要一个目录}"; shift 2 ;;
    --dry-run)   DRY="--dry-run"; shift ;;
    -h|--help)   usage; exit 0 ;;
    -*)          die "未知选项: $1（用 -h 看用法）" ;;
    *)           [ -z "$SRC" ] || die "只能给一个包/目录（已经给了: $SRC）"; SRC="$1"; shift ;;
  esac
done
[ -n "$SRC" ] || { usage >&2; die "缺少参数：模型包（.zip）或已解压的目录"; }

# ---------------------------------------------------------------- 解释器
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python; do
    p=$(command -v "$c" 2>/dev/null || true)
    [ -n "$p" ] || continue
    "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null && { PY="$p"; break; }
  done
fi
[ -n "$PY" ] || die "找不到 Python ≥3.9。可用 PYTHON=/path/to/python3 指定。"
ok "Python: $("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')  ·  $PY"

# ---------------------------------------------------------------- 输入检查
[ -e "$SRC" ] || die "找不到: $SRC"

if [ -f "$SRC" ]; then
  case "$SRC" in
    *.zip|*.ZIP) : ;;
    *)
      # 实测：从 Matrix 模型库点下载按钮保存下来的文件常常没有扩展名，
      # 文件名是一长串十六进制。导入脚本按 .zip 判断，缺后缀时会把文件
      # 当成目录而报「不是目录」——那种报错不是文件坏了。
      die "这个文件没有 .zip 后缀: $(basename "$SRC")
      从 matrix.openharmony.cn 下载的包有时就是没有扩展名的（文件名是一串十六进制）。
      请先补上后缀再运行，例如：
        mv '$(basename "$SRC")' Qwen25-Coder-7B-Instruct-OMC-20251024.zip
        $0 Qwen25-Coder-7B-Instruct-OMC-20251024.zip${DEST:+ -d $DEST}" ;;
  esac
  info "模型包: $SRC  ($(du -h "$SRC" 2>/dev/null | cut -f1))"
  if [ -z "$DEST" ]; then
    DEST="${SRC%.*}"
    command -v unzip >/dev/null 2>&1 || true
  fi
else
  info "已解压目录: $SRC"
fi

# ---------------------------------------------------------------- 转换
# POSIX sh 没有数组：直接用位置参数累积，最后原样传给 Python。
set -- "$SRC"
[ -n "$DEST" ] && set -- "$@" --dest "$DEST"
[ -n "$DRY" ]  && set -- "$@" "$DRY"

info "运行: import_omc_package.py $*"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$ROOT/scripts/import_omc_package.py" "$@"

