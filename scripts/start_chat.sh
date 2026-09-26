#!/usr/bin/env bash
# 一键启动交互式对话（逐字流式输出）。
#
#   scripts/start_chat.sh -d /path/to/model_dir
#   scripts/start_chat.sh -d /path/to/model_dir --temp 0 --topk 1
#   scripts/start_chat.sh -d /path/to/model_dir -p "你好"      # 单轮模式
#
# 模型目录也可以从环境变量 CANN_LLM_MODEL_DIR 取。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 解释器选择：显式给了 PYTHON 就用它；否则在候选里挑第一个与 CANN NDK 库
# **libc 兼容**的。本机系统 libc 是 musl、引擎按 musl 编；某些第三方 Python
# （如 harmonybrew 的）是 glibc 构建 + libmusl_compat 垫片，加载引擎会段错误。
py_libc_ok() {   # $1 = 解释器路径
    "$1" - <<'PYEOF' 2>/dev/null
import sys
try:
    maps = open("/proc/self/maps").read()
except OSError:
    raise SystemExit(0)          # 判不了就别拦
raise SystemExit(1 if "libmusl_compat" in maps else 0)
PYEOF
}

resolve_python() {
    if [[ -n "${PYTHON:-}" ]]; then printf '%s' "$PYTHON"; return; fi
    local cand
    for cand in python3 /data/service/hnp/bin/python3; do
        if command -v "$cand" >/dev/null 2>&1 && py_libc_ok "$cand"; then
            printf '%s' "$cand"; return
        fi
    done
    printf '%s' python3          # 都不兼容就原样返回，交给后端报清楚
}

PY="$(resolve_python)"
NDK_LIB="${CANN_LLM_LIB:-/system/lib64/ndk/libcann_llm_engine.so}"

die()  { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

选项: 直接透传给 cann_llm.cli.chat，常用:
  -d, --model-dir DIR   模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json）
  -c, --config FILE     TOML 配置文件
  -b, --backend NAME    后端（默认 cann）
  -t, --template NAME   对话模板（chatml / plain）
  -s, --system TEXT     system prompt
  -p, --prompt TEXT     单轮模式：生成一次后退出
      --temp F          采样温度（0 = 贪心）
      --topk N          top-k
      --topp F          top-p
      --rep F           重复惩罚
      --maxtok N        单轮最大生成 token 数
      --no-stream       关闭逐字输出
      --list-backends   列出可用后端与模板
  -h, --help            显示本帮助

环境变量: CANN_LLM_MODEL_DIR / CANN_LLM_LIB / PYTHON
EOF
}

MODEL_DIR="${CANN_LLM_MODEL_DIR:-}"
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -d|--model-dir) MODEL_DIR="${2:?}"; PASSTHRU+=("$1" "$2"); shift 2 ;;
        --list-backends) PASSTHRU+=("$1"); shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

command -v "$PY" >/dev/null 2>&1 || die "找不到 $PY；可用 PYTHON=/path/to/python3 指定"

if [[ ! -f "$NDK_LIB" ]]; then
    die "找不到 CANN NDK 库 $NDK_LIB
     本后端需要鸿蒙设备（NPU）环境。"
fi
ok "CANN NDK 库: $NDK_LIB"

if [[ -z "$MODEL_DIR" ]]; then
    for cand in "$ROOT"/models/*/ "$ROOT"/../models/*/; do
        [[ -f "${cand}executor.json" ]] && { MODEL_DIR="${cand%/}"; break; }
    done
fi

if [[ -n "$MODEL_DIR" && ! -d "$MODEL_DIR" ]]; then
    die "模型目录不存在: $MODEL_DIR"
fi
if [[ -z "$MODEL_DIR" ]]; then
    info "未指定模型目录，将交由程序提示。用 -d 指定，或把模型放到 $ROOT/models/<名字>/ 下。"
else
    ok "模型目录: $MODEL_DIR"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CANN_LLM_LIB="$NDK_LIB"

exec "$PY" -m cann_llm.cli.chat "${PASSTHRU[@]}"
