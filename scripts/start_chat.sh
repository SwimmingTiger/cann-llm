#!/bin/sh
# 一键启动交互式对话（逐字流式输出）。
#
#   scripts/start_chat.sh -d /path/to/model_dir
#   scripts/start_chat.sh -d /path/to/model_dir --temp 0 --topk 1
#   scripts/start_chat.sh -d /path/to/model_dir -p "你好"      # 单轮模式
#
# 本脚本只做一件事：**找到一个与引擎 libc 兼容的 python，然后把参数原样转发**。
# 其余逻辑（引擎库选择、模型目录探测、预检提示）都在 cann_llm.launcher 里 ——
# 这样这里永远是纯 POSIX sh，不需要 bash。
#
# 环境变量:
#   PYTHON                       指定解释器（跳过自动发现）
#   CANN_LLM_PYTHON_CANDIDATES    候选解释器列表，默认 "python3 python"
#   CANN_LLM_MODEL_DIR           模型目录（等同 -d）
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# hnp（鸿蒙包服务）装的工具在 /data/service/hnp/bin，默认不在 PATH 里。
# 补在**末尾**：用户自己的 python3 仍优先，只有它不兼容时才用到这个。
if [ -d /data/service/hnp/bin ]; then
    case ":$PATH:" in
        *:/data/service/hnp/bin:*) ;;
        *) PATH="$PATH:/data/service/hnp/bin"; export PATH ;;
    esac
fi

# 与引擎 libc 是否兼容：glibc 构建的 python（靠 libmusl_compat 垫片）会段错误
libc_ok() {
    "$1" -c 'import sys
try:
    maps = open("/proc/self/maps").read()
except OSError:
    raise SystemExit(0)
raise SystemExit(1 if "libmusl_compat" in maps else 0)' 2>/dev/null
}

# 找解释器：显式 PYTHON 优先；否则按候选顺序取第一个存在的，再逐个验 libc
PY=""
if [ -n "${PYTHON:-}" ]; then
    command -v "$PYTHON" >/dev/null 2>&1 || {
        printf '\033[31m错误\033[0m PYTHON 指定的解释器不存在: %s\n' "$PYTHON" >&2
        exit 1
    }
    PY="$PYTHON"
else
    first=""
    for cand in ${CANN_LLM_PYTHON_CANDIDATES:-python3 python}; do
        full=$(command -v "$cand" 2>/dev/null || true)
        [ -n "$full" ] || continue
        [ -n "$first" ] || first="$full"
        if libc_ok "$full"; then
            PY="$full"
            break
        fi
        printf '\033[33m!\033[0m 跳过 %s：glibc 构建（带 libmusl_compat 垫片），与按 musl 编译的引擎不兼容\n' "$full" >&2
    done
    [ -n "$PY" ] || PY="${first:-python3}"
fi

command -v "$PY" >/dev/null 2>&1 || {
    printf '\033[31m错误\033[0m 找不到 %s；可用 PYTHON=/path/to/python3 指定\n' "$PY" >&2
    exit 1
}

export CANN_LLM_RESOLVED_PYTHON="$PY"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
exec "$PY" -X faulthandler -m cann_llm.launcher chat "$@"
