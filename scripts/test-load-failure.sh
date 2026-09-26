#!/bin/bash
# 加载失败的复现脚本 —— 构造几种"能加载失败"的模型目录，跑后端看提示。
#
# 用法: bash scripts/test-load-failure.sh [模型目录（用来抄 api_config.json 等）]
#   · 默认从 models/qwen25_coder_7b_omc1024 抄；抄不到也能跑（会退化）。
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-}"
if [ -z "$SRC" ]; then
    for c in "$ROOT"/../models/*/ "$ROOT"/models/*/; do
        [ -f "${c}api_config.json" ] && { SRC="${c%/}"; break; }
    done
fi
PY="${PYTHON:-/data/service/hnp/bin/python3}"
WORK="${TMPDIR:-/storage/Users/currentUser/work/llm/.tmp}/load-fail-demo"
rm -rf "$WORK"; mkdir -p "$WORK"

echo "=============================================="
echo " 加载失败复现（工作目录 $WORK）"
echo " 抄素材自: ${SRC:-（无，功能会退化）}"
echo "=============================================="

mk() {  # mk <目录名> <executor.json 内容>
    local d="$WORK/$1"
    mkdir -p "$d"
    [ -n "${SRC:-}" ] && for f in api_config.json tokenizer.json; do
        [ -f "$SRC/$f" ] && cp "$SRC/$f" "$d/"
    done
    for f in "$SRC"/*.json; do
        b="$(basename "$f")"
        case "$b" in api_config.json|executor.json|context.json) ;; *)
            cp "$f" "$d/" 2>/dev/null ;; esac
    done
    for f in "$SRC"/*.omc "$SRC"/SubGraph_*; do
        [ -f "$f" ] && cp "$f" "$d/" 2>/dev/null
    done
    printf '%s\n' "$2" > "$d/executor.json"
    echo "$d"
}

run_one() {  # run_one <标题> <目录> [后端]
    local title="$1" d="$2" be="${3:-hiai}"
    echo
    echo "──────── $title  [-b $be] ────────"
    PYTHONPATH="$ROOT/src" timeout 120 "$PY" - "$d" "$be" <<'PYEOF' 2>&1 \
        | grep -avE "Unknown class|^avc:|could not determine" | head -12 | sed 's/^/  /'
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, __import__("os").environ.get("ROOT_SRC", ""))
d, be = sys.argv[1], sys.argv[2]
from cann_llm.backends import create_backend
try:
    create_backend(be, model_dir=d).load()
    print("（意料之外）加载成功")
except Exception as e:
    print(f"{type(e).__name__}:")
    for ln in str(e).splitlines()[:10]:
        print("    " + ln[:150])
PYEOF
}

# ① executor.json 的 llm_config 值坏了 → Executor 创建失败（会附日志）
D1=$(mk "01-坏-executor" '{"version":1,"engine_type":"autoregressive","llm_config":{"num_hidden_layers":"坏值"}}')
run_one "① executor.json 值坏了（Executor 创建失败）" "$D1" hiai

# ② 完全没有 api_config.json → 走"缺少打包配置"路径（提示怎么补）
D2="$WORK/02-缺-api-config"; mkdir -p "$D2"
printf '{"version":1,"engine_type":"autoregressive","llm_config":{}}\n' > "$D2/executor.json"
run_one "② 缺 api_config.json（提示如何补齐）" "$D2" hiai

# ③ 目录根本不存在
run_one "③ 目录不存在" "$WORK/不存在的目录" hiai

# ④ cann 后端看同样两个目录（cann 需要 context.json，会有自己的报错）
run_one "④ cann：executor.json 值坏了" "$D1" cann
run_one "⑤ cann：缺上下文配置" "$D2" cann

echo
echo "=============================================="
echo " 说明"
echo " · ①②③ 走的是 hiai 的加载失败路径 —— 其中①会附上原始 hilog"
echo " · ②③ 不需要日志：提示本身已经说明了缺什么、怎么补"
echo " · 读 hilog 需要当前终端在 log 组里；否则会打印'读不到引擎日志：…'"
echo "   用 id 看自己有没有 (log)"
echo "=============================================="
