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

die()  { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }

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

# 解析结果放进全局（不用子 shell，否则带不出警告信息）
#   PY       —— 选中的解释器
#   PY_WARN  —— 非空表示"没找到兼容的，回退了"，内容是要提示给用户的话
PY=""
PY_WARN=""

# 一个解释器为什么不能用（用户据此判断该装什么）
py_reject_reason() {   # $1 = 解释器路径
    if ! command -v "$1" >/dev/null 2>&1; then printf '找不到'; return; fi
    if py_libc_ok "$1"; then printf ''; return; fi
    printf 'glibc 构建（带 libmusl_compat 垫片），与按 musl 编译的引擎不兼容'
}

py_conflict_msg() {    # $1 = 选中的解释器；$2 = 已检查候选的说明
    cat <<EOF
$1 与 CANN NDK 库的 libc 不兼容，加载引擎时大概率会**直接段错误**（core dumped）。
    本机系统 libc 是 musl，libcann_llm_engine.so 按 musl 编译；而 glibc 构建的
    Python（靠 libmusl_compat 垫片运行）把 musl 版引擎加载进来就会崩。
$2
    → 请到 **应用市场** 安装「**Python安装器**」，装完重新运行本脚本；
      或用 PYTHON=/path/to/python3 指定一个可用的解释器。
EOF
}

resolve_python() {
    # 显式指定：照用，但兼容性问题要提示出来
    if [[ -n "${PYTHON:-}" ]]; then
        command -v "$PYTHON" >/dev/null 2>&1 \
            || die "PYTHON 指定的解释器不存在: $PYTHON"
        PY="$PYTHON"
        py_libc_ok "$PY" || PY_WARN="$(py_conflict_msg "$PY" "")"
        return 0
    fi

    # 候选：就走 PATH 查 python3 / python，但要把**所有**同名解释器都枚举出来
    # （type -a -p），而不是只取第一个 —— 否则把 hnp 目录追加到 PATH 末尾毫无
    # 意义（command -v 只看第一个）。全枚举才能：用上末尾那个可用的，并把前面
    # 那些为什么被跳过讲清楚。
    # 不猜目录布局、不做 glob：装了就在 PATH 里，就能被发现。
    # 需要额外位置时用 CANN_LLM_PYTHON_CANDIDATES="..." 覆盖。
    local candidates="${CANN_LLM_PYTHON_CANDIDATES:-python3 python}"

    local cand p first_seen="" tried="" seen=""
    local hits=()
    for cand in $candidates; do
        # 含 / 的当作显式路径；否则列出 PATH 里所有同名项（按 PATH 顺序）
        if [[ "$cand" == */* ]]; then
            hits=("$cand")
        else
            read -r -a hits <<< "$(type -a -p "$cand" 2>/dev/null | tr '\n' ' ')"
        fi
        for p in "${hits[@]:-}"; do
            [[ -n "$p" ]] || continue
            case "$seen" in *"|$p|"*) continue ;; esac   # python3 与 python 可能同一个
            seen+="|$p|"
            [[ -n "$first_seen" ]] || first_seen="$p"
            if py_libc_ok "$p"; then
                PY="$p"
                PY_SKIPPED="$tried"      # 解释"为什么跳过了前面那些"
                return 0
            fi
            tried+="      · $p：$(py_reject_reason "$p")"$'\n'
        done
    done

    # 一个兼容的都没有：**仍然回退**（免得用户什么都干不了），但把问题和安装
    # 建议说清楚 —— 真崩了用户也知道为什么、下一步该做什么。
    PY="${first_seen:-python3}"
    PY_WARN="$(py_conflict_msg "$PY" "    已检查过的候选：
${tried}")"
    return 0
}

# hnp（鸿蒙包服务）装的工具都落在 /data/service/hnp/bin，但那个目录**默认不在
# PATH 里**（用户常要自己 export）。这里主动补上 —— 但补在**末尾**，不动用户
# 原有的优先顺序：用户自己的 python3 仍然先被看到，只有它不兼容时才会用到
# hnp 里的。这样"为什么最终选了某个 python"才解释得清楚。
if [[ -d /data/service/hnp/bin ]]; then
    case ":$PATH:" in
        *:/data/service/hnp/bin:*) ;;
        *) PATH="$PATH:/data/service/hnp/bin"; export PATH ;;
    esac
fi

resolve_python

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

选项: 直接透传给 cann_llm.cli.chat，常用:
  -d, --model-dir DIR   模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json）
  -c, --config FILE     TOML 配置文件
  -b, --backend NAME    后端：hiai | cann（默认 hiai）
                        hiai = 系统内部引擎，更快、输出干净、支持停止序列（推荐）
                        cann = 官方 NDK 后端
  -t, --template NAME   对话模板（chatml / plain）
  -s, --system TEXT     system prompt
  -p, --prompt TEXT     单轮模式：生成一次后退出
      --temp F          采样温度（0 = 贪心）
      --topk N          top-k
      --topp F          top-p
      --rep F           重复惩罚
      --max-tokens N    单轮最大生成 token 数（--maxtok 亦可）
      --no-stream       关闭逐字输出
      --list-backends   列出可用后端与模板
  -h, --help            显示本帮助

环境变量: CANN_LLM_MODEL_DIR / CANN_LLM_BACKEND
          CANN_LLM_LIB（cann 后端） / CANN_LLM_HIAI_LIB（hiai 后端） / PYTHON
EOF
}

MODEL_DIR="${CANN_LLM_MODEL_DIR:-}"
BACKEND="${CANN_LLM_BACKEND:-hiai}"        # 与 CLI 默认一致
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -d|--model-dir) MODEL_DIR="${2:?}"; PASSTHRU+=("$1" "$2"); shift 2 ;;
        -b|--backend)   BACKEND="${2:?}";   PASSTHRU+=("$1" "$2"); shift 2 ;;
        --list-backends) PASSTHRU+=("$1"); shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

# 引擎库按【后端】选：两个后端用的是不同的库，检查/提示/导出都要对应
if [[ "$BACKEND" == "cann" ]]; then
    ENGINE_LIB="${CANN_LLM_LIB:-/system/lib64/ndk/libcann_llm_engine.so}"
    ENGINE_KIND="cann NDK 库"
else
    ENGINE_LIB="${CANN_LLM_HIAI_LIB:-/system/lib64/libhiai_llm_engine.so}"
    ENGINE_KIND="hiai 引擎"
fi

command -v "$PY" >/dev/null 2>&1 || die "找不到 $PY；可用 PYTHON=/path/to/python3 指定"

# NPU 访问权限预检：有些终端（如 MKCode 的）拿不到 /dev/npu* —— 早点说清楚。
# 只警告不阻断：设备节点可能因环境而异，真失败时后端还会再报一次更明确的错。
_npu="$(PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "from cann_llm.npucheck import npu_unavailable_reason as f; print(f() or '')" 2>/dev/null || true)"
if [[ -n "$_npu" ]]; then
    info "NPU 访问权限提示"
    printf '%s\n\n' "$_npu" >&2
fi


# 记录解释器选择结果：路径、版本，以及"为什么是它"
PY_PATH="$(command -v "$PY" 2>/dev/null || printf '%s' "$PY")"
PY_VER="$("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?')"

# 一律带 -X faulthandler：段错误时能打出 Python 栈，
# 而不是只有一句 "segmentation fault (core dumped)"。
PY_FLAGS=(-X faulthandler)

ok "Python $PY_VER  ·  $PY_PATH"
# 如果跳过了更靠前的候选，说明原因 —— 否则用户会奇怪"为什么不用我的 python3"
if [[ -n "${PY_SKIPPED:-}" ]]; then
    info "为什么不是更靠前的那个："
    printf '%s' "$PY_SKIPPED"
fi
# 没找到兼容解释器时已回退 —— 一定要把问题和安装建议说清楚
if [[ -n "$PY_WARN" ]]; then
    printf '\033[33m!\033[0m \033[33m解释器兼容性提示\033[0m\n' >&2
    printf '%s\n' "$PY_WARN" >&2
    printf '\n' >&2
fi

if [[ ! -f "$ENGINE_LIB" ]]; then
    die "找不到 $ENGINE_KIND $ENGINE_LIB
     后端 $BACKEND 需要鸿蒙设备（NPU）环境。"
fi
ok "$ENGINE_KIND: $ENGINE_LIB"

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
    # 上下文窗口 / 最大输出 —— 填别的工具（如 DSH）的模型配置时要用
    _mi="$("$PY" "$ROOT/scripts/model_info.py" "$MODEL_DIR" 2>/dev/null || true)"
    _win="$(printf '%s\n' "$_mi" | sed -n 's/^context_window=//p')"
    if [ -n "$_win" ]; then
        ok "上下文窗口: ${_win} token  (= kv_cache_max_len，输入 + 输出之和)"
        info "最大输出没有固定值：= 窗口 − 本次输入长度"
        info "改单轮上限：--max-tokens <n>   例：scripts/start_chat.sh -d … --max-tokens 512"
    fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# 只导出所选后端对应的库变量（另一个保持默认，不干扰）
if [[ "$BACKEND" == "cann" ]]; then
    export CANN_LLM_LIB="$ENGINE_LIB"
else
    export CANN_LLM_HIAI_LIB="$ENGINE_LIB"
fi

exec "$PY" "${PY_FLAGS[@]}" -m cann_llm.cli.chat "${PASSTHRU[@]}"
