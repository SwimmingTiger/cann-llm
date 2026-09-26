#!/usr/bin/env bash
# 一键启动 OpenAI 兼容推理服务。
#
#   scripts/start_server.sh -d /path/to/model_dir              # 前台
#   scripts/start_server.sh -d /path/to/model_dir -b           # 后台 + 等就绪
#   scripts/start_server.sh --status                           # 看运行状态
#   scripts/start_server.sh --stop                             # 停止后台服务
#
# 模型目录也可以从环境变量 CANN_LLM_MODEL_DIR 或配置文件（-c）里取。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# die/ok/info/warn 要在 resolve_python 之前定义好（它会用 die）
die()  { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!\033[0m %s\n' "$*" >&2; }

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
# PATH 里**（用户常要自己 export）。这里主动补上 —— 但补在**末尾**，不动用户原有
# 的优先顺序：用户自己的 python3 仍先被看到，只有它不兼容时才用到 hnp 里的。
# 这样"为什么最终选了某个 python"才解释得清楚。
if [[ -d /data/service/hnp/bin ]]; then
    case ":$PATH:" in
        *:/data/service/hnp/bin:*) ;;
        *) PATH="$PATH:/data/service/hnp/bin"; export PATH ;;
    esac
fi

resolve_python
RUNDIR="${CANN_LLM_RUNDIR:-$ROOT/.run}"
PIDFILE="$RUNDIR/server.pid"
STATEFILE="$RUNDIR/server.state"       # 记录启动时的 host/port，供 --status/--stop 使用
LOGFILE="${CANN_LLM_LOG:-$RUNDIR/server.log}"

MODEL_DIR="${CANN_LLM_MODEL_DIR:-}"
CONFIG="${CANN_LLM_CONFIG:-}"
HOST="${CANN_LLM_HOST:-127.0.0.1}"
PORT="${CANN_LLM_PORT:-8000}"
API_KEY="${CANN_LLM_API_KEY:-}"
BACKEND="${CANN_LLM_BACKEND:-hiai}"
BACKGROUND=0
WAIT_SECS=90

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

选项:
  -d, --model-dir DIR   模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json）
  -c, --config FILE     TOML 配置文件（examples/config.example.toml）
  -b, --backend NAME    后端：hiai | cann（默认 hiai）
                        hiai = 系统内部引擎，更快、输出干净、支持停止序列（推荐）
                        cann = 官方 NDK 后端
      --host HOST       监听地址（默认 127.0.0.1）
  -p, --port PORT       监听端口（默认 8000）
  -k, --api-key KEY     要求 Authorization: Bearer <KEY>
  -B, --background      后台运行，并等到服务就绪后才返回
      --wait SECS       后台启动的等待上限（默认 90）
      --status          只看运行状态后退出
      --stop            停止后台服务后退出
  -h, --help            显示本帮助

环境变量: CANN_LLM_MODEL_DIR / CANN_LLM_PORT / CANN_LLM_API_KEY /
          CANN_LLM_CONFIG / CANN_LLM_BACKEND / CANN_LLM_LOG / PYTHON
EOF
}

# ---------------------------------------------------------------- 参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--model-dir) MODEL_DIR="${2:?}"; shift 2 ;;
        -c|--config)    CONFIG="${2:?}";    shift 2 ;;
        -b|--backend)   BACKEND="${2:?}";   shift 2 ;;
        --host)         HOST="${2:?}";      shift 2 ;;
        -p|--port)      PORT="${2:?}";      shift 2 ;;
        -k|--api-key)   API_KEY="${2:?}";   shift 2 ;;
        -B|--background) BACKGROUND=1;      shift ;;
        --wait)         WAIT_SECS="${2:?}"; shift 2 ;;
        --status)       ACTION=status;      shift ;;
        --stop)         ACTION=stop;        shift ;;
        -h|--help)      usage; exit 0 ;;
        *) die "未知参数: $1（用 -h 看帮助）" ;;
    esac
done
ACTION="${ACTION:-run}"

# ---------------------------------------------------------------- 小工具
# 用 python3 做健康检查，避免依赖 curl
probe() {
    "$PY" - "$HOST" "$PORT" <<'PY' 2>/dev/null
import json, sys, urllib.request
host, port = sys.argv[1], sys.argv[2]
url = f"http://{host if host not in ('0.0.0.0','') else '127.0.0.1'}:{port}/healthz"
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        d = json.load(r)
    print(f"{d.get('status')} model={d.get('model')} uptime={d.get('uptime_s')}s")
except Exception:
    sys.exit(1)
PY
}

alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

lan_ip() {
    "$PY" - <<'LANPY' 2>/dev/null
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
except Exception:
    pass
finally:
    s.close()
LANPY
}

# 打印访问地址。**base_url 只到 /v1**，端点单独列，
# 避免把端点路径误当成 base_url（客户端会再往后拼路径）。
print_endpoints() {
    local shown="$1" port="$2"
    local base="http://$shown:$port/v1"
    info "base_url（OpenAI 客户端填这个）:"
    echo "      $base"
    if [[ "$HOST" == "0.0.0.0" || -z "$HOST" ]]; then
        local ip; ip="$(lan_ip)"
        [[ -n "$ip" ]] && echo "      http://$ip:$port/v1      # 局域网其它机器访问"
    fi
    echo
    info "端点:"
    echo "      GET  $base/models"
    echo "      POST $base/chat/completions"
    echo "      POST $base/completions"
    echo "      GET  http://$shown:$port/healthz            # 健康检查，无需鉴权"
    echo "      GET  http://$shown:$port/                   # 端点索引（浏览器可直接打开）"
    if [[ -n "$API_KEY" ]]; then
        echo
        info "鉴权: Authorization: Bearer <已设置>"
    fi
}

load_state() {
    # 从状态文件恢复 host/port（用户不必在 --status 时再敲一遍）
    [[ -f "$STATEFILE" ]] || return 0
    local k v
    while IFS='=' read -r k v; do
        case "$k" in
            host) HOST="$v" ;;
            port) PORT="$v" ;;
        esac
    done <"$STATEFILE"
}

do_stop() {
    if ! alive; then
        warn "没有在运行的后台服务（状态文件: $STATEFILE）"
        rm -f "$PIDFILE" "$STATEFILE"
        return 0
    fi
    pid="$(cat "$PIDFILE")"
    info "停止服务 (pid $pid)…"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -0 "$pid" 2>/dev/null && { warn "未退出，强制结束"; kill -9 "$pid" 2>/dev/null || true; }
    rm -f "$PIDFILE" "$STATEFILE"
    ok "已停止"
}

do_status() {
    load_state
    if alive; then
        ok "运行中 (pid $(cat "$PIDFILE"))"
    else
        warn "未在运行"
        # 没有在跑的后台服务就别去探默认端口，免得误导
        [[ -f "$STATEFILE" ]] || return 1
    fi
    if out="$(probe)"; then
        ok "健康检查通过: $out"
        return 0
    fi
    warn "健康检查不通（http://$HOST:$PORT/healthz）"
    return 1
}

[[ "$ACTION" == stop   ]] && { do_stop;   exit 0; }
if [[ "$ACTION" == status ]]; then do_status; exit $?; fi

# ---------------------------------------------------------------- 预检
info "检查运行环境…"
command -v "$PY" >/dev/null 2>&1 || die "找不到 $PY；可用 PYTHON=/path/to/python3 指定"

"$PY" - <<'PY' || die "Python 版本过低（需要 >= 3.9，因为用到了 tomllib）"
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY

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
    warn "解释器兼容性提示"
    printf '%s\n\n' "$PY_WARN" >&2
fi

NDK_LIB="${CANN_LLM_LIB:-/system/lib64/ndk/libcann_llm_engine.so}"
# 只有【显式】指定 cann 时才强制要求 NDK 库；auto 未定时不假定后端
if [[ "$BACKEND" == "cann" ]]; then
    if [[ -f "$NDK_LIB" ]]; then
        ok "CANN NDK 库: $NDK_LIB"
    else
        die "找不到 CANN NDK 库 $NDK_LIB
     这个后端需要鸿蒙设备（NPU）环境；在别的机器上请用 -b 指定其它后端。"
    fi
fi

# 模型目录：参数 > 环境变量 > 配置文件 > 自动探测
if [[ -z "$MODEL_DIR" ]]; then
    for cand in "$ROOT"/models/*/ "$ROOT"/../models/*/; do
        [[ -f "${cand}executor.json" ]] && { MODEL_DIR="${cand%/}"; break; }
    done
fi
if [[ -z "$MODEL_DIR" ]]; then
    die "未指定模型目录。用 -d/--model-dir 指定，或设 CANN_LLM_MODEL_DIR，
     或把模型放到 $ROOT/models/<名字>/ 下（需含 executor.json）。"
fi
MODEL_DIR="${MODEL_DIR%/}"
[[ -d "$MODEL_DIR" ]] || die "模型目录不存在: $MODEL_DIR"

missing=()
for f in executor.json context.json tokenizer.json; do
    [[ -f "$MODEL_DIR/$f" ]] || missing+=("$f")
done
shopt -s nullglob
omc=("$MODEL_DIR"/*.omc)
weights=("$MODEL_DIR"/SubGraph_*.weight)
shopt -u nullglob
[[ ${#omc[@]}    -gt 0 ]] || missing+=("*.omc")
[[ ${#weights[@]} -gt 0 ]] || missing+=("SubGraph_*.weight")
if [[ ${#missing[@]} -gt 0 ]]; then
    die "模型目录不完整: $MODEL_DIR
     缺少: ${missing[*]}
     （模型产物怎么来的见 docs/cann-engine-notes.md）"
fi
ok "模型目录: $MODEL_DIR"
ok "模型文件: $(basename "${omc[0]}") + $(basename "${weights[0]}")"

# 上下文窗口 / 最大输出 —— 填别的工具（如 DSH）的模型配置时要用
_mi="$("$PY" "$ROOT/scripts/model_info.py" "$MODEL_DIR" 2>/dev/null || true)"
_win="$(printf '%s\n' "$_mi" | sed -n 's/^context_window=//p')"
if [ -n "$_win" ]; then
    ok "上下文窗口: ${_win} token  (= kv_cache_max_len，输入 + 输出之和)"
    info "最大输出没有固定值：= 窗口 − 本次输入长度"
    info "改默认输出窗口：--max-tokens <n>   例：scripts/start_server.sh -d … --max-tokens 512"
    info "（请求体里的 max_tokens 优先生效）"
fi

# 端口占用
if "$PY" - "$HOST" "$PORT" <<'PY'
import socket, sys
h, p = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(1)
raise SystemExit(0 if s.connect_ex((h if h not in ("0.0.0.0", "") else "127.0.0.1", p)) == 0 else 1)
PY
then
    die "端口 $PORT 已被占用（服务已在跑？用 --status 看状态）"
fi

# ---------------------------------------------------------------- 启动
mkdir -p "$RUNDIR"
ARGS=(-m cann_llm.api.server -d "$MODEL_DIR" --host "$HOST" --port "$PORT")
[[ -n "$CONFIG"  ]] && ARGS+=(-c "$CONFIG")
[[ -n "$BACKEND" ]] && ARGS+=(-b "$BACKEND")
[[ -n "$API_KEY" ]] && ARGS+=(-k "$API_KEY")

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CANN_LLM_LIB="$NDK_LIB"

SHOWN="$HOST"; [[ "$HOST" == "0.0.0.0" || -z "$HOST" ]] && SHOWN="127.0.0.1"

echo
if [[ $BACKGROUND -eq 0 ]]; then
    rm -f "$PIDFILE" "$STATEFILE"
    info "前台启动（Ctrl-C 停止），正在加载模型…"
    echo
    # 这里**不**打印监听地址与端点：此刻还没 bind，打印出来是"提前宣称"。
    # serve() 会在 LlmHttpServer 绑定成功之后自己打印一次完整横幅
    # （cann-llm x.y.z · <模型> / 监听 / base_url / 端点 / 鉴权），
    # 在这里再打印一遍就会重复，所以前台分支完全交给它。
    exec "$PY" "${PY_FLAGS[@]}" "${ARGS[@]}"
fi

info "后台启动…"
nohup "$PY" "${PY_FLAGS[@]}" "${ARGS[@]}" >>"$LOGFILE" 2>&1 &
pid=$!
echo "$pid" >"$PIDFILE"
{ echo "pid=$pid"; echo "host=$HOST"; echo "port=$PORT"; echo "model_dir=$MODEL_DIR"; } >"$STATEFILE"

printf '  等待服务就绪'
for _ in $(seq 1 "$WAIT_SECS"); do
    if out="$(probe)"; then
        echo
        ok "已就绪 (pid $pid)"
        ok "$out"
        echo
        print_endpoints "$SHOWN" "$PORT"
        echo
        info "日志: $LOGFILE"
        info "停止: $0 --stop    状态: $0 --status"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo
        rm -f "$PIDFILE"
        warn "服务进程已退出，日志尾部:"
        tail -n 30 "$LOGFILE" >&2
        exit 1
    fi
    printf '.'
    sleep 1
done

echo
warn "等待超过 ${WAIT_SECS}s 仍未就绪，日志尾部:"
tail -n 20 "$LOGFILE" >&2
warn "服务可能仍在启动（模型加载较慢），可用 $0 --status 复查"
exit 1
