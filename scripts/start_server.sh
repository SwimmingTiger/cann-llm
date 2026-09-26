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
PY="${PYTHON:-python3}"
RUNDIR="${CANN_LLM_RUNDIR:-$ROOT/.run}"
PIDFILE="$RUNDIR/server.pid"
STATEFILE="$RUNDIR/server.state"       # 记录启动时的 host/port，供 --status/--stop 使用
LOGFILE="${CANN_LLM_LOG:-$RUNDIR/server.log}"

MODEL_DIR="${CANN_LLM_MODEL_DIR:-}"
CONFIG="${CANN_LLM_CONFIG:-}"
HOST="${CANN_LLM_HOST:-127.0.0.1}"
PORT="${CANN_LLM_PORT:-8000}"
API_KEY="${CANN_LLM_API_KEY:-}"
BACKEND="${CANN_LLM_BACKEND:-}"
BACKGROUND=0
WAIT_SECS=90

die()  { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m›\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!\033[0m %s\n' "$*" >&2; }

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

选项:
  -d, --model-dir DIR   模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json）
  -c, --config FILE     TOML 配置文件（examples/config.example.toml）
  -b, --backend NAME    后端（默认 cann）
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
ok "$("$PY" -c 'import sys; print("Python", sys.version.split()[0])')"

NDK_LIB="${CANN_LLM_LIB:-/system/lib64/ndk/libcann_llm_engine.so}"
if [[ "${BACKEND:-cann}" == "cann" || -z "$BACKEND" ]]; then
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
    info "前台启动（Ctrl-C 停止）"
    info "  http://$SHOWN:$PORT/v1/chat/completions"
    [[ -n "$API_KEY" ]] && info "  鉴权: Bearer <已设置>"
    echo
    exec "$PY" "${ARGS[@]}"
fi

info "后台启动…"
nohup "$PY" "${ARGS[@]}" >>"$LOGFILE" 2>&1 &
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
        info "OpenAI 兼容端点:"
        echo "      http://$SHOWN:$PORT/v1/models"
        echo "      http://$SHOWN:$PORT/v1/chat/completions"
        [[ -n "$API_KEY" ]] && echo "      鉴权: Authorization: Bearer <已设置>"
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
