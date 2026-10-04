#!/bin/sh
# 编排「gdbserver 起进程 + lldb 自动打大模型补丁」——由 `--large-mem` 调用。
#
#   scripts/large_mem_run.sh <python> [python 的参数…]
#
# 做三件事：
#   1. 用 huawei-debug-lldb-server 起被调试进程（它会先停住等调试器）；
#   2. 用 lldb 批处理接上去（gdb-remote），装断点（命中 OH_AI_ModelBuildFromFile
#      时把 4 处 2 GiB 上限检查改成 nop），然后放行；
#   3. 等进程跑完，把它的退出码原样返回。
#
# ★ 为什么必须绕这么一圈：本机 `lldb -- <bin>` 直接拉起进程会报
#   `error: 'A' packet returned an error: 8`（平台限制），只有这份 gdbserver 能调试 ✓
#
# ★ 退出码用【状态文件】传递（CANN_LLM_LARGE_MEM_STATUS_FILE）：
#   lldb 的 stdout 必须直通终端，对话/服务要流式输出，不能为了取退出码去捕获它 ✗
#
# 环境变量:
#   CANN_LLM_LLDB_SERVER   gdbserver 路径（默认 $ROOT/bin/… 或 PATH）
#   CANN_LLM_LLDB          lldb 路径（默认 $ROOT/bin/lldb 或 PATH）
#   CANN_LLM_LLDB_PORT     端口（默认 5092）
#   CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT  等 gdbserver 就绪的上限（默认 5 秒）
set -eu

SELF=$0
ROOT=$(CDPATH= cd -- "$(dirname -- "$SELF")/.." && pwd)

if [ "$#" -lt 1 ]; then
    printf '用法: %s <python> [参数…]\n' "$SELF" >&2
    exit 2
fi

PORT=${CANN_LLM_LLDB_PORT:-5092}
WAIT_MAX=${CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT:-5}

# 随包资产优先（与 start_*.sh 同一套规则；被直接调用也能自洽 ✓）
for _d in "$ROOT/bin" "$ROOT/python3/bin"; do
    [ -d "$_d" ] || continue
    case ":$PATH:" in *":$_d:"*) ;; *) PATH="$_d:$PATH" ;; esac
done
export PATH
_ld=""
for _d in "$ROOT/lib" "$ROOT/python3/lib"; do
    [ -d "$_d" ] || continue
    _ld="${_ld:+$_ld:}$_d"
done
if [ -n "$_ld" ]; then
    LD_LIBRARY_PATH="$_ld${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi
unset _d _ld

# lldb 的内嵌 python 要 import cann_llm.large_mem（补丁表的单一来源）⇒ 必须给 PYTHONPATH
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

GDB=${CANN_LLM_LLDB_SERVER:-}
if [ -z "$GDB" ]; then
    GDB="$ROOT/bin/huawei-debug-lldb-server"
    [ -x "$GDB" ] || GDB=$(command -v huawei-debug-lldb-server || true)
fi
LLDB=${CANN_LLM_LLDB:-}
if [ -z "$LLDB" ]; then
    LLDB="$ROOT/bin/lldb"
    [ -x "$LLDB" ] || LLDB=$(command -v lldb || true)
fi
[ -n "$GDB" ] && [ -x "$GDB" ] || { printf '✗ 找不到 huawei-debug-lldb-server（可用 CANN_LLM_LLDB_SERVER=… 指定）\n' >&2; exit 127; }
[ -n "$LLDB" ] && [ -x "$LLDB" ] || { printf '✗ 找不到 lldb（可用 CANN_LLM_LLDB=… 指定）\n' >&2; exit 127; }

SCRIPT="$ROOT/src/cann_llm/large_mem_lldb.py"
[ -f "$SCRIPT" ] || { printf '✗ 找不到 %s\n' "$SCRIPT" >&2; exit 127; }

# ★ 状态文件不能用 mktemp 的默认位置：本机 /tmp 是【只读】的 ✗
#   放仓库的 .run/（已 gitignore；launcher 也用这个目录放 pid/state）
RUNDIR=${CANN_LLM_RUNDIR:-$ROOT/.run}
STATUS=""
if mkdir -p "$RUNDIR" 2>/dev/null; then
    STATUS="$RUNDIR/large_mem_status.$$"
    : >"$STATUS" 2>/dev/null || STATUS=""
fi
if [ -n "$STATUS" ]; then
    CANN_LLM_LARGE_MEM_STATUS_FILE="$STATUS"
    export CANN_LLM_LARGE_MEM_STATUS_FILE
fi

# ── 1) gdbserver 起被调试进程（后台；它先把进程停住等调试器）──
printf '[large-mem] gdbserver 起进程（127.0.0.1:%s），随后 lldb 自动打补丁并放行…\n' "$PORT"
"$GDB" gdbserver --native-regs "127.0.0.1:$PORT" -- "$@" &
GDBPID=$!
trap 'kill "$GDBPID" 2>/dev/null || true; [ -n "$STATUS" ] && rm -f "$STATUS"' INT TERM HUP

# ── 2) lldb 批处理接入 ──
#   ★ 只能有一条"python 实现的命令"★：实测批处理模式下，python 命令之后的
#     `-o` 就不再执行了（`process continue` 会根本没跑、进程一直停着 ✗）
#     ⇒ 所以 `large_mem_run` 一条命令里做完「装断点 + 放行 + 等结束 + 取退出码」✓
#   连不上时重试：这时进程仍停在 gdbserver 里、什么都没发生，重试是安全的 ✓
i=0
LLDBRC=1
while [ "$i" -lt 40 ]; do
    i=$((i + 1))
    if "$LLDB" --batch \
        -o "gdb-remote 127.0.0.1:$PORT" \
        -o "command script import $SCRIPT" \
        -o "large_mem_run"; then
        LLDBRC=0
        break
    fi
    sleep 0.1
done
[ "$LLDBRC" -eq 0 ] || printf '[large-mem] ⚠ lldb 接入失败（已重试 %d 次）\n' "$i" >&2

# ── 3) 收尾：优先用状态文件里的退出码（被调试进程的），否则退回等 gdbserver ──
GDBRC=0
wait "$GDBPID" 2>/dev/null || GDBRC=$?
ST=""
if [ -n "$STATUS" ]; then
    ST=$(head -1 "$STATUS" 2>/dev/null || true)
    rm -f "$STATUS"
fi

if [ -n "$ST" ] && [ "$ST" -ge 0 ] 2>/dev/null; then
    exit "$ST"
fi
exit "$GDBRC"
