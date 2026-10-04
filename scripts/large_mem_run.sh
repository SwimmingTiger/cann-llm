#!/bin/sh
# 编排「--large-mem」：**让程序正常启动、保住它自己的终端**，再用 gdbserver 附着上去
# 在运行时打「大模型补丁」，补完立刻 detach，程序继续原样运行。
#
#   scripts/large_mem_run.sh <python> <module> [模块参数…]
#
# ★为什么不用 `gdbserver -- prog`（由调试器拉起进程）★
#   那样 lldb-server 会给被调试进程【另开一个 pty】：实测程序 fd0/fd1 指向
#   /dev/pts/N 而不是当前终端 —— 程序的**输出**靠 lldb 转发还能看见，但
#   **用户的键盘输入永远进不去** ⇒ 交互式程序（对话）会卡死在提示符 ✗
#   （`-p "…"` 那种单轮、以及服务端不读键盘，所以看不出来 ✗）
#   attach 模式下程序是被"我们"正常启动的，终端自始至终是它自己的 ✓
#
# ★为什么要"启动即自停"★
#   补丁必须在第一次 OH_AI_ModelBuildFromFile **之前/当时**打上。让程序一启动就
#   SIGSTOP 自己（用 -c 包一层），调试器从容 attach 后再放行 —— 无竞态 ✓
#
# ★POSIX 冷知识★：非交互 shell 里 `cmd &` 的 **stdin 默认被指到 /dev/null** ✗
#   ⇒ 必须显式重定向一次（显式重定向优先于那条默认规则）✓
#
# 环境变量:
#   CANN_LLM_LLDB_SERVER   gdbserver 路径（默认 $ROOT/bin/… 或 PATH）
#   CANN_LLM_LLDB          lldb 路径（默认 $ROOT/bin/lldb 或 PATH）
#   CANN_LLM_LLDB_PORT     调试端口（默认 5092）
#   CANN_LLM_LARGE_MEM_ATTACH_TIMEOUT  等补丁打上的上限秒数（默认 60）
set -eu

SELF=$0
ROOT=$(CDPATH= cd -- "$(dirname -- "$SELF")/.." && pwd)

if [ "$#" -lt 2 ]; then
    printf '用法: %s <python> <module> [模块参数…]\n' "$SELF" >&2
    exit 2
fi
PY=$1
MOD=$2
shift 2

PORT=${CANN_LLM_LLDB_PORT:-5092}

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

# lldb 的内嵌 python 要 import cann_llm.large_mem（补丁表的单一来源）⇒ 给 PYTHONPATH
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

# ── 1) 起程序：先自停、保留自己的终端 ──
exec 3<&0                      # ★存下真正的 stdin（见文件头：`&` 会把它换成 /dev/null）★
WRAPPER='import os, signal, sys, runpy
os.kill(os.getpid(), signal.SIGSTOP)
mod = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)'

"$PY" -X faulthandler -c "$WRAPPER" "$MOD" "$@" 0<&3 &
APP=$!
GDBPID=""
trap 'kill "$APP" 2>/dev/null || true; [ -n "$GDBPID" ] && kill "$GDBPID" 2>/dev/null || true' INT TERM HUP

# 等它自停（/proc/<pid>/stat 第 3 列 = T）
i=0
while [ "$i" -lt 400 ]; do
    if ! kill -0 "$APP" 2>/dev/null; then
        printf '[large-mem] 程序在自停前就退出了 —— 原样返回它的退出码\n' >&2
        RC=0; wait "$APP" || RC=$?
        exit "$RC"
    fi
    [ "$(awk '{print $3}' "/proc/$APP/stat" 2>/dev/null || echo '?')" = "T" ] && break
    i=$((i + 1))
    sleep 0.05
done

# ── 2) gdbserver 附着 ──
printf '[large-mem] 附着到 pid=%s（程序已自停，等打补丁）…\n' "$APP"
"$GDB" gdbserver --native-regs "127.0.0.1:$PORT" --attach "$APP" >/dev/null 2>&1 &
GDBPID=$!
sleep 0.3

# ── 3) lldb：import 时装断点 → 原生 `process continue` 放行 → 断点回调打补丁 ──
#   ★lldb 的 stdin 显式指到 /dev/null★：程序用【它自己的终端】读键盘，
#     若 lldb 也握着同一个 tty，用户敲的字会被它的命令解释器抢走 ✗
#     （batch 模式下 lldb 只在 -o 命令之间才碰 stdin，指到 /dev/null 最省事 ✓）
#   ★只用 lldb 原生命令 + 一条 python 命令（放在最后）★
#     实测：批处理模式下"注册的 python 命令"会让其后的 -o 失效 ✗
#     ⇒ 断点改在 `command script import` 时安装（import 不截断 ✓）
i=0
RC=1
while [ "$i" -lt 40 ]; do
    i=$((i + 1))
    if "$LLDB" --batch \
        -o "gdb-remote 127.0.0.1:$PORT" \
        -o "process handle SIGSTOP -s false -p false" \
        -o "command script import $SCRIPT" \
        -o "process continue" </dev/null; then
        RC=0
        break
    fi
    sleep 0.1
done
if [ "$RC" -ne 0 ]; then
    printf '[large-mem] ⚠ lldb 接入失败（重试 %d 次）—— 让程序不带补丁继续跑\n' "$i" >&2
    kill -CONT "$APP" 2>/dev/null || true
fi

# ── 4) 等程序自然结束，原样返回它的退出码（程序一直是我们的子进程 ✓）──
RC=0
wait "$APP" || RC=$?
[ -n "$GDBPID" ] && kill "$GDBPID" 2>/dev/null || true
exit "$RC"
