"""`scripts/start_server.sh` 的实际逻辑（外壳只负责找 python 再转发）。

与 :mod:`cann_llm.launcher` 同样的分工：shell 只发现解释器 + 转发，
参数解析、预检、后台/状态管理都在这里。

用法（由外壳转过来）::

    python -m cann_llm.launcher server -d <模型目录> [--port 8000] [-B] [--status|--stop]
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional

from .launcher import (die, find_model_dir, info, libc_compatible, model_info_line,
                       ok, pick_engine_lib, warn)

__all__ = ["run_server"]

#: 后台启动后等就绪的最长秒数
WAIT_SECS = 90


# ---------------------------------------------------------------- 运行时文件

def _stamp() -> str:
    """``20260927-1444``：日志文件名用的时间戳（本地时间）。"""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def rundir(root: str) -> str:
    """运行时**控制文件**的目录（``server.pid`` / ``server.state``）。

    ★ 日志【不】放这里 —— 日志在 :func:`logdir`，这样目录各司其职：
      ``.run/`` 是指针（谁在跑、监听哪个端口），``log/`` 才是给人看的日志。
    原先两者混在一个目录，想挪日志就会连 pid/state 一起挪走。
    """
    d = os.environ.get("CANN_LLM_RUNDIR") or os.path.join(root, ".run")
    os.makedirs(d, exist_ok=True)
    return d


def logdir(root: str) -> str:
    """日志目录（默认 ``log/``，**不存在会自动创建**）。

    用非隐藏名字：日志是给人看的，藏在 ``.run`` 里没人找得到。
    """
    d = os.environ.get("CANN_LLM_LOGDIR") or os.path.join(root, "log")
    os.makedirs(d, exist_ok=True)
    return d


def paths(root: str) -> "Dict[str, str]":
    d = rundir(root)          # 控制文件（pid/state）
    g = logdir(root)          # 日志
    return {
        "pid": os.path.join(d, "server.pid"),
        "state": os.path.join(d, "server.state"),
        # 后台进程的 stdout/stderr。带时间戳 —— 固定名会让两个实例互相覆盖。
        # （进程自己的 pid 这时还不知道，所以这里只用时间戳；
        #   诊断日志由服务端自己按 <时间>-<pid>.log 命名，见 api/server.py）
        "log": os.environ.get("CANN_LLM_LOG")
               or os.path.join(g, _stamp() + ".server.log"),
    }


def _read_pid(p: "Dict[str, str]") -> Optional[int]:
    try:
        with open(p["pid"], encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def alive(p: "Dict[str, str]") -> bool:
    pid = _read_pid(p)
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_state(p: "Dict[str, str]") -> "Dict[str, str]":
    st: "Dict[str, str]" = {}
    try:
        with open(p["state"], encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    st[k] = v
    except OSError:
        pass
    return st


# ---------------------------------------------------------------- 探测

def port_busy(host: str, port: int) -> bool:
    h = host if host not in ("0.0.0.0", "") else "127.0.0.1"
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex((h, port)) == 0
    finally:
        s.close()


def health(host: str, port: int) -> Optional[dict]:
    h = host if host not in ("0.0.0.0", "") else "127.0.0.1"
    try:
        with urllib.request.urlopen(f"http://{h}:{port}/healthz", timeout=3) as r:
            return json.load(r)
    except Exception:            # noqa: BLE001
        return None


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def print_endpoints(host: str, port: int, api_key: str) -> None:
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    base = f"http://{shown}:{port}"
    print(f"  监听      : {base}")
    print(f"  base_url  : {base}/v1      ← OpenAI 客户端的 base_url 填这个")
    print(f"  端点      : GET {base}/v1/models · POST {base}/v1/chat/completions"
          f" · POST {base}/v1/completions")
    print(f"  健康检查  : {base}/healthz（无需鉴权）"
          f"    鉴权: {'开' if api_key else '关闭'}")
    if host in ("0.0.0.0", ""):
        print(f"      http://{lan_ip()}:{port}/v1      # 局域网其它机器访问")


# ---------------------------------------------------------------- status / stop

def do_status(root: str) -> int:
    p = paths(root)
    if not alive(p):
        print("未运行（没有活跃的服务器进程）")
        return 1
    pid = _read_pid(p)
    st = load_state(p)
    host = st.get("host", "127.0.0.1")
    port = int(st.get("port", "8000") or 8000)
    print(f"运行中  pid={pid}  {host}:{port}")
    if st.get("model_dir"):
        print(f"  模型目录: {st['model_dir']}")
    h = health(host, port)
    if h:
        print(f"  健康检查: {h.get('status')} model={h.get('model')} "
              f"uptime={h.get('uptime_s')}s")
    else:
        print("  健康检查: 无响应")
    print(f"  日志: {p['log']}")
    return 0


def do_stop(root: str) -> int:
    p = paths(root)
    pid = _read_pid(p)
    if not pid or not alive(p):
        print("未运行（无需停止）")
        for f in (p["pid"], p["state"]):
            try:
                os.remove(f)
            except OSError:
                pass
        return 0
    info(f"停止 pid={pid} …")
    try:
        os.kill(pid, 15)
    except OSError as e:
        die(f"发送信号失败: {e}")
    for _ in range(50):
        if not alive(p):
            break
        time.sleep(0.2)
    else:
        warn("进程未在 10 秒内退出，发送 SIGKILL")
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        time.sleep(0.5)
    for f in (p["pid"], p["state"]):
        try:
            os.remove(f)
        except OSError:
            pass
    ok("已停止")
    return 0


# ---------------------------------------------------------------- 参数

_USAGE = """用法: scripts/start_server.sh [选项]

  -d, --model-dir DIR   模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json）
  -c, --config FILE     TOML 配置文件
  -b, --backend NAME    后端：hiai | cann（默认 hiai）
      --host HOST       监听地址（默认 127.0.0.1；0.0.0.0 表示允许局域网访问）
      --port PORT       监听端口（默认 8000）
  -k, --api-key KEY     API key（设了就要求鉴权）
  -B, --background      后台启动，等就绪后返回
      --debug           诊断模式：记录 HTTP 请求/响应与引擎的原始输入输出
                        （写到 log/<日期>-<时间>-<pid>.log；也可用 CANN_LLM_DEBUG=1 打开）
      --lldb            在 lldb 下前台启动（抓崩溃现场）。进 lldb 后敲 run，
                        崩溃时 bt 看栈。lldb 路径可用 CANN_LLM_LLDB 指定；
                        设 CANN_LLM_LLDB_BATCH=1 则非交互：run→bt→quit
      --status          查看状态
      --stop            停止后台服务
  -h, --help            显示本帮助

环境变量: CANN_LLM_MODEL_DIR / CANN_LLM_CONFIG / CANN_LLM_HOST / CANN_LLM_PORT
          CANN_LLM_API_KEY / CANN_LLM_BACKEND / CANN_LLM_LOG
          CANN_LLM_RUNDIR（pid/state 目录，默认 .run/）
          CANN_LLM_LOGDIR（日志目录，默认 log/）
          CANN_LLM_LIB（cann） / CANN_LLM_HIAI_LIB（hiai） / PYTHON
"""


def parse_args(argv: "List[str]") -> "Dict[str, object]":
    env = os.environ
    o: "Dict[str, object]" = {
        "model_dir": env.get("CANN_LLM_MODEL_DIR") or None,
        "config": env.get("CANN_LLM_CONFIG") or "",
        "backend": env.get("CANN_LLM_BACKEND") or "hiai",
        "host": env.get("CANN_LLM_HOST") or "127.0.0.1",
        "port": int(env.get("CANN_LLM_PORT") or 8000),
        "api_key": env.get("CANN_LLM_API_KEY") or "",
        "background": False,
        "debug": False,
        "lldb": False,
        "action": "run",
        "rest": [],
    }
    i = 0
    while i < len(argv):
        a = argv[i]

        def val() -> str:
            nonlocal i
            if i + 1 >= len(argv):
                die(f"{a} 需要一个值")
            i += 1
            return argv[i]

        if a in ("-h", "--help"):
            print(_USAGE)
            raise SystemExit(0)
        elif a in ("-d", "--model-dir"):
            o["model_dir"] = val()
        elif a in ("-c", "--config"):
            o["config"] = val()
        elif a in ("-b", "--backend"):
            o["backend"] = val()
        elif a == "--host":
            o["host"] = val()
        elif a == "--port":
            o["port"] = int(val())
        elif a in ("-k", "--api-key"):
            o["api_key"] = val()
        elif a == "--debug":
            # 只用来决定"日志写哪" + 在帮助里露出来；参数本身也原样转给服务端
            o["debug"] = True
            o["rest"].append(a)
        elif a == "--lldb":
            # 在调试器下前台启动（抓崩溃现场用）。这是启动器自己的选项，不透传。
            o["lldb"] = True
        elif a in ("-B", "--background"):
            o["background"] = True
        elif a == "--status":
            o["action"] = "status"
        elif a == "--stop":
            o["action"] = "stop"
        else:
            o["rest"].append(a)          # 其余透传给 API server
        i += 1
    return o


# ---------------------------------------------------------------- 主流程

def run_server(root: str, argv: "List[str]") -> int:
    o = parse_args(argv)
    action = o["action"]
    if action == "status":
        return do_status(root)
    if action == "stop":
        return do_stop(root)

    backend = str(o["backend"])
    host, port = str(o["host"]), int(o["port"])
    p = paths(root)

    py = os.environ.get("CANN_LLM_RESOLVED_PYTHON") or sys.executable or "python3"
    if sys.version_info < (3, 9):
        die("Python 版本过低（需要 >= 3.9，因为用到了 tomllib）")
    if not libc_compatible(py):
        warn("解释器兼容性提示")
        print(f"    {py} 是 glibc 构建（带 libmusl_compat 垫片），"
              f"与按 musl 编译的引擎不兼容，加载引擎时大概率段错误。", file=sys.stderr)
    ok(f"Python {sys.version.split()[0]}  ·  {py}")

    var, lib, kind = pick_engine_lib(backend)
    if not os.path.isfile(lib):
        die(f"找不到 {kind} {lib}\n     后端 {backend} 需要鸿蒙设备（NPU）环境。")
    ok(f"{kind}: {lib}")

    d = find_model_dir(root, o["model_dir"] if isinstance(o["model_dir"], str) else None)
    if d is None:
        die("未指定模型目录：用 -d 指定，或把模型放到 "
            f"{root}/models/<名字>/ 下（需含 executor.json）")
    ok(f"模型目录: {d}")
    for need in ("executor.json", "context.json", "tokenizer.json"):
        if not os.path.isfile(os.path.join(d, need)):
            die(f"模型目录缺少 {need}：{d}")
    import glob as _glob
    if not (_glob.glob(os.path.join(d, "*.omc"))
            and _glob.glob(os.path.join(d, "SubGraph_*.weight"))):
        die(f"模型目录缺少 *.omc 或 SubGraph_*.weight：{d}")
    ok("模型文件: " + ", ".join(
        [os.path.basename(x) for x in _glob.glob(os.path.join(d, "*.omc"))[:1]]
        + [os.path.basename(x) for x in _glob.glob(os.path.join(d, "SubGraph_*.weight"))[:1]]))
    kv = model_info_line(d)
    if kv:
        ok(f"上下文窗口: {kv} token  (= kv_cache_max_len，输入 + 输出之和)")
        info("最大输出没有固定值：= 窗口 − 本次输入长度")
        info("改默认输出窗口：--max-tokens <n>   例：scripts/start_server.sh -d … --max-tokens 512")
        info("（请求体里的 max_tokens 优先生效）")

    if port_busy(host, port):
        die(f"端口 {port} 已被占用（服务已在跑？用 --status 看状态）")

    args = ["-m", "cann_llm.api.server", "-d", d,
            "--host", host, "--port", str(port)]
    if o["config"]:
        args += ["-c", str(o["config"])]
    if backend:
        args += ["-b", backend]
    if o["api_key"]:
        args += ["-k", str(o["api_key"])]
    args += list(o["rest"])              # 其余参数原样透传

    env = dict(os.environ)
    if o.get("debug"):
        # ★ 只给【目录】：文件名要带 pid，而 pid 只有子进程自己知道
        #   （服务端会生成 log/<时间>-<pid>.log，见 api/server.py）。
        env["CANN_LLM_LOGDIR"] = logdir(root)
    env["PYTHONPATH"] = os.path.join(root, "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env[var] = lib

    if o.get("lldb") and o["background"]:
        die("--lldb 不能和 -B 一起用：调试器要前台交互")

    print()
    if not o["background"]:
        for f in (p["pid"], p["state"]):
            try:
                os.remove(f)
            except OSError:
                pass
        info("前台启动（Ctrl-C 停止），正在加载模型…")
        print()
        # 不在这里打印监听地址 —— 此刻还没 bind，交给 serve() 在绑定成功后自己打印
        # ★ execve 会换掉整个进程，Python 的 stdout 缓冲区不会自动 flush；
        #   重定向/接管道时（块缓冲）上面那些提示会全丢，必须手动刷。
        sys.stdout.flush()
        sys.stderr.flush()
        if o.get("lldb"):
            from .lldb_launch import build_debug_argv
            argv_dbg, hints, err = build_debug_argv(py, args)
            if argv_dbg is None:
                die(err)
            for line in hints:
                info(line) if line else print()
            print()
            sys.stdout.flush()
            sys.stderr.flush()
            os.execve(argv_dbg[0], argv_dbg, env)
            return 0
        os.execve(py, [py, "-X", "faulthandler"] + args, env)
        return 0

    info("后台启动…")
    log = open(p["log"], "a", encoding="utf-8")     # noqa: SIM115
    proc = subprocess.Popen([py, "-X", "faulthandler"] + args, env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    with open(p["pid"], "w", encoding="utf-8") as fh:
        fh.write(f"{proc.pid}\n")
    with open(p["state"], "w", encoding="utf-8") as fh:
        fh.write(f"pid={proc.pid}\nhost={host}\nport={port}\nmodel_dir={d}\n")

    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    for i in range(WAIT_SECS):
        if proc.poll() is not None:
            print(f"\n服务进程已退出（退出码 {proc.returncode}），日志尾部：")
            try:
                with open(p["log"], encoding="utf-8", errors="replace") as fh:
                    print("".join(fh.readlines()[-25:]))
            except OSError:
                pass
            return 1
        h = health(host, port)
        if h:
            print()
            ok(f"已就绪（{i + 1}s）  pid={proc.pid}")
            print(f"cann-llm  ·  {os.path.basename(d)}")
            print_endpoints(host, port, str(o["api_key"]))
            print(f"\n  日志: {p['log']}\n  停止: scripts/start_server.sh --stop")
            return 0
        time.sleep(1)
    warn(f"等了 {WAIT_SECS}s 仍未就绪（进程还在，可能还在加载模型）")
    print(f"  日志: tail -f {p['log']}")
    return 0
