#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式对话命令行（逐字流式输出）。

    python3 -m cann_llm.cli.chat -d /path/to/model_dir
    cann-llm-chat -d /path/to/model_dir --temp 0 --topk 1
    cann-llm-chat -p "你好"                  # 单轮模式
    cann-llm-chat --list-backends

交互命令见 ``/help``。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

from .. import version
from ..backends import available_backends, create_backend
from ..backends.base import EngineBackend
from ..chat.session import ChatSession
from ..chat.template import available_templates, get_template
from ..config import AppConfig, load_config
from ._commands import handle_command
from ..errors import CannLlmError
from ..types import GenerationParams

HELP = """\
可用命令:
  /help                  显示本帮助
  /reset                 清空对话历史
  /system <文本>         设置 system prompt（并清空历史）
  /temp <f>              采样温度（0 = 贪心；推荐 0.7）
  /topk <n>              top-k（推荐 20）
  /topp <f>              top-p（推荐 0.95）
  /rep <f>               重复惩罚（推荐 1.1）
  /maxtok <n>            单轮最大生成 token 数
  /stream on|off         开关逐字输出
  /params                查看当前采样参数
  /stats                 上一轮耗时与用量
  /save <文件>           保存对话记录（Markdown）
  /quit                  退出
直接输入文字即可对话。"""


def build_engine(cfg: AppConfig) -> EngineBackend:
    """按配置创建后端。"""
    mc = cfg.model
    if not mc.model_dir:
        raise SystemExit("未指定模型目录：用 -d/--model-dir、CANN_LLM_MODEL__MODEL_DIR "
                         "或配置文件里的 model.model_dir")
    kw = {
        "model_dir": mc.model_dir,
        "model_id": mc.model_id,
        "context_length": mc.context_length,
        "max_prompt_tokens": mc.max_prompt_tokens,
        "default_params": GenerationParams(
            max_tokens=mc.max_tokens, temperature=mc.temperature, top_k=mc.top_k,
            top_p=mc.top_p, repetition_penalty=mc.repetition_penalty),
    }
    if mc.backend == "cann":
        from ..backends.cann import CANN_NDK_LIB   # noqa: F401  (可用 --lib 覆盖)
        kw["lib_path"] = os.environ.get("CANN_LLM_LIB", version.CANN_NDK_LIB)
    backend = create_backend(mc.backend, **kw)
    backend.load()
    return backend


def print_params(params: GenerationParams, stream: bool) -> None:
    mode = "贪心" if params.greedy else "采样"
    print(f"  temperature={params.temperature}  top-k={params.top_k}  top-p={params.top_p}  "
          f"repetition_penalty={params.repetition_penalty}  "
          f"max_tokens={params.max_tokens}  ({mode})  流式={'on' if stream else 'off'}")


def _stream_reply(session: ChatSession, text: str, stream: bool) -> None:
    """跑一轮并输出。"""
    if stream:
        sys.stdout.write("bot> ")
        sys.stdout.flush()
        for chunk in session.ask(text):
            if chunk.text:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
        print()
    else:
        reply = session.ask_sync(text)
        print(f"bot> {reply}")
    st = session.last_stats
    if st is not None:
        print(f"  [in {st.prompt_tokens} tok · out {st.completion_tokens} tok · "
              f"prefill {st.prefill_ms:.0f} ms · decode {st.decode_ms:.0f} ms "
              f"({st.tokens_per_second:.1f} tok/s)]", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cann-llm-chat",
        description="华为 CANN LLM Engine 交互式对话（逐字流式）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--model-dir", help="模型目录")
    ap.add_argument("-c", "--config", help="TOML 配置文件")
    ap.add_argument("-b", "--backend", help=f"后端，可用：{', '.join(available_backends())}")
    ap.add_argument("-t", "--template", help=f"对话模板，可用：{', '.join(available_templates())}")
    ap.add_argument("-s", "--system", help="system prompt")
    ap.add_argument("-p", "--prompt", help="单轮模式：生成一次后退出")
    ap.add_argument("--temp", type=float, help="采样温度，0=贪心")
    ap.add_argument("--topk", type=int, help="top-k")
    ap.add_argument("--topp", type=float, help="top-p")
    ap.add_argument("--rep", type=float, help="重复惩罚")
    ap.add_argument("--maxtok", type=int, help="单轮最大生成 token 数")
    ap.add_argument("--no-stream", action="store_true", help="关闭逐字输出")
    ap.add_argument("--list-backends", action="store_true", help="列出可用后端后退出")
    ap.add_argument("-V", "--version", action="version", version=f"cann-llm {version.__version__}")
    args = ap.parse_args(argv)

    if args.list_backends:
        print("后端:", ", ".join(available_backends()) or "(无)")
        print("模板:", ", ".join(available_templates()))
        return 0

    # 配置合并：命令行 > 环境变量 > 文件 > 默认值
    cfg = load_config(args.config)
    over = {}
    for key, val in (("model_dir", args.model_dir), ("backend", args.backend),
                     ("chat_template", args.template), ("system_prompt", args.system),
                     ("max_tokens", args.maxtok), ("temperature", args.temp),
                     ("top_k", args.topk), ("top_p", args.topp),
                     ("repetition_penalty", args.rep)):
        if val is not None:
            over[key] = val
    if over:
        cfg = cfg.merged(model=over)

    stream = not args.no_stream

    try:
        t0 = time.time()
        backend = build_engine(cfg)
        load_s = time.time() - t0
    except CannLlmError as e:
        print(f"启动失败: {e}", file=sys.stderr)
        return 1

    template = get_template(cfg.model.chat_template)
    params = GenerationParams(max_tokens=cfg.model.max_tokens,
                              temperature=cfg.model.temperature,
                              top_k=cfg.model.top_k, top_p=cfg.model.top_p,
                              repetition_penalty=cfg.model.repetition_penalty,
                              stop=tuple(template.stop_strings()))
    session = ChatSession(backend=backend, template=template,
                          system_prompt=cfg.model.system_prompt, params=params,
                          max_prompt_tokens=cfg.model.max_prompt_tokens)

    if args.prompt:                     # 单轮模式
        _stream_reply(session, args.prompt, stream)
        backend.close()
        return 0

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print(f"cann-llm {version.__version__}  ·  {cfg.model.model_id}"
          f"  ·  {cfg.model.backend}  ·  加载 {load_s:.1f}s")
    print_params(params, stream)
    print("输入 /help 看命令，/quit 退出。\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            stream, quit_now = handle_command(line, session, cfg, stream)
            if quit_now:
                break
            continue

        try:
            _stream_reply(session, line, stream)
        except CannLlmError as e:
            print(f"  [失败] {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [错误] {type(e).__name__}: {e}")

    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
