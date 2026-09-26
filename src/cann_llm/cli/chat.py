#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式对话命令行（逐字流式输出，支持工具调用）。

    python3 -m cann_llm.cli.chat -d /path/to/model_dir
    cann-llm-chat -d /path/to/model_dir --tools all
    cann-llm-chat -p "现在几点？" --tools get_current_time
    cann-llm-chat --list-tools

交互命令见 ``/help``。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

from .. import version
from ..agent.loop import (
    AgentConfig,
    AgentLoop,
    Final,
    StepStarted,
    TextDelta,
    ToolCallDone,
    ToolCallReady,
)
from ..backends import available_backends, create_backend
from ..backends.base import EngineBackend
from ..chat.template import Message, available_templates, get_template
from ..config import AppConfig, load_config
from ..errors import CannLlmError
from ..tools import ToolRegistry, default_registry
from ..types import GenerationParams
from ._commands import handle_command
from .state import CliState

HELP = """\
可用命令:
  /help                  显示本帮助
  /reset                 清空对话历史
  /system <文本>         设置 system prompt（并清空历史）
  /tools                 列出工具及启用情况
  /tools <名字>…         切换工具的启用状态（all / none）
  /temp <f>              采样温度（0 = 贪心；推荐 0.7）
  /topk <n>              top-k（推荐 20）
  /topp <f>              top-p（推荐 0.95）
  /rep <f>               重复惩罚（推荐 1.1）
  /max-tokens <n>        单轮最大生成 token 数（/maxtok 亦可）
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
        "model_id": mc.resolved_id,
        "context_length": mc.context_length,
        "default_params": GenerationParams(
            max_tokens=mc.max_tokens, temperature=mc.temperature, top_k=mc.top_k,
            top_p=mc.top_p, repetition_penalty=mc.repetition_penalty),
    }
    if mc.backend == "cann":
        kw["lib_path"] = os.environ.get("CANN_LLM_LIB", version.CANN_NDK_LIB)
    backend = create_backend(mc.backend, **kw)
    backend.load()
    return backend


def resolve_tools(spec: Optional[str], *, allow_dangerous: bool = False) -> ToolRegistry:
    """解析 ``--tools`` 的取值。

    ``all`` / ``safe`` → 全部非 dangerous 工具；``none`` / 空 → 空集合；
    否则按逗号或空格分隔的工具名。
    """
    reg = default_registry()
    spec = (spec or "").strip()
    if not spec or spec == "none":
        return reg.select([])
    if spec in ("all", "safe"):
        return reg.select()
    names = [x for x in spec.replace(",", " ").split() if x]
    return reg.select(names, include_dangerous=allow_dangerous)


def print_params(state: CliState) -> None:
    p = state.params
    mode = "贪心" if p.greedy else "采样"
    tools = ", ".join(state.tool_names) if state.tool_names else "关闭"
    print(f"  temperature={p.temperature}  top-k={p.top_k}  top-p={p.top_p}  "
          f"repetition_penalty={p.repetition_penalty}  max_tokens={p.max_tokens}  "
          f"({mode})  流式={'on' if state.stream else 'off'}")
    print(f"  工具: {tools}")


def _short(value, limit: int = 80) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def run_turn(engine: EngineBackend, cfg: AppConfig, state: CliState,
             text: str, registry: ToolRegistry) -> None:
    """跑一轮：交给 agent 循环，把过程打到终端。"""
    state.messages.append(Message("user", text))
    loop = AgentLoop(
        engine, get_template(cfg.model.chat_template), registry,
        config=AgentConfig(max_steps=cfg.agent.max_steps),
        system_prompt=state.system_prompt, params=state.params,
)

    final: Optional[Final] = None
    started = False
    tool_count = 0
    for ev in loop.run(state.messages, stream=state.stream):
        if isinstance(ev, StepStarted):
            if ev.step > 1:
                print(f"  [第 {ev.step}/{ev.max_steps} 轮]")
        elif isinstance(ev, TextDelta):
            if not started:
                sys.stdout.write("bot> ")
                sys.stdout.flush()
                started = True
            sys.stdout.write(ev.text)
            sys.stdout.flush()
        elif isinstance(ev, ToolCallReady):
            tool_count += 1
            print(f"  → 调用 {ev.call.name}({_short(ev.call.arguments)})")
        elif isinstance(ev, ToolCallDone):
            mark = "✓" if ev.result.ok else "✗"
            print(f"    {mark} {ev.duration_ms:.0f}ms  {_short(ev.result.content, 120)}")
        elif isinstance(ev, Final):
            final = ev

    if started:
        print()
    elif final is not None and final.text:
        print(f"bot> {final.text}")
    elif final is not None:
        # 如实报告，不去替模型圆场。用满 max_steps 时模型可能仍停在
        # 「还想调用工具」的状态 —— 这是它的真实行为，说清楚即可。
        if final.tool_calls:
            names = ", ".join(dict.fromkeys(c.name or "?" for c in final.tool_calls))
            print(f"  [已达步数上限 {final.steps}，模型仍在请求工具调用（{names}），"
                  f"没有最终回答]")
        else:
            print("  [模型没有输出文本]")

    if final is None:
        return
    # 历史里不含 system（每次渲染时由 AgentLoop 注入）
    state.messages = [m for m in final.messages if m.role != "system"]
    state.last_stats = final.stats
    state.last_finish_reason = final.finish_reason
    state.last_steps = final.steps

    if final.stats:
        if tool_count:
            print(f"  [{tool_count} 次工具调用 · {final.steps} 轮 · "
                  f"{final.stats.tokens_per_second:.1f} tok/s]")
        else:
            print(f"  [in {final.stats.prompt_tokens} tok · out {final.stats.completion_tokens} "
                  f"tok · prefill {final.stats.prefill_ms:.0f} ms · "
                  f"decode {final.stats.decode_ms:.0f} ms "
                  f"({final.stats.tokens_per_second:.1f} tok/s)]", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cann-llm-chat",
        description="华为 CANN LLM Engine 交互式对话（逐字流式 + 工具调用）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--model-dir", help="模型目录")
    ap.add_argument("--model-id",
                    help="对外暴露的模型 id（默认取模型目录名）")
    ap.add_argument("-c", "--config", help="TOML 配置文件")
    ap.add_argument(
        "-b", "--backend", default="hiai",
        help=f"后端，可用：{', '.join(available_backends())}（默认 hiai）")
    ap.add_argument("-t", "--template", help=f"对话模板，可用：{', '.join(available_templates())}")
    ap.add_argument("-s", "--system", help="system prompt")
    ap.add_argument("-p", "--prompt", help="单轮模式：生成一次后退出")
    ap.add_argument("--tools", help="启用的工具：all / none / 逗号分隔的名字")
    ap.add_argument("--allow-dangerous-tools", action="store_true",
                    help="允许启用标记为 dangerous 的工具（自行评估风险）")
    ap.add_argument("--temp", type=float, help="采样温度，0=贪心")
    ap.add_argument("--topk", type=int, help="top-k")
    ap.add_argument("--topp", type=float, help="top-p")
    ap.add_argument("--rep", type=float, help="重复惩罚")
    ap.add_argument("--max-tokens", "--maxtok", type=int, dest="maxtok",
                    help="单轮最大生成 token 数")
    ap.add_argument("--no-stream", action="store_true", help="关闭逐字输出")
    ap.add_argument("--list-backends", action="store_true", help="列出可用后端后退出")
    ap.add_argument("--list-tools", action="store_true", help="列出可用工具后退出")
    ap.add_argument("-V", "--version", action="version", version=f"cann-llm {version.__version__}")
    args = ap.parse_args(argv)

    if args.list_backends:
        print("后端:", ", ".join(available_backends()) or "(无)")
        print("模板:", ", ".join(available_templates()))
        return 0
    if args.list_tools:
        reg = default_registry()
        for name in reg.names():
            tool = reg.get(name)
            flag = " [dangerous]" if tool.dangerous else ""
            print(f"{name}{flag}: {tool.description.splitlines()[0]}")
        return 0

    # 配置合并：命令行 > 环境变量 > 文件 > 默认值
    cfg = load_config(args.config)
    over = {}
    for key, val in (("model_dir", args.model_dir), ("model_id", args.model_id),
                     ("backend", args.backend),
                     ("chat_template", args.template), ("system_prompt", args.system),
                     ("max_tokens", args.maxtok), ("temperature", args.temp),
                     ("top_k", args.topk), ("top_p", args.topp),
                     ("repetition_penalty", args.rep)):
        if val is not None:
            over[key] = val
    if over:
        cfg = cfg.merged(model=over)

    try:
        registry = resolve_tools(args.tools, allow_dangerous=args.allow_dangerous_tools)
    except (KeyError, PermissionError) as e:
        print(f"工具配置有误: {e}", file=sys.stderr)
        return 2

    try:
        t0 = time.time()
        engine = build_engine(cfg)
        load_s = time.time() - t0
    except CannLlmError as e:
        print(f"启动失败: {e}", file=sys.stderr)
        return 1

    template = get_template(cfg.model.chat_template)
    state = CliState(
        params=GenerationParams(
            max_tokens=cfg.model.max_tokens, temperature=cfg.model.temperature,
            top_k=cfg.model.top_k, top_p=cfg.model.top_p,
            repetition_penalty=cfg.model.repetition_penalty,
            stop=tuple(template.stop_strings())),
        system_prompt=cfg.model.system_prompt,
        tool_names=registry.names(),
        stream=not args.no_stream,
        model_id=cfg.model.resolved_id, backend=cfg.model.backend)

    if args.prompt:                     # 单轮模式
        try:
            run_turn(engine, cfg, state, args.prompt, registry)
        except CannLlmError as e:
            print(f"生成失败: {e}", file=sys.stderr)
            return 1
        engine.close()
        return 0

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print(f"cann-llm {version.__version__}  ·  {cfg.model.resolved_id}"
          f"  ·  {cfg.model.backend}  ·  加载 {load_s:.1f}s")
    print_params(state)
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
            if handle_command(line, state, cfg):
                break
            # 命令可能改了工具集，重新落到注册表
            try:
                registry = resolve_tools(",".join(state.tool_names) or "none",
                                         allow_dangerous=args.allow_dangerous_tools)
            except (KeyError, PermissionError):
                registry = resolve_tools("none")
            continue

        try:
            run_turn(engine, cfg, state, line, registry)
        except CannLlmError as e:
            print(f"  [失败] {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [错误] {type(e).__name__}: {e}")

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
