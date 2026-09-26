#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式自检：确认后端是「逐 token 增量到达」而不是最后一次性返回。

    PYTHONPATH=src python3 scripts/stream_check.py -d /path/to/model_dir

判断依据：首个增量与最后一个增量之间的时间跨度。真正的逐 token 流式输出
会有明显跨度；如果所有增量几乎同时到达，说明底层只支持整块返回。
"""

from __future__ import annotations

import argparse
import sys
import time

from cann_llm.backends import create_backend
from cann_llm.chat.template import Message, get_template
from cann_llm.types import GenerationParams, GenerationRequest

PROMPT = "List three colors, one word each, separated by commas."


def main() -> int:
    ap = argparse.ArgumentParser(description="CANN LLM 流式输出自检")
    ap.add_argument("-d", "--model-dir", required=True)
    ap.add_argument("-b", "--backend", default="cann")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--maxtok", type=int, default=64)
    args = ap.parse_args()

    backend = create_backend(args.backend, model_dir=args.model_dir)
    info = backend.load()
    print(f"模型: {info.id}  后端: {info.backend}  支持流式: {backend.supports_streaming}")

    prompt = get_template("chatml").render([
        Message("system", "You are a helpful assistant."),
        Message("user", PROMPT),
    ])
    req = GenerationRequest(prompt=prompt, params=GenerationParams(
        max_tokens=args.maxtok, temperature=args.temp))

    t0 = time.time()
    arrivals = []
    print("\n--- 增量到达时间线 ---")
    for chunk in backend.generate(req):
        if chunk.text:
            arrivals.append((time.time() - t0, chunk.text))
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        if chunk.is_final:
            stats = chunk.stats
            print(f"\n\nfinish_reason = {chunk.finish_reason}")
            if stats:
                print(f"in={stats.prompt_tokens} out={stats.completion_tokens} "
                      f"prefill={stats.prefill_ms:.0f}ms decode={stats.decode_ms:.0f}ms "
                      f"({stats.tokens_per_second:.1f} tok/s)")
    print(f"\n共 {len(arrivals)} 个增量分块")
    for ts, text in arrivals[:8]:
        print(f"  {ts:6.3f}s  {text!r}")

    backend.close()

    if len(arrivals) < 3:
        print("\n⚠ 分块太少，无法判断（模型回答过短）")
        return 0
    span = arrivals[-1][0] - arrivals[0][0]
    print(f"\n首次→末次跨度: {span:.3f}s")
    if span > 0.2:
        print("★ 真流式（逐 token 增量到达）")
        return 0
    print("⚠ 疑似最后一次性到达 —— 检查 callback_freq 是否为 1")
    return 1


if __name__ == "__main__":
    sys.exit(main())
