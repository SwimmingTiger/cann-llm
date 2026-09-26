#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读模型目录里的关键规格，供启动脚本打印（也方便直接看）。

用法::

    python3 scripts/model_info.py <模型目录>          # key=value 形式
    python3 scripts/model_info.py <模型目录> --pretty  # 人读形式

关注两个数：

* **上下文窗口** = ``kv_cache_max_len``
  —— 这是 NPU 上**真正可用**的窗口：KV 缓存的张量就那么大，输入 + 输出之和不能超过它。
  （模型配置里可能还写着 ``max_position_embeddings``（原生上限，如 131072）与
  ``max_io_tokens``（单次 IO 限制，如 32768）—— 那**不是** NPU 上能用的窗口。）
* **最大输出 token** —— 没有固定值：= 窗口 − 本次输入长度。
  要改单轮上限用 CLI 的 ``--max-tokens``（``--maxtok`` 亦可），服务端同名。
"""
import json
import os
import sys


def _load_llm_config(model_dir):
    """从模型目录读出 llm_config（含 kv_cache_max_len）。

    优先我们自己的 executor.json（结构化，带 llm_config）；
    否则扫目录里的 .json —— 官方 <model>.json 是**扁平**的，我们生成的会嵌套在
    llm_config 下，两种都认。
    """
    p = os.path.join(model_dir, "executor.json")
    if os.path.isfile(p):
        try:
            ll = (json.load(open(p)) or {}).get("llm_config") or {}
            if ll.get("kv_cache_max_len"):
                return ll, "executor.json"
        except Exception:  # noqa: BLE001
            pass
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".json"):
            continue
        try:
            cfg = json.load(open(os.path.join(model_dir, name))) or {}
        except Exception:  # noqa: BLE001
            continue
        ll = cfg.get("llm_config") if isinstance(cfg.get("llm_config"), dict) else cfg
        if isinstance(ll, dict) and ll.get("kv_cache_max_len"):
            return ll, name
    return {}, ""


def main(argv):
    if len(argv) < 2:
        print("用法: model_info.py <模型目录> [--pretty]", file=sys.stderr)
        return 2
    model_dir, pretty = argv[1], ("--pretty" in argv[2:])
    if not os.path.isdir(model_dir):
        print(f"不是目录: {model_dir}", file=sys.stderr)
        return 1

    ll, src = _load_llm_config(model_dir)
    win = ll.get("kv_cache_max_len")
    native = ll.get("max_position_embeddings")
    io_max = ll.get("max_io_tokens")

    if pretty:
        if win:
            print(f"上下文窗口(kv_cache_max_len)={win}")
        print(f"来源文件={src or '(未找到)'}")
        if native:
            print(f"模型原生上限(max_position_embeddings)={native}  # 不是 NPU 上能用的窗口")
        if io_max:
            print(f"单次 IO 限制(max_io_tokens)={io_max}")
    else:
        print(f"context_window={win if win else ''}")
        print(f"source={src}")
        print(f"native_max={native if native else ''}")
        print(f"max_io_tokens={io_max if io_max else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
