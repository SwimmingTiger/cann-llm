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


#: 我们生成的 executor/context 文件名的特征（这些是派生物，不是权威来源）
_DERIVED_NAMES = ("executor.json", "context.json")
_DERIVED_PREFIXES = ("executor_", "context_")


def _looks_derived(name):
    """判断某个 json 是不是本工具链生成的派生物（executor*.json / context*.json）。"""
    return name in _DERIVED_NAMES or name.startswith(_DERIVED_PREFIXES)


def _read_llm(name, path):
    """把一份 json 读成 llm_config 视图（官方扁平 / 我们嵌套两层，两种都认）。"""
    cfg = json.load(open(path)) or {}
    ll = cfg.get("llm_config") if isinstance(cfg.get("llm_config"), dict) else cfg
    return ll if isinstance(ll, dict) else {}


def scan_all(model_dir):
    """扫描目录，返回 [(文件, llm_config, 是否派生物)]，只保留有 kv_cache_max_len 的。"""
    out = []
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".json"):
            continue
        try:
            ll = _read_llm(name, os.path.join(model_dir, name))
        except Exception:  # noqa: BLE001
            continue
        if ll.get("kv_cache_max_len"):
            out.append((name, ll, _looks_derived(name)))
    return out


def _load_llm_config(model_dir):
    """挑一份**权威**来源，并报告冲突。

    优先级：
      1. **非派生的**（官方 `<model>.json` 等）—— 这是模型自带的真相
      2. 我们生成的 executor.json —— 方便，但可能是旧的（陈旧风险）
    同优先级里按文件名排序取第一个；若多份来源数值不一致，一并报告出来。
    """
    found = scan_all(model_dir)
    if not found:
        return {}, "", []
    authoritative = [f for f in found if not f[2]]
    chosen = authoritative[0] if authoritative else found[0]
    conflicts = [(n, ll.get("kv_cache_max_len")) for n, ll, _ in found
                 if ll.get("kv_cache_max_len") != chosen[1].get("kv_cache_max_len")]
    return chosen[1], chosen[0], conflicts


def main(argv):
    if len(argv) < 2:
        print("用法: model_info.py <模型目录> [--pretty]", file=sys.stderr)
        return 2
    model_dir, pretty = argv[1], ("--pretty" in argv[2:])
    if not os.path.isdir(model_dir):
        print(f"不是目录: {model_dir}", file=sys.stderr)
        return 1

    ll, src, conflicts = _load_llm_config(model_dir)
    win = ll.get("kv_cache_max_len")
    native = ll.get("max_position_embeddings")
    io_max = ll.get("max_io_tokens")

    if pretty:
        if win:
            print(f"上下文窗口(kv_cache_max_len)={win}")
        print(f"来源文件={src or '(未找到)'}"
              + ("  # 模型自带，权威" if src and not _looks_derived(src) else
                 "  # 本工具链生成，若为旧文件可能过时"))
        if conflicts:
            print("⚠ 以下文件里的值与此不一致（请确认模型目录是否混入了别的模型/旧文件）:")
            for n, v in conflicts:
                print(f"    {n}: {v}")
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
