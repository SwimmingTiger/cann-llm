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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


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
    """返回 [(文件名, llm_config 视图, 是否本工具链生成的派生物)]。

    判定逻辑与后端共用 ``cann_llm.modelcfg``，避免"脚本说 4096、后端认 2048"
    这种两套标准漂移。
    """
    from cann_llm.modelcfg import scan as _scan
    out = []
    for name, _val, derived in _scan(model_dir):
        out.append((name, _read_llm(name, os.path.join(model_dir, name)), derived))
    return out


def _load_llm_config(model_dir):
    """挑一份**权威**来源并报告冲突（详见 cann_llm.modelcfg 的说明）。

    优先级：模型自带的（非派生）> 本工具链生成的 executor.json。
    派生物若为旧文件会给出过时值，所以只作兜底。
    """
    from cann_llm.modelcfg import read_kv_cache_max_len, scan as _scan
    val, src = read_kv_cache_max_len(model_dir)
    if not src:
        return {}, "", []
    conflicts = [(n, v) for n, v, _d in _scan(model_dir) if v != val]
    return _read_llm(src, os.path.join(model_dir, src)), src, conflicts


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
