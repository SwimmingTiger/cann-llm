# -*- coding: utf-8 -*-
"""从模型目录读出**权威**的 KV 缓存上限（= NPU 上真正可用的上下文窗口）。

为什么需要单独一个模块
----------------------
同一个值散落在好几个地方：官方模型自带的扁平 ``<model>.json``、本工具链生成的
``executor.json`` / ``context.json``，以及历史实验留下的 ``executor_*.json``。
它们可能**不一致**（派生物是旧的就过时了），所以：

* **优先非派生的**（模型自带的真相），派生物只作兜底
* 读不到就返回 ``None`` —— **绝不猜一个数字** ✗
  （猜出来的值会出现在 ``/v1/models`` 与"输入超出 KV 缓存"的提示里，
   拿错误的上限去误导人比不显示更糟）

注意这个值是**编译期固化**在模型里的：它等于 ONNX 里 ``past_key_in*`` 占位符的
形状第 0 维，转换/量化时由 ``kv_cache_max_len`` 定下，**改它要重新转换模型**。
它不是引擎的常量，所以不同模型不同（实测 7B=4096、1.5B=2048）。
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

#: 本工具链生成的派生物特征
_DERIVED_NAMES = ("executor.json", "context.json")
_DERIVED_PREFIXES = ("executor_", "context_")


def is_derived(name: str) -> bool:
    """这个 json 是不是本工具链生成的派生物（而非模型自带）。"""
    return name in _DERIVED_NAMES or name.startswith(_DERIVED_PREFIXES)


def _view(cfg: dict) -> dict:
    """把一份 json 归一到 llm_config 视图。

    官方 ``<model>.json`` 是**扁平**的（键直接在顶层）；本工具链生成的在
    ``llm_config`` 下嵌套一层 —— 两种都认。
    """
    inner = cfg.get("llm_config") if isinstance(cfg.get("llm_config"), dict) else None
    return inner if inner is not None else cfg


def scan(model_dir: str) -> list:
    """扫描目录，返回 ``[(文件名, 值, 是否派生物)]``（只含有该字段的文件）。"""
    out = []
    try:
        names = sorted(os.listdir(model_dir))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(model_dir, name)) as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(cfg, dict):
            continue
        val = _view(cfg).get("kv_cache_max_len")
        try:
            val = int(val) if val is not None else None
        except (TypeError, ValueError):
            val = None
        if val:
            out.append((name, val, is_derived(name)))
    return out


def read_kv_cache_max_len(model_dir: str) -> Tuple[Optional[int], str]:
    """返回 ``(上限, 来源文件名)``；读不到时是 ``(None, "")`` —— 不猜。"""
    found = scan(model_dir)
    if not found:
        return None, ""
    authoritative = [f for f in found if not f[2]]
    name, val, _ = (authoritative or found)[0]
    return val, name
