#!/usr/bin/env python3
"""把 cann-local-lm 模型目录里的**绝对路径改成相对路径**（目录可随意迁移/对外发布）。

改哪些：
  · ``model.json`` → ``embedding_config.embedding_weights``（只留文件名）
  · ``prefill/weight/meta.json`` / ``decode/weight/meta.json`` →
    ``hash_to_weight_file`` 的值（只留 ``delegate_graph_weight_combined``）

为什么这样能跑：
  · 引擎解析 ``weight/meta.json`` 里的相对文件名时，按该 meta 所在目录解析 ✓
  · ``model.json`` 的 embedding 路径引擎**不做相对解析**（会静默降级 ✗），
    所以 cann-llm 后端在 load() 时用 ``resolve_meta_paths`` 补成绝对再交给引擎 ✓

用法：
    python3 scripts/make_model_portable.py <模型目录> [<模型目录> ...] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

WEIGHT_META = "delegate_graph_weight_combined"


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def make_portable(model_dir: str, check: bool = False) -> "list[str]":
    """就地改写；返回改动的描述列表（check=True 只统计不改）。"""
    changes = []

    p = os.path.join(model_dir, "model.json")
    if os.path.isfile(p):
        m = _load(p)
        ec = m.get("embedding_config") or {}
        w = ec.get("embedding_weights")
        if isinstance(w, str) and os.path.isabs(w):
            rel = os.path.basename(w)
            if not os.path.isfile(os.path.join(model_dir, rel)):
                raise SystemExit(f"{model_dir}: 找不到 {rel}（embedding 文件缺失）")
            if not check:
                ec["embedding_weights"] = rel
                _dump(p, m)
            changes.append(f"model.json: embedding_weights -> {rel}")

    for stage in ("prefill", "decode"):
        p = os.path.join(model_dir, stage, "weight", "meta.json")
        if not os.path.isfile(p):
            continue
        m = _load(p)
        htw = m.get("hash_to_weight_file") or {}
        n = 0
        for k, v in list(htw.items()):
            if isinstance(v, str) and os.path.isabs(v):
                htw[k] = WEIGHT_META
                n += 1
        if n:
            if not check:
                _dump(p, m)
            changes.append(f"{stage}/weight/meta.json: {n} 处 -> {WEIGHT_META}")
    return changes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="模型目录绝对路径 → 相对路径")
    ap.add_argument("dirs", nargs="+", help="模型目录（可多个）")
    ap.add_argument("--check", action="store_true", help="只检查不修改")
    a = ap.parse_args(argv)
    total = 0
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"跳过（不是目录）：{d}")
            continue
        ch = make_portable(os.path.abspath(d), check=a.check)
        total += len(ch)
        tag = "待改" if a.check else "已改"
        print(f"{tag} {d}")
        for c in ch:
            print(f"  · {c}")
        if not ch:
            print("  （已经是可移植的相对路径 ✓）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
