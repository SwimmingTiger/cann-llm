#!/usr/bin/env python3
"""把 cann-local-lm 导出的 host 权重拼成引擎要的【四段】embedding 文件。

引擎的 embedding 模块只吃**一个文件、四段紧排**（无文件头、无段间 padding）：

===========  ==============================================  ==================
段           长度                                             来源
===========  ==============================================  ==================
seg1 主表    vocab × hidden × elem(main_type)                 host_weights 里的 embed_tokens
seg2 per层表  vocab × W × elem(identity_type)                 host_weights 里的 per_layer 表（W=行宽）
seg3 主scale  vocab × groups × elem(scale_type)               **填 1.0**
seg4 idscale  vocab × elem(identity_scale_type)               **填 1.0**
===========  ==============================================  ==================

★scale 段为什么填 1.0★
    图里已经乘过一次 ``embed_scale = sqrt(hidden_size)``；scale 工位再填 sqrt 就变成
    平方 ⇒ 输出乱码 ✗（实测：填 1.0 才语义正确 ✓）。

用法::

    python3 scripts/make_embedding_weights.py \\
        <epair目录>/host_weights_0.bin <out>/embedding_weights.bin

``host_weights_info.json``（同目录）用来取两个表的形状；没有时按 gemma-4-E2B-it
的默认形状（262144×1536 与 262144×8960，fp16）。

注意：本脚本只做**拼接 + 校验总长**，不改变权重数值 —— 权重本身来自 converter。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

#: gemma-4-E2B-it 的默认形状（host_weights_info.json 缺失时兜底）
DEFAULT_SHAPES = {
    "embed_tokens": (262144, 1536),
    "embed_tokens_per_layer": (262144, 8960),
}

ELEM = {"fp16": 2, "fp32": 4}          # 本脚本只支持非量化（无 scale 段）的导出


def _shapes_from_info(info_path: str) -> "list[tuple[str, tuple[int, int]]]":
    """从 host_weights_info.json 读 [(名字, (rows, cols))]，按出现顺序。"""
    with open(info_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: "list[tuple[str, tuple[int, int]]]" = []
    for _file, entries in data.items():
        for entry in entries:
            shape = entry.get("shape") or []
            if len(shape) == 2:
                out.append((str(entry.get("name", "")), (int(shape[0]), int(shape[1]))))
    return out


def build(src: str, dst: str, main_type: str, identity_type: str,
          vocab: int | None, hidden: int | None, width: int | None) -> int:
    """拼接四段并返回文件总长。"""
    info = os.path.join(os.path.dirname(os.path.abspath(src)), "host_weights_info.json")
    shapes: "list[tuple[str, tuple[int, int]]]" = []
    if os.path.isfile(info):
        try:
            shapes = _shapes_from_info(info)
        except (OSError, ValueError) as exc:
            print(f"  ! 读不了 {info}（{exc}），回退默认形状", file=sys.stderr)
    if not shapes:
        shapes = list(DEFAULT_SHAPES.items())

    v = vocab or shapes[0][1][0]
    h = hidden or shapes[0][1][1]
    w = width or (shapes[1][1][1] if len(shapes) > 1 else 0)

    seg1 = v * h * ELEM[main_type]
    seg2 = v * w * ELEM[identity_type] if w else 0
    # scale 段统一按 fp16 写 1.0（与 model.json 的 *_scale_type: fp16 对应）
    seg3 = v * 2
    seg4 = v * 2 if w else 0
    expect = seg1 + seg2 + seg3 + seg4

    size = os.path.getsize(src)
    if size < seg1 + seg2:
        raise SystemExit(
            f"host_weights 太小：{size} B < seg1+seg2={seg1 + seg2} B"
            f"（vocab={v}, hidden={h}, W={w}）")

    with open(src, "rb") as f, open(dst, "wb") as o:
        left = seg1 + seg2
        while left:
            block = f.read(min(1 << 20, left))
            if not block:
                raise SystemExit("host_weights 提前结束")
            o.write(block)
            left -= len(block)
        one = struct.pack("<e", 1.0)       # fp16 的 1.0
        o.write(one * v)                   # seg3 主 scale
        if w:
            o.write(one * v)               # seg4 identity scale

    got = os.path.getsize(dst)
    print(f"  seg1={seg1} seg2={seg2} seg3={seg3} seg4={seg4} expect={expect} got={got}")
    if got != expect:
        raise SystemExit("总长与四段和不一致（scale 段/形状对不上）")
    return got


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="拼四段 embedding_weights")
    ap.add_argument("src", help="epair 的 host_weights_0.bin")
    ap.add_argument("dst", help="输出 embedding_weights.bin（设备上用的那个）")
    ap.add_argument("--main-type", default="fp16", choices=sorted(ELEM))
    ap.add_argument("--identity-type", default="fp16", choices=sorted(ELEM))
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--width", type=int, default=None, help="per_layer 行宽 W（Ple 宽度×层数）")
    args = ap.parse_args(argv)
    n = build(args.src, args.dst, args.main_type, args.identity_type,
              args.vocab, args.hidden, args.width)
    print(f"✓ 写出 {args.dst}（{n} B）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
