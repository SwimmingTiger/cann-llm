"""ONNX 图级 lowering：把 DDK 没有的算子换成等价的支持算子组合 ✓。

目前只处理一个：
    IsNaN(x)  →  Not(Equal(x, x))        （NaN 检测的等价写法 ✓，Equal/Not 都是支持算子 ✓）

用法：
    python onnx_lower.py 输入.onnx 输出.onnx
或在导出脚本里直接调用 `lower_model(model)` ✓。
"""
from __future__ import annotations

import sys

import onnx
from onnx import helper


def lower_model(model: onnx.ModelProto) -> dict:
    """就地改写，返回 {算子名: 替换次数} ✓。"""
    g = model.graph
    counts: dict[str, int] = {}
    used = {o for n in g.node for o in n.output}
    new_nodes = []
    for n in g.node:
        if n.op_type == "IsNaN":
            x = n.input[0]
            out = n.output[0]
            base = n.name or (out + "_isnan")
            eq = base + "_eq"
            while eq in used:
                eq += "_"
            used.add(eq)
            new_nodes.append(helper.make_node("Equal", [x, x], [eq], name=base + "_eq"))
            new_nodes.append(helper.make_node("Not", [eq], [out], name=base + "_not"))
            counts["IsNaN"] = counts.get("IsNaN", 0) + 1
        else:
            new_nodes.append(n)
    if counts:
        del g.node[:]
        g.node.extend(new_nodes)
    return counts


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    counts = lower_model(m)
    onnx.save(m, dst)
    print("lowering: %s ⇒ 已写 %s" % (counts or "无需改写 ✓", dst))


if __name__ == "__main__":
    main()
