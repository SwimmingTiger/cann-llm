"""ONNX 图级 lowering：把 DDK 没有的算子换成等价的支持算子组合 ✓。

目前处理三个（都是 DDK 平台库没有的 ✗）：
    IsNaN(x)        →  Not(Equal(x, x))                 （NaN 检测等价写法 ✓）
    LessOrEqual(a,b)→  Not(Greater(a, b))               （等价 ✓）
    ConstantOfShape →  常量 initializer                 （形状是常量 ⇒ 直接物化 ✓）

用法：
    python onnx_lower.py 输入.onnx 输出.onnx
或在导出脚本里直接调用 `lower_model(model)` ✓。
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def lower_model(model: onnx.ModelProto) -> dict:
    """就地改写，返回 {算子名: 替换次数} ✓。"""
    g = model.graph
    counts: dict[str, int] = {}
    new_inits: list = []
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
        elif n.op_type == "LessOrEqual":
            a, b, out = n.input[0], n.input[1], n.output[0]
            base = n.name or (out + "_le")
            gt = base + "_gt"
            while gt in used:
                gt += "_"
            used.add(gt)
            new_nodes.append(helper.make_node("Greater", [a, b], [gt], name=base + "_gt"))
            new_nodes.append(helper.make_node("Not", [gt], [out], name=base + "_not"))
            counts["LessOrEqual"] = counts.get("LessOrEqual", 0) + 1
        else:
            new_nodes.append(n)
    if counts:
        del g.node[:]
        g.node.extend(new_nodes)

    # ★ConstantOfShape → Expand(标量, shape)★
    #   注意：它的 shape 输入常常【不是】常量 initializer ✗（实测 ✓），
    #   所以不能直接物化成 initializer；改成 Expand 是等价且受支持的 ✓
    inits = {i.name: i for i in g.initializer}
    used2 = used | {i.name for i in g.initializer}
    new_nodes = []
    k = 0
    for n in g.node:
        if n.op_type == "ConstantOfShape":
            val = 0.0
            for attr in n.attribute:
                if attr.name == "value":
                    val = float(numpy_helper.to_array(attr.t).reshape(-1)[0])
            cname = (n.name or n.output[0]) + "_scalar"
            while cname in used2:
                cname += "_"
            used2.add(cname)
            new_inits.append(numpy_helper.from_array(np.array([val], dtype=np.float32), cname))
            new_nodes.append(helper.make_node("Expand", [cname, n.input[0]], [n.output[0]],
                                              name=(n.name or n.output[0]) + "_expand"))
            counts["ConstantOfShape"] = counts.get("ConstantOfShape", 0) + 1
            k += 1
        else:
            new_nodes.append(n)
    if k:
        del g.node[:]
        g.node.extend(new_nodes)
        g.initializer.extend(new_inits)
    return counts


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    counts = lower_model(m)
    onnx.save(m, dst)
    print("lowering: %s ⇒ 已写 %s" % (counts or "无需改写 ✓", dst))


if __name__ == "__main__":
    main()


def fix_static_shapes(model: onnx.ModelProto, verbose: bool = True) -> dict:
    """修掉旧导出器留下的 `0` 维（如输出 `[0,0,2048]` ✗），并用 shape inference 补全 ✓。"""
    g = model.graph
    fixed = {"dims": 0}
    # ① 图输入是准的 ⇒ 用它去修同形状的输出 ✓
    in_shapes = {}
    for i in g.input:
        t = i.type.tensor_type
        in_shapes[i.name] = [d.dim_value for d in t.shape.dim]
    for o in g.output:
        t = o.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if any(v == 0 for v in dims):
            # 同名的输入（body 模型里输入输出同形 ✓）或首个输入的形状 ✓
            cand = in_shapes.get(o.name) or next(iter(in_shapes.values()))
            if len(cand) == len(dims):
                for k, d in enumerate(t.shape.dim):
                    if d.dim_value == 0:
                        d.dim_value = cand[k]
                        fixed["dims"] += 1
    # ② 用 shape inference 补中间张量 ✓
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        model.CopyFrom(inferred)
    except Exception as e:                      # noqa: BLE001
        if verbose:
            print("   shape inference 失败（忽略）：%s" % e)
    # ③ 仍有 0 维的中间张量 ⇒ 改成 1（避免 OMG 解析失败 ✗）
    for vi in list(model.graph.value_info):
        t = vi.type.tensor_type
        for d in t.shape.dim:
            if d.HasField("dim_value") and d.dim_value == 0:
                d.dim_value = 1
                fixed["dims"] += 1
    if verbose:
        print("   修了 %d 个 0 维 ✓" % fixed["dims"])
    return fixed
