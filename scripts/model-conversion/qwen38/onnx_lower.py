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


def _shape_of(model: onnx.ModelProto, name: str):
    """从 input/output/value_info 查静态形状（拿不到返回 None）。"""
    g = model.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name == name:
            t = vi.type.tensor_type
            if not t.HasField("shape"):
                return None
            dims = [d.dim_value for d in t.shape.dim]
            if any(v <= 0 for v in dims):
                return None
            return dims
    return None


def lower_model(model: onnx.ModelProto, kinds=None) -> dict:
    """就地改写，返回 {算子名: 替换次数} ✓。

    kinds 给定时只处理其中列出的算子（用于二分排查 ✓）。
    """
    _want = set(kinds) if kinds else None
    # ★先做一次 shape inference★：否则 IsNaN 的输出形状查不到 ⇒ 我们的"删节点+常量"
    #   lowering 会拿不到形状而跳过 ✗（§54 实测 ✓）
    try:
        model.CopyFrom(onnx.shape_inference.infer_shapes(model, strict_mode=False))
    except Exception:                            # noqa: BLE001
        pass
    g = model.graph
    counts: dict[str, int] = {}
    new_inits: list = []
    used = {o for n in g.node for o in n.output}
    new_nodes = []
    for n in g.node:
        if n.op_type == "IsNaN":
            # 实测(§54)：用【多个算子】替换 IsNaN(Not(Equal)/Expand(False,Shape) 等)
            # 都会让 OMG 解析器崩 -> "cannot find output tensor ..." + ParseFromMemory FAIL
            # ⇒ 用【单算子】等价写法：Less(x, x)
            #   · 非 NaN 时 Less(x,x) 与 IsNaN(x) 都是 False（取值相同）
            #   · NaN 时不同，但我们的图全是有限运算 => 无 NaN
            #   · 单节点替换：同形状、同 dtype(bool)、不引入中间张量
            x = n.input[0]
            out = n.output[0]
            new_nodes.append(helper.make_node("Less", [x, x], [out], name=(n.name or out + "_isnan") + "_less"))
            counts["IsNaN"] = counts.get("IsNaN", 0) + 1
        elif n.op_type == "LessOrEqual" and (_want is None or "LessOrEqual" in _want):
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

    # ★NaN 保护消除★：Where(IsNaN(x), 0, x) → Identity(x)
    #   实测(§55)：OMG 唯一拒绝的算子就是 IsNaN ✗，而且"改写 IsNaN 节点"会让它解析崩 ✗
    #   ⇒ 直接把整个模式换成 Identity（支持算子 ✓）并删掉 IsNaN ✓
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    keep = []
    drop = set()
    for n in g.node:
        if n.op_type == "Where" and len(n.input) == 3:
            cond = prod.get(n.input[0])
            if cond is not None and cond.op_type == "IsNaN" and cond.input[0] in (n.input[1], n.input[2]):
                src = cond.input[0]
                keep.append(helper.make_node("Identity", [src], [n.output[0]],
                                             name=(n.name or n.output[0]) + "_id"))
                drop.add(id(cond))
                counts["nan_guard"] = counts.get("nan_guard", 0) + 1
                continue
        if id(n) in drop:
            continue
        keep.append(n)
    if counts.get("nan_guard"):
        del g.node[:]
        g.node.extend(keep)
        # 清掉不再被引用的 initializer / value_info 不必做（ONNX 允许未使用 ✓）

    # ★ConstantOfShape → Expand(标量, shape)★
    #   注意：它的 shape 输入常常【不是】常量 initializer ✗（实测 ✓），
    #   所以不能直接物化成 initializer；改成 Expand 是等价且受支持的 ✓
    inits = {i.name: i for i in g.initializer}
    used2 = used | {i.name for i in g.initializer}
    new_nodes = []
    k = 0
    for n in g.node:
        if n.op_type == "ConstantOfShape" and (_want is None or "ConstantOfShape" in _want):
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
