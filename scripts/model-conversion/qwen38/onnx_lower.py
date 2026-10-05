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


def _handle_isnan(model, isnan_outs, counts, new_inits):
    """处理所有 IsNaN（整表重建 ✓）。

    ① 若它的消费者是 `Where(cond, a, b)`（cond 即它）⇒ 把该 Where 重接成 `Identity(b)`
       （非 NaN 时 IsNaN=False ⇒ Where 取 b ✓，语义等价 ✓，且解析安全 ✓）
    ② 剩下的 ⇒ 删节点 + 用常量 False 的 bool initializer 顶替它的输出 ✓
    """
    g = model.graph
    if not isnan_outs:
        return
    outs = set(isnan_outs)
    nodes = list(g.node)
    cond_by_tensor = {o: n for n in nodes for o in n.output}
    # ① 先找 Where 消费者
    keep_nodes = []
    for m in nodes:
        if m.op_type == "Where" and len(m.input) == 3 and m.input[0] in outs:
            cond = cond_by_tensor.get(m.input[0])
            if cond is not None and cond.input[0] in (m.input[1], m.input[2]):
                src = m.input[2] if m.input[2] != cond.input[0] else m.input[1]
                keep_nodes.append(helper.make_node("Identity", [src], [m.output[0]],
                                                   name=(m.name or m.output[0]) + "_noguard"))
                counts["IsNaN(重接)"] = counts.get("IsNaN(重接)", 0) + 1
                continue
        keep_nodes.append(m)
    nodes = keep_nodes
    # ② 剩下的 IsNaN：删节点 + 常量输出
    rest = []
    handled = set()
    for m in nodes:
        if m.op_type == "IsNaN":
            out = m.output[0]
            shp = _shape_of(model, out)
            if shp is not None:
                new_inits.append(numpy_helper.from_array(np.zeros(shp, dtype=np.bool_), out))
                handled.add(out)
                counts["IsNaN(常量)"] = counts.get("IsNaN(常量)", 0) + 1
                continue
            counts["IsNaN(未处理)"] = counts.get("IsNaN(未处理)", 0) + 1
        rest.append(m)
    del g.node[:]
    g.node.extend(rest)


def _lower_softplus(g, counts):
    """Softplus(x) → Log(Add(1, Exp(Min(x, 20)))) ✓（CPUCL 不认 Softplus 激活 ✗）。"""
    if not any(n.op_type == "Softplus" for n in g.node):
        return
    used = {i.name for i in g.initializer}
    for n in g.node:
        used.update(n.output)
    new_inits = []
    nodes = []
    k = 0
    for n in g.node:
        if n.op_type != "Softplus":
            nodes.append(n)
            continue
        x = n.input[0]
        base = n.name or (n.output[0] + "_sp")
        names = []
        for suf in ("_c20", "_min", "_exp", "_one", "_add", "_log"):
            nm = base + suf
            while nm in used:
                nm += "_"
            used.add(nm)
            names.append(nm)
        c20, mn, ex, one, add, lg = names
        new_inits.append(numpy_helper.from_array(np.array(20.0, dtype=np.float32), c20))
        new_inits.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), one))
        nodes.append(helper.make_node("Min", [x, c20], [mn], name=base + "_min"))
        nodes.append(helper.make_node("Exp", [mn], [ex], name=base + "_exp"))
        nodes.append(helper.make_node("Add", [ex, one], [add], name=base + "_add"))
        nodes.append(helper.make_node("Log", [add], [n.output[0]], name=base + "_log"))
        counts["Softplus"] = counts.get("Softplus", 0) + 1
        k += 1
    if k:
        del g.node[:]
        g.node.extend(nodes)
        g.initializer.extend(new_inits)


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
    isnan_nodes: list = []          # IsNaN 的输出张量名 ✓
    used = {o for n in g.node for o in n.output}
    new_nodes = []
    for n in g.node:
        if n.op_type == "IsNaN":
            isnan_nodes.append(n.output[0])    # 记【输出名】而不是节点对象（重建后会失效 ✗）
            continue
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

    # ★统一处理 IsNaN★（§54/§55）
    if isnan_nodes:
        _handle_isnan(model, isnan_nodes, counts, new_inits)

    # ★Softplus → Log(1+Exp(Min(x,20)))★（§71：CPUCL 不认这个激活 ✗）
    _lower_softplus(g, counts)

    # ★MatMul(标量, x) / MatMul(x, 标量) → Mul(x, 标量)★（§57）
    _init_shape = {}
    for i in g.initializer:
        _init_shape[i.name] = list(i.dims)
    for _n in g.node:                       # ★Constant 节点也算★（legacy 导出器很爱用它 ✗）
        if _n.op_type == "Constant":
            for _a in _n.attribute:
                if _a.name == "value":
                    _init_shape[_n.output[0]] = list(_a.t.dims)
                    break
    new_nodes = []
    for n in g.node:
        if n.op_type == "MatMul" and len(n.input) == 2:
            a, bb = n.input[0], n.input[1]
            sa, sb = _init_shape.get(a), _init_shape.get(bb)
            if sa is not None and (len(sa) == 0 or (len(sa) == 1 and sa[0] == 1)):
                new_nodes.append(helper.make_node("Mul", [bb, a], [n.output[0]], name=n.name + "_mul"))
                counts["MatMul(标量)"] = counts.get("MatMul(标量)", 0) + 1
                continue
            if sb is not None and (len(sb) == 0 or (len(sb) == 1 and sb[0] == 1)):
                new_nodes.append(helper.make_node("Mul", [a, bb], [n.output[0]], name=n.name + "_mul"))
                counts["MatMul(标量)"] = counts.get("MatMul(标量)", 0) + 1
                continue
        new_nodes.append(n)
    if counts.get("MatMul(标量)"):
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
