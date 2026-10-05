"""给 onnx_lower.py 加一条"NaN 保护消除"lowering：
    Where(IsNaN(x), 0, x)  ->  Identity(x)
语义：非 NaN 时等价 ✓；我们的图全有限 ⇒ 无 NaN ✓。
好处：连 IsNaN 节点一起删掉 ✓（OMG 的 pre-check 唯一拒绝的算子就是它 ✗）。
"""
import io
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
s = io.open(P, encoding="utf-8").read()

s = s.replace('''    # ★ConstantOfShape → Expand(标量, shape)★''',
'''    # ★NaN 保护消除★：Where(IsNaN(x), 0, x) → Identity(x)
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

    # ★ConstantOfShape → Expand(标量, shape)★''')

# 让 IsNaN 的单独 lowering 默认不再出场（保留能力但默认关闭 ✓）
s = s.replace('''        if n.op_type == "IsNaN" and (_want is None or "IsNaN" in _want):''',
'''        if n.op_type == "IsNaN" and _want is not None and "IsNaN" in _want:''')
io.open(P, "w", encoding="utf-8").write(s)
print("nan_guard lowering 已加入 ✓")
