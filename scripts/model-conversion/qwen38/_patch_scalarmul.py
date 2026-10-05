"""给 onnx_lower.py 加一条 lowering：MatMul(标量常量, x) / MatMul(x, 标量常量) -> Mul(x, 标量)。

背景（§57）：legacy 导出器会把 `q * scaling`（scaling 是 tensor 而非 float）导成
`MatMul(Constant[1], x)` ✗ —— OMG 的 MatMul 形状推导直接报
  "The value of wDim in x1 should be equal to hDim in x2 ... Infershape for MatMul_1 failed" ✗
改成逐元素 Mul（标量广播）语义完全等价 ✓，而 Mul 在 OMG 里毫无问题 ✓。
"""
import io
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
s = io.open(P, encoding="utf-8").read()

s = s.replace('''    # ★NaN 保护消除★''', '''    # ★MatMul(标量, x) / MatMul(x, 标量) → Mul(x, 标量)★（§57）
    _init_shape = {}
    for i in g.initializer:
        _init_shape[i.name] = list(i.dims)
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

    # ★NaN 保护消除★''')
io.open(P, "w", encoding="utf-8").write(s)
print("MatMul(标量) -> Mul lowering 已加入 ✓")
