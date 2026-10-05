"""重写 onnx_lower.py 的 IsNaN 处理：整表重建、不持有旧节点引用。

背景（§54/§55）：OMG 唯一拒绝 IsNaN ✗；用算子替换 IsNaN 节点会让解析崩 ✗；
但"重接消费者"（Where(IsNaN(x),a,b) → 取 b）解析安全 ✓。
这里两条路都做，且都用【整表重建】避免 protobuf 引用失效 ✗。
"""
import io

P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
s = io.open(P, encoding="utf-8").read()

# 主循环里：IsNaN 先记下名字
s = s.replace('''        if n.op_type == "IsNaN":
            isnan_nodes.append(n)          # 统一在后面处理（需要先知道所有消费者）
            continue
''', '''        if n.op_type == "IsNaN":
            isnan_nodes.append(n.output[0])    # 记【输出名】而不是节点对象（重建后会失效 ✗）
            continue
''')

s = s.replace('''    isnan_nodes: list = []''', '''    isnan_nodes: list = []          # IsNaN 的输出张量名 ✓''')

# 用新的实现替换 _handle_isnan
start = s.index('def _handle_isnan(')
end = s.index('def _shape_of(')
new = '''def _handle_isnan(model, isnan_outs, counts, new_inits):
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


'''
s = s[:start] + new + s[end:]
io.open(P, "w", encoding="utf-8").write(s)
print("_handle_isnan 已改为整表重建 ✓")
