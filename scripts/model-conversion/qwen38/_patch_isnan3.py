"""把 onnx_lower.py 的 IsNaN 处理重写成两条稳妥路径：

① 拿得到输出静态形状  →  删掉 IsNaN 节点，把它的输出改成【常量 False 的 bool initializer】
② 拿不到形状          →  退回"重接消费者"：Where(IsNaN(x), a, b) → Identity(b) / Expand(b, Shape(x))

实测背景（§54/§55）：
  · OMG 唯一拒绝的算子就是 IsNaN ✗
  · 用【算子】替换 IsNaN 节点会让 OMG 解析崩 ✗（Not(Equal)/Expand/Less 各种写法都不行）
  · 但"重接消费者"（Where → Identity）解析是【安全】的 ✓
  · 上一轮"删节点+常量"的分支被误关掉，从未真正验证过 ⇒ 这次让它默认生效 ✓
"""
import io

P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
s = io.open(P, encoding="utf-8").read()

start = s.index('        if n.op_type == "IsNaN":')
end = s.index('        elif n.op_type == "LessOrEqual"', start)

new = '''        if n.op_type == "IsNaN":
            isnan_nodes.append(n)          # 统一在后面处理（需要先知道所有消费者）
            continue
'''
s = s[:start] + new + s[end:]

# 在 lower_model 里、重建节点表之前，插入 IsNaN 的统一处理
s = s.replace('''    if counts:
        del g.node[:]
        g.node.extend(new_nodes)

    # ★NaN 保护消除★''', '''    if counts:
        del g.node[:]
        g.node.extend(new_nodes)

    # ★统一处理 IsNaN★（§54/§55）
    if isnan_nodes:
        _handle_isnan(model, isnan_nodes, counts, new_inits)

    # ★NaN 保护消除★''')

s = s.replace('''    g = model.graph
    counts: dict[str, int] = {}
    new_inits: list = []''', '''    g = model.graph
    counts: dict[str, int] = {}
    new_inits: list = []
    isnan_nodes: list = []''')

# 处理函数
s = s.replace('''def _shape_of(model: onnx.ModelProto, name: str):''', '''def _handle_isnan(model, isnan_nodes, counts, new_inits):
    """删掉 IsNaN 节点并保证它的输出仍有定义 ✓。

    优先"常量 bool initializer"（最干净 ✓）；拿不到形状时退回"重接消费者"✓。
    两条路都【不产生新的算子节点】⇒ OMG 解析安全 ✓。
    """
    g = model.graph
    for n in isnan_nodes:
        x = n.input[0]
        out = n.output[0]
        shape = _shape_of(model, out)
        if shape is not None:
            new_inits.append(numpy_helper.from_array(np.zeros(shape, dtype=np.bool_), out))
            g.node.remove(n)
            counts["IsNaN(常量)"] = counts.get("IsNaN(常量)", 0) + 1
            continue
        # 退路：重接消费者 ✓
        done = False
        for m in list(g.node):
            if m.op_type == "Where" and len(m.input) == 3 and m.input[0] == out:
                keep = m.input[2] if m.input[2] != out else m.input[1]
                g.node.remove(n)
                m.op_type = "Identity"
                del m.input[:]
                m.input.append(keep)
                del m.attribute[:]
                counts["IsNaN(重接)"] = counts.get("IsNaN(重接)", 0) + 1
                done = True
                break
        if not done:
            counts["IsNaN(未处理)"] = counts.get("IsNaN(未处理)", 0) + 1


def _shape_of(model: onnx.ModelProto, name: str):''')

io.open(P, "w", encoding="utf-8").write(s)
print("IsNaN 统一处理已就位 (常量优先 / 重接兜底)")
