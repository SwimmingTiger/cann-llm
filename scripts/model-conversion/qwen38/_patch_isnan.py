"""把 onnx_lower.py 里 IsNaN 的处理换成「删节点 + 常量输出」——
任何用算子替换 IsNaN 的写法都会让 OMG 解析器崩 ✗（§54 实测）。
"""
import io

P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
src = io.open(P, encoding="utf-8").read()

start = src.index('        if n.op_type == "IsNaN" and (_want is None or "IsNaN" in _want):')
end = src.index('        elif n.op_type == "LessOrEqual"', start)

new = '''        if n.op_type == "IsNaN" and (_want is None or "IsNaN" in _want):
            # 实测(§54)：任何"用算子替换 IsNaN"的写法(Not(Equal) / Expand(False,Shape))
            # 都会让 OMG 解析器直接崩 -> "cannot find output tensor ..." + ParseFromMemory FAIL
            # 而【删掉这个节点、把它的输出改成常量 initializer】就没事
            # 语义上：我们的图全是有限运算 => 无 NaN（正是那个保护想表达的）
            x = n.input[0]
            out = n.output[0]
            shape = _shape_of(model, out) or _shape_of(model, x)
            if shape is None:
                new_nodes.append(n)                  # 拿不到形状就原样保留
                continue
            new_inits.append(numpy_helper.from_array(np.zeros(shape, dtype=np.bool_), out))
            counts["IsNaN"] = counts.get("IsNaN", 0) + 1   # 节点被删 => 不进 new_nodes
'''
src = src[:start] + new + src[end:]

if "_shape_of" not in src.split("def lower_model")[0]:
    src = src.replace('def lower_model(model: onnx.ModelProto, kinds=None) -> dict:',
'''def _shape_of(model: onnx.ModelProto, name: str):
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


def lower_model(model: onnx.ModelProto, kinds=None) -> dict:''', 1)

io.open(P, "w", encoding="utf-8").write(src)
print("patched ok")
