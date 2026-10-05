"""修正：标量常量也可能来自 Constant 节点（不止 initializer）✓。"""
import io
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
s = io.open(P, encoding="utf-8").read()
s = s.replace('''    _init_shape = {}
    for i in g.initializer:
        _init_shape[i.name] = list(i.dims)''',
'''    _init_shape = {}
    for i in g.initializer:
        _init_shape[i.name] = list(i.dims)
    for _n in g.node:                       # ★Constant 节点也算★（legacy 导出器很爱用它 ✗）
        if _n.op_type == "Constant":
            for _a in _n.attribute:
                if _a.name == "value":
                    _init_shape[_n.output[0]] = list(_a.t.dims)
                    break''')
io.open(P, "w", encoding="utf-8").write(s)
print("已把 Constant 节点纳入 ✓")
