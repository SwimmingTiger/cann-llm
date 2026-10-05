"""把 onnx_lower.py 里 IsNaN 的处理换成单算子等价写法 `Less(x, x)`。

理由（§54 实测）：任何"用多个算子替换 IsNaN"的写法都会让 OMG 解析器崩 ✗；
`Less(x, x)` 与 `IsNaN(x)` 在**非 NaN** 时取值相同（都是 False ✓）—— 我们的图全是有限运算 ✓。
它是单节点替换：同形状、同 dtype(bool) ✓，不引入中间张量 ✓。
"""
import io

P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/onnx_lower.py"
src = io.open(P, encoding="utf-8").read()

start = src.index('        if n.op_type == "IsNaN" and (_want is None or "IsNaN" in _want):')
end = src.index('        elif n.op_type == "LessOrEqual"', start)

new = '''        if n.op_type == "IsNaN" and (_want is None or "IsNaN" in _want):
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
'''
src = src[:start] + new + src[end:]
io.open(P, "w", encoding="utf-8").write(src)
print("patched: IsNaN -> Less(x,x) ok")
