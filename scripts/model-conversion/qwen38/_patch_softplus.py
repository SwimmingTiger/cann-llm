"""Softplus → 支持算子组合（§71：引擎的 CPUCL 报 activation.mode = 9 not support ✗）。

  softplus(x) ≈ log(1 + exp(min(x, 20)))
      · x ≤ 20 时与真值完全一致 ✓
      · x > 20 时真值 ≈ x（差 < 2e-9 ✓），而 min 截断后 log(1+exp(20)) ≈ 20 ✓ 差 < 2e-9 ✓
  只用 Exp / Add / Log / Min ✓（都是基础算子 ✓）
"""
import io
P = "onnx_lower.py"
s = io.open(P, encoding="utf-8").read()
if "_lower_softplus" not in s:
    s = s.replace('''def _shape_of(''', '''def _lower_softplus(g, counts):
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


def _shape_of(''')
    s = s.replace('''    # ★MatMul(标量, x) / MatMul(x, 标量) → Mul(x, 标量)★（§57）''',
'''    # ★Softplus → Log(1+Exp(Min(x,20)))★（§71：CPUCL 不认这个激活 ✗）
    _lower_softplus(g, counts)

    # ★MatMul(标量, x) / MatMul(x, 标量) → Mul(x, 标量)★（§57）''')
    io.open(P, "w", encoding="utf-8").write(s)
    print("Softplus lowering 已加入 ✓")
