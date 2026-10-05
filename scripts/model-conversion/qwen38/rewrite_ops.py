#!/usr/bin/env python3
"""图级算子改写：把 NPU kernel 库里【没有实现】的算子换成等价写法（§101 ✓）。

  Sub(A,B)          ⇒ Add(A, Mul(B, -1))          ✓
  Pow(x, 2)         ⇒ Mul(x, x)                    ✓（指数是常量 2 时）
用法：python rewrite_ops.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _const_scalar(name, value, dtype=np.float32):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name=name)


def main(src, dst):
    m = onnx.load(src)                      # 连外置权重一起读入 ✓（后面统一重存 ✓）
    g = m.graph
    new_nodes, extra_init = [], []
    n_sub = n_pow = 0
    for i, nd in enumerate(g.node):
        if nd.op_type == "Sub" and len(nd.input) == 2:
            # ★Sub(A,B) ⇒ Add(A, Mul(B,-1))★ —— 多一个常量初始化器 ✓
            cname = "__neg_one_%d" % i
            extra_init.append(_const_scalar(cname, -1.0))
            tmp = nd.output[0] + "__negB"
            new_nodes.append(helper.make_node("Mul", [nd.input[1], cname], [tmp], name=nd.name + "_mulneg"))
            new_nodes.append(helper.make_node("Add", [nd.input[0], tmp], list(nd.output), name=nd.name + "_add"))
            n_sub += 1
        elif nd.op_type == "Pow" and len(nd.input) == 2:
            # 指数是常量 2 时 ⇒ Mul(x,x) ✓
            ini = {t.name: t for t in g.initializer}.get(nd.input[1])
            if ini is not None and ini.dims == [1] if hasattr(ini, "dims") else False:
                try:
                    v = float(numpy_helper.to_array(ini).reshape(-1)[0])
                except Exception:
                    v = None
                if v == 2.0:
                    new_nodes.append(helper.make_node("Mul", [nd.input[0], nd.input[0]], list(nd.output),
                                                      name=nd.name + "_sq"))
                    n_pow += 1
                    continue
            new_nodes.append(nd)
        else:
            new_nodes.append(nd)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(extra_init)
    onnx.checker.check_model(m)
    # 大模型：统一存成单个外置权重文件 ✓
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.split("/")[-1] + ".weights", size_threshold=1024)
    print("  改写：Sub %d 处 · Pow(2) %d 处 ⇒ 写出 %s" % (n_sub, n_pow, dst))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
