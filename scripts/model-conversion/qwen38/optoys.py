"""造一批"单算子小图"ONNX，用来在设备上二分：哪个算子真正被 NPU 内核拒收 ✗。

背景（§51）：qwen3_5 改写的图在 converter_lite 里转得出来 ✓，但设备上
`OH_AI_ModelBuildFromFile` 静默失败 ✗（rc=-1/-2）⇒ 需要逐个算子试 ✓。
注意：平台库里"出现该算子的字符串"≠"内核实现了它" ✗ —— 这是本轮的教训 ✓。

跑法：~/q38env/bin/python optoys.py     # 生成 toys/<op>.onnx
"""
from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = os.path.expanduser("~/q38/toys")
SHAPE = [1, 64]


def _graph(nodes, name, inputs=None, outputs=None, inits=None, extra_inputs=()):
    inputs = inputs or [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)]
    outputs = outputs or [helper.make_tensor_value_info("y", TensorProto.FLOAT, SHAPE)]
    g = helper.make_graph(nodes, name, inputs, outputs, inits or [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    return m


def _c(name, arr, dtype=None):
    arr = np.asarray(arr, dtype=dtype)
    return numpy_helper.from_array(arr, name)


def toys():
    """返回 {名字: ModelProto}，每个只包含一个（或一小簇）待测算子 ✓。"""
    t = {}
    # 已知能建图的对照组 ✓（gemma4 用过的算子）
    t["matmul"] = _graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "matmul",
                         inits=[_c("w", np.random.randn(64, 64).astype(np.float32) * 0.1)])
    t["pow_reducemean"] = _graph(
        [helper.make_node("Pow", ["x", "two"], ["p"]),
         helper.make_node("ReduceMean", ["p"], ["y"], keepdims=1)], "pow_reducemean",
        inits=[_c("two", np.array(2.0, dtype=np.float32))])
    # 待测算子 ✓
    t["softplus"] = _graph([helper.make_node("Softplus", ["x"], ["y"])], "softplus")
    t["greater"] = _graph([helper.make_node("Greater", ["x", "zero"], ["b"]),
                           helper.make_node("Cast", ["b"], ["y"], to=TensorProto.FLOAT)],
                          "greater", inits=[_c("zero", np.array(0.0, dtype=np.float32))])
    t["equal"] = _graph([helper.make_node("Equal", ["x", "zero"], ["b"]),
                         helper.make_node("Cast", ["b"], ["y"], to=TensorProto.FLOAT)],
                        "equal", inits=[_c("zero", np.array(0.0, dtype=np.float32))])
    t["not"] = _graph([helper.make_node("Equal", ["x", "zero"], ["b"]),
                       helper.make_node("Not", ["b"], ["nb"]),
                       helper.make_node("Cast", ["nb"], ["y"], to=TensorProto.FLOAT)],
                      "not", inits=[_c("zero", np.array(0.0, dtype=np.float32))])
    t["and"] = _graph([helper.make_node("Greater", ["x", "zero"], ["a"]),
                       helper.make_node("Less", ["x", "one"], ["b"]),
                       helper.make_node("And", ["a", "b"], ["c"]),
                       helper.make_node("Cast", ["c"], ["y"], to=TensorProto.FLOAT)],
                      "and", inits=[_c("zero", np.array(0.0, dtype=np.float32)),
                                    _c("one", np.array(1.0, dtype=np.float32))])
    t["where"] = _graph([helper.make_node("Greater", ["x", "zero"], ["b"]),
                         helper.make_node("Where", ["b", "x", "zero"], ["y"])],
                        "where", inits=[_c("zero", np.array(0.0, dtype=np.float32))])
    t["expand"] = _graph([helper.make_node("Shape", ["x"], ["s"]),
                          helper.make_node("Expand", ["x", "s"], ["y"])], "expand")
    t["split"] = _graph([helper.make_node("Split", ["x"], ["a", "b"], axis=1, num_outputs=2),
                         helper.make_node("Concat", ["a", "b"], ["y"], axis=1)], "split")
    t["squeeze_unsqueeze"] = _graph([helper.make_node("Unsqueeze", ["x", "ax"], ["u"]),
                                     helper.make_node("Squeeze", ["u", "ax"], ["y"])],
                                    "squeeze_unsqueeze",
                                    inits=[_c("ax", np.array([0], dtype=np.int64))])
    t["sqrt_recip"] = _graph([helper.make_node("Abs", ["x"], ["a"]),
                              helper.make_node("Add", ["a", "one"], ["a1"]),
                              helper.make_node("Sqrt", ["a1"], ["s"]),
                              helper.make_node("Reciprocal", ["s"], ["y"])], "sqrt_recip",
                             inits=[_c("one", np.array(1.0, dtype=np.float32))])
    t["div_sub_exp_neg"] = _graph([helper.make_node("Exp", ["x"], ["e"]),
                                   helper.make_node("Neg", ["e"], ["n"]),
                                   helper.make_node("Sub", ["n", "x"], ["s"]),
                                   helper.make_node("Div", ["s", "two"], ["y"])], "div_sub_exp_neg",
                                  inits=[_c("two", np.array(2.0, dtype=np.float32))])
    t["cos_sin"] = _graph([helper.make_node("Cos", ["x"], ["c"]),
                           helper.make_node("Sin", ["c"], ["y"])], "cos_sin")
    t["pad"] = _graph([helper.make_node("Pad", ["x", "pads"], ["y"], mode="constant")], "pad",
                      inits=[_c("pads", np.array([0, 0, 0, 0], dtype=np.int64))])
    t["gather"] = _graph([helper.make_node("Gather", ["x", "idx"], ["y"], axis=1)], "gather",
                         inits=[_c("idx", np.arange(64, dtype=np.int64))])
    t["reducesum"] = _graph([helper.make_node("ReduceSum", ["x", "ax"], ["y"], keepdims=1)],
                            "reducesum", inits=[_c("ax", np.array([1], dtype=np.int64))])
    t["sigmoid_softmax"] = _graph([helper.make_node("Sigmoid", ["x"], ["s"]),
                                   helper.make_node("Softmax", ["s"], ["y"], axis=-1)],
                                  "sigmoid_softmax")
    toy_ops = sorted(t)
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    t = toys()
    for name, m in t.items():
        p = os.path.join(OUT, name + ".onnx")
        onnx.save(m, p)
        print("  %-20s %6d B" % (name, os.path.getsize(p)))
    print("共 %d 个 ✓ ⇒ %s" % (len(t), OUT))


if __name__ == "__main__":
    main()
