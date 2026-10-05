#!/usr/bin/env python3
"""★通用 ONNX → NNRt 在线建图【表】生成器★（§134 方案 A ✓）

输出一个紧凑的文本表（graph.txt ✓），由 nnrt_build.cpp 读取并调用 NNRt 在线 API 建图 ✓。
这样几千个节点的模型也不会生成巨型 C++ ✓。

表格式（每行一条，字段空格分隔 ✓）：
  T <idx> <名称> <dtype> <rank> <dim0> <dim1> …          # 一张张量（dtype: F32/I32/I64/B)
  I <idx> <权重文件> <偏移> <字节数>                    # 该张量是常量 ⇒ 从文件读数据 SetTensorData ✓
  N <op> <参数个数> <参数idx…> <输入个数> <输入idx…> <输出个数> <输出idx…>
  P <模型输入idx…> | <模型输出idx…>                     # 用 'I' 与 'O' 两行分别给出 ✓
  IN <idx…>       # 模型输入
  OUT <idx…>      # 模型输出

★参数张量表（PARAM_TABLE ✓）★：每个算子要哪些参数张量、什么类型、什么取值 —— 一次写好长期可用 ✓
★注意★：NNRt 对秩 0 张量的 dimensions 必须传 nullptr ✓（§132 踩过 ✓）
"""
import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

DT = {TensorProto.FLOAT: "F32", TensorProto.FLOAT16: "F16", TensorProto.INT32: "I32",
      TensorProto.INT64: "I64", TensorProto.BOOL: "B", TensorProto.INT8: "I8",
      TensorProto.UINT8: "U8", TensorProto.DOUBLE: "F64"}
# ONNX elem_type → NNRt OH_NN_DataType（数值取自头文件 ✓ OH_NN_FLOAT32=11 等）
NN_DT = {"F32": 11, "F16": 0, "I32": 4, "I64": 7, "B": 12, "I8": 3, "U8": 6, "F64": 13}

# ★算子 → NNRt op 号★（数值取自 neural_network_runtime_type.h 的 OH_NN_OPS_* ✓）
OP = {  # ★算子号一律从 neural_network_runtime_type.h 的 OH_NN_OPS_* 自动抽取✓★（§137 修正 ✗ 之前手写的几乎全错）
    "Add": 1, "Cast": 6, "Concat": 7, "ConstantOfShape": 68, "Div": 11, "Exp": 60,
    "Gather": 16, "Log": 80, "MatMul": 19, "Mul": 22, "Neg": 84, "Pad": 24, "Pow": 25,
    "ReduceMean": 42, "ReduceSum": 101, "Reshape": 45, "Rsqrt": 44, "Sigmoid": 28,
    "Slice": 29, "Softmax": 30, "Split": 32, "Sqrt": 33, "Squeeze": 35, "Sub": 38,
    "Tanh": 39, "Transpose": 41, "Unsqueeze": 55, "Where": 62,   # Where ⇒ NNRt 的 SELECT ✓
    "Tile": 40, "Maximum": 20, "Minimum": 97, "Greater": 71, "Less": 61, "Equal": 70,
    "Softplus": -1, "Constant": -1,
}
# ★每个算子需要的参数张量：参数枚举名 → (类型, 取值)，取值 None 表示从 ONNX 属性取✓★
PARAM_TABLE = {
    "MatMul":   [("OH_NN_MATMUL_TRANSPOSE_A", "B", 0), ("OH_NN_MATMUL_TRANSPOSE_B", "B", 0)],
    "Add":      [("OH_NN_ADD_ACTIVATIONTYPE", "I32", 0)],
    "Sub":      [("OH_NN_SUB_ACTIVATIONTYPE", "I32", 0)],
    "Mul":      [("OH_NN_MUL_ACTIVATION_TYPE", "I32", 0)],
    "Div":      [("OH_NN_DIV_ACTIVATIONTYPE", "I32", 0)],
    "Concat":   [("OH_NN_CONCAT_AXIS", "I32", None)],
    "ReduceMean": [("OH_NN_REDUCE_MEAN_KEEP_DIMS", "B", 1)],
    "ReduceSum":  [("OH_NN_REDUCE_SUM_KEEP_DIMS", "B", 1)],
    "Unsqueeze": [("OH_NN_UNSQUEEZE_AXIS", "I32", None)],
    "Squeeze":  [("OH_NN_SQUEEZE_AXIS", "I32", None)],
    "Pad":      [("OH_NN_PAD_PADDING_MODE", "I32", 0)],
    "Split":    [("OH_NN_SPLIT_AXIS", "I32", None), ("OH_NN_SPLIT_OUTPUT_NUM", "I32", None)],
    "Pow":      [("OH_NN_POW_SCALE", "F32", 1.0), ("OH_NN_POW_SHIFT", "F32", 0.0)],
    "Slice":    [("OH_NN_SLICE_AXES", "I32", None)],
}

PARAM_ENUM = {
    "OH_NN_MATMUL_TRANSPOSE_A": 33,
    "OH_NN_MATMUL_TRANSPOSE_B": 34,
    "OH_NN_CONCAT_AXIS": 10,
    "OH_NN_ADD_ACTIVATIONTYPE": 1,
    "OH_NN_SUB_ACTIVATIONTYPE": 59,
    "OH_NN_MUL_ACTIVATION_TYPE": 41,
    "OH_NN_DIV_ACTIVATIONTYPE": 29,
    "OH_NN_EXP_BASE": 89,
    "OH_NN_EXP_SCALE": 90,
    "OH_NN_EXP_SHIFT": 91,
    "OH_NN_POW_SCALE": 107,
    "OH_NN_POW_SHIFT": 108,
    "OH_NN_REDUCE_MEAN_KEEP_DIMS": 60,
    "OH_NN_REDUCE_MEAN_REDUCE_TO_END": 117,
    "OH_NN_REDUCE_MEAN_COEFF": 118,
    "OH_NN_UNSQUEEZE_AXIS": 77,
    "OH_NN_SQUEEZE_AXIS": 52,
    "OH_NN_PAD_PADDING_MODE": 116,
    "OH_NN_PAD_CONSTANT_VALUE": 43,
    "OH_NN_SPLIT_AXIS": 49,
    "OH_NN_SPLIT_OUTPUT_NUM": 50,
    "OH_NN_SPLIT_SIZE_SPLITS": 51,
    "OH_NN_SLICE_AXES": 127,
}

# NNRt 参数枚举的字面名 → 我们输出给 C++ 的整数（C++ 里用同名常量 ✓ 所以直接写字面名 ✓）


def pre_lower(m):
    """把 NNRt 没有的算子就地展开 ✓（目前：Softplus = Log(1 + Exp(x)) ✓）"""
    from onnx import helper, TensorProto
    g = m.graph
    new_nodes = []
    extra = []
    n_low = 0
    for n in g.node:
        if n.op_type == "Softplus":
            x, y = n.input[0], n.output[0]
            one = y + "_one"
            ex = y + "_exp"
            ad = y + "_add"
            extra.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), one))
            new_nodes.append(helper.make_node("Exp", [x], [ex], name=n.name + "_exp"))
            new_nodes.append(helper.make_node("Add", [ex, one], [ad], name=n.name + "_add"))
            new_nodes.append(helper.make_node("Log", [ad], [y], name=n.name + "_log"))
            n_low += 1
        else:
            new_nodes.append(n)
    if n_low:
        del g.node[:]
        g.node.extend(new_nodes)
        g.initializer.extend(extra)
        print("  ✓ 展开了 %d 个 Softplus（⇒ Exp/Add/Log ✓）" % n_low)
    return m


def materialize_constants(m):
    """★把 Constant / ConstantOfShape 物化成 initializer✓★（否则它们的输出没有生产者 ✗）"""
    from onnx import helper, TensorProto
    g = m.graph
    kept, made, failed = [], 0, []
    for n in g.node:
        if n.op_type == "Constant":
            val = next((a for a in n.attribute if a.name == "value"), None)
            if val is None:
                failed.append(n.name or "Constant"); continue
            g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(val.t), n.output[0]))
            made += 1
        elif n.op_type == "ConstantOfShape":
            shp = next((i for i in n.input), None)
            if shp is None or shp not in {t.name: t for t in g.initializer}:
                failed.append(n.name or "ConstantOfShape"); continue
            want = numpy_helper.to_array({t.name: t for t in g.initializer}[shp]).astype(np.int64)
            val = next((a for a in n.attribute if a.name == "value"), None)
            fill = numpy_helper.to_array(val.t).flatten()[0] if val is not None else 0.0
            g.initializer.append(numpy_helper.from_array(
                np.full([int(v) for v in want], fill, dtype=np.float32), n.output[0]))
            made += 1
        else:
            kept.append(n)
    if made or failed:
        del g.node[:]
        g.node.extend(kept)
        print("  ✓ 物化常量 %d 个 ✓" % made + ("  ✗ 失败 %d 个：%s" % (len(failed), failed[:3]) if failed else ""))
    return m


def main(onnx_path, out_path, weights_path):
    m = onnx.load(onnx_path)                     # 连外置权重一起读入 ✓
    m = pre_lower(m)                             # ★先展开 NNRt 没有的算子✓★
    m = materialize_constants(m)                 # ★再物化常量✓★
    # ★做形状推断✓★：NNRt 要求静态形状（不能有 -1 ✗ §123 ✓），必须把每个张量的真实形状填对 ✓
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        m = SymbolicShapeInference.infer_shapes(m, auto_merge=True, guess_output_rank=True, verbose=0)
        print("  ✓ 用 symbolic_shape_infer 推断形状 ✓")
    except Exception as e:
        print("  ⚠ symbolic_shape_infer 不可用（%s）⇒ 退回 onnx.shape_inference" % str(e)[:60])
        try:
            from onnx import shape_inference
            m = shape_inference.infer_shapes(m)
        except Exception as e2:
            print("  ⚠ 形状推断失败：%s" % str(e2)[:80])
    g = m.graph
    SHAPE = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            SHAPE[vi.name] = [d.dim_value for d in t.shape.dim]
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    # 权重落地：把所有常量拼成一个大 blob ✓（C++ 侧 mmap/读取 ✓）
    lines, tensors, blob_off = [], [], 0
    idx_of, next_idx = {}, 0

    def add_tensor(name, np_dtype, shape, ptype=0):
        nonlocal next_idx
        if name in idx_of:
            return idx_of[name]
        idx_of[name] = next_idx
        rank = len(shape)
        dt = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int32"): "I32",
              np.dtype("int64"): "I64", np.dtype("bool"): "B", np.dtype("int8"): "I8",
              np.dtype("uint8"): "U8"}.get(np_dtype, "F32")
        lines.append("T %d %s %s %d %s%s" % (idx_of[name], name, dt, rank,
                                           " ".join(str(int(d)) for d in shape) if rank else "",
                                           (" PARAM %d" % ptype) if ptype else ""))
        next_idx += 1
        return idx_of[name]

    # ① 权重（initializer）✓
    blob = open(weights_path, "wb")
    for name, arr in inits.items():
        a = np.ascontiguousarray(arr)
        idx = add_tensor(name, a.dtype, a.shape)
        blob.write(a.tobytes())
        lines.append("I %d %s %d %d" % (idx, os.path.basename(weights_path), blob_off, a.nbytes))
        blob_off += a.nbytes
    blob.close()
    # ② 图输入 ✓
    vin = []
    for i in g.input:
        if i.name in inits:
            continue
        shp = [d.dim_value for d in i.type.tensor_type.shape.dim]
        vin.append(add_tensor(i.name, np.dtype("float32"), shp))
    # ③ 节点 ✓
    vout = []
    for n in g.node:
        if n.op_type in ("Constant", "ConstantOfShape"):
            lines.append("# 跳过常量算子 %s（常量已由 initializer 处理 ✓）" % n.op_type)
            continue
        opnum = OP.get(n.op_type, -1)
        if opnum < 0:
            lines.append("# ★不支持的算子 %s（节点 %s）⇒ 需要展开或换写法" % (n.op_type, n.name))
            continue
        # 参数张量 ✓
        params = []
        for pname, ptype, pval in PARAM_TABLE.get(n.op_type, []):
            if pval is None:                      # 取值来自 ONNX 属性 ✓
                av = next((a for a in n.attribute if a.name in ("axis", "num_outputs")), None)
                pval = int(av.i) if av is not None else 0
            tid = add_tensor("__p_%s_%s_%d" % (n.name or n.op_type, pname, len(params)),
                             {"I32": np.dtype("int32"), "B": np.dtype("bool"),
                              "F32": np.dtype("float32")}.get(ptype, np.dtype("int32")),
                             [], PARAM_ENUM.get(pname, 0))          # ★带参数枚举值✓★
            # 参数张量是常量 ⇒ 写进 blob ✓
            npdt = {"I32": np.int32, "B": np.bool_, "F32": np.float32}.get(ptype, np.int32)
            arr = np.array(pval, dtype=npdt)
            blob = open(weights_path, "ab")
            off = os.path.getsize(weights_path)
            blob.write(np.ascontiguousarray(arr).tobytes())
            blob.close()
            lines.append("I %d %s %d %d" % (tid, os.path.basename(weights_path), off, arr.nbytes))
            params.append(tid)
        missing = [i for i in n.input if i and i not in idx_of]
        if missing:
            print("  ★✗ 节点 %s(%s) 缺少输入张量：%s ⇒ 跳过该节点（请修生成器 ✓）"
                  % (n.op_type, n.name, missing[:2]))
            continue
        ins = [idx_of[i] for i in n.input if i]
        outs = []
        for o in n.output:
            shp = SHAPE.get(o)
            if shp is None or any((not isinstance(d, int)) or d <= 0 for d in shp):
                print("  ⚠ %s 的输出 %s 形状未知/动态 ⇒ %s" % (n.op_type, o, shp))
                shp = [1]
            outs.append(add_tensor(o, np.dtype("float32"), shp))
        lines.append("N %d %d %s %d %s %d %s" % (opnum, len(params), " ".join(map(str, params)),
                                                len(ins), " ".join(map(str, ins)),
                                                len(outs), " ".join(map(str, outs))))
        vout.extend(outs)
    lines.append("IN " + " ".join(map(str, vin)))
    lines.append("OUT " + " ".join(map(str, vout[-1:])))           # 先只取最后一个输出 ✓
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("  张量 %d 个 · 节点行 %d 条 · 权重 blob %d B" %
          (next_idx, sum(1 for l in lines if l.startswith("N ")), os.path.getsize(weights_path)))
    print("  已写 %s" % out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
