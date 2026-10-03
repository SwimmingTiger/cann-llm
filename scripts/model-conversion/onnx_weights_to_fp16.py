"""把 ONNX 权重转 FP16（体积减半），但★只转消费者全是 MatMul/Gemm 的权重★。

为什么：上一版给所有大权重都插了 Cast ✗，结果连 RMSNorm 的权重也被插上，
ReduceMean 变成混合精度 ⇒ NPU 的 reduce 路径直接拒收 ✗
（报错就是 reduce_check_support / CheckElemSupportV0 ✓）

现在只动 MatMul/Gemm 的权重 —— 它们是体积的 99%，且不经过任何 reduce ✓
"""
import sys, os, collections
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

# ① 统计每个张量被哪些算子消费
consumers = collections.defaultdict(set)
for n in g.node:
    for x in n.input:
        consumers[x].add(n.op_type)

# ② 只转"消费者全是 MatMul/Gemm"的权重（且足够大）
n_conv = n_skip = 0
saved = 0
new_inits, new_nodes = [], []
for init in list(g.initializer):
    ops = consumers.get(init.name, set())
    arr = None
    if (init.data_type == TensorProto.FLOAT and ops and ops <= {"MatMul", "Gemm"}):
        arr = numpy_helper.to_array(init)
    if arr is None or arr.size < 4096:
        new_inits.append(init); n_skip += 1; continue
    saved += arr.nbytes
    new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name + "_h"))
    new_nodes.append(helper.make_node("Cast", [init.name + "_h"], [init.name + "_f32"],
                                      to=TensorProto.FLOAT))
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == init.name:
                n.input[i] = init.name + "_f32"
    n_conv += 1

del g.initializer[:]
g.initializer.extend(new_inits)
g.node.extend(new_nodes)
for f in (dst + ".data",):
    if os.path.exists(f): os.remove(f)
convert_model_to_external_data(m, all_tensors_to_one_file=True,
                               location=os.path.basename(dst) + ".data", size_threshold=1024)
onnx.save(m, dst)
print("  ✓ 转 FP16: %d 个（跳过 %d 个非 MatMul 权重 ✓）· 原 %d 字节" % (n_conv, n_skip, saved), flush=True)
print("  data: %.2f GB" % (os.path.getsize(dst + ".data") / 1e9), flush=True)
