"""权重转 FP16（★让 MatMul 真在 fp16 下算★）—— 就地插入 Cast，保持拓扑顺序 ✓

背景（实测教训）：
  · v0：给所有大权重插 Cast ⇒ 连 RMSNorm 的权重也被插 ⇒ ReduceMean 混合精度 ⇒ OMG 拒收 ✗
  · v1：只给 MatMul 权重插 Cast（权重后 → fp32）⇒ OMG 过 ✓ 体积减半 ✓
        但 ★推理慢 2.4 倍★ ✗ —— 因为 MatMul 仍算 fp32，
        每次调用都要把几百 MB 的权重从 fp16 转成 fp32 ✗
  · v2（本版）：★把 Cast 挪到【激活】侧★ ⇒ MatMul 在 fp16 下算 ✓
        权重 fp16 不动；激活侧 Cast(fp32→fp16) + 输出侧 Cast(fp16→fp32) ✓
        ⇒ 每次只转"激活"（几十 KB）⇒ 开销可忽略 ✓✓
  · v2 第一次失败：Cast 节点 append 到图末尾 ⇒ 破坏拓扑顺序 ⇒ OMG 建 IR 失败 ✗
        ⇒ 本版改为★就地插入★（Cast → MatMul → Cast 紧跟原节点）✓
"""
import sys, os, collections
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

inits = {i.name: i for i in g.initializer}
consumers = collections.defaultdict(list)
for n in g.node:
    for k, x in enumerate(n.input):
        consumers[x].append((n, k))

# ① 挑出"权重是 initializer、且该权重只被 MatMul/Gemm 消费"的节点
pick = {}          # id(node) -> weight name
for n in g.node:
    if n.op_type not in ("MatMul", "Gemm") or len(n.input) < 2:
        continue
    w = n.input[1]
    init = inits.get(w)
    if init is None or init.data_type != TensorProto.FLOAT:
        continue
    others = [op.op_type for op, _ in consumers[w] if op is not n]
    if others and any(o not in ("MatMul", "Gemm") for o in others):
        continue
    pick[id(n)] = w

if not pick:
    print("  ✗ 没有可转的 MatMul 权重"); sys.exit(1)

# ② 权重转 fp16（同名同形状，仅 dtype 变 ✓）
nw = saved = 0
new_inits = []
for init in g.initializer:
    if init.name in set(pick.values()):
        arr = numpy_helper.to_array(init)
        saved += arr.nbytes
        new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
        nw += 1
    else:
        new_inits.append(init)
del g.initializer[:]; g.initializer.extend(new_inits)

# ③ ★就地插入 Cast★，保持拓扑顺序 ✓
new_nodes = []
ncast = 0
act_done = {}          # ★ 激活名 -> Cast 输出名；同一个激活【只插一次】★
#   否则：Gated MLP 里 gate_proj 与 up_proj 共用同一个激活 ✗
#   ⇒ 会生成两个同名 Cast ⇒ OMG 报 "The name …__h is repeated." ✗（实测踩过 ✓）
for n in list(g.node):
    if id(n) in pick:
        a = n.input[0]
        if a in act_done:
            n.input[0] = act_done[a]           # 复用已有的 Cast 输出 ✓
        else:
            ah = a + "__h"
            act_done[a] = ah
            new_nodes.append(helper.make_node("Cast", [a], [ah], to=TensorProto.FLOAT16))
            n.input[0] = ah
            ncast += 1
        out = n.output[0]
        oh = out + "__h"
        n.output[0] = oh
        new_nodes.append(n)
        new_nodes.append(helper.make_node("Cast", [oh], [out], to=TensorProto.FLOAT))
        ncast += 1
    else:
        new_nodes.append(n)
del g.node[:]; g.node.extend(new_nodes)

for f in (dst + ".data",):
    if os.path.exists(f): os.remove(f)
convert_model_to_external_data(m, all_tensors_to_one_file=True,
                               location=os.path.basename(dst) + ".data", size_threshold=1024)
onnx.save(m, dst)
print("  ✓ 权重转 fp16: %d 个（原 %d 字节）· 就地插 Cast: %d 个 ✓" % (nw, saved, ncast), flush=True)
print("  data: %.2f GB" % (os.path.getsize(dst + ".data") / 1e9), flush=True)
