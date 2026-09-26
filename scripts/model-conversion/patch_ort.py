"""Patch the CANN-LLM ONNX so ONNXRuntime can load it, then run a numerical check.

ORT requires ScatterND indices to be int64; the CANN export feeds int32.
This inserts a Cast before each such indices input and re-saves.

usage: ort_check2.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper as H

if len(sys.argv) < 3:
    sys.exit(__doc__ + "\n用法: patch_ort.py <in.onnx> <out.onnx>")

src = sys.argv[1]
dst = sys.argv[2]

m = onnx.load(src, load_external_data=True)
g = m.graph

vt = {vi.name: vi.type.tensor_type.elem_type for vi in list(g.input) + list(g.output) + list(g.value_info)}

patched = 0
new_nodes = []
for n in g.node:
    if n.op_type == "ScatterND":
        idx = n.input[1]
        et = vt.get(idx)
        if et == TensorProto.INT32:
            cast_out = idx + "_i64"
            new_nodes.append(H.make_node("Cast", [idx], [cast_out],
                                         name=idx + "_cast64",
                                         to=TensorProto.INT64))
            n.input[1] = cast_out
            patched += 1
    new_nodes.append(n)

del g.node[:]
g.node.extend(new_nodes)
print("patched %d ScatterND index inputs" % patched)

onnx.save_model(m, dst)          # keep external_data refs pointing at model.pb
print("saved", dst)
