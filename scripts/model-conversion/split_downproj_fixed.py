"""Split down_proj (and gate/up) MatMuls along K into chunks, then sum.

FIX vs the original split_downproj.py
-------------------------------------
The original built the replacement nodes and did

    g.node.remove(node)
    ...
    for n in new_nodes:
        g.node.append(n)          # <-- appended at the END of the node list

so every layer's rewritten down_proj subgraph landed at the tail of the graph
(layer 0's `down_proj_ksplit` was node #2005 of 2117) while the consumers of
`down_proj`'s output stayed near the front.  The graph was no longer
topologically sorted and the compiled model consumed not-yet-computed data.

This version rebuilds the node list, substituting the replacement subgraph at
the ORIGINAL position of each rewritten node, so the export stays topologically
correct.

The math is unchanged (for Y = X @ W with W:[K,N], splitting K and summing
X_k @ W_k reproduces X @ W exactly).

Usage:
    split_downproj_fixed.py <src.onnx> <dst.onnx> <nlayers> <chunk>
"""
import sys

import onnx
import numpy as np
from onnx import helper as H
from onnx import numpy_helper as NH

if len(sys.argv) < 3:
    sys.exit(__doc__ + "\n用法: split_downproj_fixed.py <in.onnx> <out.onnx> [nlayers] [k_chunk]")

src = sys.argv[1]
dst = sys.argv[2]
nlayers = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("NLAYERS", "28"))
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("K_CHUNK", "2048"))
m = onnx.load(src)
g = m.graph
inits = {i.name: i for i in g.initializer}
# Only down_proj is rewritten -- this is what the working model actually did.
# (Rewriting gate_proj/up_proj is a no-op anyway: their K = hidden_size = 1536 is
#  smaller than the chunk size, so they would come out as a single chunk and the
#  accumulator loop would never emit the original output name.)
targets = {"model.layers.%d.mlp.down_proj" % i for i in range(nlayers)}

new_nodes = []          # ordered replacement nodes
rewritten = 0
kept = []


def rewrite(node):
    """Return the list of nodes that replaces `node`."""
    global rewritten
    xname, wname = node.input
    out = node.output[0]
    W = NH.to_array(inits[wname])
    K, N = W.shape
    parts = [(i, min(i + chunk, K)) for i in range(0, K, chunk)]

    # Nothing to gain from a single chunk: that would just rename the output and
    # leave the original tensor name unproduced (dangling reference).
    if len(parts) < 2:
        return [node]

    names = []
    for idx, (a, b) in enumerate(parts):
        w_nm = "%s_k%d" % (wname, idx)
        g.initializer.append(NH.from_array(W[a:b], w_nm))
        names.append(w_nm)

    nodes = []
    split_outs = ["%s_k%d" % (xname, i) for i in range(len(parts))]
    nodes.append(H.make_node("Split", [xname], split_outs,
                             name=node.name + "_ksplit", axis=-1,
                             split=[b - a for a, b in parts]))

    mm_outs = []
    for idx, w_nm in enumerate(names):
        o = "%s_kmm%d" % (out, idx)
        mm_outs.append(o)
        nodes.append(H.make_node("MatMul", [split_outs[idx], w_nm], [o],
                                 name="%s_k%d" % (node.name, idx)))

    acc = mm_outs[0]
    for idx in range(1, len(mm_outs)):
        o = out if idx == len(mm_outs) - 1 else "%s_kacc%d" % (out, idx)
        nodes.append(H.make_node("Add", [acc, mm_outs[idx]], [o],
                                 name="%s_kadd%d" % (node.name, idx)))
        acc = o

    rewritten += 1
    return nodes


# --- rebuild the node list, substituting in place (topological order kept) ---
for node in list(g.node):
    if node.name in targets and node.op_type == "MatMul":
        new_nodes.extend(rewrite(node))
    else:
        new_nodes.append(node)

del g.node[:]
g.node.extend(new_nodes)

print("rewrote %d MatMuls, K split into chunks of %d" % (rewritten, chunk))
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=dst.split("/")[-1][:-5] + ".pb", size_threshold=1024)
print("saved", dst)
