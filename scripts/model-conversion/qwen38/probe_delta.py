#!/usr/bin/env python3
"""delta rule 单独探针 —— ★布局与我们实现一致：[B,S,H,D] ✓★"""
import os, sys, torch, torch.nn as nn
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.expanduser("~/q38"))
from npu_gated_delta import npu_chunk_gated_delta_rule   # noqa: E402
B, S, H, D = 1, 8, 16, 128

class DeltaProbe(nn.Module):
    def forward(s, q, k, v, g, beta, st):
        o, st2 = npu_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=S, initial_state=st,
                                           output_final_state=True, use_qk_l2norm_in_kernel=True,
                                           seq_len=S, batch=B)
        return o, st2

args = (torch.randn(B, S, H, D), torch.randn(B, S, H, D), torch.randn(B, S, H, D),
        torch.randn(B, S, H), torch.randn(B, S, H), torch.randn(B, H, D, D))
try:
    torch.onnx.export(DeltaProbe().eval(), args, "delta.onnx",
                      input_names=["q", "k", "v", "g", "beta", "st"],
                      output_names=["o", "st_out"], opset_version=14, dynamo=False)
    print("  导出 delta ✓ 输入 6 输出 2")
    import onnx
    from collections import Counter
    g = onnx.load("delta.onnx", load_external_data=False).graph
    print("  节点总数：%d" % len(g.node))
    print("  算子分布：", dict(Counter(n.op_type for n in g.node).most_common(14)))
except Exception as e:
    import traceback; traceback.print_exc(); print("  ✗", str(e)[:110])
