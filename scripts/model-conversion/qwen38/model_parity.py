"""整模型对拍：打完 NPU 补丁 vs 原版 transformers，比对 logits。

跑法：~/q38env/bin/python model_parity.py
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import npu_gated_delta as N
import npu_attention as AP

torch.set_grad_enabled(False)
MODEL = os.path.expanduser("~/q38")


def logits_for(tag):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = torch.arange(1, 129, dtype=torch.long).unsqueeze(0)      # 128 = 2×chunk ✓
    pos = torch.arange(128, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = m(input_ids=ids, position_ids=pos, use_cache=False).logits
    print("  %s: logits %s" % (tag, tuple(out.shape)), flush=True)
    del m
    return out


def main():
    print("① 原版（参考实现）…", flush=True)
    ref = logits_for("ref")

    print("② 打补丁（NPU 友好改写）…", flush=True)
    N.install(M)
    AP.install(M)
    # 命中计数：确认层的调用真的走到了补丁 ✓
    import numpy as _np
    hits = {"chunk": 0, "conv": 0, "rec": 0}
    _oc, _ov = M.torch_chunk_gated_delta_rule, M.causal_conv1d_fn
    M.torch_chunk_gated_delta_rule = lambda *a, **k: (hits.__setitem__("chunk", hits["chunk"] + 1), _oc(*a, **k))[1]
    M.causal_conv1d_fn = lambda *a, **k: (hits.__setitem__("conv", hits["conv"] + 1), _ov(*a, **k))[1]
    if hasattr(M, "torch_recurrent_gated_delta_rule"):
        _orr = M.torch_recurrent_gated_delta_rule
        M.torch_recurrent_gated_delta_rule = lambda *a, **k: (hits.__setitem__("rec", hits["rec"] + 1), _orr(*a, **k))[1]
    HITS = hits   # 前向跑完后再打印 ✓（上次打印放早了，误报 0 命中 ✗）
    try:
        npu = logits_for("npu")
    finally:
        N.uninstall(M)

    print("   补丁命中:", HITS, flush=True)
    d = (npu - ref).abs()
    scale = ref.abs().max().item()
    print("\n=== 结果 ===")
    print("  logits 最大绝对差 : %.3e" % d.max().item())
    print("  logits 最大相对差 : %.3e" % (d.max().item() / max(1e-9, scale)))
    print("  参考 logits 量级  : %.3f（argmax 一致性见下）" % scale)
    agree = (npu.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print("  ★argmax 一致率    : %.4f★" % agree)
    ok = agree >= 0.999 and (d.max().item() / max(1e-9, scale)) < 1e-3
    print("\n%s" % ("★★ 整模型对拍通过 ✓★" if ok else "✗ 未达标（看上面数字定夺）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
