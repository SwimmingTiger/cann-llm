"""对拍：NPU 改写版 vs transformers 参考实现（逐步比对 ✓）。

跑法（hu60tx 上）：
    ~/q38env/bin/python parity_check.py
"""
import sys
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import npu_gated_delta as N


def mk_inputs(b, s, hk, dk, hv, dv, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        0.3 * torch.randn(b, s, hk, dk, generator=g),
        0.3 * torch.randn(b, s, hk, dk, generator=g),
        0.3 * torch.randn(b, s, hv, dv, generator=g),
        -0.1 * torch.rand(b, s, hv, generator=g),    # 衰减在 log 空间 ⇒ 必须 <= 0 ✓
        0.1 + 0.8 * torch.rand(b, s, hv, generator=g),   # beta ∈ (0.1, 0.9) ✓
    )


def compare(tag, a, b):
    d = (a - b).abs().max().item()
    rel = d / max(1e-9, b.abs().max().item())
    ok = d < 2e-4
    print("  %-28s 最大绝对差 %.3e （相对 %.2e）%s" % (tag, d, rel, "✓" if ok else "✗"))
    return ok


def main():
    torch.set_grad_enabled(False)
    allok = True
    for (b, s, hk, dk, hv, dv) in [(1, 256, 4, 128, 4, 128),      # 正好 4 个 chunk ✓
                                   (1, 100, 4, 128, 4, 128),      # 非 64 倍数 ⇒ 走 padding ✓
                                   (1, 64, 2, 64, 2, 64)]:        # 单 chunk ✓
        print("=== 形状 B=%d S=%d Hk=%d Dk=%d Hv=%d Dv=%d ===" % (b, s, hk, dk, hv, dv))
        q, k, v, g, beta = mk_inputs(b, s, hk, dk, hv, dv)
        r_out, r_st = M.torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, output_final_state=True)
        n_out, n_st = N.npu_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, output_final_state=True)
        print("  形状: out %s vs %s | state %s vs %s"
              % (tuple(r_out.shape), tuple(n_out.shape), tuple(r_st.shape), tuple(n_st.shape)))
        print("  有限性: ref out %s state %s | npu out %s state %s"
              % (bool(torch.isfinite(r_out).all()), bool(torch.isfinite(r_st).all()),
                 bool(torch.isfinite(n_out).all()), bool(torch.isfinite(n_st).all())))
        allok &= compare("输出 out", n_out, r_out)
        allok &= compare("末状态 state", n_st, r_st)

    print("=== causal_conv1d_fn（深度因果卷积）===")
    h = 32
    x = torch.randn(1, h, 128)
    w = torch.randn(h, 4) * 0.3      # 参考实现要 [H, k] ✓
    bias = torch.randn(h) * 0.1
    for act in (None, "silu"):
        r = M.causal_conv1d_fn(x, w, bias, activation=act)
        n = N.npu_causal_conv1d_fn(x, w, bias, activation=act)
        allok &= compare("conv1d activation=%s" % act, n, r)

    print("\n%s" % ("★★ 全部对拍通过 ✓★" if allok else "✗ 有对拍不通过"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
