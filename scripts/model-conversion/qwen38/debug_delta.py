"""逐步对拍 Gated DeltaNet 的中�间量，定位改写版的偏差出在哪一步。

跑法：~/q38env/bin/python debug_delta.py
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import npu_gated_delta as N

torch.set_grad_enabled(False)
B, S, H, D = 1, 128, 2, 32
C = 64
g = torch.Generator().manual_seed(1)
q = 0.3 * torch.randn(B, S, H, D, generator=g)
k = 0.3 * torch.randn(B, S, H, D, generator=g)
v = 0.3 * torch.randn(B, S, H, D, generator=g)
dec = -0.1 * torch.rand(B, S, H, generator=g)
beta = 0.1 + 0.8 * torch.rand(B, S, H, generator=g)


def ref_stage1():
    """照抄参考实现的第一阶段（只到 ut / inv / new_values ✓）。"""
    qt, kt, vt, bt, dt = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (q, k, v, beta, dec)]
    qt = qt * (qt.shape[-1] ** -0.5)
    nc = S // C
    v_beta = vt * bt.unsqueeze(-1)
    k_beta = kt * bt.unsqueeze(-1)
    qt, kt, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], nc, C, x.shape[-1]) for x in (qt, kt, k_beta, v_beta)]
    dt = dt.reshape(dt.shape[0], dt.shape[1], nc, C)
    mask = torch.ones(C, C, dtype=torch.bool).triu(1)
    cum = dt.cumsum(dim=3)
    pw = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(mask, float("-inf")).exp()
    ut = (k_beta @ kt.transpose(-1, -2)) * pw
    intra = (qt @ kt.transpose(-1, -2)) * pw
    dkb = k_beta * cum.exp().unsqueeze(-1)
    nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    kcd = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    return cum, pw, ut, nv, kcd, intra, qt, kt


def npu_stage1():
    qt, kt, vt, bt, dt = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (q, k, v, beta, dec)]
    qt = qt * (qt.shape[-1] ** -0.5)
    nc = S // C
    v_beta = vt * bt.unsqueeze(-1)
    k_beta = kt * bt.unsqueeze(-1)
    qt, kt, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], nc, C, x.shape[-1]) for x in (qt, kt, k_beta, v_beta)]
    dt = dt.reshape(dt.shape[0], dt.shape[1], nc, C)
    cum = dt @ N.tril_ones(C, torch.float32, dt.device)  # tril_ones 现在返回 triu(ones) ✓
    up = N.strict_upper(C, dt.device)
    pw = torch.where(up, torch.full((), float("-inf"), dtype=torch.float32), cum.unsqueeze(4) - cum.unsqueeze(3)).exp()
    ut = (k_beta @ kt.transpose(-1, -2)) * pw
    intra = (qt @ kt.transpose(-1, -2)) * pw
    dkb = k_beta * cum.exp().unsqueeze(-1)
    lo = N.strict_lower(C, ut.device)
    lower = torch.where(lo, -ut, torch.zeros((), dtype=ut.dtype))
    eye = torch.eye(C, dtype=ut.dtype)
    inv = eye + lower
    power = lower
    for _ in range(max(1, (C - 1).bit_length())):
        power = power @ power
        inv = inv + inv @ power
    return cum, pw, ut, inv @ v_beta, inv @ dkb, intra, qt, kt, inv


r_cum, r_pw, r_ut, r_nv, r_kcd, r_intra, r_q, r_k = ref_stage1()
n_cum, n_pw, n_ut, n_nv, n_kcd, n_intra, n_q, n_k, n_inv = npu_stage1()


def d(tag, a, b):
    diff = (a - b).abs().max().item()
    print("  %-22s 最大差 %.3e %s" % (tag, diff, "✓" if diff < 1e-4 else "✗"))
    return diff < 1e-4


print("=== 第一阶段逐步对拍 ===")
ok = True
ok &= d("cum_decay", n_cum, r_cum)
ok &= d("pairwise_decay", n_pw, r_pw)
ok &= d("ut_system", n_ut, r_ut)
ok &= d("intra_chunk_attn", n_intra, r_intra)
# 参考解的逆矩阵：solve(A, I) 即 A^{-1} ✓
ref_inv = torch.linalg.solve_triangular(r_ut, torch.eye(C).expand_as(r_ut), upper=False, unitriangular=True)
ok &= d("inv 矩阵 (I-L)^-1", n_inv, ref_inv)
ok &= d("new_values", n_nv, r_nv)
ok &= d("k_cumdecay", n_kcd, r_kcd)
print("\n%s" % ("★ 第一阶段完全一致 ✓" if ok else "✗ 偏差就在上面标出的那一步"))
