"""qwen3_5 线性注意力的 **NPU 友好改写** —— 只使用 DDK 平台库支持的算子。

参考实现（transformers/models/qwen3_5/modeling_qwen3_5.py）：
  · `causal_conv1d_fn`            (268)  —— depthwise 因果 Conv1D
  · `torch_chunk_gated_delta_rule` (299) —— Gated DeltaNet 分块实现
  · `torch_recurrent_gated_delta_rule`

原实现里有 4 个算子 DDK 平台库【没有】（见 docs/maintainer-notes.md §49）：
  ✗ CumSum · Trilu · ScatterElements · ScatterND      （Conv1D 也只有 Conv2D ✓）

本模块把 4 个缺失算子全部消掉，**语义保持一致**（`parity_check.py` 逐步对拍 ✓）：

| 原写法 | 缺失算子 | 这里的替换 |
|---|---|---|
| `decay.cumsum(dim=3)` | CumSum | ★常量下三角矩阵乘法★ `decay @ tril(ones)` ✓ |
| `torch.ones(C,C).triu(1)` / `.tril(-1)` | Trilu | ★常量布尔掩码★（`torch.where` / 乘法）✓ |
| 循环里 `out[:, :, i] = …`（切片赋值） | Scatter* | ★列表收集 + `torch.cat`★ ✓ |
| `F.conv1d`（depthwise，k=4） | Conv1D | ★移位切片 + Mul/Add★（零卷积 ✓）|

另外两点让图更"干净"：
  · ★两处循环都完全展开★（chunk 数、块大小是编译期常量）⇒ ONNX 里没有 Loop / Scan ✓
  · ★UT 变换用"平方-乘积恒等式"★替代参考实现的 63 步递推：
      L 是严格下三角 ⇒ L^C = 0 ⇒ (I−L)^−1 = (I+L)(I+L²)(I+L⁴)…
      ⇒ C=64 只需 ★6 次 matmul★ ✓（节点数远少于逐行展开 ✓）

用法（导出时 monkey-patch，不改 site-packages ✓）：

    from npu_gated_delta import install
    install()          # 把 qwen3_5 模块里的两个函数换成 NPU 友好版
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "npu_causal_conv1d_fn", "npu_chunk_gated_delta_rule",
    "install", "uninstall", "tril_ones", "strict_lower", "strict_upper",
]

# ---------------------------------------------------------------- 常量（编译期固定 ✓）

def tril_ones(c: int, dtype: torch.dtype, device) -> torch.Tensor:
    """★前缀和用的常量矩阵★：`M[j,t] = 1 ⟺ j <= t`，即 `triu(ones)` ✓

    ★坑★：直觉上会写成 `tril(ones)`，但 `tril(ones)[j,t] = 1 ⟺ j >= t`
    ⇒ `x @ tril(ones)` 算出来是【后缀和】✗（实测正好反了 ✓，见 §51）。
    cumsum 是前缀和 ⇒ 必须用 `triu(ones)` ✓。
    """
    return torch.triu(torch.ones(c, c, dtype=dtype, device=device))


def strict_lower(c: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(c, c, dtype=torch.bool, device=device), -1)


def strict_upper(c: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(c, c, dtype=torch.bool, device=device), 1)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """与参考实现一致（纯基础算子 ✓）"""
    return x / (x.pow(2).sum(dim=dim, keepdim=True) + eps).sqrt()


# ---------------------------------------------------------------- ⓪ 分块前代（数值稳定）

def _solve_unit_lower(L: torch.Tensor, X: torch.Tensor, block: int = 16) -> torch.Tensor:
    """解 `(I - L) Y = X`（L 严格下三角），只用一个常量块大小的 matmul ✓。

    ★为什么不用"整体平方-乘积"★：`(I−L)^-1 = Π(I+L^{2^k})` 数学上精确 ✓，
    但它会把条件数**平方** ✗ —— 实测真实模型里 `beta≈0.995`、衰减很小的层，
    UT 系统近乎奇异 ⇒ 逐层误差冲到 ★2.3e+04 / 6.6e+16★ ✗（§51 记录 ✓）。
    分块前代只在【16×16 小块】上用平方-乘积（块小 ⇒ 条件数低 ✓），
    块间用普通 matmul 传播 ✓ ⇒ 数值稳定 ✓，而且块数固定（C=64 ⇒ 4 块）⇒ 全部可展开 ✓。
    """
    c = L.shape[-1]
    nblk = (c + block - 1) // block
    ys = []
    for i in range(nblk):
        r0 = i * block
        r1 = min(c, r0 + block)
        acc = X[..., r0:r1, :]
        for j in range(i):                                  # 前面的块 ✓（常量次数 ✓）
            c0 = j * block
            c1 = min(c, c0 + block)
            acc = acc + L[..., r0:r1, c0:c1] @ ys[j]
        Lii = L[..., r0:r1, r0:r1]
        m = r1 - r0
        inv = torch.eye(m, dtype=L.dtype, device=L.device) + Lii
        pw = Lii
        for _ in range(max(1, (m - 1).bit_length())):       # 16 ⇒ 4 次 ✓
            pw = pw @ pw
            inv = inv + inv @ pw
        ys.append(inv @ acc)
    return torch.cat(ys, dim=-2)


# ---------------------------------------------------------------- ⓪′ M-RoPE 分节（函数式）

def npu_recomposition_frequencies(self, freq):
    """`Qwen3_5TextRotaryEmbedding.recomposition_frequencies` 的等价改写 ✓。

    原实现是**原地写**：
        freqs_thw[..., idx] = freq[dim, ..., idx]      # idx = slice(offset, length, 3)
    ⇒ dynamo 会把它降成 `select_scatter` / `ScatterND` ✗（DDK 都没有 ✓）。

    这里改成**常量掩码 + Where**：掩码只取决于维数下标（编译期常量 ✓），
    结果一样，但全程只有 `Where`/`Cat` ✓（都是支持算子 ✓）。
    """
    freqs_thw = freq[0]
    dim_size = freqs_thw.shape[-1]
    idx = torch.arange(dim_size, device=freq.device)
    for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        mask = (idx >= offset) & (idx < length) & (((idx - offset) % 3) == 0)
        freqs_thw = torch.where(mask, freq[dim], freqs_thw)     # 广播 ✓
    return torch.cat((freqs_thw, freqs_thw), dim=-1)


# ---------------------------------------------------------------- ① 因果深度卷积

def npu_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """`causal_conv1d_fn` 的等价改写：depthwise 因果卷积（左侧补 k-1）。

    原实现用 `F.conv1d(..., groups=hidden_size)`（DDK 只有 Conv2D ✗）⇒
    这里换成 **移位切片 + Mul/Add**（k 固定 = 4 ✓，纯基础算子 ✓）。

    输入：hidden_states [B, H, S]、weight [H, 1, k]（depthwise）、bias [H]
    """
    b, h, s = hidden_states.shape
    k = weight.shape[-1]
    x = hidden_states.to(weight.dtype)
    xpad = F.pad(x, (k - 1, 0))                        # Pad ✓ 左侧补 k-1（因果）
    w = weight.reshape(h, k)                           # [H, k]
    out = None
    for j in range(k):                                 # k 是编译期常量 ⇒ 展开 ✓
        part = xpad[:, :, j:j + s] * w[:, j].reshape(1, h, 1)
        out = part if out is None else out + part
    if bias is not None:
        out = out + bias.reshape(1, h, 1)
    if activation is not None:
        out = F.silu(out)                              # Sigmoid + Mul ✓
    return out.to(hidden_states.dtype)


# ---------------------------------------------------------------- ② Gated DeltaNet（分块）

def npu_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """`torch_chunk_gated_delta_rule` 的等价改写（只用支持算子 ✓）。

    形状约定与参考实现一致：
      query/key [B, S, Hk, Dk] · value [B, S, Hv, Dv] · g/beta [B, S, Hv]
      返回 (out [B, S, Hv, Dv], final_state [B, Hv, Dk, Dv] 或 None)
    """
    initial_dtype = query.dtype
    batch_size, seq_len, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)

    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size:
        query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total_len = seq_len + pad_size
    num_chunks = total_len // chunk_size
    c = chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], num_chunks, c, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], num_chunks, c)

    # ★CumSum → 一次常量下三角矩阵乘法★
    cum_decay = decay @ tril_ones(c, torch.float32, decay.device)          # [B,Hv,nc,C]

    # ★triu 掩码 → 常量布尔掩码 + where★
    pairwise = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)             # [.., i, j]
    up = strict_upper(c, pairwise.device)
    pairwise = torch.where(up, torch.full((), float("-inf"), dtype=pairwise.dtype,
                                          device=pairwise.device), pairwise)
    pairwise = pairwise.exp()

    ut = (k_beta @ key.transpose(-1, -2)) * pairwise                      # [B,Hv,nc,C,C]
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # ★UT 变换：L 严格下三角 ⇒ 幂零 ⇒ (I−L)^-1 = (I+L)(I+L²)(I+L⁴)…★
    #   参考实现写成 63 步逐行递推（还会引入 scatter）✗；这里 6 次 matmul ✓
    zero = torch.zeros((), dtype=ut.dtype, device=ut.device)
    lo = strict_lower(c, ut.device)
    lower = torch.where(lo, -ut, zero)                 # = -(ut.tril(-1)) ✓（免 Trilu ✓）
    # ★分块前代★（小块平方-乘积 + 块间 matmul ⇒ 数值稳定 ✓，块数常量 ⇒ 可展开 ✓）
    new_values = _solve_unit_lower(lower, v_beta, block=16)
    k_cumdecay = _solve_unit_lower(lower, decayed_k_beta, block=16)

    if initial_state is None:
        state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # ★分块顺序扫描：用列表 + cat 取代原实现的 in-place 切片赋值（免 Scatter ✓）★
    outs = []
    for i in range(num_chunks):                        # num_chunks 是编译期常量 ⇒ 展开 ✓
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter = query[:, :, i] @ state
        outs.append(inter + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    core = torch.cat(outs, dim=2)                      # [B,Hv,total_len,Dv] ✓
    core = core.reshape(batch_size, num_v_heads, total_len, v_head_dim)
    core = core[:, :, :seq_len]
    core = core.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core, (state if output_final_state else None)


# ---------------------------------------------------------------- 安装 / 卸载

_PATCHED = {}


def install(modeling_module=None):
    """把 qwen3_5 建模模块里的两个函数换成 NPU 友好版（导出时用 ✓）。"""
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    for name, new in (("causal_conv1d_fn", npu_causal_conv1d_fn),
                      ("torch_chunk_gated_delta_rule", npu_chunk_gated_delta_rule)):
        if name not in _PATCHED:
            _PATCHED[name] = getattr(modeling_module, name, None)
        setattr(modeling_module, name, new)
    # 参考实现上挂着"从 hub 取 kernel"的装饰器 ⇒ 模型内部引用的是被装饰后的名字，
    # 因此还要把模块里别处引用到的别名一并替换（见 README 的说明 ✓）
    # M-RoPE 分节：函数式替换（去 select_scatter ✗）
    rope_cls = getattr(modeling_module, "Qwen3_5TextRotaryEmbedding", None)
    if rope_cls is not None and "rope" not in _PATCHED:
        _PATCHED["rope"] = rope_cls.recomposition_frequencies
        rope_cls.recomposition_frequencies = npu_recomposition_frequencies
    for alias in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
        if hasattr(modeling_module, alias):
            setattr(modeling_module, alias, npu_chunk_gated_delta_rule if "chunk" in alias
                    else npu_causal_conv1d_fn)
    return modeling_module


def uninstall(modeling_module=None):
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    for name, old in _PATCHED.items():
        if old is not None:
            setattr(modeling_module, name, old)
    _PATCHED.clear()
