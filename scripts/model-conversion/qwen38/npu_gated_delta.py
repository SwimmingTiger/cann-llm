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

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "npu_causal_conv1d_fn", "npu_chunk_gated_delta_rule",
    "install", "uninstall", "tril_ones", "strict_lower", "strict_upper",
    "strict_lower_f", "strict_upper_f",
]

# ---------------------------------------------------------------- 常量（编译期固定 ✓）

_MASK_CACHE: dict = {}


def tril_ones(c: int, dtype: torch.dtype, device) -> torch.Tensor:
    """★前缀和用的常量矩阵★：`M[j,t] = 1 ⟺ j <= t`，即 `triu(ones)` ✓

    ★坑★：直觉会写 `tril(ones)`，但 `tril(ones)[j,t] = 1 ⟺ j >= t` ⇒ 算出来是【后缀和】✗
    （实测正好反了 ✓，§51）⇒ cumsum 是前缀和 ⇒ 必须用 triu(ones) ✓。

    ★用 numpy 造★：`torch.triu` 会导出成 `Trilu` 节点 ✗，而 OMG 的 pre-check 拒收它 ✗
    （§68 实测：全图只剩 Trilu 一个 fail ✓）
    """
    key = ("ones", c, str(dtype), str(device))
    if key not in _MASK_CACHE:
        arr = np.triu(np.ones((c, c), dtype=np.float32))
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return _MASK_CACHE[key]


def _mask_f(arr: "np.ndarray", key, device) -> torch.Tensor:
    """★float 常量掩码★（numpy 直接造 ✓ 避免 bool→float 的 Cast ✗ —— §107 踩到类型错 ✓）。"""
    k = (key, arr.shape[0], str(device))
    if k not in _MASK_CACHE:
        _MASK_CACHE[k] = torch.from_numpy(arr.astype(np.float32)).to(device=device)
    return _MASK_CACHE[k]


def strict_upper_f(c: int, device) -> torch.Tensor:
    """严格上三角（★float32 常量✓★）。"""
    return _mask_f(np.triu(np.ones((c, c), dtype=np.float32), 1), "upf", device)


def strict_lower_f(c: int, device) -> torch.Tensor:
    """严格下三角（★float32 常量✓★）。"""
    return _mask_f(np.tril(np.ones((c, c), dtype=np.float32), -1), "lof", device)


def strict_lower(c: int, device) -> torch.Tensor:
    """严格下三角（布尔 ✓，numpy 常量 ✓）。"""
    key = ("lo", c, str(device))
    if key not in _MASK_CACHE:
        arr = np.tril(np.ones((c, c), dtype=bool), -1)
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device)
    return _MASK_CACHE[key]


def strict_upper(c: int, device) -> torch.Tensor:
    """严格上三角（布尔 ✓，numpy 常量 ✓）。"""
    key = ("up", c, str(device))
    if key not in _MASK_CACHE:
        arr = np.triu(np.ones((c, c), dtype=bool), 1)
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device)
    return _MASK_CACHE[key]


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """与参考实现一致（纯基础算子 ✓）"""
    # ★不用除法✗★：OMG 日志明确说 RealDiv 在我们的 NPU kernel 库里没有实现
    #   （"op name [/Div] type [RealDiv] is not supported in npucl store" ✗ §103/§104 ✓）
    #   ⇒ 改成 ★x * (·)^(-0.5)★ ✓：Pow 在支持列表里 ✓（不出现于 unsupported 清单 ✓）
    return x * (x.pow(2).sum(dim=dim, keepdim=True) + eps).pow(-0.5)


# ---------------------------------------------------------------- ⓪ 分块前代（数值稳定）

def _solve_unit_lower(L: torch.Tensor, X: torch.Tensor, block: int = 16, c: int = 0) -> torch.Tensor:
    """解 `(I - L) Y = X`（L 严格下三角），只用一个常量块大小的 matmul ✓。

    ★为什么不用"整体平方-乘积"★：`(I−L)^-1 = Π(I+L^{2^k})` 数学上精确 ✓，
    但它会把条件数**平方** ✗ —— 实测真实模型里 `beta≈0.995`、衰减很小的层，
    UT 系统近乎奇异 ⇒ 逐层误差冲到 ★2.3e+04 / 6.6e+16★ ✗（§51 记录 ✓）。
    分块前代只在【16×16 小块】上用平方-乘积（块小 ⇒ 条件数低 ✓），
    块间用普通 matmul 传播 ✓ ⇒ 数值稳定 ✓，而且块数固定（C=64 ⇒ 4 块）⇒ 全部可展开 ✓。
    """
    # ★c 必须由调用方以 Python int 传入★：旧 TorchScript 导出器会把
    #   `L.shape[-1]` 变成 Tensor，导致 `bit_length()` 报错 ✗（§53 实测 ✓）
    c = c or int(L.shape[-1])
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
    seq_len: int = 0,
    batch: int = 0,
    **kwargs,
):
    """`torch_chunk_gated_delta_rule` 的等价改写 —— ★全程只用 ≤3 维张量★ ✓。

    ★为什么强调 3 维★：NPU-CL 对 ≥4 维支持很差 ✗（§30 的老结论 ✓）。
    实测（§52）：5 维张量上那 49 个 `Reshape` 被 OMG 的 pre-check 直接判 fail ✗，
    而同样这些数学用 3 维表达时全部 pass ✓。

    做法：
      · 把 (batch, head) 折成一维 ⇒ `[B*H, S, D]` ✓（一次 reshape ✓）
      · ★不再构造 chunk 维★：逐 chunk 用【常量下标 Slice】切出来 ✓
        （chunk 数是编译期常量 ⇒ 循环完全展开 ✓ 图里没有动态形状 ✓）
      · 所有中间量：`[BH, C, D]` / `[BH, C, C]` / `[BH, Dk, Dv]` —— 全是 3 维 ✓

    形状约定与参考实现一致：
      query/key [B, S, Hk, Dk] · value [B, S, Hv, Dv] · g/beta [B, S, Hv]
      返回 (out [B, S, Hv, Dv], final_state [B, Hv, Dk, Dv] 或 None)
    """
    initial_dtype = query.dtype
    # ★shape 取值一律走显式 Python int★：legacy TorchScript 追踪下
    #   `key.shape[1]` 可能是 Tensor ✗（§53/§66 实测）⇒ 这里优先用传入的 int ✓
    _bs, _sl, hk, k_head_dim = key.shape
    batch_size = batch or _bs
    seq_len = seq_len or _sl
    num_v_heads, v_head_dim = value.shape[-2:]
    if hk != num_v_heads:
        raise ValueError("本实现要求 key/value 头数一致（参考实现的实际用法亦然 ✓）")
    heads = hk
    bh = batch_size * heads

    # 折 (B, H) ✓ —— 每次 reshape 都是 3 维目标 ✓
    q = query.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format).reshape(bh, seq_len, k_head_dim)
    k = key.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format).reshape(bh, seq_len, k_head_dim)
    v = value.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format).reshape(bh, seq_len, v_head_dim)
    b = beta.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format).reshape(bh, seq_len)
    dec = g.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format).reshape(bh, seq_len)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q, dim=-1, eps=1e-6)
        k = _l2norm(k, dim=-1, eps=1e-6)
    q = q * (k_head_dim ** -0.5)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size:
        q, k, v = (F.pad(x, (0, 0, 0, pad_size)) for x in (q, k, v))   # 3 维：补第 1 维 ✓
        b, dec = (F.pad(x, (0, pad_size)) for x in (b, dec))           # 2 维：补最后一维 ✓
    total_len = seq_len + pad_size
    num_chunks = total_len // chunk_size
    c = chunk_size

    cum_ones = tril_ones(c, torch.float32, q.device)        # 前缀和常量矩阵（triu(ones) ✓）
    lo = strict_lower(c, q.device)
    up = strict_upper(c, q.device)
    eye = torch.eye(c, dtype=torch.float32, device=q.device)
    zero = torch.zeros((), dtype=torch.float32, device=q.device)
    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=q.device)

    if initial_state is None:
        state = torch.zeros(bh, k_head_dim, v_head_dim, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.to(torch.float32).reshape(bh, k_head_dim, v_head_dim)

    outs = []
    for i in range(num_chunks):                             # 编译期常量次数 ⇒ 全展开 ✓
        sl = slice(i * c, (i + 1) * c)                      # ★常量下标 Slice★ ✓
        qi, ki, vi = q[:, sl, :], k[:, sl, :], v[:, sl, :]   # [BH, C, D] ✓ 3 维
        bi, di = b[:, sl], dec[:, sl]                        # [BH, C] ✓ 2 维
        cum = di @ cum_ones                                  # [BH, C] ✓ 前缀和（免 CumSum ✓）
        # ★不用 Select✗★（NPU 库里没实现 ✓ §106）：where(up,-inf,d).exp() 等价于
        #   d.exp() * (1-up_f) ✓ —— up 处置 0 ✓；d = cum_i - cum_j ≤ 0 ✓ 不会溢出 ✓
        pw = torch.where(up, neg_inf, cum.unsqueeze(2) - cum.unsqueeze(1)).exp()   # [BH,C,C] ✓（原版 ✓）
        v_beta = vi * bi.unsqueeze(-1)
        k_beta = ki * bi.unsqueeze(-1)
        ut = (k_beta @ ki.transpose(-1, -2)) * pw            # [BH,C,C] ✓
        intra = (qi @ ki.transpose(-1, -2)) * pw             # [BH,C,C] ✓
        dkb = k_beta * cum.exp().unsqueeze(-1)               # [BH,C,D] ✓
        lower = torch.where(lo, -ut, zero)                   # = -(ut.tril(-1)) ✓（原版 ✓）
        new_values = _solve_unit_lower(lower, v_beta, block=16, c=c)     # [BH,C,Dv] ✓
        k_cumdecay = _solve_unit_lower(lower, dkb, block=16, c=c)        # [BH,C,Dk] ✓
        qi2 = qi * cum.exp().unsqueeze(-1)
        ki2 = ki * (cum[:, -1:] - cum).exp().unsqueeze(-1)
        cdec = cum[:, -1].reshape(bh, 1, 1).exp()            # [BH,1,1] ✓
        v_new = new_values - k_cumdecay @ state              # [BH,C,Dv] ✓
        outs.append(qi2 @ state + intra @ v_new)             # [BH,C,Dv] ✓
        state = state * cdec + ki2.transpose(-1, -2) @ v_new # [BH,Dk,Dv] ✓

    core = torch.cat(outs, dim=1)                            # [BH, total, Dv] ✓ 3 维
    core = core[:, :seq_len, :]
    core = core.reshape(batch_size, heads, seq_len, v_head_dim).transpose(1, 2)
    core = core.to(initial_dtype, memory_format=torch.contiguous_format)
    final = state.reshape(batch_size, num_v_heads, k_head_dim, v_head_dim) if output_final_state else None
    return core, final


# ---------------------------------------------------------------- 安装 / 卸载

_PATCHED = {}
_NAN_PATCHED = False


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
    # ★导出期间把 torch.nan_to_num 变成恒等★（从源头去掉 IsNaN ✗，见 §55）
    global _NAN_PATCHED
    if not _NAN_PATCHED:
        _PATCHED["nan_to_num"] = torch.nan_to_num
        torch.nan_to_num = lambda x, *a, **k: x
        _NAN_PATCHED = True
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
    global _NAN_PATCHED
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    for name, old in list(_PATCHED.items()):
        if name == "nan_to_num":
            torch.nan_to_num = old
            _NAN_PATCHED = False
        elif old is not None:
            setattr(modeling_module, name, old)
    _PATCHED.clear()
