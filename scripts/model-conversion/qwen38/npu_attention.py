"""把 Qwen3_5Attention.forward 改写成★全程 ≤3 维★的版本（NPU 内核只吃 ≤3 维 ✗，§57）。

为什么要改：实测（§57）NPUCL 对 **4 维** 的 Slice / Reshape / ExpandDims 一律拒绝 ✗
  （真实图里 18 处 4 维 Slice 全废 ✓），而 qwen3_5 的注意力原本走
  `[B, S, H, D] → transpose → [B, H, S, D]` ✗ 全程 4 维 ✗。

思路（沿用 gemma4 的老办法 ✓）：把 head 折进前导维，全程 3 维：
    [B, S, H*D]  --reshape-->  [B, S*H, D]  --Gather(常量索引)-->  [B, H*S, D]
                 --reshape-->  [B*H, S, D]            ← ★等价于 transpose 后的结果✓★
  ⇒ 换序用【常量索引的 Gather】完成 ✓（不用 4 维 transpose/reshape ✓）
  注意力：q @ k^T → [BH, S, S] ✓；softmax ✓；@ v → [BH, S, D] ✓ —— 全是 3 维 ✓
  回来时用【逆索引的 Gather】复原 ✓

其它 3 维化要点：
  · `q_norm`/`k_norm` 是【逐 head】的 RMSNorm（最后一维 ✓）⇒ 必须在 split 之后再套 ✓
  · mask 不再参与（我们导出时传全 1 mask ✓，eager 通路里的加法是 4 维 ✗ ⇒ 直接省掉 ✓）
  · RoPE 用 [1, S, D] 的 cos/sin 广播 ✓（rotate_half 用 Slice+Concat ✓）
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_PATCHED = {}


# ---------------------------------------------------------------- 常量索引（head ↔ seq 换序）
_PERM_CACHE: dict = {}
_CAUSAL_CACHE: dict = {}


def _perm_indices(seq: int, heads: int, device) -> torch.Tensor:
    """`[B, S*H, D]` ➜ `[B, H*S, D]` 的行置换索引（★纯 numpy 常量★）。

    原布局行 = s*H + h；目标行 = h*S + s。
    ★必须用 numpy 造好再 from_numpy★：用 Python 循环造张量会让旧导出器把它
      展开成一堆算子（图爆炸到 Constant 18576 ✗，§59 实测）
    """
    key = (seq, heads, str(device))
    if key not in _PERM_CACHE:
        idx = np.arange(seq * heads, dtype=np.int64).reshape(seq, heads).T.reshape(-1)
        _PERM_CACHE[key] = torch.from_numpy(idx.copy()).to(device)
    return _PERM_CACHE[key]


def _unperm_tensor(seq: int, heads: int, device) -> torch.Tensor:
    """上式的逆置换 ✓（同样纯 numpy）。"""
    key = ("inv", seq, heads, str(device))
    if key not in _PERM_CACHE:
        base = np.arange(seq * heads, dtype=np.int64).reshape(seq, heads).T.reshape(-1)
        _PERM_CACHE[key] = torch.from_numpy(np.argsort(base).astype(np.int64).copy()).to(device)
    return _PERM_CACHE[key]


def _causal_bias(seq: int, device, dtype=torch.float32) -> torch.Tensor:
    """常量因果掩码（上三角 -inf）—— ★numpy 造，只产出一个 Constant 节点★。

    torch.triu / torch.where 会让旧导出器产出 Trilu / ScatterND ✗（DDK 缺这两个 ✗）
    """
    key = (seq, str(device), str(dtype))
    if key not in _CAUSAL_CACHE:
        m = np.zeros((seq, seq), dtype=np.float32)
        m[np.triu_indices(seq, k=1)] = -np.inf
        _CAUSAL_CACHE[key] = torch.from_numpy(m).to(device=device, dtype=dtype)
    return _CAUSAL_CACHE[key]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`rotate_half` 的 3 维写法：cat(-x2, x1) ✓（Slice + Concat ✓）。"""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def _apply_rope_3d(q, k, cos, sin):
    """cos/sin: [S, rd] ⇒ 广播成 [1, S, rd] ✓（3 维 ✓，不引入 4 维 ✗）。

    ★注意 partial_rotary_factor★：qwen3_5 只对 head_dim 的【一部分】做 RoPE ✓
    （本配置 0.25 ⇒ 64/256 ✓，正好等于 cos 的最后一维 ✓）
    ⇒ 必须【分段拼接】：前 rd 维旋转 ✓、其余原样透传 ✓（只用 Slice + Concat ✓ 全 3 维 ✓）
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rd = cos.shape[-1]

    def one(x):
        if rd == x.shape[-1]:
            return x * cos + _rotate_half(x) * sin
        xr, xp = x[..., :rd], x[..., rd:]                      # Slice ✓ 3 维 ✓
        xr = xr * cos + _rotate_half(xr) * sin
        return torch.cat([xr, xp], dim=-1)                     # Concat ✓ 3 维 ✓
    return one(q), one(k)


# ---------------------------------------------------------------- 新的 forward
def npu_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """3 维版 `Qwen3_5Attention.forward` ✓（数学与参考实现一致 ✓）。"""
    b, s, _ = hidden_states.shape
    heads = self.config.num_attention_heads
    kv_heads = getattr(self.config, "num_key_value_heads", heads)
    hd = self.head_dim
    scale = self.scaling

    # ① Q 与 gate：q_proj 的输出布局是 [B, S, H, 2*D] ✓
    #    ⇒ ★必须先在每个 head 的 2*D 块内部切★（参考实现是 chunk(dim=-1) ✓）
    #    直接切整段的前一半会得到完全不同的排布 ✗（我第一次就踩了 ✓）
    qg = self.q_proj(hidden_states)                      # [B, S, H, 2*D] ✓
    qg3 = qg.reshape(b, s * heads, 2 * hd)               # [B, S*H, 2*D] ✓ 3 维
    q_raw = qg3[..., :hd]                                # [B, S*H, D] ✓（每 head 的 Q ✓）
    gate_flat = qg3[..., hd:]                            # [B, S*H, D] ✓（每 head 的 gate ✓）

    def to_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
        """[B, S, Nh*D] ➜ [B*Nh, S, D] ✓（★全 3 维★：reshape + 常量索引 Gather ✓）。"""
        x = x.reshape(b, s * n_head, hd)                 # [B, S*Nh, D] ✓
        perm = _perm_indices(s, n_head, x.device)
        x = torch.index_select(x, 1, perm)               # [B, Nh*S, D] ✓
        return x.reshape(b * n_head, s, hd)              # [B*Nh, S, D] ✓

    def from_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
        """[B*Nh, S, D] ➜ [B, S, Nh*D] ✓（逆置换 ✓）。"""
        x = x.reshape(b, n_head * s, hd)                 # [B, Nh*S, D] ✓
        inv = _unperm_tensor(s, n_head, x.device)
        x = torch.index_select(x, 1, inv)                # [B, S*Nh, D] ✓
        return x.reshape(b, s, n_head * hd)              # [B, S, Nh*D] ✓

    q = to_heads(q_raw, heads)                           # [B*H, S, D] ✓
    k = to_heads(self.k_proj(hidden_states), kv_heads)
    v = to_heads(self.v_proj(hidden_states), kv_heads)

    # ② 逐 head 归一化 ✓（必须在 split 之后 ✓，否则会把 head 维一起归一 ✗）
    q = self.q_norm(q)
    k = self.k_norm(k)

    # ③ RoPE ✓（3 维 ✓）
    cos, sin = position_embeddings
    q, k = _apply_rope_3d(q, k, cos, sin)

    # ④ 注意力 ✓（全 3 维 ✓；mask 略去——导出时传的是全 1 ✓）
    if kv_heads != heads:                                # GQA：把 KV 头复制到 Q 头 ✓
        rep = heads // kv_heads
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
    scores = (q @ k.transpose(-1, -2)) * scale                        # [BH, S, S] ✓
    scores = scores + _causal_bias(s, scores.device, scores.dtype)     # ★因果掩码★ ✓（3 维广播 ✓）
    attn = torch.softmax(scores, dim=-1)                              # [BH, S, S] ✓
    out = attn @ v                                       # [BH, S, D] ✓

    # ⑤ 复原布局 ✓ → gate ✓ → o_proj ✓
    out = from_heads(out, heads)                         # [B, S, H*D] ✓
    # ★gate 不需要换序★：它本来就是 [B, S*H, D] 布局 ✓
    #   直接 reshape 就是 [B, S, H*D] ✓（对它做置换反而会把内存解释乱 ✗ —— §59 实测 ✓）
    gate = gate_flat.reshape(b, s, heads * hd)           # [B, S, H*D] ✓
    out = out * torch.sigmoid(gate)
    return self.o_proj(out), None


def install(modeling_module=None):
    """把参考实现替换成 3 维版 ✓（与 npu_gated_delta 的 install 风格一致 ✓）。"""
    global _PATCHED
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    cls = modeling_module.Qwen3_5Attention
    if "forward" not in _PATCHED:
        _PATCHED["forward"] = cls.forward
        cls.forward = npu_attention_forward
    return _PATCHED


def uninstall(modeling_module=None):
    global _PATCHED
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    if "forward" in _PATCHED:
        modeling_module.Qwen3_5Attention.forward = _PATCHED.pop("forward")
