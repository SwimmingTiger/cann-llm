"""带状态的层实现（hiai 接口用 ✓）：全注意力层走 KV 缓存 ✓、线性注意力层走卷积窗口+递归状态 ✓。

设计见 maintainer-notes §63：
  · 两种层的状态都塞进同一对 past_key_in{i}/past_value_in{i} 槽位 ✓
  · 全注意力：key 槽 = K 缓存、value 槽 = V 缓存（官方布局 [kv_max, kv_heads, B, head_dim] ✓）
  · 线性注意力：key 槽 = 因果卷积窗口 [B, conv_dim, K-1] ✓
                value 槽 = gated delta rule 递归状态 [B, H, Dk, Dv] ✓
  · ★不写 ScatterND✗★：缓存更新一律用【静态切片 + Concat 位移】✓
  · 掩码用官方给的 attention_mask [B,1,S,kv] ✓（引擎负责算 ✓，我们不自己造 ✓）

对内一律 ≤3 维（§57：NPU 内核只吃 ≤3 维 ✗），只在【缓存边界】做必要的 4 维转置 ✓。
"""
from __future__ import annotations

import torch

import npu_attention as A
from npu_gated_delta import npu_causal_conv1d_fn, npu_chunk_gated_delta_rule


def rope_cos_sin(position_ids, inv_freq, dtype=torch.float32):
    """★纯浮点 RoPE★ ✓（position_ids 只做 Cast 成 float ✓，没有整数算术 ✗）。

    position_ids: [B,S]（int64 或 float 均可 ✓）· inv_freq: [dim/2] 常量 ✓
    返回 (cos, sin)，形状 [B,S,dim] ✓ —— 实测与模型 rotary_emb 输出差 6.3e-08 ✓（§69）
    """
    pos_f = position_ids.float() if position_ids.dtype != torch.float32 else position_ids
    if inv_freq.dtype != torch.float32:
        inv_freq = inv_freq.float()
    freqs = pos_f.unsqueeze(-1) * inv_freq                     # [B,S,dim/2] ✓ 纯浮点 ✓
    emb = torch.cat([freqs, freqs], dim=-1)                    # [B,S,dim] ✓
    return emb.cos(), emb.sin()


def _to_heads(x: torch.Tensor, b: int, s: int, n_head: int, hd: int) -> torch.Tensor:
    """[B, S, Nh*D] ➜ [B*Nh, S, D] ✓（常量索引 Gather 换序 ✓，见 npu_attention §58）。"""
    x = x.reshape(b, s * n_head, hd)
    perm = A._perm_indices(s, n_head, x.device)
    return torch.index_select(x, 1, perm).reshape(b * n_head, s, hd)


def _from_heads(x: torch.Tensor, b: int, s: int, n_head: int, hd: int) -> torch.Tensor:
    """逆运算 ✓。"""
    x = x.reshape(b, n_head * s, hd)
    inv = A._unperm_tensor(s, n_head, x.device)
    return torch.index_select(x, 1, inv).reshape(b, s, n_head * hd)


# ---------------------------------------------------------------- 全注意力层（带 KV 缓存）
def full_attention_layer(layer, hidden, mask, cos, sin, past_key, past_value, kv_max, heads, kv_heads, hd):
    att = layer.self_attn
    b, s, _ = hidden.shape
    scale = att.scaling

    qg = att.q_proj(hidden).reshape(b, s * heads, 2 * hd)      # 每 head 的 [q|gate] 块 ✓
    q = _to_heads(qg[..., :hd], b, s, heads, hd)
    gate = qg[..., hd:].reshape(b, s, heads * hd)              # gate 无需换序 ✓（§59）
    k = _to_heads(att.k_proj(hidden), b, s, kv_heads, hd)
    v = _to_heads(att.v_proj(hidden), b, s, kv_heads, hd)
    q = att.q_norm(q)
    k = att.k_norm(k)
    q, k = A._apply_rope_3d(q, k, cos, sin)                    # partial rope 0.25 ✓

    # ★顺序★：先更新缓存（官方语义：past_key_in{i} 是本步【之前】的缓存 ✓），再在更新后的
    #   缓存上做注意力 ✓ —— 否则本步 token 不参与注意力 ✗（§65 踩过 ✓）
    #
    # ★移位必须在 3 维里做✗✗★：NPUCL 拒收 4 维 Slice（strided_slice ✗，§69）；
    #   而 4 维的 Transpose/Reshape 是通过的 ✓（官方 dopt 实现也是 4 维转置 ✓）
    #   这里统一转到 [B*kv_heads, hd, kv_max] 的 3 维布局做移位 ✓
    ck = past_key.permute(2, 1, 0, 3).reshape(b * kv_heads, kv_max, hd).permute(0, 2, 1)   # [BH,hd,kv] ✓
    cv = past_value.permute(2, 1, 0, 3).reshape(b * kv_heads, kv_max, hd).permute(0, 2, 1)
    k_new = k.permute(0, 2, 1) if False else k.reshape(b * kv_heads, s, hd).permute(0, 2, 1)   # [BH,hd,S] ✓
    v_new = v.reshape(b * kv_heads, s, hd).permute(0, 2, 1)
    new_ck = torch.cat([k_new, ck[:, :, s:]], dim=2)        # ★3 维切片+Concat✓★ 新 token 在前 ✓
    new_cv = torch.cat([v_new, cv[:, :, s:]], dim=2)
    # ★输出缓存（kv_heads 份 ✓，用 repeat 之前的值 ✓）
    out_k = new_ck.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2)   # [kv,B? 官方布局 ✓]
    out_v = new_cv.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2)

    # 注意力（GQA 时把缓存复制到 heads 份 ✓）
    ck_att, cv_att = new_ck, new_cv
    if kv_heads != heads:
        rep = heads // kv_heads
        ck_att = ck_att.repeat_interleave(rep, dim=0)
        cv_att = cv_att.repeat_interleave(rep, dim=0)
    q3 = q.reshape(b * heads, s, hd)                        # [BH, S, hd] ✓
    scores = (q3 @ ck_att) * scale                          # [BH, S, kv_max] ✓ 3 维
    if mask is not None:
        m = mask.reshape(b, s, kv_max)                      # [B,S,kv] ✓
        scores = scores + m
    probs = torch.softmax(scores, dim=-1)
    out = probs @ cv_att.transpose(-1, -2)                  # [BH, S, hd] ✓
    out = _from_heads(out, b, s, heads, hd)
    out = att.o_proj(out * torch.sigmoid(gate))

    return out, out_k, out_v


# ---------------------------------------------------------------- 线性注意力层（卷积窗口 + 递归状态）
def linear_attention_layer(layer, hidden, conv_state, rec_state, heads, kv_heads, hd, seq: int = 0,
                           batch: int = 0, trace: list | None = None, skip: tuple = ()):
    """skip: 用于★探针二分★，可跳过 ('conv','delta','norm','out') 里的部件 ✓（生产路径不传 ✓）。"""
    la = layer.linear_attn
    # ★同样只用显式 int★（避免 legacy 追踪把 shape 变 Tensor ✗）
    b = batch or hidden.shape[0]
    s = seq or hidden.shape[1]
    k_heads = getattr(la, "num_k_heads", getattr(la, "num_key_heads", heads))
    v_heads = getattr(la, "num_v_heads", getattr(la, "num_value_heads", heads))
    k_dim = getattr(la, "head_k_dim", getattr(la, "key_head_dim", hd))
    v_dim = getattr(la, "head_v_dim", getattr(la, "value_head_dim", hd))
    ksize = la.conv_kernel_size if hasattr(la, "conv_kernel_size") else la.conv1d.weight.shape[-1]

    mixed = la.in_proj_qkv(hidden).transpose(1, 2)             # [B, conv_dim, S] ✓
    if trace is not None:
        trace.append(("mixed", mixed))
    z = la.in_proj_z(hidden).reshape(b, s, v_heads, v_dim)     # [B,S,Hv,Dv] ✓
    beta = la.in_proj_b(hidden).sigmoid()
    g = -la.A_log.float().exp() * torch.nn.functional.softplus(la.in_proj_a(hidden).float() + la.dt_bias)

    # ★卷积：把缓存窗口拼在序列前面★ ⇒ 新窗口 = 拼接后的最后 K-1 个 ✓（全静态切片 ✓）
    x = torch.cat([conv_state, mixed], dim=2)                  # [B, conv_dim, K-1+S] ✓
    if trace is not None:
        trace.append(("conv_in", x))
    y = npu_causal_conv1d_fn(x, la.conv1d.weight.squeeze(1), la.conv1d.bias,
                             activation=getattr(la, "activation", None))
    # ★切片下标一律用【正数】★ ✗✗：负下标切片在 opset>=10 的 ONNX 导出里语义会错
    #   （"Only steps=1 can be constant folded for opset >= 10 onnx::Slice" ✓）—— §65 实测踩到 ✓
    #   x 的长度是 (K-1)+S ✓ ⇒ 最后 K-1 个 = 从下标 S 开始 ✓
    new_conv = x[:, :, seq:] if ksize > 1 else x[:, :, :0]
    # ★只取【新 token】对应的输出★：状态那 K-1 个位置的卷积结果在上一步已经算过 ✓
    #   （下标用 Python int ✓ 避免旧 tracer 把 shape 变成 Tensor ✗，§53）
    y = y[:, :, (ksize - 1):]                       # ★正下标✓★：丢掉状态那 K-1 个位置 ✓
    if trace is not None:
        trace.append(("conv_out_sliced", y))
    y = y.transpose(1, 2)                                      # [B, S, conv_dim] ✓

    kd = k_heads * k_dim
    vd = v_heads * v_dim
    query, key, value = torch.split(y, [kd, kd, vd], dim=-1)
    query = query.reshape(b, s, k_heads, k_dim)
    key = key.reshape(b, s, k_heads, k_dim)
    value = value.reshape(b, s, v_heads, v_dim)
    if v_heads // k_heads > 1:
        rep = v_heads // k_heads
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    if trace is not None:
        trace.extend([("query", query), ("key", key), ("value", value), ("g", g), ("beta", beta)])
    # ★chunk_size 随 S 自适应✓★：写死 64 会在 S<64 时补齐 ✗
    #   ⇒ 最后一块没有有效行 ⇒ 三角掩码产生 0 维空张量[0,S,0] ✗
    #   ⇒ NPUCL 判 "dimCnt 2 != 3" ⇒ OMG 编译失败（§94 实测 ✓）
    #   decode(S=1) 用 chunk=1 ⇒ 退化为逐 token 递归更新 ✓ 且不补齐 ✓
    chunk = 64 if s >= 64 else max(s, 1)
    core, new_rec = npu_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=chunk, initial_state=rec_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True, seq_len=s, batch=b)

    # ★gated RMSNorm 要额外传 gate（z）★ ✓（类名形如 Qwen3_5RMSNormGated ⇒ 用 in 判断 ✓）
    if trace is not None:
        trace.append(("core", core))
    if "rmsnormgated" in type(la.norm).__name__.lower():
        core = la.norm(core, z)
    else:
        core = la.norm(core)
    # ★out_proj 要 [B,S,Hv*Dv]★ ✓（delta rule 给的是 [B,S,Hv,Dv] ✓，参考实现也 reshape ✓）
    core = core.reshape(b, s, v_heads * v_dim)
    if trace is not None:
        trace.append(("normed", core))
    out = la.out_proj(core)
    if trace is not None:
        trace.append(("out", out))
    return out, new_conv, new_rec



# ---------------------------------------------------------------- 统一入口（给导出脚本用）
def layer_forward(layer, hidden, mask, cos, sin, k_slot, v_slot, idx, layer_type,
                  kv_max, heads, kv_heads, hd, seq: int = 0, batch: int = 0):
    """返回 (hidden_out, 新 key 槽, 新 value 槽) ✓。"""
    residual = hidden
    h = layer.input_layernorm(hidden)
    if layer_type == "full_attention":
        h, nk, nv = full_attention_layer(layer, h, mask, cos, sin, k_slot, v_slot,
                                         kv_max, heads, kv_heads, hd)
    else:
        h, nk, nv = linear_attention_layer(layer, h, k_slot, v_slot, heads, kv_heads, hd, seq, batch=batch)
    hidden = residual + h
    residual = hidden
    hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, nk, nv
