"""线性层的状态改用【官方 KV 形状的缓冲】✓。

原因（§72）：引擎按官方约定给每层分配 past_key_in{i}/past_value_in{i} 的
  形状 [kv_max, kv_heads, B, head_dim]（4 MB ✓），而我们的 conv/rec 状态尺寸完全不同 ✗
  ⇒ CPUCL 报 "param["size"] is less than["dataSize"]" ✗

做法：线性层的两个槽位也声明成官方 KV 形状 ✓，内部
  · 拉平成 1 维 ✓（4 维→1 维的 Reshape 是允许的 ✓，被拒的只有 4 维 Slice ✗）
  · ★1 维切片★取出 conv 窗口与递归状态 ✓
  · 算完再用 ★1 维 Concat + 常量零★ 填回同样长度 ✓ ⇒ 输出仍是 KV 形状 ✓
全流程 ≤3 维 ✓（1 维/2 维/3 维 ✓）。
"""
import io
P = "npu_layers.py"
s = io.open(P, encoding="utf-8").read()

# 线性层：入口加官方缓冲形状参数 ✓
s = s.replace('''def linear_attention_layer(layer, hidden, conv_state, rec_state, heads, kv_heads, hd, seq: int = 0,
                           batch: int = 0, trace: list | None = None):''',
'''def linear_attention_layer(layer, hidden, conv_state, rec_state, heads, kv_heads, hd, seq: int = 0,
                           batch: int = 0, trace: list | None = None):
    """★conv_state/rec_state 现在是【官方 KV 形状的缓冲】✓★（§73）

    引擎按官方约定分配 [kv_max, kv_heads, B, head_dim] ✓，
    所以线性层的两个槽位也用这个形状 ✓ —— 内部拉平成 1 维再切片 ✓。
    """''')

s = s.replace('''    mixed = la.in_proj_qkv(hidden).transpose(1, 2)             # [B, conv_dim, S] ✓''',
'''    # ★从官方 KV 形状的缓冲里取出真正的状态★（1 维切片 ✓ 全 ≤3 维 ✓）
    n_k = getattr(la, "num_k_heads", getattr(la, "num_key_heads", heads))
    n_v0 = getattr(la, "num_v_heads", getattr(la, "num_value_heads", heads))
    d_k0 = getattr(la, "head_k_dim", getattr(la, "key_head_dim", hd))
    d_v0 = getattr(la, "head_v_dim", getattr(la, "value_head_dim", hd))
    ksize0 = la.conv1d.weight.shape[-1]
    conv_sz = (2 * n_k * d_k0 + n_v0 * d_v0) * (ksize0 - 1)
    rec_sz = n_v0 * d_k0 * d_v0
    flat_k = conv_state.reshape(-1)                            # 4维→1维 ✓ 允许 ✓
    flat_v = rec_state.reshape(-1)
    conv_state = flat_k[:conv_sz].reshape(b, conv_sz // max(ksize0 - 1, 1), ksize0 - 1) \\
        if ksize0 > 1 else flat_k[:0].reshape(b, 0, 0)
    rec_state = flat_v[:rec_sz].reshape(b, n_v0, d_k0, d_v0)

    mixed = la.in_proj_qkv(hidden).transpose(1, 2)             # [B, conv_dim, S] ✓''')

# 出口：把新状态填进同长度的 1 维缓冲 ✓
s = s.replace('''    core = core.reshape(b, s, v_heads * v_dim)
    if trace is not None:
        trace.append(("normed", core))
    out = la.out_proj(core)
    if trace is not None:
        trace.append(("out", out))
    return out, new_conv, new_rec''',
'''    core = core.reshape(b, s, v_heads * v_dim)
    if trace is not None:
        trace.append(("normed", core))
    out = la.out_proj(core)
    if trace is not None:
        trace.append(("out", out))
    # ★把新状态填回【官方 KV 形状】的缓冲★（1 维 Concat + 常量零 ✓ 长度不变 ✓）
    tot = conv_state.numel() // 1 if False else None
    tot_k = int(flat_k.shape[0])
    tot_v = int(flat_v.shape[0])
    nk = new_conv.reshape(-1).shape[0]
    nv = new_rec.reshape(-1).shape[0]
    zk = torch.zeros(tot_k - nk, dtype=flat_k.dtype, device=flat_k.device)
    zv = torch.zeros(tot_v - nv, dtype=flat_v.dtype, device=flat_v.device)
    new_k_slot = torch.cat([new_conv.reshape(-1), zk], dim=0).reshape(conv_state_shape)
    new_v_slot = torch.cat([new_rec.reshape(-1), zv], dim=0).reshape(rec_state_shape)
    return out, new_k_slot, new_v_slot''')

# 入口处记下原始形状 ✓（供出口 reshape 用）
s = s.replace('''    flat_k = conv_state.reshape(-1)                            # 4维→1维 ✓ 允许 ✓''',
'''    conv_state_shape = conv_state.shape                        # 记下官方形状 ✓
    rec_state_shape = rec_state.shape
    flat_k = conv_state.reshape(-1)                            # 4维→1维 ✓ 允许 ✓''')
io.open(P, "w", encoding="utf-8").write(s)
print("线性层状态已改用官方 KV 形状缓冲 ✓")
