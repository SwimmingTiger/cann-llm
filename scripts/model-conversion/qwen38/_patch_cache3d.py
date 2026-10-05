"""把 KV 缓存的"移位"从 4 维挪到 3 维做 ✓。

实测（§70）：NPUCL 只拒 4 维的 **Slice**（strided_slice ✗），而 4 维的
Transpose/Reshape 是【通过】的 ✓（官方 dopt 实现里也是 4 维转置 + index_put ✓）。
⇒ 边界仍是官方 4 维布局 [kv_max, kv_heads, B, hd] ✓，
   进层后立刻转成 3 维 [B*kv_heads, kv_max, hd] ✓，
   ★在 3 维里做 cat([新 K, 旧缓存的后半])★ ✓，算完再转回 4 维输出 ✓。
"""
import io
P = "npu_layers.py"
s = io.open(P, encoding="utf-8").read()
start = s.index("    # ★顺序很关键★")
end = s.index("    out = att.o_proj(out * torch.sigmoid(gate))")
new = '''    # ★顺序很关键★：先把本步的新 K/V 写进缓存（官方语义：past_key_in{i} 是本步【之前】的缓存 ✓），
    #   再在【更新后的缓存】上做注意力 ✓ —— 否则本步的 token 根本参与不了注意力 ✗（§65 实测踩到 ✓）
    #
    # ★缓存移位必须在 3 维里做★ ✗✗：NPUCL 拒收 4 维 Slice（strided_slice ✗，§69）
    #   而 4 维的 Transpose/Reshape 是通过的 ✓（官方 dopt 实现同样是 4 维转置 ✓）
    k3 = k.reshape(b, kv_heads, s, hd)                     # [B, kh, S, hd] ✓
    v3 = v.reshape(b, kv_heads, s, hd)
    ck = past_key.permute(2, 1, 0, 3).reshape(b * kv_heads, kv_max, hd)     # 4维→3维 ✓
    cv = past_value.permute(2, 1, 0, 3).reshape(b * kv_heads, kv_max, hd)
    k_new3 = k3.permute(0, 1, 3, 2).reshape(b * kv_heads, hd, s)            # [BH, hd, S] ✓
    v_new3 = v3.permute(0, 1, 3, 2).reshape(b * kv_heads, hd, s)
    # ★3 维切片 + Concat★：新 token 放前面，旧缓存保留 [S:] ✓（全静态 ✓）
    new_ck = torch.cat([k_new3, ck[:, :, s:]], dim=2)      # [BH, hd, kv_max] ✓ 3 维
    new_cv = torch.cat([v_new3, cv[:, :, s:]], dim=2)      # ★切片在 2/3 维上 ✓★

    q3 = q.reshape(b * heads, s, hd)                        # [BH, S, hd] ✓
    if kv_heads != heads:                                   # GQA ✓
        rep = heads // kv_heads
        new_ck = new_ck.repeat_interleave(rep, dim=0)
        new_cv = new_cv.repeat_interleave(rep, dim=0)
    scores = (q3 @ new_ck) * scale                          # [BH, S, kv_max] ✓ 3 维
    if mask is not None:
        m = mask.reshape(b, s, kv_max)                      # [B,S,kv] ✓
        scores = scores + m
    probs = torch.softmax(scores, dim=-1)
    out = probs @ new_cv.transpose(-1, -2)                  # [BH, S, hd] ✓
    out = _from_heads(out, b, s, heads, hd)

    # 回官方 4 维布局 ✓（Transpose/Reshape 在 4 维上没问题 ✓）
    new_k = new_ck.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2) if kv_heads == heads else None
    new_v = new_cv.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2) if kv_heads == heads else None
    if new_k is None:                                       # GQA：缓存只存 kv_heads 份 ✓
        k_src3 = k_new3
        v_src3 = v_new3
        new_k = torch.cat([k_src3, past_key.permute(2, 1, 0, 3).reshape(b * kv_heads, hd, kv_max)[:, :, s:]], dim=2)
        new_v = torch.cat([v_src3, past_value.permute(2, 1, 0, 3).reshape(b * kv_heads, hd, kv_max)[:, :, s:]], dim=2)
        new_k = new_k.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2)
        new_v = new_v.reshape(b, kv_heads, hd, kv_max).permute(3, 1, 0, 2)
'''
s = s[:start] + new + s[end:]
io.open(P, "w", encoding="utf-8").write(s)
print("缓存移位已改到 3 维 ✓")
