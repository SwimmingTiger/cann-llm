"""把 full_attention_layer 的缓存段重写干净：3 维移位 ✓ + 输出回 4 维 ✓。"""
import io
P = "npu_layers.py"
s = io.open(P, encoding="utf-8").read()
start = s.index("    # ★顺序很关键★")
end = s.index("    out = att.o_proj(out * torch.sigmoid(gate))")
new = '''    # ★顺序★：先更新缓存（官方语义：past_key_in{i} 是本步【之前】的缓存 ✓），再在更新后的
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
'''
s = s[:start] + new + s[end:]
s = s.replace('''    out = att.o_proj(out * torch.sigmoid(gate))

    return out, new_k, new_v''', '''    out = att.o_proj(out * torch.sigmoid(gate))

    return out, out_k, out_v''')
io.open(P, "w", encoding="utf-8").write(s)
print("缓存段已重写 ✓")
