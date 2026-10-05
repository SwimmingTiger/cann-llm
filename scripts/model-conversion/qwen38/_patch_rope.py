"""纯浮点版 RoPE（图内不放整数运算 ✓）。

背景（§68）：图内调用模型的 rotary_emb 会带来 FloorMod 等整型算子 ✗，
OMG 直接拒收（"Input[0] DataType INT32 is wrong" ✗）。
实测反推（§69）：文本输入的 M-RoPE 三段完全相同 ✓，最终形状就是
    cos/sin = cos/sin( cat([pos*inv_freq, pos*inv_freq], -1) )    [B,S,64] ✓
（与模型输出差 6.3e-08 ✓）。⇒ 自己在图内用纯浮点算 ✓，inv_freq 作为常量 ✓。
"""
import io
P = "npu_layers.py"
s = io.open(P, encoding="utf-8").read()
if "def rope_cos_sin" not in s:
    s = s.replace('''def _to_heads(''', '''def rope_cos_sin(position_ids, inv_freq, dtype=torch.float32):
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


def _to_heads(''')
    io.open(P, "w", encoding="utf-8").write(s)
    print("npu_layers 已加纯浮点 rope ✓")

P2 = "export_hiai_q35.py"
t = io.open(P2, encoding="utf-8").read()
t = t.replace('''            # ★M-RoPE 的 position_ids 是 [3,B,S]★ ✓（官方接口给 [B,S] ⇒ 这里展开 ✓）
            pos3 = position_ids
            if pos3.dim() == 2:
                pos3 = pos3.unsqueeze(0).expand(3, -1, -1)
            cos, sin = self.body.rotary_emb(inputs_embeds, pos3)''',
'''            # ★纯浮点 RoPE★（不用模型的 rotary_emb ✗ —— 它会引入 FloorMod 等整型算子 ✗）
            pos1 = position_ids[0] if position_ids.dim() == 3 else position_ids
            cos, sin = npu_layers.rope_cos_sin(pos1, self._inv_freq)''')
t = t.replace('''    class Wrap(torch.nn.Module):
        """官方接口的一版实现 ✓（embedding 可在图内或图外 ✓）。"""

        def __init__(self, m):
            super().__init__()
            self.body = m.model''',
'''    class Wrap(torch.nn.Module):
        """官方接口的一版实现 ✓（embedding 可在图内或图外 ✓）。"""

        def __init__(self, m):
            super().__init__()
            self.body = m.model
            # inv_freq 是常量 ✓（纯浮点 RoPE 用 ✓）
            self.register_buffer("_inv_freq",
                                 m.model.rotary_emb.inv_freq.detach().clone().float(),
                                 persistent=False)''')
io.open(P2, "w", encoding="utf-8").write(t)
print("导出脚本已换用纯浮点 rope ✓")
