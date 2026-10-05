"""全注意力层逐步对拍 ✓（定位 §64 的那处 bug）。"""
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM
import npu_attention as A

torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
m = Qwen3_5ForCausalLM(tc).eval(); body = m.model
B, S = 1, 64
heads, kv_heads, hd = tc.num_attention_heads, tc.num_key_value_heads, 256
layer = body.layers[3]; att = layer.self_attn
h = torch.randn(B, S, tc.hidden_size)
hn = layer.input_layernorm(h)
pos = torch.arange(S).unsqueeze(0)
cos, sin = body.rotary_emb(hn, pos.unsqueeze(0).expand(3, -1, -1))

def md(tag, x, y):
    d = (x - y).abs().max().item()
    print("  %-22s 最大差 %.4e  %s" % (tag, d, "✓" if d < 1e-3 else "✗"))

# ---- 参考路径（照 Qwen3_5Attention.forward ✓，无缓存 ✓，因果 ✓）----
qg = att.q_proj(hn).view(B, S, -1, hd * 2)
q_ref, gate_ref = torch.chunk(qg, 2, dim=-1)
gate_ref = gate_ref.reshape(B, S, -1)
q_ref = att.q_norm(q_ref).transpose(1, 2)                  # [B,H,S,D]
k_ref = att.k_norm(att.k_proj(hn).view(B, S, kv_heads, hd)).transpose(1, 2)
v_ref = att.v_proj(hn).view(B, S, kv_heads, hd).transpose(1, 2)
q_ref, k_ref = A._apply_rope_3d(q_ref, k_ref, cos, sin)
kk = k_ref.repeat_interleave(heads // kv_heads, dim=1)
vv = v_ref.repeat_interleave(heads // kv_heads, dim=1)
probs_ref = torch.softmax((q_ref @ kk.transpose(-1, -2)) * att.scaling
                          + A._causal_bias(S, q_ref.device), dim=-1)
out_ref = (probs_ref @ vv).transpose(1, 2).reshape(B, S, -1)

# ---- 我的路径（经 full_attention_layer ✓，缓存置零 ✓）----
import npu_layers as NL
KV = 256
mask = torch.full((B, 1, S, KV), -1e9)
mask[0, 0, :, :S] = torch.triu(torch.full((S, S), -1e9), 1)
o_mine, nk, nv = NL.full_attention_layer(
    layer, hn, mask, cos, sin, torch.zeros(KV, kv_heads, B, hd), torch.zeros(KV, kv_heads, B, hd),
    KV, heads, kv_heads, hd)

# 我的中间量（复算一遍以便逐步比 ✓）
q3 = NL._to_heads(att.q_proj(hn).reshape(B, S * heads, 2 * hd)[..., :hd], B, S, heads, hd)
print("  ① q 换序（rope 前）  最大差 %.4e" % (q3 - att.q_norm(q_ref.reshape(B*heads, S, hd))).abs().max().item())
md("② 注意力输出（变换前）", NL._from_heads(o_mine, B, S, heads, hd), out_ref)
print("  附：参考 out 量级 %.4f | 我的 %.4f" % (out_ref.abs().max().item(), o_mine.abs().max().item()))
