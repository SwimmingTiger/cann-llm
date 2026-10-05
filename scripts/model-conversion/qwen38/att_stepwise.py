"""注意力 3 维版的★逐步数值对拍★：把参考实现每一阶段都复现一遍，和 npu_attention 比。

用于钉死 §58 剩下的那处差异（argmax 0.9583 ✗）。
跑法（hu60tx）：~/q38env/bin/python att_stepwise.py
"""
import torch

from transformers import AutoConfig, Qwen3_5ForCausalLM

import npu_attention as A


def md(x, y):
    x = x.float()
    y = y.float()
    if x.shape != y.shape:
        return "形状不同 %s vs %s" % (tuple(x.shape), tuple(y.shape))
    d = (x - y).abs().max().item()
    scale = max(x.abs().max().item(), 1e-9)
    return "最大差 %.3e（相对 %.2e）" % (d, d / scale)


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained("/home/hu60/q38")
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc.num_hidden_layers = 1
    tc.layer_types = ["full_attention"]
    tc.hidden_size = 256
    tc.intermediate_size = 512
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    m = Qwen3_5ForCausalLM(tc).eval()
    att = m.model.layers[0].self_attn
    b, s = 1, 24
    h = torch.randn(b, s, 256)
    # ★M-RoPE 的 position_ids 是 [3, B, S]★（模型内部由 attention_mask 的 cumsum 得到 ✓）
    pos = torch.arange(s).view(1, 1, s).expand(3, b, s)
    cos, sin = m.model.rotary_emb(h, pos)
    print("  cos/sin 形状:", tuple(cos.shape), tuple(sin.shape))

    heads, kv_heads, hd = att.config.num_attention_heads, att.config.num_key_value_heads, att.head_dim
    scale = att.scaling

    # ---------------- 参考实现各阶段 ----------------
    qg = att.q_proj(h).view(b, s, -1, hd * 2)
    q_ref, gate_ref = torch.chunk(qg, 2, dim=-1)          # [B,S,H,D] ✓
    gate_ref = gate_ref.reshape(b, s, -1)                 # [B,S,H*D] ✓
    q_ref = att.q_norm(q_ref).transpose(1, 2)             # [B,H,S,D] ✓
    k_ref = att.k_norm(att.k_proj(h).view(b, s, kv_heads, hd)).transpose(1, 2)
    v_ref = att.v_proj(h).view(b, s, kv_heads, hd).transpose(1, 2)
    q_ref, k_ref = A._apply_rope_3d(q_ref, k_ref, cos, sin)   # 用同一个 rope ✓
    kk = k_ref.repeat_interleave(heads // kv_heads, dim=1)
    vv = v_ref.repeat_interleave(heads // kv_heads, dim=1)
    probs_ref = torch.softmax((q_ref @ kk.transpose(-1, -2)) * scale
                              + A._causal_bias(s, q_ref.device), dim=-1)
    out_ref = (probs_ref @ vv).transpose(1, 2).reshape(b, s, -1)   # [B,S,H*D] ✓
    out_ref = att.o_proj(out_ref * torch.sigmoid(gate_ref))

    # ---------------- 3 维版各阶段 ----------------
    qg3 = att.q_proj(h).reshape(b, s * heads, 2 * hd)
    q3 = qg3[..., :hd]
    g3 = qg3[..., hd:]
    perm = A._perm_indices(s, heads, h.device)

    def to_heads(x, nh):
        x = x.reshape(b, s * nh, hd)
        return torch.index_select(x, 1, A._perm_indices(s, nh, x.device)).reshape(b * nh, s, hd)

    def from_heads(x, nh):
        x = x.reshape(b, nh * s, hd)
        return torch.index_select(x, 1, A._unperm_tensor(s, nh, x.device)).reshape(b, s, nh * hd)

    q3 = to_heads(q3, heads)
    k3 = to_heads(att.k_proj(h), kv_heads)
    v3 = to_heads(att.v_proj(h), kv_heads)
    print("  ① Q 换序          :", md(q3, q_ref.reshape(b * heads, s, hd)))
    print("  ② K 换序          :", md(k3, k_ref.reshape(b * kv_heads, s, hd)))
    print("  ③ V 换序          :", md(v3, v_ref.reshape(b * kv_heads, s, hd)))
    q3n, k3n = att.q_norm(q3), att.k_norm(k3)
    print("  ④ q_norm/k_norm   :", md(q3n, att.q_norm(q_ref.reshape(b * heads, s, hd))))
    q3r, k3r = A._apply_rope_3d(q3n, k3n, cos, sin)
    kk3 = k3r.repeat_interleave(heads // kv_heads, dim=0)
    vv3 = v3.repeat_interleave(heads // kv_heads, dim=0)
    print("  ⑤ rope 之后 Q     :", md(q3r, q_ref.reshape(b * heads, s, hd)))
    probs3 = torch.softmax((q3r @ kk3.transpose(-1, -2)) * scale
                           + A._causal_bias(s, q3r.device), dim=-1)
    print("  ⑥ 注意力概率      :", md(probs3, probs_ref.reshape(b * heads, s, s)))
    out3 = probs3 @ vv3
    print("  ⑦ 注意力输出      :", md(out3, (probs_ref @ vv).reshape(b * heads, s, hd)))
    back3 = from_heads(out3, heads)
    print("  ⑧ 复原布局        :", md(back3, (probs_ref @ vv).transpose(1, 2).reshape(b, s, heads * hd)))
    g3b = g3.reshape(b, s, heads * hd)      # gate 无需换序 ✓
    print("  ⑨ gate            :", md(g3b, gate_ref))
    print("  ⑩ 最终输出        :", md(att.o_proj(back3 * torch.sigmoid(g3b)), out_ref))


if __name__ == "__main__":
    main()
