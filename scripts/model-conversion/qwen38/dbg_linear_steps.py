"""线性注意力层逐步对拍：我的 npu_layers vs 参考实现的每一步 ✓（都用零状态 ✓）。"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn
import npu_gated_delta as G

torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
m = Qwen3_5ForCausalLM(tc).eval()
B, S = 1, 64
h = torch.randn(B, S, tc.hidden_size)
layer = m.model.layers[0]                      # 第 0 层是线性注意力 ✓
la = layer.linear_attn
hn = layer.input_layernorm(h)
print("层 0 类型:", tc.layer_types[0])

# ---- 参考路径（无缓存：整段卷积 ✓）----
mixed = la.in_proj_qkv(hn).transpose(1, 2)                  # [B, conv_dim, S]
z_ref = la.in_proj_z(hn).reshape(B, S, -1, la.head_v_dim)
beta_ref = la.in_proj_b(hn).sigmoid()
g_ref = -la.A_log.float().exp() * F.softplus(la.in_proj_a(hn).float() + la.dt_bias)
conv_ref = causal_conv1d_fn(mixed, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation)
conv_ref = conv_ref.transpose(1, 2)                          # [B, S, conv_dim]

# ---- 我的路径（缓存窗口拼前面 ✓）----
ksize = la.conv1d.weight.shape[-1]
state = torch.zeros(B, mixed.shape[1], ksize - 1)
x = torch.cat([state, mixed], dim=2)
conv_mine = G.npu_causal_conv1d_fn(x, la.conv1d.weight.squeeze(1), la.conv1d.bias,
                                   activation=getattr(la, "activation", None))
conv_mine = conv_mine[:, :, -S:].transpose(1, 2)             # 取最后 S 个 ✓

def md(tag, a, b):
    d = (a - b).abs().max().item()
    print("  %-22s 最大差 %.4e  %s" % (tag, d, "✓" if d < 1e-4 else "✗"))

md("① 卷积输出", conv_mine, conv_ref)
# ② 切分/reshape
kd = la.key_dim // 1 if hasattr(la, "key_dim") else None
print("  la.key_dim=%s value_dim=%s head_k_dim=%s head_v_dim=%s num_k_heads=%s num_v_heads=%s" % (
    getattr(la, "key_dim", "—"), getattr(la, "value_dim", "—"), getattr(la, "head_k_dim", "—"),
    getattr(la, "head_v_dim", "—"), getattr(la, "num_k_heads", "—"), getattr(la, "num_v_heads", "—")))
q_ref, k_ref, v_ref = torch.split(conv_ref, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
q_ref = q_ref.reshape(B, S, -1, la.head_k_dim)
k_ref = k_ref.reshape(B, S, -1, la.head_k_dim)
v_ref = v_ref.reshape(B, S, -1, la.head_v_dim)
# ③ delta rule（参考实现 ✓）
core_ref, _ = G2 = None, None
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
core_ref, st_ref = torch_chunk_gated_delta_rule(q_ref, k_ref, v_ref, g=g_ref, beta=beta_ref,
                                                initial_state=None, output_final_state=True,
                                                use_qk_l2norm_in_kernel=True)
# 我的（带零初始状态 ✓）
core_mine, st_mine = G.npu_chunk_gated_delta_rule(q_ref, k_ref, v_ref, g_ref, beta_ref, chunk_size=64,
                                                  initial_state=torch.zeros(B, la.num_v_heads, la.head_k_dim, la.head_v_dim),
                                                  output_final_state=True, use_qk_l2norm_in_kernel=True)
md("③ delta rule 输出", core_mine, core_ref)
md("③ 递归状态", st_mine, st_ref)
# ④ gated norm + out_proj
o_ref = la.out_proj(la.norm(core_ref, z_ref).reshape(B, S, -1))
o_mine = la.out_proj(la.norm(core_mine, z_ref).reshape(B, S, -1))
md("④ out_proj 输出", o_mine, o_ref)
ref_layer = layer(h, position_embeddings=(torch.zeros(1, S, 64), torch.zeros(1, S, 64)),
                  attention_mask=None, position_ids=torch.arange(S).unsqueeze(0), use_cache=False)[0]
print("  （参考整层输出量级 %.3f）" % ref_layer.abs().max().item())
