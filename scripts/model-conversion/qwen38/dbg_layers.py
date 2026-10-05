"""逐层定位：拿参考实现的每一层 vs 我们 npu_layers 的每一层比 ✓（零状态 ✓）。"""
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM
import npu_layers as NL
import npu_attention as A

torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = 4; tc.layer_types = list(tc.layer_types)[:4]
m = Qwen3_5ForCausalLM(tc).eval()
body = m.model
B, S, KV = 1, 64, 256
heads, kv_heads, hd = tc.num_attention_heads, tc.num_key_value_heads, 256
h = torch.randn(B, S, tc.hidden_size)
pos = torch.arange(S).unsqueeze(0)
cos, sin = body.rotary_emb(h, pos.unsqueeze(0).expand(3, -1, -1))
am = torch.ones(B, S, dtype=torch.long)
mask_causal = torch.full((B, 1, S, KV), -1e9)
mask_causal[0, 0, :, :S] = torch.triu(torch.full((S, S), -1e9), 1)

for i, layer in enumerate(body.layers):
    lt = tc.layer_types[i]
    ref = layer(h, position_embeddings=(cos, sin), attention_mask=am, position_ids=pos, use_cache=False)[0]
    if lt == "full_attention":
        k0 = torch.zeros(KV, kv_heads, B, hd); v0 = torch.zeros(KV, kv_heads, B, hd)
    else:
        k0 = torch.zeros(B, 6144, 3); v0 = torch.zeros(B, 16, 128, 128)
    if lt == "full_attention":
        got, _, _ = NL.layer_forward(layer, h, mask_causal, cos, sin, k0, v0, i, lt, KV, heads, kv_heads, hd, S)
    else:
        got, _, _ = NL.layer_forward(layer, h, mask_causal, cos, sin, k0, v0, i, lt, KV, heads, kv_heads, hd, S)
    d = (got - ref).abs().max().item()
    print("  层 %d %-17s 最大差 %.4e   %s" % (i, lt, d, "✓" if d < 1e-2 else "✗"))
