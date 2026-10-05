"""整层图（ORT）vs ★HF 参考实现那一层★ —— 同一个进程、同一份真权重 ✓。"""
import numpy as np, torch, onnxruntime as ort
from transformers import Qwen3_5ForCausalLM
S, KV = 64, 256
torch.set_grad_enabled(False); torch.manual_seed(0)
m = Qwen3_5ForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32).eval()
m.model.layers = m.model.layers[:1]; m.requires_grad_(False)
tc = m.config.text_config if hasattr(m.config, "text_config") else m.config
layer = m.model.layers[0]
embeds = torch.randn(1, S, tc.hidden_size)
pos = torch.arange(S).unsqueeze(0)
cos, sin = m.model.rotary_emb(embeds, pos.unsqueeze(0).expand(3, -1, -1))
# ★参考：直接跑那一层（走它自己的通路 ✓）
ref = layer(embeds, position_embeddings=(cos, sin), attention_mask=None,
            position_ids=pos, use_cache=False)[0]
# ★图★
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("/home/hu60/q38/q35_real_L1.onnx", so, providers=["CPUExecutionProvider"])
mask = np.full((1, 1, S, KV), -1e9, dtype=np.float32)
mask[0, 0, :, :S] = np.triu(np.full((S, S), -1e9, dtype=np.float32), 1)
print("  图的输入:", [i.name for i in sess.get_inputs()])
feeds = {}
for i in sess.get_inputs():
    n = i.name
    if n == "input_embed": feeds[n] = embeds.numpy()
    elif n == "attention_mask": feeds[n] = mask
    elif n == "position_ids": feeds[n] = np.tile(np.arange(S, dtype=np.int64), (3, 1, 1))
    elif n == "new_kv_cache_pos": feeds[n] = np.arange(S, dtype=np.int64)
    elif n == "past_key_in0": feeds[n] = np.zeros((1, 6144, 3), dtype=np.float32)
    else: feeds[n] = np.zeros((1, 16, 128, 128), dtype=np.float32)
got = sess.run(None, feeds)[0]
d = np.abs(got - ref.numpy()).max()
print("  图 vs 参考层：最大绝对差 %.4e | 参考量级 %.3f | 相对 %.3e ⇒ %s" % (
    d, ref.abs().max().item(), d / ref.abs().max().item(), "✓" if d / ref.abs().max().item() < 1e-2 else "✗"))
