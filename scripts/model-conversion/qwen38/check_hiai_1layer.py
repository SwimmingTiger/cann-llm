"""1 层版对拍：图的 hidden_states vs 参考实现第 0 层的输出 ✓（定位图管线问题 ✓）。"""
import sys
import numpy as np, torch, onnxruntime as ort
from transformers import AutoConfig, Qwen3_5ForCausalLM
onnx_path, nlayers = sys.argv[1], int(sys.argv[2])
S, KV = 64, 256
torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = nlayers; tc.layer_types = list(tc.layer_types)[:nlayers]
m = Qwen3_5ForCausalLM(tc).eval()
embeds = torch.randn(1, S, tc.hidden_size)
pos = torch.arange(S).unsqueeze(0)
h = embeds
for layer in m.model.layers:                        # 逐层手跑（等价于整模型 ✓）
    out = layer(h, position_embeddings=m.model.rotary_emb(h, pos.unsqueeze(0).expand(3, -1, -1)),
                attention_mask=None, position_ids=pos, use_cache=False)
    h = out[0]
ref = h
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
kv_heads, hd = tc.num_key_value_heads, 256
feeds = {}
for n in [i.name for i in sess.get_inputs()]:
    if n == "input_embed":
        feeds[n] = embeds.numpy().astype(np.float32)
    elif n == "attention_mask":
        mm = np.full((1, 1, S, KV), -1e9, dtype=np.float32)
        mm[0, 0, :, :S] = np.triu(np.full((S, S), -1e9, dtype=np.float32), 1)
        feeds[n] = mm
    elif n == "position_ids":
        feeds[n] = np.tile(np.arange(S, dtype=np.int64), (3, 1, 1))
    elif n == "new_kv_cache_pos":
        feeds[n] = np.arange(S, dtype=np.int64)
    elif n.startswith("past_key_in") or n.startswith("past_value_in"):
        k = int(n.split("in")[-1]); lt = tc.layer_types[k]
        if lt == "full_attention":
            feeds[n] = np.zeros((KV, kv_heads, 1, hd), dtype=np.float32)
        elif n.startswith("past_key_in"):
            feeds[n] = np.zeros((1, 6144, 3), dtype=np.float32)
        else:
            feeds[n] = np.zeros((1, 16, 128, 128), dtype=np.float32)
outs = sess.run(None, feeds)
got = outs[0]
d = np.abs(got - ref.numpy()).max()
print("  %d 层：最大绝对差 %.4e | 参考量级 %.3f | 相对 %.3e ⇒ %s" % (
    nlayers, d, ref.abs().max().item(), d / ref.abs().max().item(),
    "✓" if d / ref.abs().max().item() < 1e-2 else "✗"))
