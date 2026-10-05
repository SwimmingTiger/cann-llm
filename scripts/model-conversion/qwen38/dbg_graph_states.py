"""图 vs Python：比第 0 层的两路状态输出 + 中间量 ✓（定位图管线差异 ✗）。"""
import numpy as np, torch, onnxruntime as ort
from transformers import AutoConfig, Qwen3_5ForCausalLM
import npu_layers as NL
S, KV = 64, 256
torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = 1; tc.layer_types = list(tc.layer_types)[:1]
m = Qwen3_5ForCausalLM(tc).eval(); layer = m.model.layers[0]
embeds = torch.randn(1, S, tc.hidden_size)
pos = torch.arange(S).unsqueeze(0)
cos, sin = m.model.rotary_emb(embeds, pos.unsqueeze(0).expand(3, -1, -1))
kv_heads, hd = tc.num_key_value_heads, 256
conv0 = torch.zeros(1, 6144, 3); rec0 = torch.zeros(1, 16, 128, 128)
mask = torch.full((1, 1, S, KV), -1e9); mask[0, 0, :, :S] = torch.triu(torch.full((S, S), -1e9), 1)
h_py, nk_py, nv_py = NL.layer_forward(layer, embeds, mask, cos, sin, conv0, rec0, 0,
                                       "linear_attention", KV, tc.num_attention_heads, kv_heads, hd, S)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("/home/hu60/q38/q35_hiai_L1.onnx", so, providers=["CPUExecutionProvider"])
print("  图输出名:", [o.name for o in sess.get_outputs()])
feeds = {}
for n in [i.name for i in sess.get_inputs()]:
    if n == "input_embed": feeds[n] = embeds.numpy().astype(np.float32)
    elif n == "attention_mask": feeds[n] = mask.numpy().astype(np.float32)
    elif n == "position_ids": feeds[n] = np.tile(np.arange(S, dtype=np.int64), (3, 1, 1))
    elif n == "new_kv_cache_pos": feeds[n] = np.arange(S, dtype=np.int64)
    elif n == "past_key_in0": feeds[n] = conv0.numpy()
    elif n == "past_value_in0": feeds[n] = rec0.numpy()
outs = sess.run(None, feeds)
names = [o.name for o in sess.get_outputs()]
for nm, val, ref in zip(names, outs, [h_py, nk_py, nv_py]):
    r = ref.numpy() if torch.is_tensor(ref) else ref
    if r.shape != val.shape:
        print("  %-14s 形状不同：图 %s vs Python %s" % (nm, val.shape, r.shape)); continue
    d = np.abs(val - r).max()
    print("  %-14s 最大差 %.4e  %s" % (nm, d, "✓" if d < 2e-2 else "✗"))
