"""对拍：导出的 hiai 接口图（ONNX Runtime）vs 未打补丁的 HF 参考实现 ✓。

要点：图的注意力是在【缓存】上做的（kv_len 列 ✓）⇒
  · 缓存输入置零 ✓（等于没有历史 ✓）
  · mask 输入要把【前 S 列之外的列】压到 -1e9 ✓（否则 softmax 会平摊到空位 ✗）
"""
import sys
import numpy as np
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

onnx_path = sys.argv[1] if len(sys.argv) > 1 else "/home/hu60/q38/q35_hiai_L4.onnx"
S, KV = 64, 256

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = 4
tc.layer_types = list(tc.layer_types)[:4]
m = Qwen3_5ForCausalLM(tc).eval()                       # ★未打补丁=参考实现★ ✓

embeds = torch.randn(1, S, tc.hidden_size)
pos = torch.arange(S).unsqueeze(0)
am = torch.ones(1, S, dtype=torch.long)
ref = m.model(inputs_embeds=embeds, position_ids=pos, use_cache=False, attention_mask=am).last_hidden_state

# ---- ORT ----
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
feeds = {}
kv_heads = tc.num_key_value_heads
hd = getattr(tc, "head_dim", 256)
for n in names:
    if n == "input_embed":
        feeds[n] = embeds.numpy().astype(np.float32)
    elif n == "attention_mask":
        mm = np.full((1, 1, S, KV), -1e9, dtype=np.float32)
        mm[0, 0, :, :S] = np.triu(np.full((S, S), -1e9, dtype=np.float32), 1)   # 因果 ✓
        feeds[n] = mm
    elif n == "position_ids":
        feeds[n] = np.arange(S, dtype=np.int32)[None, :]        # ★2 维✓★（§69 我们导的是 2 维 ✓）
    elif n == "new_kv_cache_pos":
        feeds[n] = np.arange(S, dtype=np.int32)[None, :]   # ★2 维 int32✓★
    elif n.startswith("past_key_in") or n.startswith("past_value_in"):
        idx = int(n.split("in")[-1])
        lt = tc.layer_types[idx]
        if lt == "full_attention":
            feeds[n] = np.zeros((KV, kv_heads, 1, hd), dtype=np.float32)
        elif n.startswith("past_key_in"):
            feeds[n] = np.zeros((1, 6144, 3), dtype=np.float32)
        else:
            feeds[n] = np.zeros((1, 16, 128, 128), dtype=np.float32)
    else:
        raise SystemExit("未知输入 %s" % n)
# ★按图的声明转换输入 dtype★（我们的图把 position_ids / new_kv_cache_pos 设成 INT32 ✓ §87）
for _k in ("position_ids", "new_kv_cache_pos"):
    if _k in feeds:
        feeds[_k] = np.asarray(feeds[_k], dtype=np.int32)
outs = sess.run(None, feeds)
got = outs[0]
print("  参考 %s | ORT %s" % (tuple(ref.shape), got.shape))
d = np.abs(got.astype(np.float64) - ref.numpy().astype(np.float64))
print("  最大绝对差 %.4e | 参考量级 %.3f | 相对 %.3e" % (d.max(), np.abs(ref.numpy()).max(),
                                                    d.max() / max(np.abs(ref.numpy()).max(), 1e-9)))
print("  ★对拍通过 ✓★" if d.max() / max(np.abs(ref.numpy()).max(), 1e-9) < 2e-2 else "  ✗ 不过")
