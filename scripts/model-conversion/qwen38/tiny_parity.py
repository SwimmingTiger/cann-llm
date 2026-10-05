"""微型模型对拍 + 补丁命中统计（快 ✓）。

跑法：~/q38env/bin/python tiny_parity.py
"""
import sys
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import npu_gated_delta as N

torch.set_grad_enabled(False)
torch.manual_seed(0)

cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = 2
tc.hidden_size = 256
tc.intermediate_size = 512
tc.num_attention_heads = 4
for k in ("linear_num_key_heads", "linear_num_value_heads"):
    if hasattr(tc, k):
        setattr(tc, k, 4)
tc.layer_types = ["linear_attention", "full_attention"]
model = Qwen3_5ForCausalLM(tc).eval()
print("微模型 %.1f M 参数 | layer_types=%s" % (sum(p.numel() for p in model.parameters()) / 1e6, tc.layer_types))

ids = torch.arange(1, 129, dtype=torch.long).unsqueeze(0)
pos = torch.arange(128, dtype=torch.long).unsqueeze(0)

print("\n① 原版 …", flush=True)
ref = model(input_ids=ids, position_ids=pos, use_cache=False).logits

print("② 打补丁（带命中计数）…", flush=True)
hits = {"chunk": 0, "conv": 0, "recurrent": 0}
N.install(M)
_oc, _ov = M.torch_chunk_gated_delta_rule, M.causal_conv1d_fn


def w_chunk(*a, **k):
    hits["chunk"] += 1
    return _oc(*a, **k)


def w_conv(*a, **k):
    hits["conv"] += 1
    return _ov(*a, **k)


M.torch_chunk_gated_delta_rule = w_chunk
M.causal_conv1d_fn = w_conv
if hasattr(M, "torch_recurrent_gated_delta_rule"):
    _orr = M.torch_recurrent_gated_delta_rule
    M.torch_recurrent_gated_delta_rule = lambda *a, **k: (hits.__setitem__("recurrent", hits["recurrent"] + 1), _orr(*a, **k))[1]

npu = model(input_ids=ids, position_ids=pos, use_cache=False).logits
N.uninstall(M)

print("  补丁命中次数:", hits)
d = (npu - ref).abs()
scale = ref.abs().max().item()
agree = (npu.argmax(-1) == ref.argmax(-1)).float().mean().item()
print("\n=== 结果 ===")
print("  最大绝对差 %.3e | 最大相对差 %.3e | argmax 一致率 %.4f"
      % (d.max().item(), d.max().item() / max(1e-9, scale), agree))
ok = agree >= 0.999 and hits["chunk"] + hits["recurrent"] > 0 and hits["conv"] > 0
print("\n%s" % ("★★ 微模型对拍通过 ✓★" if ok and d.max().item() / max(1e-9, scale) < 1e-3
                else "✗ 见上面的命中次数与误差"))
sys.exit(0 if ok else 1)
