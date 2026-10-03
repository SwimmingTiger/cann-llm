"""设备跑出的 hidden → 主机侧 norm + lm_head + softcap → 与 HF 的 top-5 对拍（目标验收）。"""
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_model as G

m, tm = G.load()
ids = torch.tensor([[1, 42, 777, 9000]], dtype=torch.long)
mask4 = torch.zeros([1, 1, 4, 4])
with torch.no_grad():
    ref = m(input_ids=ids, attention_mask=mask4, use_cache=False).logits

raw = torch.from_numpy(__import__("numpy").fromfile(
    os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "gemma4"), dtype="float32"))
h = raw.view(1, 4, 1536)
print("  设备 hidden:", tuple(h.shape), flush=True)

with torch.no_grad():
    hn = tm.norm(h)                      # ★ 段链不做 norm，这里做 ✓
    logits = m.lm_head(hn)
    cap = getattr(tm.config, "final_logit_softcapping", None)
    if cap: logits = torch.tanh(logits / cap) * cap

a = logits[0, -1].float(); b = ref[0, -1].float()
ta = torch.topk(a, 5).indices.tolist(); tb = torch.topk(b, 5).indices.tolist()
print("  ★ logits 最大绝对差 = %.4e（参考量级 %.2f）" % ((a-b).abs().max().item(), b.abs().max().item()), flush=True)
print("  ★ 我的 top5 ids: %s" % ta, flush=True)
print("  ★ HF   top5 ids: %s" % tb, flush=True)
print("  ★ 我的 top5 值: %s" % [round(float(x),3) for x in torch.topk(a,5).values], flush=True)
print("  ★ HF   top5 值: %s" % [round(float(x),3) for x in torch.topk(b,5).values], flush=True)
print("  ★★ %s" % ("★★ 设备端全链 top-5 与 transformers 完全一致 ⇒ 目标达成 ★★"
                  if ta == tb else ("集合一致" if set(ta)==set(tb) else "✗ 不一致")), flush=True)
