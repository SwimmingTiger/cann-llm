"""全链 → logits → 与 HF 做 top-k 对拍（目标里的验收标准）。

注意：词表 262144 × 4B = 1 MB，正好顶到设备"单张量 <1MB"的上限 ✗
⇒ lm_head 放在【主机侧】做（本脚本就是在验证这条路线 ✓）。
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_model as G

NO, SEQ = 4, 4
m, tm = G.load()
print("  载入 ✓", flush=True)
ids = torch.tensor([[1, 42, 777, 9000]], dtype=torch.long)
mask4 = torch.zeros([1, 1, SEQ, SEQ])

# ① HF 参考 logits（含 final_logit_softcapping）
with torch.no_grad():
    ref = m(input_ids=ids, attention_mask=mask4, use_cache=False).logits
print("  HF logits:", tuple(ref.shape), flush=True)

# ② 我的全链（9 段）→ norm → lm_head → softcap
with torch.no_grad():
    x = tm.embed_tokens(ids)
    ple = tm.project_per_layer_inputs(x, tm.get_per_layer_inputs(ids, x))
    pos = torch.arange(SEQ).unsqueeze(0)
    sk = sv = fk = fv = torch.zeros(1)
    NTOT = len(tm.layers)
    st = 0
    while st < NTOT:
        no = min(NO, NTOT - st)
        w = G.Seg3D(tm, st, no).eval()
        cs = []
        for i in range(st, st + no):
            c, s2 = tm.rotary_emb(x, pos, tm.config.layer_types[i])
            cs += [c[0], s2[0]]
        x, sk, sv, fk, fv = w(x, torch.zeros([1, SEQ, SEQ]), sk, sv, fk, fv,
                              *cs, *[ple[:, :, i, :] for i in range(st, st + no)])
        st += no
    h = tm.norm(x)                       # ★ 段内不做 norm，全链跑完在这里做 ✓
    logits = m.lm_head(h)
    cap = getattr(tm.config, "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
print("  我的 logits:", tuple(logits.shape), "· softcap =", cap, flush=True)

# ③ 对拍：最后位置的 top-k
a = logits[0, -1].float(); b = ref[0, -1].float()
d = (a - b).abs().max().item()
ta = torch.topk(a, 5).indices.tolist(); tb = torch.topk(b, 5).indices.tolist()
print("  ★ logits 最大绝对差 = %.4e（参考量级 %.2f）" % (d, b.abs().max().item()), flush=True)
print("  ★ 我的 top5 ids: %s" % ta, flush=True)
print("  ★ HF 的 top5 ids: %s" % tb, flush=True)
print("  ★★ top5 %s ⇒ %s" % ("完全一致" if ta == tb else "集合一致" if set(ta) == set(tb) else "不一致",
      "★★ 对拍通过 ★★" if set(ta) == set(tb) else "✗"))
