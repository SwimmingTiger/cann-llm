"""正确验证：把段【串联】起来，在每个边界与 HF 对应层的输出对比。

（之前逐段单独测都喂 embedding 是错的 —— 段 k 的输入必须是段 k-1 的输出 ✓）
"""
import os, sys, torch
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
import g4_seg3d as G

NO, SEQ, NSEG = 4, 4, int(os.environ.get("NSEG", "3"))
m, tm = G.load()
print("  载入 ✓", flush=True)
ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
mask4 = torch.zeros([1, 1, SEQ, SEQ])

# ① HF 逐层输出（作为各边界的参考）
caps = {}
hs = [tm.layers[i].register_forward_hook(
        (lambda i: lambda mo, inp, o: caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().clone()))(i))
      for i in range(NSEG * NO)]
with torch.no_grad():
    tm(input_ids=ids, attention_mask=mask4, use_cache=False)
for h in hs: h.remove()

with torch.no_grad():
    x = tm.embed_tokens(ids)
    ple = tm.project_per_layer_inputs(x, tm.get_per_layer_inputs(ids, x))
    pos = torch.arange(SEQ).unsqueeze(0)
    print("  HF embedding 与我的差 = %.3e" % (x - tm.embed_tokens(ids)).abs().max().item(), flush=True)
    for seg in range(NSEG):
        st = seg * NO
        w = G.Seg3D(tm, st, NO).eval()
        cs = []
        for i in range(st, st + NO):
            c, s2 = tm.rotary_emb(x, pos, tm.config.layer_types[i])   # ★ 用当前 hidden 算 ✓
            cs += [c[0], s2[0]]
        x = w(x, torch.zeros([1, SEQ, SEQ]), *cs, *[ple[:, :, i, :] for i in range(st, st + NO)])
        last = st + NO - 1
        d = (x - caps[last]).abs().max().item()
        base = caps[last].abs().max().item()
        print("  段 [%d:%d] → 与 HF layer%d 差 = %.3e（参考量级 %.2f）%s"
              % (st, st + NO, last, d, base, "✓" if d / base < 1e-3 else "✗"), flush=True)
