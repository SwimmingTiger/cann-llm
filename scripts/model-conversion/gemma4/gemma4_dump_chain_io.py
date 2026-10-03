"""为设备端 9 段串联准备输入：hidden / 每段的 per_layer 切片 / 两组 cos-sin / mask3 / HF 参考 top5。"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_model as G

SEQ, NO = 4, 4
OUT = os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "gemma4"); os.makedirs(OUT, exist_ok=True)
m, tm = G.load()
ids = torch.tensor([[1, 42, 777, 9000]], dtype=torch.long)
mask4 = torch.zeros([1, 1, SEQ, SEQ])
with torch.no_grad():
    ref = m(input_ids=ids, attention_mask=mask4, use_cache=False).logits
tb = torch.topk(ref[0, -1].float(), 5).indices.tolist()
with torch.no_grad():
    hidden = tm.embed_tokens(ids)
    ple = tm.project_per_layer_inputs(hidden, tm.get_per_layer_inputs(ids, hidden))   # [1,4,35,256]
    pos = torch.arange(SEQ).unsqueeze(0)
    cs_sl = tm.rotary_emb(hidden, pos, "sliding_attention")
    cs_fu = tm.rotary_emb(hidden, pos, "full_attention")
def w(name, t):
    t.detach().float().contiguous().cpu().numpy().tofile(os.path.join(OUT, name + ".bin"))
w("hidden", hidden)
w("mask3", torch.zeros([1, SEQ, SEQ]))
w("cos_sl", cs_sl[0][0]); w("sin_sl", cs_sl[1][0])
w("cos_fu", cs_fu[0][0]); w("sin_fu", cs_fu[1][0])
NTOT = len(tm.layers); st = 0
while st < NTOT:
    no = min(NO, NTOT - st)
    for i in range(no):
        w("pl_%d_%d" % (st, i), ple[:, :, st + i, :])
    st += no
open(os.path.join(OUT, "ref_top5.txt"), "w").write(" ".join(map(str, tb)))
print("  ✓ io 已备：%d 个文件 · 参考 top5 = %s" % (len(os.listdir(OUT)) - 1, tb), flush=True)
print("  cos_sl %s · cos_fu %s · hidden %s" % (tuple(cs_sl[0][0].shape), tuple(cs_fu[0][0].shape), tuple(hidden.shape)), flush=True)
