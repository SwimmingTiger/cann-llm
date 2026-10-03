"""导出【真实输入】与 HF 参考输出，供设备端对拍。

输入 = hidden(embed_tokens) + cos/sin + mask3 + per_layer_0..3
参考 = HF 整模型的 layer3 输出
全部存成 float32 裸二进制（设备端用纯 Python 读，不依赖 numpy）。
"""
import os, sys, torch, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_model as G
from transformers import AutoModelForCausalLM

tm = G.tm
SEQ, NO = 4, 4
OUT = os.path.join(os.environ.get("OUTDIR") or os.path.join(os.getcwd(), "out"), "gemma4"); os.makedirs(OUT, exist_ok=True)

# 用一组【非平凡】的 token，避免全 0 导致看不出问题
ids = torch.tensor([[1, 42, 777, 9000]], dtype=torch.long)
mask4 = torch.zeros([1, 1, SEQ, SEQ])
ref = {}
h = tm.layers[3].register_forward_hook(
        lambda mo, i, o: ref.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
with torch.no_grad():
    tm(input_ids=ids, attention_mask=mask4, use_cache=False)
h.remove()

torch.manual_seed(0)
with torch.no_grad():
    hidden = tm.embed_tokens(ids)
    ple = tm.get_per_layer_inputs(ids, hidden)
    ple = tm.project_per_layer_inputs(hidden, ple)            # [1,4,35,256]
    cos, sin = tm.rotary_emb(hidden, torch.arange(SEQ).unsqueeze(0), tm.config.layer_types[0])
    mask3 = torch.zeros([1, SEQ, SEQ])

def dump(name, t):
    a = t.detach().float().contiguous().cpu()
    p = os.path.join(OUT, name + ".bin")
    a.numpy().tofile(p)
    print("  %-16s %-16s → %d 字节" % (name, str(tuple(a.shape)), os.path.getsize(p)), flush=True)
    return p

dump("hidden", hidden); dump("cos", cos[0]); dump("sin", sin[0]); dump("mask3", mask3)
for i in range(NO):
    dump("per_layer_%d" % i, ple[:, :, i, :])
dump("ref_layer3", ref["h"])
with open(os.path.join(OUT, "ids.txt"), "w") as f:
    f.write(" ".join(str(int(x)) for x in ids[0]))
print("  ids:", ids[0].tolist(), flush=True)
print("  参考 layer3 前 6 个值:", [round(float(x),4) for x in ref["h"][0,0,:6]], flush=True)
