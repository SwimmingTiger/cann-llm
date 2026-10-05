"""看图上输出与 Python 的数值模式 ✓（转置？置换？还是彻底不对 ✓）+ 零输入对照 ✓。"""
import numpy as np, torch, onnxruntime as ort
from transformers import AutoConfig, Qwen3_5ForCausalLM
import npu_layers as NL
S, KV = 64, 256
torch.set_grad_enabled(False); torch.manual_seed(0)
cfg = AutoConfig.from_pretrained("/home/hu60/q38"); tc = cfg.text_config if hasattr(cfg,"text_config") else cfg
tc.num_hidden_layers = 1; tc.layer_types = list(tc.layer_types)[:1]
m = Qwen3_5ForCausalLM(tc).eval(); layer = m.model.layers[0]
kv_heads, hd = tc.num_key_value_heads, 256
def run(embeds, onnx):
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = m.model.rotary_emb(embeds, pos.unsqueeze(0).expand(3,-1,-1))
    conv0 = torch.zeros(1,6144,3); rec0 = torch.zeros(1,16,128,128)
    mask = torch.full((1,1,S,KV), -1e9); mask[0,0,:,:S] = torch.triu(torch.full((S,S),-1e9),1)
    h,nk,nv = NL.layer_forward(layer, embeds, mask, cos, sin, conv0, rec0, 0, "linear_attention",
                               KV, tc.num_attention_heads, kv_heads, hd, S, 1)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
    feeds = {}
    for n in [i.name for i in sess.get_inputs()]:
        if n=="input_embed": feeds[n]=embeds.numpy().astype(np.float32)
        elif n=="attention_mask": feeds[n]=mask.numpy().astype(np.float32)
        elif n=="position_ids": feeds[n]=np.tile(np.arange(S,dtype=np.int64),(3,1,1))
        elif n=="new_kv_cache_pos": feeds[n]=np.arange(S,dtype=np.int64)
        elif n=="past_key_in0": feeds[n]=conv0.numpy()
        else: feeds[n]=rec0.numpy()
    outs = sess.run(None, feeds)
    return (h,nk,nv), outs
for tag, embeds in [("随机", torch.randn(1,S,tc.hidden_size)), ("全零", torch.zeros(1,S,tc.hidden_size))]:
    py, gr = run(embeds, "/home/hu60/q38/q35_hiai_L1dyn.onnx")
    print("══ %s 输入 ══" % tag)
    for nm, p, g in zip(("hidden","conv_state","rec_state"), py, gr):
        p = p.numpy()
        print("  %-11s 图量级 %.4f | py 量级 %.4f | 最大差 %.4e | 首元素 图 %.4f vs py %.4f" % (
            nm, np.abs(g).max(), np.abs(p).max(), np.abs(g-p).max(), g.reshape(-1)[0], p.reshape(-1)[0]))
