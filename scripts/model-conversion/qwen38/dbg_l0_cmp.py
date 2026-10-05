"""整层中间量对拍：ORT vs 同进程 Python ✓。"""
import numpy as np, torch, onnxruntime as ort
from transformers import Qwen3_5ForCausalLM
import npu_layers as NL
S = 64
torch.set_grad_enabled(False); torch.manual_seed(0)
m = Qwen3_5ForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32).eval()
m.model.layers = m.model.layers[:1]; m.requires_grad_(False)
tc = m.config.text_config if hasattr(m.config, "text_config") else m.config
layer = m.model.layers[0]
embeds = torch.randn(1, S, tc.hidden_size)
conv0 = torch.zeros(1, 6144, 3); rec0 = torch.zeros(1, 16, 128, 128)
h = layer.input_layernorm(embeds); tr = []
out, nconv, nrec = NL.linear_attention_layer(layer, h, conv0, rec0, 8, 2, 256, S, 1, tr)
py = [out, nconv, nrec] + [v for _, v in tr]
names = ["out", "new_conv", "new_rec", "mixed", "conv_in", "conv_out_sliced",
         "query", "key", "value", "g", "beta", "core", "normed"]
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession("/home/hu60/q38/dbg_l0.onnx", so, providers=["CPUExecutionProvider"])
o = sess.run(None, {"input_embed": embeds.numpy(), "conv_state": conv0.numpy(), "rec_state": rec0.numpy()})
for nm, g, p in zip(names, o, py):
    p = p.numpy()
    if g.shape != p.shape:
        print("  %-16s 形状不同 图%s vs py%s" % (nm, g.shape, p.shape)); continue
    d = np.abs(g - p).max()
    print("  %-16s 最大差 %.4e %s" % (nm, d, "✓" if d < 1e-3 else "✗"))
