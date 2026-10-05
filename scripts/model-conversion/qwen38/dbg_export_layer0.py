"""只导第 0 层（线性 ✓）并把每个中间量都作为图输出 ✓（逐段定位图上失真 ✗）。"""
import sys, torch
from transformers import AutoConfig, Qwen3_5ForCausalLM
import npu_layers as NL

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained("/home/hu60/q38")
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc.num_hidden_layers = 1; tc.layer_types = list(tc.layer_types)[:1]
m = Qwen3_5ForCausalLM(tc).eval()
layer = m.model.layers[0]
S = 64

class L0(torch.nn.Module):
    def forward(self, input_embed, conv_state, rec_state):
        h = layer.input_layernorm(input_embed)
        tr = []
        out, nconv, nrec = NL.linear_attention_layer(layer, h, conv_state, rec_state, 8, 2, 256, S, tr)
        return (out, nconv, nrec) + tuple(v for _, v in tr)

names = ["out", "new_conv", "new_rec", "mixed", "conv_in", "conv_out_sliced",
         "query", "key", "value", "g", "beta", "core", "normed"]
torch.onnx.export(L0().eval(),
                  (torch.zeros(1, S, tc.hidden_size), torch.zeros(1, 6144, 3), torch.zeros(1, 16, 128, 128)),
                  "/home/hu60/q38/dbg_l0.onnx", input_names=["input_embed", "conv_state", "rec_state"],
                  output_names=names, opset_version=14, dynamo=False)
print("导出 dbg_l0.onnx ✓ 输出 %d 个" % len(names))
