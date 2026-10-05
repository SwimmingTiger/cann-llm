"""分段导出定位：把线性注意力层切成 5 段，各自导出小图并在 ORT 里跑，
与【同一进程内】的 Python 计算逐步对比 ✓（权重/输入完全一致 ✓）。

段划分：
  seg1: input_embed          -> mixed（in_proj_qkv + transpose ✓）
  seg2: mixed + conv_state   -> conv_out（拼接 + 因果卷积 + 截取 + transpose ✓）
  seg3: conv_out             -> query/key/value/beta/g（split/reshape/repeat ✓）
  seg4: q/k/v + beta/g + rec -> core + new_rec（delta rule ✓）
  seg5: core + z             -> out（gated norm + out_proj ✓）
跑法：~/q38env/bin/python dbg_segments.py
"""
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForCausalLM

import npu_gated_delta as G
import npu_layers as NL

S = 64
torch.set_grad_enabled(False)
torch.manual_seed(0)


def ort_run(path, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def export(mod, inputs, names, out_names, path):
    torch.onnx.export(mod, inputs, path, input_names=names, output_names=out_names,
                      opset_version=14, dynamo=False)
    return path


def main():
    model = Qwen3_5ForCausalLM.from_pretrained("/home/hu60/q38", dtype=torch.float32).eval()
    model.model.layers = model.model.layers[:1]
    model.requires_grad_(False)          # ★冻结★：否则 tracer 报 "Cannot insert a Tensor that requires grad" ✗
    layer = model.model.layers[0]
    la = layer.linear_attn
    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    embeds = torch.randn(1, S, H)
    conv_state = torch.zeros(1, 6144, 3)
    rec_state = torch.zeros(1, 16, 128, 128)
    ksize = la.conv1d.weight.shape[-1]

    # ---------- seg1: input_layernorm + in_proj_qkv + transpose ----------
    def seg1_fn(e):
        return la.in_proj_qkv(layer.input_layernorm(e)).transpose(1, 2)

    mixed_py = seg1_fn(embeds)
    m1 = export(_Wrap(seg1_fn), (embeds,), ["input_embed"], ["mixed"], "/home/hu60/q38/seg1.onnx")
    got = ort_run(m1, {"input_embed": embeds.numpy()})[0]
    print("  seg1 mixed         最大差 %.4e %s" % (np.abs(got - mixed_py.numpy()).max(),
                                                "✓" if np.abs(got - mixed_py.numpy()).max() < 1e-3 else "✗"))

    # ---------- seg2: 拼接 + 卷积 + 截取 + transpose ----------
    def seg2_fn(mixed, cs):
        x = torch.cat([cs, mixed], dim=2)
        y = G.npu_causal_conv1d_fn(x, la.conv1d.weight.squeeze(1), la.conv1d.bias,
                                   activation=getattr(la, "activation", None))
        y = y[:, :, (ksize - 1):]
        return y.transpose(1, 2), x[:, :, S:]

    conv_py, nc_py = seg2_fn(mixed_py, conv_state)
    m2 = export(_Wrap2(seg2_fn), (mixed_py, conv_state), ["mixed", "conv_state"],
                ["conv_out", "new_conv"], "/home/hu60/q38/seg2.onnx")
    o = ort_run(m2, {"mixed": mixed_py.numpy(), "conv_state": conv_state.numpy()})
    print("  seg2 conv_out      最大差 %.4e %s" % (np.abs(o[0] - conv_py.numpy()).max(),
                                                "✓" if np.abs(o[0] - conv_py.numpy()).max() < 1e-3 else "✗"))
    print("  seg2 new_conv      最大差 %.4e %s" % (np.abs(o[1] - nc_py.numpy()).max(),
                                                "✓" if np.abs(o[1] - nc_py.numpy()).max() < 1e-3 else "✗"))

    # ---------- seg3: split/reshape/repeat ----------
    k_heads, v_heads = la.num_k_heads, la.num_v_heads
    k_dim, v_dim = la.head_k_dim, la.head_v_dim

    def seg3_fn(c):
        b, s, _ = c.shape
        z = la.in_proj_z(layer.input_layernorm(embeds)).reshape(b, s, v_heads, v_dim)
        beta = la.in_proj_b(layer.input_layernorm(embeds)).sigmoid()
        g = -la.A_log.float().exp() * F.softplus(la.in_proj_a(layer.input_layernorm(embeds)).float() + la.dt_bias)
        kd, vd = k_heads * k_dim, v_heads * v_dim
        q, k, v = torch.split(c, [kd, kd, vd], dim=-1)
        q = q.reshape(b, s, k_heads, k_dim)
        k = k.reshape(b, s, k_heads, k_dim)
        v = v.reshape(b, s, v_heads, v_dim)
        return q, k, v, beta, g, z

    py3 = seg3_fn(conv_py)
    m3 = export(_Wrap3(seg3_fn), (conv_py,), ["conv_out"],
                ["query", "key", "value", "beta", "g", "z"], "/home/hu60/q38/seg3.onnx")
    o3 = ort_run(m3, {"conv_out": conv_py.numpy()})
    for nm, a, bb in zip(("query", "key", "value", "beta", "g", "z"), o3, py3):
        d = np.abs(a - bb.numpy()).max()
        print("  seg3 %-6s         最大差 %.4e %s" % (nm, d, "✓" if d < 1e-3 else "✗"))

    # ---------- seg4: delta rule ----------
    def seg4_fn(q, k, v, beta, g, rec):
        core, new_rec = G.npu_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, initial_state=rec, output_final_state=True,
            use_qk_l2norm_in_kernel=True, seq_len=S, batch=1)
        return core, new_rec

    core_py, nr_py = seg4_fn(*py3[:5], rec_state)
    m4 = export(_Wrap6(seg4_fn), (py3[0], py3[1], py3[2], py3[3], py3[4], rec_state),
                ["query", "key", "value", "beta", "g", "rec_state"], ["core", "new_rec"],
                "/home/hu60/q38/seg4.onnx")
    o4 = ort_run(m4, {"query": py3[0].numpy(), "key": py3[1].numpy(), "value": py3[2].numpy(),
                      "beta": py3[3].numpy(), "g": py3[4].numpy(), "rec_state": rec_state.numpy()})
    print("  seg4 delta core    最大差 %.4e %s" % (np.abs(o4[0] - core_py.numpy()).max(),
                                                "✓" if np.abs(o4[0] - core_py.numpy()).max() < 1e-2 else "✗"))
    print("  seg4 new_rec       最大差 %.4e %s" % (np.abs(o4[1] - nr_py.numpy()).max(),
                                                "✓" if np.abs(o4[1] - nr_py.numpy()).max() < 1e-2 else "✗"))

    # ---------- seg5: gated norm + out_proj ----------
    z = py3[5]

    def seg5_fn(core, zz):
        normed = la.norm(core, zz) if "rmsnormgated" in type(la.norm).__name__.lower() else la.norm(core)
        return la.out_proj(normed.reshape(1, S, -1))

    out_py = seg5_fn(core_py, z)
    m5 = export(_Wrap2(seg5_fn), (core_py, z), ["core", "z"], ["out"], "/home/hu60/q38/seg5.onnx")
    o5 = ort_run(m5, {"core": core_py.numpy(), "z": z.numpy()})[0]
    print("  seg5 out           最大差 %.4e %s" % (np.abs(o5 - out_py.numpy()).max(),
                                                "✓" if np.abs(o5 - out_py.numpy()).max() < 1e-2 else "✗"))


class _Wrap(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a):
        return self.fn(a)


class _Wrap2(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn(a, b)


class _Wrap3(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a):
        return self.fn(a)


class _Wrap6(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b, c, d, e, f):
        return self.fn(a, b, c, d, e, f)


if __name__ == "__main__":
    main()
