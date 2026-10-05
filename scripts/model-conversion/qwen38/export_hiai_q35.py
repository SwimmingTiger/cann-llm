"""按【hiai LLM 引擎的官方接口】导出 qwen3_5 —— 混合架构的状态怎么接（★核心设计★）。

官方接口（抄自 cannkit 样例的 export_model_single_qwen3.py ✓）：
    输入  input_ids [B,S] int64 · attention_mask [B,1,S,kv] · position_ids [B,S]
          past_key_in{i} / past_value_in{i}   [kv_max, kv_heads, B, head_dim]
          new_kv_cache_pos [S] int64
    输出  lm_logits [B,S,V] · past_key{i} / past_value{i}

★qwen3_5 是混合架构（18 层线性注意力 + 6 层全注意力）✗：
    全注意力层的"缓存"就是 K/V ✓（形状照官方 ✓）；
    线性注意力层没有 K/V ✗，它要传的是
        · 因果卷积的卷积窗口（最近 kernel-1 个输入 ✓）
        · gated delta rule 的递归状态 [B, H, Dk, Dv] ✓
    ⇒ ★我们把这两样塞进同一对 past_key_in{i}/past_value_in{i} 槽位★ ✓
      （引擎把这些张量当不透明的每层缓存透传 ✓ —— 这正是它驱动任意结构模型的方式 ✓）

用法（先只导 body、不导 embedding/lm_head ✓）：
    python3 export_hiai_q35.py --hf /home/hu60/q38 --seq 64 --kv-len 2048 --layers 2 \
        --no-embed-head --out q35_hiai_L2.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/home/hu60/q38", help="HF 检查点目录")
    ap.add_argument("--seq", type=int, default=64, help="本次喂入的 token 数（prefill 长度 ✓）")
    ap.add_argument("--kv-len", type=int, default=2048, help="KV/状态的缓存长度（编译期常量 ✓）")
    ap.add_argument("--layers", type=int, default=0, help="只导前 N 层（0=全部 ✓，调试用 ✓）")
    ap.add_argument("--out", default="q35_hiai.onnx")
    ap.add_argument("--no-embed-head", action="store_true",
                    help="不导 embedding / lm_head（先把图跑通 ✓）")
    ap.add_argument("--legacy", action="store_true", help="用旧 TorchScript 导出器（opset 14 ✓ 必须 ✓）")
    args = ap.parse_args()

    from transformers import AutoConfig, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    import npu_attention
    import npu_gated_delta as N

    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(args.hf)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if args.layers:
        tc.num_hidden_layers = args.layers
        tc.layer_types = tuple(tc.layer_types)[: args.layers]
    tc.kv_cache_max_len = args.kv_len
    model = Qwen3_5ForCausalLM(tc).eval()

    N.install(M)          # delta rule 3 维化 ✓
    npu_attention.install(M)

    heads = tc.num_attention_heads
    kv_heads = tc.num_key_value_heads
    hd = getattr(tc, "head_dim", tc.hidden_size // heads)
    b, s, kv = 1, args.seq, args.kv_len
    layer_types = list(tc.layer_types)
    print("层型:", layer_types)
    print("heads=%d kv_heads=%d head_dim=%d hidden=%d" % (heads, kv_heads, hd, tc.hidden_size))

    class Wrap(torch.nn.Module):
        """官方接口的一版实现 ✓（embedding 可在图内或图外 ✓）。"""

        def __init__(self, m):
            super().__init__()
            self.body = m.model

        def forward(self, input_ids, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = self.body.embed_tokens(input_ids) if not args.no_embed_head else input_ids
            hidden = inputs_embeds
            outs = []
            st = list(states)
            for idx, layer in enumerate(self.body.layers):
                k_slot, v_slot = st[2 * idx], st[2 * idx + 1]
                hidden, new_k, new_v = call_layer(layer, hidden, attention_mask, position_ids,
                                                  new_kv_cache_pos, k_slot, v_slot, idx)
                outs.extend([new_k, new_v])
            hidden = self.body.norm(hidden)
            logits = self.body.embed_tokens.weight.new_zeros(1)  # 占位；--no-embed-head 时不用 ✓
            return (hidden, *outs) if args.no_embed_head else (logits, *outs)

    def call_layer(layer, hidden, mask, pos, pos_new, k_slot, v_slot, idx):
        """调用一层的★我们自己的 3 维实现★ ✓（带状态进出 ✓）。

        全注意力层：k_slot/v_slot = KV 缓存 [kv, kv_heads, B, hd] ✓
        线性注意力层：k_slot = 卷积窗口 ✓；v_slot = 递归状态 [B, H, Dk, Dv] ✓
        """
        import npu_layers

        return npu_layers.layer_forward(layer, hidden, mask, pos, pos_new, k_slot, v_slot,
                                        idx, layer_types[idx], kv, heads, kv_heads, hd)

    # 例化输入 ✓
    input_ids = torch.ones(b, s, dtype=torch.int64)
    attention_mask = torch.ones(b, 1, s, kv, dtype=torch.float32)
    position_ids = torch.arange(s).unsqueeze(0).expand(3, b, s)
    pos_new = torch.arange(s, dtype=torch.int64)
    inputs = [input_ids, attention_mask, position_ids, pos_new]
    in_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]
    out_names = ["lm_logits"]
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
        else:
            conv_dim = getattr(tc, "linear_conv_kernel_dim", 4)
            n_v = getattr(tc, "linear_num_value_heads", heads)
            d_k = getattr(tc, "linear_key_head_dim", hd)
            d_v = getattr(tc, "linear_value_head_dim", hd)
            inputs.append(torch.zeros(b, conv_dim * n_v, conv_dim))       # 卷积窗口 ✓
            inputs.append(torch.zeros(b, n_v, d_k, d_v))                  # 递归状态 ✓
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])

    kw = dict(opset_version=14, dynamo=False) if args.legacy else dict(opset_version=18, dynamo=True)
    print("导出（%s）…" % ("legacy opset14" if args.legacy else "dynamo opset18"))
    torch.onnx.export(Wrap(model), tuple(inputs), args.out, input_names=in_names,
                      output_names=out_names, **kw)
    size = os.path.getsize(args.out)
    if os.path.exists(args.out + ".data"):
        size += os.path.getsize(args.out + ".data")
    print("产物 %.1f MB ✓ 输入 %d 个 · 输出 %d 个" % (size / 1e6, len(in_names), len(out_names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
