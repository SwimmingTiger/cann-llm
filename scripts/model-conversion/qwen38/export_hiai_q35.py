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
    import npu_layers

    torch.set_grad_enabled(False)
    # ★必须加载真实权重★（原来只读了 config ⇒ 建出的是随机初始化模型 ✗，
    #   和另一进程里的参考模型根本不是同一个 ⇒ 对拍永远不可能一致 ✓ —— §66 实测踩到 ✓）
    model = Qwen3_5ForCausalLM.from_pretrained(args.hf, dtype=torch.float32).eval()
    if args.layers:
        model.model.layers = model.model.layers[: args.layers]
    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    if args.layers:
        tc.num_hidden_layers = args.layers
        tc.layer_types = list(tc.layer_types)[: args.layers]
    tc.kv_cache_max_len = args.kv_len
    _w = model.model.layers[0].linear_attn.in_proj_qkv.weight
    print("FINGERPRINT in_proj_qkv |sum|=%.6f shape=%s dtype=%s" % (float(_w.abs().sum()), tuple(_w.shape), _w.dtype))

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
            # inv_freq 是常量 ✓（纯浮点 RoPE 用 ✓）
            self.register_buffer("_inv_freq",
                                 m.model.rotary_emb.inv_freq.detach().clone().float(),
                                 persistent=False)

        def forward(self, first, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = first if args.no_embed_head else self.body.embed_tokens(first)
            # ★纯浮点 RoPE★（不用模型的 rotary_emb ✗ —— 它会引入 FloorMod 等整型算子 ✗）
            # ★不要对 int64 张量做索引✗★（会生成 data 为 INT64 的 Gather ✗，
            #   OMG 直接拒："GatherV2D Verify failed, Input[0] DataType INT64 is wrong" ✓）
            cos, sin = npu_layers.rope_cos_sin(position_ids, self._inv_freq)
            hidden = inputs_embeds
            outs = []
            st = list(states)
            for idx, layer in enumerate(self.body.layers):
                k_slot, v_slot = st[2 * idx], st[2 * idx + 1]
                hidden, new_k, new_v = call_layer(layer, hidden, attention_mask, cos, sin,
                                                  k_slot, v_slot, idx)
                outs.extend([new_k, new_v])
            hidden = self.body.norm(hidden)
            logits = self.body.embed_tokens.weight.new_zeros(1)  # 占位；--no-embed-head 时不用 ✓
            return (hidden, *outs) if args.no_embed_head else (logits, *outs)

    def call_layer(layer, hidden, mask, cos, sin, k_slot, v_slot, idx):
        """调用一层的★我们自己的 3 维实现★ ✓（带状态进出 ✓）。

        全注意力层：k_slot/v_slot = KV 缓存 [kv, kv_heads, B, hd] ✓
        线性注意力层：k_slot = 卷积窗口 ✓；v_slot = 递归状态 [B, H, Dk, Dv] ✓
        """
        import npu_layers

        return npu_layers.layer_forward(layer, hidden, mask, cos, sin, k_slot, v_slot,
                                        idx, layer_types[idx], kv, heads, kv_heads, hd, args.seq, b)

    # 例化输入 ✓
    if args.no_embed_head:
        # ★官方 embedding_separate 约定：第一路输入是 input_embed [B,S,H]★ ✓（不是 input_ids ✗）
        first = torch.zeros(b, s, tc.hidden_size, dtype=torch.float32)
        first_name = "input_embed"
    else:
        first = torch.ones(b, s, dtype=torch.int64)
        first_name = "input_ids"
    attention_mask = torch.ones(b, 1, s, kv, dtype=torch.float32)
    position_ids = torch.arange(s).unsqueeze(0).expand(b, s)      # ★2 维✓★（文本 M-RoPE 三段相同 ✓）
    pos_new = torch.arange(s, dtype=torch.int64)
    inputs = [first, attention_mask, position_ids, pos_new]
    in_names = [first_name, "attention_mask", "position_ids", "new_kv_cache_pos"]
    out_names = ["hidden_states" if args.no_embed_head else "lm_logits"]   # ★名字要对上★ ✓
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
        else:
            # ★线性层的状态也用【官方 KV 形状】的缓冲★ ✗✗（§73）
            #   原因：引擎按官方约定给每层分配 [kv_max, kv_heads, B, head_dim] ✓，
            #   尺寸对不上就会 CPUCL 报 "param[size] is less than[dataSize]" ✗（§72）
            #   内部只用到缓冲的前一小段 ✓（1 维切片存取 ✓）
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
            inputs.append(torch.zeros(kv, kv_heads, b, hd))
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
