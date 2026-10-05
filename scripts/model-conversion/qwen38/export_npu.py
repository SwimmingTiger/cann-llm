"""导出 qwen3_5 的 ONNX（★打上 NPU 友好补丁★）并清点算子，与 DDK 平台库对比。

跑法：
    ~/q38env/bin/python export_npu.py --tiny      # 微型同构模型：秒级，只看算子集 ✓
    ~/q38env/bin/python export_npu.py --full      # 真模型 2B（fp32，约数 GB）✓

判断标准：★输出的"缺失算子"应为 0★（DDK 平台库里没有的算子 ✗）
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import glob

import torch

MODEL = os.path.expanduser("~/q38")
DDK_LIBS = os.path.expanduser("~/ddk/tools/platform/kirinx90/lib64")

#: DDK 平台库里确认【没有】的算子（§49 实测）
KNOWN_MISSING = {"ScatterElements", "ScatterND", "CumSum", "Trilu", "IsNaN", "GatherND"}


def ddk_has(op: str) -> bool:
    """在 DDK 平台库里找这个算子名（字符串级 ✓，与 §49 的方法一致 ✓）。"""
    for pat in ("libai_npucore_fusionengine_internal.so", "libai_npucore_ascendc.so"):
        for f in glob.glob(os.path.join(DDK_LIBS, pat)):
            try:
                with open(f, "rb") as fh:
                    if op.encode() in fh.read():
                        return True
            except OSError:
                pass
    return False


class Wrap(torch.nn.Module):
    """只暴露 logits（transformers 5.x 输出里带 DynamicCache，dynamo 不认 ✗）"""

    def __init__(self, m, attn_mask: bool = False):
        super().__init__()
        self.m = m
        self.attn_mask = attn_mask

    def forward(self, input_ids, position_ids):
        # ★图输入用 int32★（DDK 对 int64 支持存疑 ✗）⇒ 内部 Cast 到 int64 再查表 ✓
        input_ids = input_ids.long()
        position_ids = position_ids.long()
        kw = {}
        if self.attn_mask:
            # ★显式给全 1 的 attention_mask★：绕开"从 mask 反推 position_ids"的路径
            #   （那条路会引入 CumSum / GatherND ✗，见 §51）
            kw["attention_mask"] = torch.ones_like(input_ids)
        return self.m(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kw).logits


def build_model(tiny: bool, layers: int = 0):
    from transformers import AutoConfig, Qwen3_5ForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if tiny:
        tc.num_hidden_layers = 2
        tc.hidden_size = 256
        tc.intermediate_size = 512
        tc.num_attention_heads = 8
        tc.num_key_value_heads = 2
        tc.layer_types = ["linear_attention", "full_attention"]
        m = Qwen3_5ForCausalLM(tc).eval()
    elif layers:
        # ★真实宽度、只留前 N 层★：用来做"单图建图+推理"的工具链验证 ✓
        tc.num_hidden_layers = layers
        tc.layer_types = (tc.layer_types * layers)[:layers]
        m = Qwen3_5ForCausalLM(tc).eval()
        print("   （%d 层 · 真实宽度 %d · 随机权重 —— 只为验证工具链 ✓）" % (layers, tc.hidden_size))
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seq", type=int, default=128, help="固定序列长度（chunk 的整数倍 ✓）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn-mask", action="store_true", help="显式传全 1 attention_mask（绕开 mask→position_ids ✗）")
    ap.add_argument("--layers", type=int, default=0, help=">0 时用真实宽度但只保留这么多层（做单图验证 ✓）")
    args = ap.parse_args()
    tiny = args.tiny or not args.full
    out = args.out or ("micro_qwen35.onnx" if tiny else "qwen35_2b.onnx")

    torch.set_grad_enabled(False)
    print("① 建模型（%s）…" % ("微型同构 ✓" if tiny else "真 2B ✓"), flush=True)
    m = build_model(tiny, args.layers)
    print("   %.1f M 参数" % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)

    print("② 打 NPU 友好补丁 …", flush=True)
    import npu_gated_delta as N
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    N.install(M)

    s = args.seq
    # ★dummy 用 int32★ ⇒ 图输入就是 int32 ✓（DDK 对 int64 支持存疑 ✗；wrapper 内部 Cast 到 int64 ✓）
    ids = torch.arange(1, s + 1, dtype=torch.int32).unsqueeze(0)
    pos = torch.arange(s, dtype=torch.int32).unsqueeze(0)
    print("③ 导出 ONNX（dynamo / opset18 / 固定 S=%d）…" % s, flush=True)
    torch.onnx.export(
        Wrap(m, args.attn_mask), (ids, pos), out,
        input_names=["input_ids", "position_ids"], output_names=["logits"],
        dynamo=True, opset_version=18, do_constant_folding=True,
    )
    sz = os.path.getsize(out) + (os.path.getsize(out + ".data") if os.path.exists(out + ".data") else 0)
    print("   产物 %.2f GB" % (sz / 1e9), flush=True)

    import onnx
    # ★图级 lowering：把 DDK 缺失的算子换成等价支持算子组合 ✓
    import onnx_lower
    _m = onnx.load(out)
    _c = onnx_lower.lower_model(_m)
    if _c:
        onnx.save(_m, out)
        print("   lowering 完成：%s" % _c, flush=True)
    g = onnx.load(out, load_external_data=False).graph
    ops = collections.Counter(n.op_type for n in g.node)
    print("④ 算子清点：节点 %d · %d 种" % (len(g.node), len(ops)), flush=True)
    print("   " + ", ".join("%s×%d" % kv for kv in ops.most_common(60)), flush=True)

    ddk_missing = sorted(o for o in ops if not ddk_has(o))
    print("\n⑤ 与 DDK 平台库比对：", flush=True)
    if ddk_missing:
        print("   ★缺失 %d 种：%s★" % (len(ddk_missing), ", ".join(ddk_missing)), flush=True)
    else:
        print("   ★★ 零缺失 —— 所有算子 DDK 都有 ✓★", flush=True)
    for o in sorted(set(ops) & KNOWN_MISSING):
        print("   （注意：%s 在已知缺失清单里 ✓）" % o, flush=True)
    print("\n%s" % ("★ 目标② 达成：无缺失算子 ✓" if not ddk_missing else "✗ 仍有缺失"))
    return 0 if not ddk_missing else 1


if __name__ == "__main__":
    sys.exit(main())
