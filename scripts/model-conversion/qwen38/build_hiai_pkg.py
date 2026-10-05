"""组装 hiai 模型包（§47 形态 ✓）：embedding(int8+scale) + omc + tokenizer + 配置 ✓。

实测格式（对 models/model_qwen2_1p5b_w4_2048 反推 ✓）：
  <name>_<seq>_<kv>.embedding_weights        int8，逐元素 1 字节 ✓（vocab × hidden）
  <name>_<seq>_<kv>.embedding_dequant_scale  fp32，★每行一个 scale★ ✓
      （607744 B ÷ 151936 行 = 4 B ✓ 正好对上 ✓）
  ⇒ 量化方案：逐行对称 int8 ✓（scale = max|row| / 127 ✓）
配置：<name>.json（llm_config）· executor.json · context.json · api_config.json ✓
      照 W4 包改维度 ✓（另可用 `python3 -m cann_llm.modelpkg <dir>` 补齐 ✓）

用法（hu60tx 上跑 ✓，HF 检查点在那）：
  ~/q38env/bin/python build_hiai_pkg.py --hf /home/hu60/q38 --name qwen38_2b \
      --seq 64 --kv 2048 --out ~/q38/pkg_qwen38_2b
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/home/hu60/q38")
    ap.add_argument("--name", default="qwen38_2b")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--kv", type=int, default=2048)
    ap.add_argument("--omc", default="/home/hu60/q38/omg_fp16/seg.omc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-embedding", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.hf)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg

    # ---------- ① embedding: 逐行对称 int8 ✓ ----------
    emb_prefix = "%s_%d_%d" % (args.name, args.seq, args.kv)
    wpath = os.path.join(args.out, emb_prefix + ".embedding_weights")
    spath = os.path.join(args.out, emb_prefix + ".embedding_dequant_scale")
    if not args.skip_embedding:
        from transformers import AutoModelForCausalLM
        print("  载入 HF 取 embedding …", flush=True)
        m = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32)
        W = m.get_input_embeddings().weight.detach().float().numpy()
        print("  embedding 形状:", W.shape, flush=True)
        scale = np.abs(W).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
        q.tofile(wpath)
        scale.astype(np.float32).tofile(spath)
        print("  ✓ %s %.1f MB · %s %.2f MB" % (os.path.basename(wpath), os.path.getsize(wpath) / 1e6,
                                              os.path.basename(spath), os.path.getsize(spath) / 1e6))
        del m

    # ---------- ② omc + tokenizer ----------
    omc_dst = os.path.join(args.out, args.name + ".omc")
    if not os.path.exists(omc_dst) and os.path.exists(args.omc):
        print("  复制 omc …", flush=True)
        shutil.copy2(args.omc, omc_dst)
        print("  ✓ %s %.1f MB" % (os.path.basename(omc_dst), os.path.getsize(omc_dst) / 1e6))
    tok = os.path.join(args.hf, "tokenizer.json")
    if os.path.exists(tok):
        shutil.copy2(tok, os.path.join(args.out, "tokenizer.json"))
        print("  ✓ tokenizer.json")

    # ---------- ③ <name>.json（llm_config ✓ 照 W4 包字段 ✓）----------
    def g(o, k, d=None):
        return getattr(o, k, d)

    llm = {
        "bos_token_id": g(tc, "bos_token_id", 0),
        "eos_token_id": g(tc, "eos_token_id", 0),
        "kv_cache_max_len": args.kv,
        "sliding_window_len": 0,
        "max_position_embeddings": g(tc, "max_position_embeddings", 32768),
        "num_attention_kv_heads": g(tc, "num_key_value_heads", 0),
        "num_attention_head_dims": g(tc, "head_dim", 128),
        "num_hidden_layers": g(tc, "num_hidden_layers", 0),
        "prefill_len": args.seq,
        "decode_len": 1,
        "vocab_size": g(tc, "vocab_size", 0),
        "vocab_real_size": g(tc, "vocab_size", 0),
        "use_output_pos": False,
        "max_io_tokens": 4096,
        "hidden_size": g(tc, "hidden_size", 0),
        "embedding_weights": emb_prefix + ".embedding_weights",
        "embedding_dequant_scale": emb_prefix + ".embedding_dequant_scale",
        "embedding_input_type": "int8",
        "pad_token_id": g(tc, "pad_token_id", 0) or 0,
        "num_attention_heads": g(tc, "num_attention_heads", 0),
        "intermediate_size": g(tc, "intermediate_size", 0),
        "rms_norm_eps": g(tc, "rms_norm_eps", 1e-6),
        "rope_theta": float((g(tc, "rope_parameters", {}) or {}).get("rope_theta", 1e6)),
        "model_type": g(tc, "model_type", "qwen3_5"),
        "architectures": ["Qwen3_5ForCausalLM"],
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": bool(g(tc, "tie_word_embeddings", False)),
    }
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(llm, fh, indent=4, ensure_ascii=False)
    print("  ✓ %s.json" % args.name)

    # ---------- ④ executor.json / context.json / api_config.json ✓（照 W4 包 ✓）----------
    executor = {
        "version": 1,
        "engine_type": "autoregressive",
        "llm_config": llm,
        "tokenizer": {"type": "qwen", "path": "tokenizer.json"},
        "autoregressive": {"model_path": args.name + ".omc", "weight_dir": "./"},
    }
    with open(os.path.join(args.out, "executor.json"), "w") as fh:
        json.dump(executor, fh, indent=4, ensure_ascii=False)
    context = {
        "version": 1,
        "engine_type": "autoregressive",
        "generate_options": {"max_gen_tokens": 2048, "prefill_len": args.seq, "decode_len": 1},
        "sampler": {"top_k": 20, "top_p": 0.8, "temperature": 0.7, "repetition_penalty": 1.1},
    }
    with open(os.path.join(args.out, "context.json"), "w") as fh:
        json.dump(context, fh, indent=4, ensure_ascii=False)
    api = {
        "inferType": 0, "tokenizerType": 4, "tokenizerPath": "tokenizer.json",
        "modelType": 0, "modelPath": args.name + ".omc", "weightDir": "./",
        "prefixPrompt": "", "pmtCacheOperation": "", "pfxInitTokenLen": 6, "loraCfgPath": "",
        "expect": "", "callbackFreq": 2, "sampleFlag": True, "seed": 99, "topK": 20,
        "topP": 0.8, "temperature": 0.7, "maxGenTokens": 2048, "repetitionPenalty": 1.1,
        "initTokenLen": 1024, "stopSeq": ["<|im_end|>", "<|endoftext|>"],
    }
    with open(os.path.join(args.out, "api_config.json"), "w") as fh:
        json.dump(api, fh, indent=4, ensure_ascii=False)
    print("  ✓ executor.json / context.json / api_config.json")
    print("打包完成 ⇒ %s" % args.out)
    for f in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, f)
        print("   %10.1f MB  %s" % (os.path.getsize(p) / 1e6, f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
