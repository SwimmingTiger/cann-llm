#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把官方发布的 OMC 模型包整成 cann-llm 能直接加载的模型目录。

背景：OpenHarmony 模型矩阵（matrix.openharmony.cn）提供**已经量化好、编译好**的
OMC 模型包，解压出来是：

    <name>.omc                     编译好的模型
    SubGraph_0.weight              量化权重
    <prefix>.embedding_weights     词表 embedding
    <prefix>.embedding_dequant_scale
    tokenizer.json
    <model>.json                   引擎侧 llm_config（**扁平**）
    api_config.json                引擎侧生成配置（采样/停止符/chat template）

而 cann-llm（以及华为示例代码）用的是另一种打包：

    executor.json                  结构化：version/engine_type/llm_config/tokenizer/autoregressive
    context.json                   结构化：version/engine_type/generate_options/sampler

两者字段内容基本一一对应，只是外形不同。这个脚本做转换，省得手工拼。

用法::

    # 一般不用直接调它 —— 包装脚本更方便（还能处理缺 .zip 后缀等情况）：
    scripts/import_model.sh <包.zip> -d /path/to/model_dir

    python3 import_omc_package.py <解压后的目录> [--dry-run]

    # 或者直接从 zip 解出来并转换
    python3 import_omc_package.py <包.zip> --dest /path/to/model_dir

转换后会写出 ``executor.json`` 与 ``context.json``（原文件保留，方便对照）。
"""
import argparse
import json
import os
import shutil
import sys
import zipfile


def log(msg):
    print(f"  {msg}")


def ok(msg):
    """打勾的成功提示（与 scripts/*.sh 的 ok() 同款，不带缩进）。"""
    print(f"\033[32m✓\033[0m {msg}")


def info(msg):
    """箭头的过程提示（与 scripts/*.sh 的 info() 同款，不带缩进）。"""
    print(f"\033[36m›\033[0m {msg}")


def die(msg):
    print(f"错误：{msg}", file=sys.stderr)
    raise SystemExit(1)


#: 官方 <model>.json 里我们已知的 llm_config 字段；其余原样带过去
LLM_KEYS = (
    "bos_token_id", "eos_token_id", "pad_token_id",
    "kv_cache_max_len", "max_io_tokens", "sliding_window_len",
    "max_position_embeddings", "num_attention_kv_heads",
    "num_attention_head_dims", "num_hidden_layers", "num_attention_heads",
    "hidden_size", "intermediate_size", "vocab_size", "vocab_real_size",
    "prefill_len", "decode_len", "use_output_pos",
    "embedding_weights", "embedding_dequant_scale", "embedding_input_type",
    "enable_dynamic_kv_cache", "enable_lm_head_opt", "enable_lm_head_topk",
    "is_kv_cache_merge", "tie_word_embeddings", "rms_norm_eps", "rope_theta",
    "model_type", "architectures", "torch_dtype",
)


def find_files(d):
    """在目录里认出各个角色的文件，返回 dict。"""
    out = {"omc": None, "llm_json": None, "api_json": None,
           "emb_w": None, "emb_s": None, "tokenizer": None, "weight": None}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(".omc"):
            out["omc"] = name
        elif name == "api_config.json":
            out["api_json"] = name
        elif low.endswith(".json") and out["llm_json"] is None:
            out["llm_json"] = name                 # <model>.json
        elif "embedding_weights" in low:
            out["emb_w"] = name
        elif "embedding_dequant_scale" in low:
            out["emb_s"] = name
        elif low == "tokenizer.json":
            out["tokenizer"] = name
        elif low == "subgraph_0.weight":
            out["weight"] = name
    return out


def build_executor(found, d):
    llm = json.load(open(os.path.join(d, found["llm_json"])))
    cfg = {k: llm[k] for k in LLM_KEYS if k in llm}
    # embedding 文件名以磁盘上的实际文件为准（官方的名字和 llm_config 里可能不一致）
    if found["emb_w"]:
        cfg["embedding_weights"] = found["emb_w"]
    if found["emb_s"]:
        cfg["embedding_dequant_scale"] = found["emb_s"]
    return {
        "version": 1,
        "engine_type": "autoregressive",
        "llm_config": cfg,
        "tokenizer": {"type": "qwen", "path": found["tokenizer"] or "tokenizer.json"},
        "autoregressive": {"model_path": found["omc"], "weight_path": "./"},
    }


def build_context(found, d):
    api = json.load(open(os.path.join(d, found["api_json"])))
    stop = api.get("stopSeq") or []
    if isinstance(stop, str):
        stop = [stop]
    return {
        "version": 1,
        "engine_type": "autoregressive",
        "generate_options": {
            "callback_freq": api.get("callbackFreq", 1),
            "max_gen_tokens": api.get("maxGenTokens", 128),
            "stop_sequence": stop,
            "init_token_len": api.get("initTokenLen", 0),
        },
        "sampler": {
            "do_sample": bool(api.get("sampleFlag", True)),
            "seed": api.get("seed", 99),
            "top-k": api.get("topK", 20),
            "top-p": api.get("topP", 0.95),
            "temperature": api.get("temperature", 0.7),
            "repetition_penalty": api.get("repetitionPenalty", 1.1),
        },
    }


def convert(d, dry_run=False):
    """把官方 OMC 包整成结构化目录 —— 实现统一在 src/cann_llm/omcimport.py。

    这里只负责命令行体验（打印 + dry-run），转换逻辑与 launcher 的自动导入完全同一份，
    避免两边漂移。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from cann_llm.omcimport import find_omc_files, import_omc_dir

    found = find_omc_files(d)
    missing = [k for k in ("omc", "llm_json", "api_json", "tokenizer") if not found[k]]
    if missing:
        die(f"{d} 里缺少: {', '.join(missing)}（现有: {sorted(os.listdir(d))}）")

    log(f"omc        : {found['omc']}")
    log(f"llm_config : {found['llm_json']}")
    log(f"生成配置   : {found['api_json']}")
    log(f"embedding  : {found['emb_w']} / {found['emb_s']}")
    log(f"权重       : {found['weight']}")
    log(f"tokenizer  : {found['tokenizer']}")

    ex, cx = import_omc_dir(d, dry_run=dry_run)
    lc = ex["llm_config"]
    log(f"→ kv_cache_max_len={lc.get('kv_cache_max_len')} "
        f"layers={lc.get('num_hidden_layers')} kv_heads={lc.get('num_attention_kv_heads')} "
        f"dynamic_kv={lc.get('enable_dynamic_kv_cache')}")
    if dry_run:
        log("(--dry-run，不写文件)")
    else:
        log("已写 executor.json / context.json")
    return ex, cx


def main():
    ap = argparse.ArgumentParser(description="把官方 OMC 模型包整成 cann-llm 的模型目录")
    ap.add_argument("src", help="解压后的目录，或 .zip 包")
    ap.add_argument("--dest", help="zip 包解压到哪（默认与 zip 同名的目录）")
    ap.add_argument("--dry-run", action="store_true", help="只解析并打印，不写文件")
    args = ap.parse_args()

    if args.src.lower().endswith(".zip"):
        dest = args.dest or os.path.splitext(args.src)[0]
        if os.path.isdir(dest):
            log(f"目标目录已存在，跳过解压: {dest}")
        else:
            os.makedirs(dest, exist_ok=True)
            log(f"解压 {args.src} -> {dest}")
            with zipfile.ZipFile(args.src) as z:
                z.extractall(dest)
        d = dest
    else:
        d = args.src

    if not os.path.isdir(d):
        die(f"不是目录: {d}")
    convert(d, args.dry_run)
    log(f"完成 → {d}")
    ok(f"模型目录就绪: {d}")
    info("试一下:")
    log(f"    scripts/start_chat.sh -d {d}")
    log(f"    scripts/start_server.sh -d {d} --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
