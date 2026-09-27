"""把官方发布的 OMC 模型包（**扁平布局**）整成 cann-llm 的模型目录（**结构化布局**）。

背景：OpenHarmony 模型矩阵（matrix.openharmony.cn）发的 OMC 包，解压出来是::

    <name>.omc                     编译好的模型
    SubGraph_0.weight              量化权重
    <prefix>.embedding_weights     词表 embedding
    <prefix>.embedding_dequant_scale
    tokenizer.json
    <model>.json                   引擎侧 llm_config（**扁平**）
    api_config.json                引擎侧生成配置（采样/停止符/chat template）

而 cann 后端（``libcann_llm_engine.so`` 的 ``CreateFromExecutorJson``）只读::

    executor.json   结构化：version / engine_type / llm_config / tokenizer / autoregressive
    context.json    结构化：version / engine_type / generate_options / sampler

两者字段内容一一对应，只是外形不同。本模块做这个转换，供两处共用：

* ``launcher`` / ``launcher_server`` —— ``-b cann`` 且模型目录缺那两个文件时自动生成
  （只补缺的；已经有就完全不碰）
* ``scripts/import_omc_package.py`` —— 命令行显式导入

注意：这里**只生成 executor.json / context.json**，不碰 .omc / 权重 / embedding /
tokenizer.json。`cann.py` 里那套 ``.executor.live.json`` / ``.tokenizer.live.json``
是另一件事（补 <omc 同名>.json 的标量、规范化 merges），作为兜底保留。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["needs_import", "find_omc_files", "build_executor_json",
           "build_context_json", "import_omc_dir", "ensure_model_dir"]

#: 放进 executor.json 的 llm_config 的字段（与 scripts/import_omc_package.py 一致）
LLM_KEYS: Tuple[str, ...] = (
    "bos_token_id", "eos_token_id", "kv_cache_max_len", "sliding_window_len",
    "max_position_embeddings", "num_attention_kv_heads", "num_attention_head_dims",
    "num_hidden_layers", "num_attention_heads", "hidden_size", "intermediate_size",
    "vocab_size", "vocab_real_size", "prefill_len", "decode_len", "max_io_tokens",
    "use_output_pos", "embedding_weights", "embedding_dequant_scale",
    "embedding_input_type", "enable_dynamic_kv_cache", "enable_lm_head_opt",
    "enable_lm_head_topk", "is_kv_cache_merge", "tie_word_embeddings",
    "rms_norm_eps", "rope_theta", "model_type", "architectures", "torch_dtype",
)


def needs_import(model_dir: str) -> bool:
    """模型目录是否【缺】结构化配置（缺任一即需要导入）。"""
    return not (os.path.exists(os.path.join(model_dir, "executor.json"))
                and os.path.exists(os.path.join(model_dir, "context.json")))


def find_omc_files(d: str) -> Dict[str, Optional[str]]:
    """在目录里认出各个角色的文件。"""
    out: Dict[str, Optional[str]] = {"omc": None, "llm_json": None, "api_json": None,
                                     "emb_w": None, "emb_s": None,
                                     "tokenizer": None, "weight": None}
    for name in sorted(os.listdir(d)):
        # 跳过隐藏文件：cann.py 的 .executor.live.json / .context.live.json 等
        # 也是 *.json，会被误认成 <model>.json（实测踩过）。
        if name.startswith("."):
            continue
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(".omc"):
            out["omc"] = name
        elif name == "api_config.json":
            out["api_json"] = name
        elif low.endswith(".json") and out["llm_json"] is None:
            out["llm_json"] = name                      # <model>.json
        elif "embedding_weights" in low:
            out["emb_w"] = name
        elif "embedding_dequant_scale" in low:
            out["emb_s"] = name
        elif low == "tokenizer.json":
            out["tokenizer"] = name
        elif low == "subgraph_0.weight":
            out["weight"] = name
    return out


def build_executor_json(found: Dict[str, Optional[str]], d: str) -> Dict[str, Any]:
    """由 <model>.json（扁平）组装结构化 executor。"""
    llm = json.load(open(os.path.join(d, found["llm_json"]), encoding="utf-8"))
    cfg = {k: llm[k] for k in LLM_KEYS if k in llm}
    # embedding 文件名以磁盘上的实际文件为准（官方名字与 llm_config 里可能不一致）
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


def build_context_json(found: Dict[str, Optional[str]], d: str) -> Dict[str, Any]:
    """由 api_config.json 组装结构化 context。"""
    api = json.load(open(os.path.join(d, found["api_json"]), encoding="utf-8"))
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


def import_omc_dir(d: str, dry_run: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """把官方 OMC 包目录整成结构化模型目录，返回 (executor, context)。

    只写 ``executor.json`` / ``context.json``；缺 ``api_config.json`` 时顺带补一份。
    ``ValueError`` 表示目录里缺必需的文件（调用方决定是报错还是忽略）。
    """
    found = find_omc_files(d)
    missing = [k for k in ("omc", "llm_json", "api_json", "tokenizer") if not found[k]]
    if missing:
        raise ValueError(f"{d} 里缺少: {', '.join(missing)}"
                         f"（现有: {sorted(os.listdir(d))}）")

    ex = build_executor_json(found, d)
    cx = build_context_json(found, d)

    # ★ ① 把 <omc 同名>.json 的【标量】超参也并进 llm_config。
    #   cann 引擎的 CreateFromExecutorJson 只读 executor.json，不看 <omc 同名>.json —
    #   官方包恰恰把 hidden_size / num_hidden_layers / kv_cache_max_len 放在后者，
    #   不并的话 cann 那边还得靠 .executor.live.json 兜底（多生成一个文件、多一条警告）。
    #   只并标量：dict/list（architectures、rope_scaling 之类）引擎不认，实测列表会
    #   让它抛 nlohmann type_error。
    llm_json = found.get("llm_json")
    lc = ex["llm_config"]
    if llm_json:
        try:
            mj = json.load(open(os.path.join(d, llm_json), encoding="utf-8"))
            for key, val in mj.items():
                if key not in lc and not isinstance(val, (dict, list)):
                    lc[key] = val
        except (OSError, ValueError):
            pass

    # ★ ② merges 规范化：Qwen3 的 tokenizer.json 里 merges 是「数组的数组」，
    #   引擎对每个元素调 string ⇒ 抛 type_error 并 **abort**（SIGABRT）。
    #   不动用户的 tokenizer.json，另写一份规范化副本、让 executor 指过去。
    tok_name = found.get("tokenizer") or "tokenizer.json"
    try:
        tok = json.load(open(os.path.join(d, tok_name), encoding="utf-8"))
        merges = (tok.get("model") or {}).get("merges")
        if merges and isinstance(merges[0], list):
            tok["model"]["merges"] = [" ".join(m) for m in merges]
            tok_name = "tokenizer.cann.json"
            if not dry_run:
                with open(os.path.join(d, tok_name), "w", encoding="utf-8") as fh:
                    json.dump(tok, fh, ensure_ascii=False)
            ex["tokenizer"]["path"] = tok_name
    except (OSError, ValueError, TypeError, IndexError):
        pass

    if dry_run:
        return ex, cx

    with open(os.path.join(d, "executor.json"), "w", encoding="utf-8") as fh:
        json.dump(ex, fh, indent=4, ensure_ascii=False)
    with open(os.path.join(d, "context.json"), "w", encoding="utf-8") as fh:
        json.dump(cx, fh, indent=4, ensure_ascii=False)

    # 官方包自带 api_config.json / <model>.json，通常无需补写；缺了才用 executor 反推
    try:
        from .modelpkg import is_packaged, write_package_files
        if not is_packaged(d):
            write_package_files(d, ex)
    except Exception:                                   # noqa: BLE001
        pass
    return ex, cx


def ensure_model_dir(model_dir: str, backend: str, stream: Any = None) -> bool:
    """``-b cann`` 且模型目录缺结构化配置时，就地导入（只补缺的）。

    * 幂等：``executor.json`` / ``context.json`` 都在就什么都不做。
    * ``hiai`` 后端不需要这两个文件，直接返回。
    * 不像官方包（缺必需文件）时**不报错也不写**，留给后端照原样报它自己的错。

    返回是否真的导入了。
    """
    if backend != "cann" or not model_dir or not os.path.isdir(model_dir):
        return False
    if not needs_import(model_dir):
        return False
    try:
        import_omc_dir(model_dir)
    except ValueError:
        return False
    print(f"  · cann 后端需要结构化配置，已从官方包导入："
          f"{model_dir}/executor.json + context.json", file=stream or sys.stderr)
    return True
