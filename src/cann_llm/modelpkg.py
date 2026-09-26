"""把自装配的模型目录补成 **打包布局**（带 ``api_config.json`` + ``<model>.json``）。

为什么要它
----------
``executor.json`` / ``context.json`` 是**华为引擎自己的配置格式**（官方 OMC 包里也带着
它们），所以只有这两个文件的目录并不是"另一种官方不认的布局"—— 只是少了官方包里
**额外附带**的两个 JSON：

``api_config.json``
    运行参数：tokenizer 路径/类型、模型路径、采样默认值、``initTokenLen``、``stopSeq`` …

``<model>.json``
    模型自身的结构/超参（HF 风格），``build_configs()`` 把它当 ``llm_config`` 的基底。

本模块从既有 ``executor.json`` / ``context.json`` 反推出这两个文件，
让目录与官方包**同构**，从而两个后端、以及"打包布局优先"的逻辑都无需特殊分支。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

__all__ = ["build_package_files", "write_package_files", "is_packaged"]


def is_packaged(model_dir: str) -> bool:
    """目录里是否已有 ``api_config.json``。"""
    return os.path.isfile(os.path.join(model_dir, "api_config.json"))


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _gen_from_llm(llm: Dict[str, Any], model_dir: str,
                  model_name: str) -> Dict[str, Any]:
    """从 ``llm_config`` 造 ``api_config.json``。"""
    kv = int(llm.get("kv_cache_max_len") or 2048)
    tok = "tokenizer.json" if os.path.isfile(os.path.join(model_dir, "tokenizer.json")) \
        else (llm.get("tokenizerPath") or "tokenizer.json")
    omc = llm.get("model_path") or ""
    if not omc:
        cands = [n for n in sorted(os.listdir(model_dir)) if n.endswith(".omc")]
        omc = cands[0] if cands else f"{model_name}.omc"
    return {
        "inferType": int(llm.get("inferType") or 0),
        "tokenizerType": int(llm.get("tokenizerType") or 4),     # 4 = Qwen
        "tokenizerPath": tok,
        "modelType": int(llm.get("modelType") or 0),
        "modelPath": omc,
        "weightDir": llm.get("weightDir") or "./",
        "prefixPrompt": llm.get("prefixPrompt") or "",
        "pmtCacheOperation": llm.get("pmtCacheOperation") or "",
        "pfxInitTokenLen": int(llm.get("pfxInitTokenLen") or 6),
        "loraCfgPath": "",
        "expect": "",
        "callbackFreq": int(llm.get("callback_freq") or 2),
        "sampleFlag": True,
        "seed": 99,
        "topK": 20,
        "topP": 0.8,
        "temperature": 0.7,
        # ★ maxGenTokens 受 max_io_tokens 约束：initTokenLen + maxGenTokens 不能超过它
        #   （超了会让引擎内部的 MaskAndPosManager->SetInitTokenLen 失败 →
        #    Generate 报 FAIL）。官方 7B: 2048+5000=7048 ≤ 32768 ✓；
        #   自转 1.5B: max_io_tokens 只有 4096，取 5000 就会超。
        "maxGenTokens": max(1, min(5000,
                                   int(llm.get("max_io_tokens") or 32768)
                                   - min(2048, kv))),
        "repetitionPenalty": 1.1,
        # ★ initTokenLen ≠ kv_cache_max_len：官方 7B 包是 kv=4096 / initTokenLen=2048。
        #   填成 kv 会让引擎内部的 MaskAndPosManager->SetInitTokenLen 失败。
        "initTokenLen": min(2048, kv),
        "stopSeq": llm.get("stop_sequence") or ["<|im_end|>", "<|endoftext|>"],
    }


def _model_json_from_llm(llm: Dict[str, Any]) -> Dict[str, Any]:
    """从 ``llm_config`` 造 ``<model>.json``（HF 风格的模型结构/超参）。"""
    out: Dict[str, Any] = dict(llm)
    # ★ num_attention_heads 必须显式给出（官方 qwen7b.json 里有）—— 缺它 Executor 创建会失败。
    #   它可由 hidden_size / num_attention_head_dims 推出（官方 7B: 3584/128 = 28 ✓）。
    if not out.get("num_attention_heads"):
        hs = int(llm.get("hidden_size") or 0)
        hd = int(llm.get("num_attention_head_dims") or 0)
        if hs and hd:
            out["num_attention_heads"] = hs // hd
    mt = str(llm.get("model_type") or "qwen2")
    out.setdefault("model_type", mt)
    out.setdefault("architectures", ["Qwen2ForCausalLM"])
    out.setdefault("hidden_act", "silu")
    out.setdefault("rms_norm_eps", 1e-06)
    out.setdefault("rope_theta", 10000)
    out.setdefault("torch_dtype", "bfloat16")
    out.setdefault("initializer_range", 0.02)
    out.setdefault("attention_dropout", 0.0)
    out.setdefault("use_cache", True)
    out.setdefault("tie_word_embeddings", False)
    out.setdefault("pad_token_id", 0)
    out.setdefault("max_window_layers", llm.get("num_hidden_layers", 0))
    out.setdefault("transformers_version", "4.44.0")
    # ★ 不要在这里"猜"引擎的图优化开关（enable_dynamic_kv_cache / enable_lm_head_opt /
    #   enable_lm_head_topk / is_kv_cache_merge）：它们是**图编译期**决定的，
    #   官方包之所以有，是因为官方模型的图就是按这些优化编的。
    #   给自转模型硬加会与它的图不符 → Executor 创建失败（实测）。
    #   原始 llm_config 里有就原样带上（上面的 dict(llm) 已经带过来了）。
    return out


def build_package_files(model_dir: str,
                        executor: Optional[Dict[str, Any]] = None
                        ) -> "tuple[Dict[str, Any], Dict[str, Any], str]":
    """算出该写的 ``(api_config, model_json, model_json 文件名)``，不落盘。"""
    model_name = os.path.basename(os.path.abspath(model_dir)) or "model"
    if executor is None:
        ex = _read_json(os.path.join(model_dir, "executor.json"))
    else:
        ex = dict(executor)
    llm = dict(ex.get("llm_config") or {})
    if not llm:
        raise ValueError(f"{model_dir}/executor.json 里没有 llm_config，无法反推打包配置")
    api = _gen_from_llm(llm, model_dir, model_name)
    mj = _model_json_from_llm(llm)
    # 把推出来的字段回填进 executor.json 的 llm_config（build_configs 以它为基底）
    llm.setdefault("num_attention_heads", mj.get("num_attention_heads"))
    # 文件名取模型名（build_configs 会挑非派生的那份 .json）
    return api, mj, f"{model_name}.json"


def write_package_files(model_dir: str,
                        executor: Optional[Dict[str, Any]] = None,
                        overwrite: bool = False) -> List[str]:
    """把 ``api_config.json`` 与 ``<model>.json`` 写进目录，返回写出的文件名。"""
    api, mj, mj_name = build_package_files(model_dir, executor)
    written: List[str] = []
    for name, data in (("api_config.json", api), (mj_name, mj)):
        path = os.path.join(model_dir, name)
        if os.path.exists(path) and not overwrite:
            continue
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
            fh.write("\n")
        written.append(name)
    return written
