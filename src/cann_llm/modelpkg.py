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

__all__ = ["build_package_files", "write_package_files", "is_packaged",
           "detect_layout", "PACKAGED_MARKERS", "main", "derive_model_name"]


def is_packaged(model_dir: str) -> bool:
    """目录里是否已有 ``api_config.json``。"""
    return os.path.isfile(os.path.join(model_dir, "api_config.json"))


#: 带 ``api_config.json`` 的完整包（官方导出的那套）
PACKAGED_MARKERS = ("api_config.json",)
def detect_layout(model_dir: str) -> str:
    """判断模型目录的布局。

    :return: ``"packaged"``（带 api_config.json 的完整包）/ ``"incomplete"``（缺它，
             用 :func:`write_package_files` 可补齐）/ ``"unknown"``
    """
    if not model_dir or not os.path.isdir(model_dir):
        return "unknown"
    names = set(os.listdir(model_dir))
    if any(m in names for m in PACKAGED_MARKERS):
        return "packaged"
    if "executor.json" in names:
        return "incomplete"
    return "unknown"




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
        # ★ initTokenLen 必须【严格小于】kv_cache_max_len —— 引擎的原话是
        #   "set init token len = 2048 error. it should smaller than kvCacheMaxLen 2048"
        #   （mask_and_pos_manager.cpp SetInitTokenLen），等于也会失败。
        #   官方 7B 是 kv=4096 / initTokenLen=2048，正好取一半。
        #   ⚠ 这里原来写的是 min(2048, kv)：kv=2048 时它给出 2048 —— 与 kv 相等，
        #     于是 --kv-len ≤ 2048 的模型【必然】加载失败（实测踩过）。
        "initTokenLen": max(1, min(2048, kv // 2)),
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
    # ★ 文件名必须与 **.omc 同名**（不是目录名）：引擎建 Executor 时自己按
    #   `modelPath` 去掉扩展名 + ".json" 去找这份配置
    #   （`InitOptionPacker::GetConfigFilePath`，反编译见
    #   docs/hiai-backend-handoff.md 最后一节）。目录名与 omc 名不一致的目录
    #   （例：`models/qwen25_coder_7b_omc1024/qwen7b.omc`）按目录名写就会找不到。
    omc = str(api.get("modelPath") or "")
    stem = os.path.splitext(omc)[0] if omc else ""
    return api, mj, f"{stem or model_name}.json"


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


def main(argv: "Optional[List[str]]" = None) -> int:
    """命令行入口：``python -m cann_llm.modelpkg <模型目录> [...]``

    为缺 ``api_config.json`` 的目录补齐打包配置（幂等；已存在则跳过，
    加 ``--overwrite`` 强制重写）。
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m cann_llm.modelpkg",
        description="给模型目录补齐打包配置（api_config.json + <model>.json）")
    ap.add_argument("dirs", nargs="+", help="模型目录")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重写")
    args = ap.parse_args(argv)

    rc = 0
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"✗ {d}: 不是目录")
            rc = 1
            continue
        if is_packaged(d) and not args.overwrite:
            print(f"· {d}: 已有 api_config.json，跳过（--overwrite 可强制重写）")
            continue
        try:
            written = write_package_files(d, overwrite=args.overwrite)
        except (OSError, ValueError) as e:
            print(f"✗ {d}: {e}")
            rc = 1
            continue
        print(f"✓ {d}: 已写 {', '.join(written) if written else '（无需改动）'}")
    return rc


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())

def derive_model_name(model_dir: str) -> str:
    """给模型目录起一个**像样的**展示名。

    优先级（都用得上的信息，不猜）::

        ① 目录名（先 realpath —— 否则 ``-d .`` 会得到 "."）
        ② <model>.json 的文件名去扩展名（官方包里有，如 qwen7b.json）
        ③ api_config.json 的 modelPath 去扩展名（如 qwen7b.omc）
        ④ "unknown"

    ★ 为什么不直接用 ``os.path.basename(model_dir)``：``-d .`` 或 ``-d ..`` 时
      它返回 "." / ".."，模型名就成了一个点，在 CLI / /v1/models 里都很难看。
    """
    d = os.path.abspath(model_dir or ".")
    name = os.path.basename(d)
    if name and name not in (".", "..", os.sep):
        return name
    # ② <model>.json：非派生的那份（executor/context 是我们生成的）
    try:
        names = sorted(os.listdir(d))
    except OSError:
        names = []
    for n in names:
        if n.endswith(".json") and n not in _DERIVED:
            stem = n[:-5]
            if stem:
                return stem
    # ③ modelPath 去扩展名
    api = _read_json(os.path.join(d, "api_config.json"))
    mp = str(api.get("modelPath") or "")
    if mp:
        stem = os.path.basename(mp)
        for suf in (".omc", ".json"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        if stem:
            return stem
    return "unknown"


#: 我们自己生成的两份配置，不算"模型自带的 json"
_DERIVED = ("executor.json", "context.json")
