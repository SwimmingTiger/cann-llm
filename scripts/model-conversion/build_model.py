#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条命令从 HF 检查点产出可上 NPU 的模型目录，**上下文长度是命令行参数**。

背景：KV 缓存长度是**编译期**属性 —— 它同时写死在三个地方，必须一致：

  1. 导出 yaml 的 ``kv_cache_max_len``   -> ONNX 里 ``past_key_in*`` 占位符的形状
  2. OMG 的 ``--input_shape``            -> 编译进 .omc 的图输入维度
  3. ``executor.json`` 的 ``kv_cache_max_len`` -> 运行时引擎照着分配

所以「上下文能开多大」不是运行时开关，而是要**重新走一遍导出 + OMG**。
这个脚本就是把这条链路收敛成一个 ``--kv-len`` 参数，免得手工改三处还容易漏。

用法（在转换机上，需能 import torch/onnx/transformers）：

    python3 build_model.py \\
        --hf-model   /path/to/Qwen3-4B-Instruct-2507 \\
        --quant-pth  /path/to/fake_quant_weight.pth \\
        --dopt-config /path/to/dopt_config.json \\
        --export-dir /path/to/npu_tuned_export \\
        --omg-dir    /path/to/tools_omg \\
        --asc-dir    /path/to/tools_ascendc \\
        --name       qwen3_4b \\
        --workdir    /path/to/work/quant4b \\
        --kv-len     8192

常用的分步开关：``--only-yaml``（只生成 yaml 看一眼）、``--skip-export``（复用已有
ONNX）、``--skip-omg``、``--dry-run``。

注意：KV 越大，解码时每 token 要读的 KV 张量也越大（图里是静态形状，与实际用了
多少上下文无关）。4B 大约 144 KB/token：8K≈1.1 GB、32K≈4.5 GB，同时解码变慢。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

#: HF 的 model_type -> 官方导出脚本与 npu_tuned_model 里的架构名
ARCH_BY_MODEL_TYPE = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "glm": "glm",
    "chatglm": "glm",
}

YAML_TMPL = """embedding_config:
  embedding_separate: True
  embedding_as_fp16: False
  mul_twice: False
no_gemm: True
mock_as_s16: False
output_pos:
model_arch: {arch}
hf_model_path: {hf_model}
config_file: {dopt_config}
quant_pth: {quant_pth}
run_mode: hardware
dump_path:
output_dir: {onnx_out}
lora:
  enable: False
  megatron: False
  lora_scale: 4
  lora_config:
  lora_ckpt:
lm_head:
onnx_output_model_name: {name}
onnx_opset: 12
batch: 1
kv_cache_max_len: {kv_len}
layers: {layers}
seq_len:
  - {seq_len}
"""

CONTEXT_JSON = {
    "version": 1,
    "engine_type": "autoregressive",
    "generate_options": {
        "callback_freq": 1,
        "max_gen_tokens": 128,
        "stop_sequence": ["<|im_end|>"],
        "init_token_len": 0,
    },
    "sampler": {
        "do_sample": True,
        "seed": 99,
        "top-k": 20,
        "top-p": 0.95,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
    },
}


def log(msg):
    print(f"  {msg}", flush=True)


def die(msg):
    print(f"错误：{msg}", file=sys.stderr)
    raise SystemExit(1)


def read_hf_config(path):
    p = os.path.join(path, "config.json")
    if not os.path.exists(p):
        die(f"找不到 {p} —— --hf-model 要指向 HF 检查点目录")
    with open(p) as f:
        return json.load(f)


def detect_arch(hf_cfg):
    mt = (hf_cfg.get("model_type") or "").lower()
    if mt not in ARCH_BY_MODEL_TYPE:
        die(f"不支持的 model_type={mt!r}（支持：{', '.join(ARCH_BY_MODEL_TYPE)}）")
    return ARCH_BY_MODEL_TYPE[mt]


def run(cmd, cwd, env=None):
    log("执行: " + " ".join(cmd[:3]) + (" …" if len(cmd) > 3 else ""))
    rc = subprocess.call(cmd, cwd=cwd, env=env)
    if rc != 0:
        die(f"命令返回 {rc}")


def main():
    ap = argparse.ArgumentParser(description="按指定 KV 长度构建模型目录")
    ap.add_argument("--hf-model", required=True, help="HF 检查点目录")
    ap.add_argument("--quant-pth", required=True, help="dopt 量化产物 fake_quant_weight.pth")
    ap.add_argument("--dopt-config", required=True, help="dopt_config.json")
    ap.add_argument("--export-dir", required=True, help="npu_tuned_export 目录")
    ap.add_argument("--omg-dir", required=True, help="tools_omg 目录")
    ap.add_argument("--asc-dir", required=True, help="tools_ascendc 目录")
    ap.add_argument("--name", required=True, help="模型名（omc / embedding / 目录都用它）")
    ap.add_argument("--workdir", required=True, help="工作目录（放 onnx / omc 中间产物）")
    ap.add_argument("--out-dir", help="最终模型目录（默认 <workdir>/model_<name>_<kvlen>）")

    # ★ 本次的核心参数
    ap.add_argument("--kv-len", type=int, default=2048,
                    help="KV 缓存长度（编译期属性，默认 2048）")
    ap.add_argument("--seq-len", type=int, default=64, help="prefill 每轮喂的 token 数")
    ap.add_argument("--platform", default="kirinx90")
    ap.add_argument("--weight-data-type", default="FP16",
                    help="OMG 的权重要求（官方流程是 FP16）")

    # 形状：默认从 HF config 推
    ap.add_argument("--arch")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--vocab-size", type=int)

    ap.add_argument("--skip-export", action="store_true", help="复用已有 ONNX")
    ap.add_argument("--skip-omg", action="store_true", help="不跑 OMG")
    ap.add_argument("--only-yaml", action="store_true", help="只写 yaml 后退出")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    ap.add_argument("--python", default=sys.executable, help="跑导出脚本用的解释器")
    args = ap.parse_args()

    hf = os.path.abspath(args.hf_model)
    work = os.path.abspath(args.workdir)
    cfg = read_hf_config(hf)

    arch = args.arch or detect_arch(cfg)
    layers = args.layers or cfg.get("num_hidden_layers")
    hidden = args.hidden or cfg.get("hidden_size")
    kv_heads = args.kv_heads or cfg.get("num_key_value_heads")
    head_dim = args.head_dim or cfg.get("head_dim") or 128
    vocab = args.vocab_size or cfg.get("vocab_size")
    for label, v in (("layers", layers), ("hidden", hidden),
                     ("kv_heads", kv_heads), ("vocab_size", vocab)):
        if not v:
            die(f"无法确定 {label}，请用 --{label.replace('_','-')} 显式指定")

    bos = cfg.get("bos_token_id", 151643)
    eos = cfg.get("eos_token_id", 151645)
    if isinstance(eos, list):
        eos = eos[0]

    onnx_out = os.path.join(work, "onnx_out")
    om_out = os.path.join(work, "om_out")
    out_dir = os.path.abspath(args.out_dir
                              or os.path.join(work, f"model_{args.name}_{args.kv_len}"))

    log(f"架构 {arch}  ·  {layers} 层  ·  hidden {hidden}  ·  {kv_heads} 个 KV 头 "
        f"·  head_dim {head_dim}  ·  vocab {vocab}")
    log(f"★ KV 缓存长度 = {args.kv_len}（每 token 约 "
        f"{2 * layers * kv_heads * head_dim * 2 / 1024:.0f} KB，"
        f"合计约 {2 * layers * kv_heads * head_dim * 2 * args.kv_len / 1024**3:.2f} GB）")
    log(f"输出模型目录 {out_dir}")

    if args.dry_run:
        log("(--dry-run，未做任何改动)")
        return 0

    os.makedirs(work, exist_ok=True)

    # ---- 1) 写导出 yaml ----
    yaml_path = os.path.join(work, f"model_info_{args.name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(YAML_TMPL.format(
            arch=arch, hf_model=hf,
            dopt_config=os.path.abspath(args.dopt_config),
            quant_pth=os.path.abspath(args.quant_pth),
            onnx_out=onnx_out, name=args.name,
            kv_len=args.kv_len, layers=layers, seq_len=args.seq_len))
    log(f"已写 {yaml_path}")

    if args.only_yaml:
        return 0

    # ---- 2) 导出 ONNX ----
    export_script = os.path.join(args.export_dir, f"export_model_single_{arch}.py")
    onnx_dir = None
    if not args.skip_export:
        if not os.path.exists(export_script):
            die(f"找不到导出脚本 {export_script}")

        # ★ Qwen3 的 fp32 导出里，onnx_utils.process_onnx 的 onnxsim.simplify 是内存
        #   峰值所在（16 GB 模型 + 它的工作副本 > 31 GB，实测被 OOM killer 杀，
        #   日志里只看得到一句 "命令返回 -9"）。仓库的 patch_qwen3_export_mem.py
        #   把这一段做成受 CANN_SKIP_ONNX_SIMPLIFY 控制 —— 所以：
        #     · 官方脚本必须先打过那个补丁，否则下面的变量没有任何作用；
        #     · 默认给 qwen3 设上（不设就是在 31 GB 机器上必然失败）。
        #   要保留 simplify（内存更大的机器）就自己设 CANN_SKIP_ONNX_SIMPLIFY=0。
        export_env = dict(os.environ)
        if arch == "qwen3":
            src = open(export_script, encoding="utf-8").read()
            if "CANN_SKIP_ONNX_SIMPLIFY" not in src:
                die(f"{os.path.basename(export_script)} 还没打过内存补丁 —— 先运行：\n"
                    f"      {args.python} {os.path.join(os.path.dirname(__file__), 'patch_qwen3_export_mem.py')} {args.export_dir}")
            if export_env.get("CANN_SKIP_ONNX_SIMPLIFY") is None:
                export_env["CANN_SKIP_ONNX_SIMPLIFY"] = "1"
                print("  [build_model] 设 CANN_SKIP_ONNX_SIMPLIFY=1（跳过 onnxsim，避开内存峰值）")

        if os.path.isdir(onnx_out):
            shutil.rmtree(onnx_out)
        run([args.python, export_script, yaml_path], cwd=args.export_dir, env=export_env)
        # 导出会给目录加后缀，按名字找
        for cand in sorted(os.listdir(work)):
            if cand.startswith("onnx_out"):
                onnx_dir = os.path.join(work, cand)
                break
    else:
        for cand in sorted(os.listdir(work)):
            if cand.startswith("onnx_out"):
                onnx_dir = os.path.join(work, cand)
        if not onnx_dir:
            die("--skip-export 但工作目录下没有 onnx_out* 目录")
    if not onnx_dir:
        die("导出后没找到 onnx_out* 目录")
    log(f"ONNX 目录 {onnx_dir}")

    onnx_file = os.path.join(onnx_dir, f"{args.name}.onnx")
    if not os.path.exists(onnx_file):
        found = [x for x in os.listdir(onnx_dir) if x.endswith(".onnx")]
        if not found:
            die(f"{onnx_dir} 里没有 .onnx")
        onnx_file = os.path.join(onnx_dir, found[0])

    # ---- 3) OMG ----
    omc_dir = os.path.join(om_out, args.name)
    if not args.skip_omg:
        omg_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "omg_convert.py")
        # OMG 对环境很挑：缺 PATH(bisheng/package) 或没 source set_ascendc_env.sh
        # 会以 "RmsNorm ... infershape func failed" 的形式失败（看着像算子问题，
        # 其实是环境不全）。这里完整复刻实测可用的那套环境。
        env = dict(os.environ)
        lib = os.path.join(args.omg_dir, "master", "lib64")
        tools_root = os.path.dirname(args.omg_dir)
        plat = os.path.join(tools_root, "platform", args.platform)
        plat_lib = os.path.join(plat, "lib64")

        # 解释器自带的 lib（glibc/libstdc++），从 base_prefix 推，避免写死路径
        try:
            base = subprocess.check_output(
                [args.python, "-c", "import sys; print(sys.base_prefix)"],
                text=True).strip()
            py_lib = os.path.join(base, "lib")
        except Exception:                                   # noqa: BLE001
            py_lib = ""

        ld = [lib, plat_lib] + ([py_lib] if py_lib and os.path.isdir(py_lib) else [])
        env["LD_LIBRARY_PATH"] = ":".join(ld + [env.get("LD_LIBRARY_PATH", "")]).strip(":")
        env["SOC_VERSION"] = args.platform
        env["PYTHONPATH"] = os.path.join(plat, "ops", "impl") + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PATH"] = ":".join(filter(None, [
            env.get("PATH", ""),
            os.path.join(args.asc_dir, "bisheng", "bin"),
            os.path.join(args.asc_dir, "package"),
        ]))
        env.setdefault("TMPDIR", os.path.join(work, "tmp"))
        os.makedirs(env["TMPDIR"], exist_ok=True)
        os.makedirs(om_out, exist_ok=True)

        omg_cmd = [args.python, omg_script,
                   "--onnx", onnx_file, "--out", omc_dir,
                   "--layers", str(layers), "--kv-len", str(args.kv_len),
                   "--hidden", str(hidden), "--kv-heads", str(kv_heads),
                   "--head-dim", str(head_dim), "--platform", args.platform,
                   "--weight-data-type", args.weight_data_type,
                   "--omg-dir", args.omg_dir, "--asc-dir", args.asc_dir]

        # 通过 bash 跑，好 source set_ascendc_env.sh（Python 里 source 不了）
        setup = os.path.join(args.asc_dir, "set_ascendc_env.sh")
        if os.path.exists(setup):
            log(f"source {setup}")
            wrapped = ["bash", "-c",
                       f'source "{setup}" >/dev/null 2>&1 || true; exec "$@"', "--"]
            run(wrapped + omg_cmd, cwd=args.omg_dir, env=env)
        else:
            log("警告：没找到 set_ascendc_env.sh，直接跑（可能因环境不全失败）")
            run(omg_cmd, cwd=args.omg_dir, env=env)

    # ---- 4) 装配模型目录 ----
    os.makedirs(out_dir, exist_ok=True)
    omc = os.path.join(omc_dir, f"{args.name}.omc")
    if os.path.exists(omc):
        shutil.copy2(omc, out_dir)
        log(f"复制 {os.path.basename(omc)}")
    w = os.path.join(omc_dir, "SubGraph_0.weight")
    if os.path.exists(w):
        shutil.copy2(w, out_dir)
        log("复制 SubGraph_0.weight")

    emb_prefix = f"{args.name}_{args.seq_len}_{args.kv_len}"
    for suf in (".embedding_weights", ".embedding_dequant_scale"):
        src = os.path.join(onnx_dir, emb_prefix + suf)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
        else:
            log(f"警告：没找到 {os.path.basename(src)}")

    tok_src = os.path.join(hf, "tokenizer.json")
    tok_dst = os.path.join(out_dir, "tokenizer.json")
    shutil.copy2(tok_src, tok_dst)
    log("复制 tokenizer.json")

    # 新版 tokenizer 的 merges 是「数组的数组」，引擎会 core dump，必须规范化
    norm = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "normalize_tokenizer_merges.py")
    if os.path.exists(norm):
        subprocess.call([args.python, norm, tok_dst])
    else:
        log("警告：没找到 normalize_tokenizer_merges.py，请手工确认 merges 格式")

    with open(os.path.join(out_dir, "context.json"), "w") as f:
        json.dump(CONTEXT_JSON, f, indent=4, ensure_ascii=False)

    executor = {
        "version": 1,
        "engine_type": "autoregressive",
        "llm_config": {
            "bos_token_id": bos,
            "eos_token_id": eos,
            "kv_cache_max_len": args.kv_len,          # ★ 与图和 yaml 一致
            "sliding_window_len": 0,
            "max_position_embeddings": cfg.get("max_position_embeddings", 32768),
            "num_attention_kv_heads": kv_heads,
            "num_attention_head_dims": head_dim,
            "num_hidden_layers": layers,
            "prefill_len": args.seq_len,
            "decode_len": 1,
            "vocab_size": vocab,
            "vocab_real_size": vocab,
            "use_output_pos": False,
            "max_io_tokens": 4096,
            "hidden_size": hidden,
            "embedding_weights": f"{emb_prefix}.embedding_weights",
            "embedding_dequant_scale": f"{emb_prefix}.embedding_dequant_scale",
            "embedding_input_type": "int8",
            # ★ 下面这些此前没写，而官方包（对照 models/qwen25_coder_7b_omc1024 的
            #   executor.json）里都有 —— 缺了它们引擎加载会失败：
            #       model_manager_ndk_impl.cpp PrepareModelManager(122):
            #           "executor_" "null, return FAIL."
            #       engine_executor_impl.cpp Init(153): "init model fail or load toke..."
            #   数值一律取自 HF config.json，不要自己编。
            "pad_token_id": cfg.get("pad_token_id") or 0,
            "num_attention_heads": cfg.get("num_attention_heads"),
            "intermediate_size": cfg.get("intermediate_size"),
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-06),
            "rope_theta": cfg.get("rope_theta", 10000),
            "model_type": cfg.get("model_type", arch),
            "architectures": cfg.get("architectures"),
            "torch_dtype": cfg.get("torch_dtype"),
            "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
            "enable_dynamic_kv_cache": True,
            "enable_lm_head_opt": True,
            "enable_lm_head_topk": True,
            "is_kv_cache_merge": True,
        },
        "tokenizer": {"type": "qwen", "path": "tokenizer.json"},
        "autoregressive": {"model_path": f"{args.name}.omc", "weight_path": "./"},
    }
    with open(os.path.join(out_dir, "executor.json"), "w") as f:
        json.dump(executor, f, indent=4, ensure_ascii=False)
    log("已写 context.json / executor.json")

    # ★ 再补出 api_config.json 与 <model>.json，使目录与官方包【同构】——
    #   这样 hiai / cann 两个后端、以及"打包布局优先"的逻辑都不需要特殊分支，
    #   也就不会产出所谓 bare 模型。
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "src"))
        from cann_llm.modelpkg import write_package_files
        written = write_package_files(out_dir, executor)
        if written:
            log("已写打包配置: " + ", ".join(written))
    except Exception as e:                      # noqa: BLE001
        log(f"警告：未能补写 api_config.json / <model>.json（{e}）")
        log("      这两个文件引擎要用（采样参数 / 停止符 / chat template），"
            "且 <model>.json 必须与 .omc 同名。")
        log("      转换机上一般没有仓库的 src/，所以在【设备】上补一句即可：")
        log(f"          PYTHONPATH=src python3 -m cann_llm.modelpkg {out_dir}")

    log(f"完成 → {out_dir}")
    log("三处 KV 长度已对齐：" 
        f"yaml {args.kv_len} · OMG input_shape {args.kv_len} · executor.json {args.kv_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
