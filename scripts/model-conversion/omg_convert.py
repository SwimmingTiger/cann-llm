#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成并执行华为 OMG 的转换命令（ONNX -> .omc）。

为什么要有这个脚本：官方示例里的 OMG 命令行把 28 层 / 2048 上下文 / 1536 隐藏维
全部写死成超长的 `--input_shape=...` 字符串，换成别的模型就没法用。这里按
「层数 + KV 长度 + 隐藏维」自动拼出来。

用法：
    # 只打印命令，不执行（先看看对不对）
    python3 omg_convert.py --onnx model.onnx --out ./om_out --dry-run

    # 实际执行（需要 OMG 工具与环境，见 docs/model-conversion.md）
    python3 omg_convert.py --onnx model.onnx --out ./om_out

    # 无压缩配置（rebuild_weights 走 FP16 路线时用这个）
    python3 omg_convert.py --onnx model.onnx --out ./om_out --weight-data-type FP16

环境变量：
    OMG_DIR   OMG 工具目录（含 master/omg 或 omg），默认 /path/to/ddk/tools/tools_omg
    ASC_DIR   AscendC 工具目录，默认 /path/to/ddk/tools/tools_ascendc
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import List


def build_input_shape(nlayers: int, kv_len: int, hidden: int,
                      kv_heads: int = 2, head_dim: int = 128) -> str:
    parts = [
        f"input_embed:1,-1,{hidden}",
        f"attention_mask:1,1,-1,{kv_len}",
        "position_ids:1,-1",
    ]
    for i in range(nlayers):
        parts.append(f"past_key_in{i}:{kv_len},{kv_heads},1,{head_dim}")
        parts.append(f"past_value_in{i}:{kv_len},{kv_heads},1,{head_dim}")
    parts += ["new_kv_cache_pos:-1", "embed_scales:1,-1,1"]
    return ";".join(parts)


#: ONNX 的 elem_type -> OMG 的 --input_type 名称
_ONNX2OMG = {1: "FP32", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64", 10: "FP16"}


def input_type_str(onnx_path: str, layers: int) -> str:
    """按 ONNX 里每个输入的【真实】dtype 生成 --input_type。

    实测教训：只声明 past_key_in*/past_value_in*:FP16 是不够的 —— 图里
      input_embed 是 INT8（embedding 分离后量化过）、position_ids 是 INT32、
      attention_mask 是 FP16，声明与真实不符时 OMG 的 onnx_parser 直接失败：
          E onnx_parser.cpp InsertPermuteNode(348):
              "!inputNodes.empty() || !outputNodes.empty()" "false, return FAIL"
          E general_model_compiler.cpp BeforeCompile(145): "check ir model compatibility failed"

    读不到 ONNX（没装 onnx 库等）时回退到"只有 past_* 是 FP16"的老行为。
    """
    try:
        import onnx                                       # noqa: PLC0415
        m = onnx.load(onnx_path, load_external_data=False)
        parts = []
        for v in m.graph.input:
            t = v.type.tensor_type.elem_type
            name = _ONNX2OMG.get(t)
            if name:
                parts.append(f"{v.name}:{name}")
        if parts:
            return ";".join(parts)
    except Exception:                                     # noqa: BLE001
        pass
    return build_past_type(layers, "FP16")


def build_past_type(nlayers: int, dtype: str) -> str:
    parts: List[str] = []
    for i in range(nlayers):
        parts.append(f"past_key_in{i}:{dtype}")
        parts.append(f"past_value_in{i}:{dtype}")
    return ";".join(parts)


def build_output_type(nlayers: int, dtype: str) -> str:
    parts = ["lm_logits:FP32"]
    for i in range(nlayers):
        parts.append(f"past_key{i}:{dtype}")
        parts.append(f"past_value{i}:{dtype}")
    return ";".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/执行 OMG 转换命令")
    ap.add_argument("--onnx", required=True, help="输入 ONNX 路径")
    ap.add_argument("--out", required=True, help="输出前缀（会生成 <out>.omc 等）")
    ap.add_argument("--layers", type=int, default=28, help="Transformer 层数（默认 28）")
    ap.add_argument("--kv-len", type=int, default=2048, help="KV 缓存长度（默认 2048）")
    ap.add_argument("--hidden", type=int, default=1536, help="hidden size（默认 1536）")
    ap.add_argument("--kv-heads", type=int, default=2,
                    help="KV 头数 num_key_value_heads（1.5B=2, 7B=4, Qwen3-8B=8）")
    ap.add_argument("--head-dim", type=int, default=128, help="head 维度（默认 128）")
    ap.add_argument("--platform", default="kirinx90", help="平台（默认 kirinx90）")
    ap.add_argument("--compress-conf", default=None,
                    help="dopt 量化参数文件；给了就用量化，不给就走 FP16")
    ap.add_argument("--weight-data-type", default=None,
                    help="例如 FP16（与 --compress-conf 二选一，FP16 不量化）")
    ap.add_argument("--omg-dir", default=os.environ.get("OMG_DIR", "/path/to/ddk/tools/tools_omg"))
    ap.add_argument("--omg-bin", default=None,
                    help="显式指定 OMG 可执行文件（默认用 <omg-dir>/omg 包装脚本）")
    ap.add_argument("--asc-dir", default=os.environ.get("ASC_DIR", "/path/to/ddk/tools/tools_ascendc"))
    ap.add_argument("--dynamic-dims", default="1,1,1,1,1;64,64,64,64,64",
                    help="prefill / decode 两个动态档位")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    # ★ 必须走 OMG 的【包装脚本】而不是 master/omg 二进制。实测：直接跑 master/omg 时
    #   算子插件注册不上（日志：E plugin.cc RegisterLibrary(58)::"PlugIn library
    #   :libai_npucore_ascendc.so Initialize failed"、W "dlopen so failed:
    #   libai_npucore_itf.so"、"Skip InferShapeOptimize"），产出的 omc 引擎加载时报
    #       E AI_INFRA model_manager_ndk_impl.cpp PrepareModelManager(122):
    #           "executor_" "null, return FAIL."
    #   而交叉实验证明同一个 SubGraph_0.weight 配当年的 omc 就能加载 —— 问题在 omc。
    #   包装脚本会选 HIAI_VERSION、准备 PATH/LD_LIBRARY_PATH，并用 glibc loader 的
    #   --library-path 启动真正的二进制（"host glibc >= 2.35 → use the HOST loader"）。
    #   用 --omg-bin 可以显式覆盖。
    omg_bin = args.omg_bin if getattr(args, "omg_bin", None) else os.path.join(args.omg_dir, "omg")
    if not os.path.exists(omg_bin):
        alt = os.path.join(args.omg_dir, "master", "omg")
        omg_bin = alt if os.path.exists(alt) else omg_bin

    cmd = [
        omg_bin,
        "--model", os.path.abspath(args.onnx),
        "--framework", "5",
        "--output", os.path.abspath(args.out),
        f"--input_shape={build_input_shape(args.layers, args.kv_len, args.hidden, args.kv_heads, args.head_dim)}",
        f"--dynamic_dims={args.dynamic_dims}",
        f"--input_type={input_type_str(args.onnx, args.layers)}",
        f"--output_type={build_output_type(args.layers, 'FP16')}",
    ]
    if args.compress_conf:
        cmd += ["--compress_conf", os.path.abspath(args.compress_conf)]
    # 参数顺序对齐 x570 上当年能跑的那条命令（quant/to_omc_rebuilt.sh）：
    #     … --output_type=… --weight_data_type FP16 --save_weights_as_external_data=true --platform=… --target=omc
    # ⚠ 实测：把 --weight_data_type 放到末尾【产物大小不变】（仍 5.8 G），
    #   所以顺序【不是】那个加载失败的原因 —— 这里只是保持与历史命令一致，别读成修复。
    #   真正的原因在输入 ONNX 的权重精度：当年那份是 fp16 量级（产物 3.1 G），
    #   我们这次是 fp32（产物 5.8 G）。
    if args.weight_data_type:
        cmd += ["--weight_data_type", args.weight_data_type]
    cmd += ["--save_weights_as_external_data=true",
            f"--platform={args.platform}",
            "--target=omc"]
    if not (args.compress_conf or args.weight_data_type):
        print("提示：既没给 --compress-conf 也没给 --weight-data-type，"
              "默认按不量化（FP32）走。若要 FP16 请显式加 --weight-data-type FP16。",
              file=sys.stderr)
    env = dict(os.environ)
    env.setdefault("SOC_VERSION", args.platform)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(args.omg_dir, "..", "platform", args.platform, "ops", "impl"),
         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # ★ LD_LIBRARY_PATH 必须包含 tools_omg/master/lib64 与 platform/<plat>/lib64：
    #   否则 OMG 的算子库 dlopen 失败，日志里是
    #       W ops_kernel_store_manager.cpp DlopenComputeLibrary(41):
    #         "dlopen so failed: libai_npucore_itf.so: cannot open shared object file"
    #       I model_optimizer.cpp Optimize(251)::"Skip InferShapeOptimize"
    #   RmsNorm 等算子拿不到 infershape 函数，生成的 omc 引擎【加载不了】：
    #       E AI_INFRA model_manager_ndk_impl.cpp PrepareModelManager(122):
    #           "executor_" "null, return FAIL."
    #   实测（交叉实验）：同一个 SubGraph_0.weight 配上当年生成的 omc 就能加载 ——
    #   也就是问题出在这个 omc 上。当年能跑的 to_omc_rebuilt.sh 里正是这么设的。
    # ⚠ 路径必须【规范化】：写成 ".../tools_omg/../platform/kirinx90/lib64" 这种带 ".."
    #   的字面量时，动态加载器找不到 libai_npucore_ascendc.so（实测报
    #   "PlugIn library :libai_npucore_ascendc.so Initialize failed"），
    #   于是算子插件注册不上、OMG 产出引擎不认的 omc。
    lib_dirs = [os.path.abspath(os.path.join(args.omg_dir, "master", "lib64")),
                os.path.abspath(os.path.join(args.omg_dir, os.pardir, "platform",
                                             args.platform, "lib64"))]
    # ★ TBE / te_fusion 是 python 引擎：LD_LIBRARY_PATH 里必须有 py3.10 的 libpython，
    #   否则 TbeInitialize 失败 —— 日志是
    #       E/TE_FUSION fusion_api.cc TbeInitialize(265)::"failed to initialize tbe."
    #       E/GENE generated_adaptee.cc InitializeTeFusion(174)::"TbeInitialize failed"
    #       E/AI_FMK general_model_compiler.cpp BeforeCompile(148)::"check ir model compatibility failed"
    #   当年 to_omc_rebuilt.sh 里正是这一行（uv 装的 cpython 3.10 lib），照抄：
    #       export LD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.10.21-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH
    import glob as _glob
    for pat in ("~/.local/share/uv/python/cpython-3.10*-linux-x86_64-gnu/lib",
                "~/.local/share/uv/python/cpython-3.10*/lib"):
        for d in sorted(_glob.glob(os.path.expanduser(pat))):
            if os.path.isdir(d):
                lib_dirs.append(d)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        lib_dirs + [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p])
    # 官方 set_ascendc_env.sh 还会把 ascendc 的 package / bisheng/bin 放进 PATH ——
    # 少了它 libcustom_op.so / te_fusion 之类也加载不了。
    asc = getattr(args, "asc_dir", None)
    if asc:
        extra = [os.path.join(asc, "package"), os.path.join(asc, "bisheng", "bin")]
        env["PATH"] = os.pathsep.join(
            [d for d in extra if os.path.isdir(d)] +
            [p for p in env.get("PATH", "").split(os.pathsep) if p])
        env.setdefault("TMPDIR", os.path.join(asc, "tmp"))
    # set_ascendc_env.sh 里会 `unset LD_LIBRARY_PATH` 再自己设一份，所以把我们算好的
    # 那份通过环境变量带进去，在 source 之后重新接上（当年脚本就是 source 完再 export 的）。
    env["CANN_LLM_OMG_LD"] = env["LD_LIBRARY_PATH"]
    asc_dir = getattr(args, "asc_dir", None)
    if asc_dir:
        del args.asc_dir

    print("### OMG 命令")
    print(" ".join(shlex.quote(c) for c in cmd))
    print("### 环境")
    print(f"  SOC_VERSION={env['SOC_VERSION']}")
    print(f"  PYTHONPATH={env['PYTHONPATH']}")
    print(f"  LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
    if args.dry_run:
        return 0

    if not os.path.exists(omg_bin):
        print(f"错误：找不到 OMG 可执行文件 {omg_bin}\n"
              f"      用 --omg-dir 指定，或设环境变量 OMG_DIR。", file=sys.stderr)
        return 2
    # ★ 官方 DDK 解压出来的 omg 是包装脚本，通常【没有执行权限】（文档 §1 提过这一点，
    #   但此前只能靠人记住）。不检查的话只会得到一个
    #   PermissionError: [Errno 13] Permission denied，看不出该怎么处理。
    if not os.access(omg_bin, os.X_OK):
        print(f"错误：{omg_bin} 没有执行权限（DDK 解压后常见）。\n"
              f"      先执行：chmod +x {omg_bin}\n"
              f"      （tools_omg/ 下同级的 omg 往往也是同样情况，可一并 chmod）",
              file=sys.stderr)
        return 2
    # ★ 必须先 source <ascendc>/set_ascendc_env.sh 再跑 OMG（当年 to_omc_rebuilt.sh 就是这么做的）。
    #   它把 <ascendc>/package/python 加进 PYTHONPATH（te_fusion / TBE 的 python 包在这里）、
    #   把 <ascendc>/package 与 ddk/ccec_compiler/bin 加进 PATH、并设 HIAI_VERSION。
    #   不做这一步的后果（实测）：
    #       E/GENE generated_adaptee.cc InitializeTeFusion(175)::"TbeInitialize failed"
    #       E/AI_NPUCL plugin.cc RegisterLibrary(58)::"libai_npucore_ascendc.so Initialize failed"
    #       E/AI_FMK general_model_compiler.cpp BeforeCompile(148)::"check ir model compatibility failed"
    #   —— 因为 set_ascendc_env.sh 会先 unset LD_LIBRARY_PATH，所以 source 完要把我们
    #   算好的那份接回去（用 CANN_LLM_OMG_LD 传进去）。
    # ★ 只有【老布局】的 DDK 才需要 source set_ascendc_env.sh —— 判据是
    #   <ascendc>/package/python 是否存在：老布局的 TBE 是 python 版（tbe/te/te_fusion
    #   装在那里），必须靠它设 PYTHONPATH/PATH；新布局的 TBE 是 C++ 库
    #   （libai_npucore_tefusion.so 在 tools/platform/<plat>/lib64/），source 反而坏事：
    #   它会把 PYTHONPATH 指向不存在的目录，并 unset LD_LIBRARY_PATH。
    #   实测（Qwen2.5-1.5B，DDK 6.1.1.0）：
    #       source 它  ⇒ TbeInitialize failed / libai_npucore_ascendc.so Initialize failed
    #       不 source ⇒ OMG generate offline model success ✓（产物 SubGraph_0.weight 2.9G）
    # ★ 不要 source <ascendc>/set_ascendc_env.sh：那是老布局（python 版 TBE，靠
    #   <ascendc>/package/python 里的 te_fusion）用的。当前 DDK 的 TBE 是 C++ 库
    #   （libai_npucore_tefusion.so 在 tools/platform/<plat>/lib64/），source 它会把
    #   PYTHONPATH 指向不存在的目录、并改坏 PATH —— 实测会直接让 OMG 找不到（退出码 127）。
    #   OMG 自带的包装脚本（<omg-dir>/omg）会自己准备 HIAI_VERSION / 库路径 / loader。
    # 仅保留：把我们算好的 LD_LIBRARY_PATH 传下去（包装脚本会在此基础上追加）。
    print("### 执行…", flush=True)
    rc = subprocess.call(cmd, cwd=args.omg_dir, env=env)
    print(f"### OMG_EXIT={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
