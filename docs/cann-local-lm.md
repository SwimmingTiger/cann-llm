# cann-local-lm 后端（`-b cllm`）：从 HF 模型到 NPU 出字的全流程

本文件是 **cllm 后端**（`src/cann_llm/backends/cllm.py`）的配套说明：它跑的是
**cann-local-lm 的引擎 + 我们自己编译的 `.omc`**，与仓库里另外三个后端的区别见下表。
**模型转换、编译、部署**这三段都不在 cann-llm 里做，但缺任何一段都跑不起来，
所以这里把完整流程和全部实测踩坑写清楚。

## 0. 先分清后端（名字很像，别混）

| 后端 | 加载的库 | C API 前缀 | 模型形态 | 谁执行算子 |
|---|---|---|---|---|
| `hiai` | `/system/lib64/libhiai_llm_engine.so` | `HMS_*` | 官方 LLM 引擎包 | 系统 LLM 引擎 |
| `nnrt` | `/system/lib64/ndk/libmindspore_lite_ndk.so` | `OH_AI_*` | 分段 `.ms`（按层切段） | MindSpore Lite + NNRt |
| `cann` | `/system/lib64/ndk/libcann_llm_engine.so` | `HMS_LLMEngine*` | 官方 OMC 包 | 系统 CANN 引擎 |
| **`cllm`** | **`libcann_lm_engine.so`（自己编的）** | **`CANN_LMEngine_*`** | **`prefill.omc` + `decode.omc` + 外置权重** | **设备 HiAI/NNCore** |

注意两个库名只差一个字母：`libcann_llm_engine.so`（系统的，`cann` 后端）vs
`libcann_lm_engine.so`（我们的，`cllm` 后端）。

### 模型是"两段"，不是按层切

```
prefill.omc   一次喂 N 个 token（gemma4-E2B：N=64）  ← 35 层全在这一个图里
decode.omc    每步喂 1 个 token                      ← 同样 35 层全在一个图里
```

* 每段都是**整网**；不是"每 2 层一段、主机循环调用"那种（那是 `nnrt` 的分段方案）。
* 两段的权重各自编译产物（`delegate_graph_weight_combined` 内容不同），**不能共用**。
* KV 上限 = `model.json` 的 `kv_cache_max_len`；gemma4 这类有滑窗层的模型必须取
  `sliding_window + prefill_len`（见 §2.3），否则引擎逐层校验会直接拒绝。

## 1. 全流程总览

```
① HF 模型 ──converter──► epair（prefill/decode 两张图的 JSON + 权重 bin + model.json）
② epair   ──cann_lm_compile──► prefill.omc / decode.omc + weight/（外置权重）
③ 部署    ──交叉编译 + 签名──► libcann_lm_engine.so；构造 embedding_weights.bin
④ 运行    ──cann-llm -b cllm──► 设备上出字
```

需要的两套环境：

| 阶段 | 需要什么 | 为什么 |
|---|---|---|
| ① | Linux x86_64 + Python 3.12 + `torch==2.9.0+cpu` + `torchair` + `torch_delegate` | 导出走 torchair/torch_delegate 的图捕获 |
| ② | CANN toolkit 9.3.0（**mobile-station** 变体）+ `Ascend-cann-kirinx90-ops` | 目标 SoC 是 KirinX90；`ccec`（毕昇）编 kernel |
| ③ | **`clang`（PATH 上即可）** 或 OHOS SDK；`selfsign.py`（自动 clone） | 编译 aarch64/musl 的 `.so` 并签名 |
| ④ | 设备（HarmonyOS，KirinX90） | 运行时 |

> **宿主 ABI 的坑**：如果宿主是滚动更新的发行版（glibc 2.38+/GCC 16），直接用宿主
> 编出来的 `cann_lm_compile` 在 CANN 的编译容器里会因 `GLIBCXX_3.4.32`/`gcc` 版本
> 不匹配而失败 ✗。**推荐**：在 `ubuntu:22.04` 容器里把 CANN + `torch_delegate` +
> `cann_lm_compile` **全部从源码重建**，保证编译链和被编译物 ABI 一致。

## 2. ① HF → epair

### 2.1 准备 venv（P1）

```bash
python3.12 -m venv "$HOME/p1"
"$HOME/p1/bin/pip" install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
# torchair / torch_delegate / cann-local-lm：按 cann-local-lm 仓库的 build.sh 安装
# 关键：torchair 需要 CANN 的 stub 库，见 §2.5
```

### 2.2 转换

```bash
SANDBOX=$HOME/work/cann-local-lm            # 源码目录（含 converter）
HF=$HOME/models/gemma-4-E2B-it              # HF 权重目录（safetensors）
OUT=$HOME/work/out/e2e                      # 输出（约 10 GB）

PYTHONPATH=$SANDBOX "$HOME/p1/bin/python" -m \
  cann_local_lm.converter.cann_lm_converter \
  --hf-path "$HF" --output-dir "$OUT" \
  --mode prefill-decode --target-soc KirinX90
```

`--mode prefill-decode` = 一次导出两张图（prefill + decode，共享权重）。

### 2.3 KV 长度：`kv_cache_max_len` = **目标上下文**，与 prefill 行宽解耦

两层 KV 的容量规则不同（`engine/core/component/kv_cache_pool`）：

```
full_attention   层：KV 容量 = kv_cache_max_len        ← 上下文总长由它决定 ✓
sliding_attention 层：ring 容量 = sliding_window + prefill_len
                     （新契：meta 带 sliding_window_head_dim 时引擎按此校验）
```

**两者不需要相等**。早期我把 `kv_cache_max_len` 当成"必须等于滑窗环"✗，
于是把 kv 压到 576 —— 实际可以独立放大到目标上下文，`prefill_len` 保持小：

| 配置 | kv_cache_max_len | prefill_len（图行宽） | 上下文 | 短 prompt TTFT |
|---|---|---|---|---|
| 早期（错） | 576 | 64 | 576 | 2.2 s |
| **推荐** | **2048** | **64** | **2048** | **2.5 s** |

`prefill_len` 只影响 **prefill 图行宽**（= 引擎 chunk 大小）：
短 prompt 会被 padding 到该行宽 ⇒ **越小 TTFT 越好**；与上下文长度无关 ✓。
所以"长上下文 + 低延迟"可以同时要：`kv=2048, prefill_len=64`。

导出时显式传两个值（`cann_local_lm_converter.convert()` 目前不暴露 CLI 参数）：

```python
_seq = 64                                   # 图行宽 / chunk
_kv  = 2048                                 # 目标上下文
save_epair_prefill_decode(..., kv_cache_max_len=_kv, seq_len=_seq)
```

> 注意：`max_io_tokens`（默认 8192）只是部署上限字段，实际上下文受 `kv_cache_max_len` 约束。

### 2.4 产物

| 文件 | 说明 |
|---|---|
| `prefill_ep_json_with_ascend_ir.json` / `decode_...json` | 两张图的 IR（②的输入） |
| `weight_info.json` | 图权重索引（②的输入） |
| `weights_0.bin` | 图权重（~4.5 GB） |
| `host_weights_0.bin` + `host_weights_info.json` | host 侧查表权重（embedding 等，④要用） |
| `model.json` | 引擎 meta（**部署时还要补字段**，见 §3.4） |
| `model_config.json` | HF config 快照（仅供参考，不是引擎的 meta） |
| `tokenizer.json` / `tokenizer_config.json` | 词表 |

### 2.5 torchair 的 stub 库

导出阶段 `import torchair` 需要 `libge_runner.so` 等 CANN 库；没有 ascend-sdk 时按
cann-local-lm 的说明编 stub，并把它们放进 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH="$SANDBOX/../torchair/build/stubs:$LD_LIBRARY_PATH"
```

（报 `LiteImportError: failed to import TorchAir ... native stub coverage` 就是这里缺 ✗。）

## 3. ② epair → omc

### 3.1 环境（**顺序不能错**）

```bash
# ① 先 CANN 自己的 set_env：它才会把 ccec（毕昇）加进 PATH、
#    把 opp/built-in/op_impl/ai_core/tbe 加进 PYTHONPATH
source "$CANN_HOME/set_env.sh"
# ② 再仓库的：设 SOC_VERSION / 自定义算子插件 / 融合开关
export PROJECT_ROOT="$SANDBOX"
export TORCH_DELEGATE_HOME="$SANDBOX/torch_delegate"
export CONVERTER_PYTHON=python3
export SOC_VERSION=KirinX90
source "$SANDBOX/scripts/cann_lm_compile/set_env.sh"
```

⚠ **`source` 不要接管道**（`source x | sed` 会把 source 放进子 shell，变量全丢 ✗）。
⚠ CANN 的 `set_env.sh` 会重置 `LD_LIBRARY_PATH`，source 完要**补回**工具链库目录。

### 3.2 编译

```bash
OUT=/tmp/omc          # ★必须是容器/本机内的目录，不能是 bind mount（见 3.3）
mkdir -p "$OUT/prefill" "$OUT/decode"
BIN="$SANDBOX/build/cann_lm_compile"

export TE_PARALLEL_COMPILER=0     # 串行更稳（并发版偶发主进程 abort）

"$BIN" "$EP/prefill_ep_json_with_ascend_ir.json" "$EP/weight_info.json" "$OUT/prefill/prefill"
"$BIN" "$EP/decode_ep_json_with_ascend_ir.json"  "$EP/weight_info.json" "$OUT/decode/decode"
```

成功时逐行打印：

```
[PASS] EpairModelLoader constructed
[PASS] GE Graph extracted successfully
[PASS] aclgrphBuildInitialize succeeded
[PASS] aclgrphBuildModel succeeded
[PASS] mobile model path: renamed onlinec -> .../prefill.omc (size: N bytes)
[PASS] aclgrphBuildFinalize succeeded
```

产物：`<out>/{prefill,decode}/prefill.omc|decode.omc` + `weight/`
（`delegate_graph_weight_combined`、`weight_0.weight`、`SubGraph_*.weight`、`meta.json`）。

### 3.3 两个环境级坑（都是实测）

| 坑 | 现象 | 修法 |
|---|---|---|
| 输出目录是 bind mount | `terminate called ... filesystem error: cannot rename: Invalid cross-device link [onlinec] -> ...omc` | 先写到容器/本机内目录，再 `cp` 出来 |
| 没 source CANN 的 set_env | `ModuleNotFoundError: No module named 'tbe'` / `No module named 'ge'` | 按 §3.1 顺序 source（ccec + tbe 路径都在里面） |

### 3.4 部署前一刀：`model.json` 补字段

转换器写出来的 `model.json` 有两个问题，**不改就起不来**：

```jsonc
"model_config": {
  "sliding_window": 512,            // 转换器默认写 256 ✗（要取 HF text_config 的真值）
  "sliding_window_head_dim": 256,   // 转换器不写这个字段 ✗（引擎要靠它算滑窗环宽度）
  "kv_cache_max_len": 576           // = sliding_window + prefill_len（见 §2.3）
},
"embedding_config": {
  "embedding_weights": "<设备路径>/embedding_weights.bin",
  "embedding_type": "fp16",
  "embedding_scale_type": "fp16",       // 不能是 none（图有 scale 工位时引擎直接拒绝）
  "embedding_quant_mode": "per_token",
  "embedding_load_mode": "in_memory",   // 省内存可换 "in_flash"
  "token_identity_type": "fp16",
  "token_identity_scale_type": "fp16"   // ★填 "none" 会让整段配置降级 ✗（非法值）
}
```

`tokenizer_config.tokenizer_file` 可留空 —— 运行时用 `EngineOptions.tokenizer_path`
（cann-llm 会传模型目录下的 `tokenizer.json`）覆盖它。

## 4. ③ 部署到设备

### 4.1 交叉编译引擎 `.so` 并签名

#### 4.1.1 缺库时自动构建（推荐）

`cllm` 后端发现引擎库不存在时会**自动构建**：从上游 **clone 源码**到 cann-llm 的
构建临时目录（**不在 `/tmp`**），编完签名，装到 `<模型目录>/lib/`：

```bash
# 默认不需要任何环境变量：用 PATH 上的 clang/clang++ + cmake/make（设备上就够）
# 想用 SDK 交叉编译时才需要：export OHOS_SDK_NATIVE=<ohos-sdk>/native
# 方式一：直接跑脚本（推荐，等价于"缺库时的自动构建"，也可单独用于预构建）
scripts/build_cllm_engine.sh                  # 产物 → <cann-llm>/lib/
scripts/build_cllm_engine.sh --force --refresh # 强制重编 + 重新 clone 源码/签名工具
scripts/build_cllm_engine.sh --skip-sign      # 仅本机检查（设备上加载会 Permission denied）

# 方式二：让 `-b cllm` 自己发现缺库 ⇒ 自动 clone + 构建 + 签名 + 安装
# 可选：CANN_LLM_CLLM_REPO / _REF / _BUILD_DIR / _SRC / _AUTOBUILD / _SIGN
python -m cann_llm.launcher chat -d <model>
```

目录约定（全部在仓库内 ✓）：

```
<cann-llm>/.tmp/engine-build/src/            # clone 的 cann-local-lm（--depth 1）
<cann-llm>/.tmp/engine-build/src/build_ohos/ # 构建产物（.so）
<cann-llm>/.tmp/engine-build/build-engine.log# 构建日志（失败先看它）
<cann-llm>/lib/libcann_lm_engine.so          # ★安装 + 签名的最终库★（加载器优先这里）
<cann-llm>/lib/libc++_shared.so
```

| 变量 | 作用 |
|---|---|
| `CANN_LLM_CLLM_TOOLCHAIN` | `clang`（**默认**）/ `sdk`；`--clang` 等价于 `clang` |
| `OHOS_SDK_NATIVE` | **可选**：只有显式给了才走 SDK 交叉编译；不给 ⇒ 普通 clang（**不猜** ✗） |
| `CANN_LLM_CLLM_CC` / `_CXX` | 指定编译器（默认 PATH 的 `clang` / `clang++`，会退到 `CC`/`CXX`） |
| `CANN_LLM_CLLM_REPO` | 上游地址，默认 `https://gitcode.com/cann/cann-local-lm.git` |
| `CANN_LLM_CLLM_REF` | 要构建的分支/标签（默认默认分支） |
| `CANN_LLM_CLLM_SRC` | 直接用本地源码树（跳过 clone，离线/固定版本时用） |
| `CANN_LLM_CLLM_BUILD_DIR` | 构建根目录，默认 `<cann-llm>/.tmp/engine-build` |
| `CANN_LLM_CLLM_REFRESH` | `1` = 强制重新 clone |
| `CANN_LLM_CLLM_AUTOBUILD` | `1`/`0` 强制开/关，缺省=有 SDK 就自动 |
| `CANN_LLM_CLLM_SIGN` | `auto`（clone ohos-bst-light 用其 `selfsign.py`）/ `skip` / 指定签名器路径 |
| `CANN_LLM_CLLM_SIGN_REPO` | 签名工具仓库，默认 `https://gitcode.com/hu60cn/ohos-bst-light` |

产物比源码新时**跳过重复构建**；`force=True`（或删 `.tmp/engine-build`）可强制重编。

**两套工具链，都行**：

| 模式 | 需要什么 | 做法 |
|---|---|---|
| `clang`（默认，没有 SDK 时） | PATH 上的 `clang/clang++`（目标就是 `aarch64-unknown-linux-ohos`）+ cmake/make | 直接 cmake 配 `cann_local_lm/engine/ohos`；`libc++_shared.so` 用 `clang++ -print-file-name` 找 |
| `sdk`（有 SDK 时） | OHOS SDK native（`sysroot/` + `llvm/bin/clang`） | 走上游 `build_ohos.sh --no-signed` |

设备上编译目标 == 本机 ⇒ **纯 clang 完全可行**（实测 21 s 编完、签名后直接加载出字 ✓）。
**默认就是普通 clang**（用 PATH 上的 `clang/clang++` + `cmake`）。只有你显式给
`--sdk` / `OHOS_SDK_NATIVE` 时才走 SDK 交叉编译；**不会**从 PATH 或常见目录去猜 SDK ✗。

#### 4.1.2 手工构建（调试/离线时）

```bash
export OHOS_SDK_NATIVE=<sdk>/native          # 含 sysroot/ 与 llvm/bin/clang
bash "$SANDBOX/scripts/cann_lm_engine/build_ohos.sh" --no-signed
# → build_ohos/libcann_lm_engine.so + libc++_shared.so（ELF aarch64）

# 签名：clone ohos-bst-light，用它的 selfsign.py 逐个签（就地）
git clone --depth 1 https://gitcode.com/hu60cn/ohos-bst-light .tmp/engine-build/sign-tool
python3 .tmp/engine-build/sign-tool/selfsign.py libcann_lm_engine.so --force
python3 .tmp/engine-build/sign-tool/selfsign.py libc++_shared.so --force
```

**不签名 ⇒ `dlopen` 报 `Permission denied`** ✗（不是文件权限位，是 OHOS 的代码签名校验）。
签名后 `ctypes.CDLL(path)` 能直接加载 ✓。

### 4.2 `embedding_weights.bin`：四段布局（`scale` 段填 **1.0**）

引擎的 embedding 模块要吃**单个文件、四段紧排**（无文件头、无 padding）：

| 段 | 长度（gemma-4-E2B, fp16） | 来源 |
|---|---|---|
| seg1 主表 `[vocab, hidden]` | 262144×1536×2 = 805,306,368 B | `host_weights_0.bin` 偏移 0 |
| seg2 per-layer `[vocab, W]` | 262144×8960×2 = 4,697,620,480 B | `host_weights_0.bin` 偏移 805,306,368 |
| seg3 主 scale `[vocab]` | 524,288 B | **填 1.0** |
| seg4 identity scale `[vocab]` | 524,288 B | **填 1.0** |

```bash
python3 scripts/make_embedding_weights.py \
  <epair>/host_weights_0.bin <deploy>/embedding_weights.bin
```

★**scale 段为什么是 1.0**★：图里已经乘过 `embed_scale = sqrt(hidden_size)`，scale 工位
再填 `sqrt(1536)` 就变成平方 ⇒ 输出乱码 ✗（实测：填 1.0 才语义正确 ✓）。

### 4.3 `weight/meta.json` 的绝对路径

② 编译时 `meta.json` 里 `hash_to_weight_file` 写的是**编译机路径**
（例如 `/tmp/omc/prefill/weight/delegate_graph_weight_combined`）✗。
部署时改成设备上的实际路径，否则引擎 `ExecutorCreate` 直接返回
`INVALID_ARGUMENT`，**而且 hilog 里一条日志都没有** ✗（最容易卡住的一条）。

```bash
python3 - <<'PY'
import json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
for st in ("prefill", "decode"):
    p = os.path.join(d, st, "weight", "meta.json")
    m = json.load(open(p)); tgt = os.path.join(d, st, "weight", "delegate_graph_weight_combined")
    if isinstance(m.get("hash_to_weight_file"), dict):
        for k in m["hash_to_weight_file"]:
            m["hash_to_weight_file"][k] = tgt
    json.dump(m, open(p, "w"), indent=1)
PY
```

### 4.4 目录布局（`-b cllm` 就认这个）

```
<model>/
├── prefill/prefill.omc
├── prefill/weight/{delegate_graph_weight_combined,weight_0.weight,SubGraph_*.weight,meta.json}
├── decode/decode.omc
├── decode/weight/{...同上...}
├── model.json
├── tokenizer.json
├── embedding_weights.bin
├── testcase.json          （可选：覆盖上面路径 + 采样默认值）
├── lib/libcann_lm_engine.so      ← 引擎库（也可以放 py/cann_local_lm/lib/）
└── lib/libc++_shared.so
```

`testcase.json` 字段（与 cann-local-lm 一致，路径可相对模型目录）：

```json
{
  "tokenizer_type": "BPE",
  "tokenizer_path": "tokenizer.json",
  "model_path_prefill": "prefill/prefill.omc",
  "model_path_decode": "decode/decode.omc",
  "model_meta_config": "model.json",
  "model_weight_dir_prefill": "prefill/weight",
  "model_weight_dir_decode": "decode/weight"
}
```

## 4.5 量化（W4A8；可在纯 CPU 上做，不需要 torch_npu）

> **dopt 支持 `DEVICE=cpu|cuda|npu`**（`quant/dopt_lm/opt_funcs/eco_stage/train.py:23`），
> QUICKSTART 里"GPTQ 需 NPU"的口径偏保守；4.6B 模型 CPU 上 256 样本约 4.8 小时 ✓。

```bash
# ① 校准 + GPTQ（一个只有 torch+transformers 的 python 即可；config.yaml: ptq_samples=256, cutoff_len=1024）
DEVICE=cpu python dopt_wrapper.py --model-path <hf> --optimize-config config.yaml     --output-dir ./out --hf-type-save
# ② converter 消费（不需要 NPU）
python -m cann_local_lm.converter.cann_lm_converter --hf-path <hf>     --output-dir ./output/quant --target-soc KirinX90 --quant-path ./out
# ③ 编译：量化图要 op_cpu/op_pass 插件（Kirin 的 set_env.sh 自动设
#    ASCEND_CUSTOM_OPP_PATH / ASCEND_GE_PY_PASS_PATH）
```

**三个必须知道的坑**（实测）：

| 坑 | 现象 | 处理 |
|---|---|---|
| `lm_head` 被 W4 量化 | 输出乱码 / emoji 刷屏 / 韩文；**只量化 lm_head 就崩** | **`lm_head` 保持浮点**（`swap.py` 的 `DEFAULT_SKIP` 加 `"lm_head"`），质量立恢复 ✓ |
| `<turn|>`(106) 未当停止符 | 句末无限重复 `<turn|>` | 后端把 `tokenizer_config.special_tokens` 作为 `stop_token_ids` 下发 ✓ |
| 校准样本太少（如 8 个） | 整段乱码 | **至少 256 样本**；`cutoff_len` 用默认 1024 |

实测（gemma-4-E2B-it，KirinX90，kv=2048 / prefill=64）：

| 变体 | 体积 | decode | 质量 |
|---|---|---|---|
| fp16 | 23 GB | 0.48 tok/s | ✓ |
| W4A8 全量化（256 样本） | 8.2 GB | **3.53 tok/s** | ✗（lm_head 拖累） |
| **W4A8 + lm_head 浮点** | 11 GB | 1.13 tok/s | ✓ |
| W4A8 + lm_head int8（目标） | ~9 GB | ~1.8–2.2 tok/s（预期） | 预期 ✓ |

## 5. ④ 用 cann-llm 跑

```bash
cd <cann-llm>
# 引擎库默认从模型目录找（lib/ 或 py/cann_local_lm/lib/）；也可显式指定
export CANN_LLM_CLLM_LIB=<model>/lib/libcann_lm_engine.so

python -m cann_llm.launcher chat -d <model> -b cllm
# 服务：python -m cann_llm.launcher server -d <model> -b cllm --port 8080
```

* 不给 `-b` 时 `detect_backend()` 会认出这个布局（`model.json` + 两段 `.omc`）✓，
  不给 `-t` 时模板跟随后端自报（gemma 模型 → `gemma` 模板）✓；
  注意它的判据**必须排在 nnrt 之前** —— `prefill/`、`decode/` 也以 `pre/dec` 开头 ✓。
* 上下文窗口 = `kv_cache_max_len`（gemma4 部署值 576）；`--max-tokens` 受它限制。
* 引擎不套 chat template：指令模型请自己拼
  `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`（gemma）。
* 引擎内部完成分词，cann-llm 只传文本、收文本；`tokenizer_type` 取
  `tokenizer.json` 的 `model.type`（BPE → `BPE`，Unigram → `SPM`）。

实测（gemma-4-E2B-it / KirinX90，贪心解码）：

| 路径 | prompt | 输出 | 耗时 |
|---|---|---|---|
| 原生 cllm CLI | `今天天气` | `怎么样？` | ≈15 s / 24 token |
| 原生 cllm CLI | `Hello, my name is` | ` Alex. How are you today?` | ≈20 s / 24 token |
| **cann-llm `-b cllm`** | `你好，请用一句话介绍你自己。` | `我是一个由 Google DeepMind 开发的开放权重的大型语言模型。` | ≈49 s / 23 token |
| **cann-llm `-b cllm`** | `用一句话介绍杭州。` | `杭州是一座充满活力…以美丽的西湖…闻名。` | ≈79 s / 40 token |

≈ **0.5 token/s**（含 prefill）。同一 prompt 用原生 CLI 也要 ≈89 s/40 token，
所以瓶颈在引擎/模型本身，不是 cann-llm 这层。

### 5.1 三个使用细节（cann-llm 侧已经处理）

* **gemma 模板**：`-t gemma`（`templates.py` 内置）。Gemma 没有 system 角色，
  模板会把 system 内容前置到第一条 user；`-t gemma` 时不注入 Qwen 那句默认系统提示
  （`ModelConfig.system_prompt` 默认已改成空，由模板的 `default_system` 决定）。
* **stop 串裁剪**：引擎把 `<end_of_turn>` **留在文本里** ✗，流式又收不回来 ⇒
  后端做了"保留后缀"裁剪（`split_stop()`），输出干净 ✓。
* **prompt_tokens 未知**：引擎 C API 只回传生成侧的 token 数，所以 `[in 0 tok]`；
  `tok/s` 用总耗时折算（`decode_ms` 由 wall time 填）。

## 5.5 工具调用（function calling）—— gemma4 原生支持 ✓

之前报 400「gemma 模板不支持工具声明」是**错的**：gemma4 有完整的 function calling
协议（6 个专用 token），已实现：

```
<|tool>declaration:name{description:<|"|>…<|"|>,parameters:{…}}<tool|>   ← 工具声明（进 system 回合）
<|tool_call>call:name{arg:<|"|>值<|"|>}<tool_call|>                      ← 模型请求调用
<|tool_response>response:name{…}<tool_response|>                          ← 结果回填
```

要点：
- 字符串一律用 ``<|"|>`` 定界（官方规定的结构化字面量）
- ``<|tool_response>``(50) 与 ``<|turn|>``(106) 都在 ``generation_config.eos_token_id`` 里
  ⇒ 引擎会在工具请求后自然停住（正是握手需要的行为）✓
- 模板按 tokenizer 自动识别（``<|turn>``/``<turn|>`` ⇒ gemma），不再被套成 ChatML ✗

实测（gemma-4-E2B-it-ctx2048，HTTP `/v1/chat/completions` + tools）：

```json
{"choices":[{"message":{"role":"assistant","content":null,
  "tool_calls":[{"id":"call_0","type":"function",
    "function":{"name":"get_weather","arguments":"{\"location\": \"北京\"}"}}]},
  "finish_reason":"tool_calls"}]}
```

把结果按 `role=tool` 回填后，模型继续同一回合给出:
「北京现在天气晴朗，气温是15摄氏度。」✓

## 6. 排错表（全部是实测撞过的）

| 现象 | 根因 | 修法 |
|---|---|---|
| `[PASS] GE Graph extracted` 后 `aclgrphBuildModel failed, status 1343225857` | TBE 编算子失败（多数是环境不完整） | 按 §3.1 完整 source；`TE_PARALLEL_COMPILER=0` |
| `ModuleNotFoundError: No module named 'tbe'` / `'ge'` | 没 source CANN 的 `set_env.sh` | 先 `source $CANN_HOME/set_env.sh` |
| `cannot rename: Invalid cross-device link` | 输出目录是 bind mount | 先写容器内目录再 `cp` |
| `ExecutorCreate failed: INVALID_ARGUMENT(2)`，**hilog 无任何日志** | `weight/meta.json` 里是编译机绝对路径 | §4.3 |
| `config.load.section: <private>` + `FillInputs failed` / `INVALID_STATE(3)` | `embedding_config` 段被降级：`token_identity_scale_type:"none"` 是非法值 | 全填 `fp16`（§3.4） |
| `kv_pool.ensure: kv_len dim mismatch` | `kv_cache_max_len ≠ sliding_window + prefill_len` | §2.3（576） |
| `graph has per_layer_token_identity_scale slot but embedding scale segment is absent` | `embedding_weights.bin` 只有两段 | §4.2 补 seg3/seg4 |
| 能出字但**乱码/语义错** | scale 段填了 `sqrt(hidden)`（图内已乘一次） | seg3/seg4 填 **1.0** |
| `dlopen ... Permission denied` | `.so` 未签名 | `selfsign.py <elf> --force`（自动模式会自己 clone 工具） |
| `cllm 后端需要模型目录` / `找不到 cann-local-lm 引擎库` | 模型目录或库没配 | `-d` + `CANN_LLM_CLLM_LIB` |

## 7. 限制与后续

1. **上下文 576 token**（= 当前 `kv_cache_max_len`）：要更长就按目标长度重新导出+编译
   （`kv = 目标长度 + prefill_len` 近似）。
2. **两段权重各 4.5 GB**（磁盘 ~18 GB），embedding 5.2 GB 常驻 —— 省内存可试
   `embedding_load_mode: "in_flash"`。
3. 引擎**不做 chat template**、不做工具调用 —— 这些留在 cann-llm 侧。
4. `supports_streaming = True`：cllm 后端用 `ExecutorGenerateAsync` 的逐 token 回调
   实时吐字，但引擎内部线程回调 → 本后端用队列转交。
5. 调试：设备上只有 `huawei-debug-lldb-server` 能调（普通 `lldb` 没权限）；
   `--attach` 会被拒，要用**启动模式**（`gdbserver 127.0.0.1:5555 -- <python> <script>`）。
