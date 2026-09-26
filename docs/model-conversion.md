# 模型转换：从 HuggingFace 检查点到能上 NPU 的模型

本文记录**实际跑通过**的完整链路，目标是让你能照着把受支持的模型转成 `cann-llm`
能加载的模型目录。

> [!IMPORTANT]
> **一句话版**：用 `DDK-tools-next-6.1.1.0` + `kirinx90-plugin-next-6.1.1.0`，
> 配置里 `quant_param_2 = False`，**官方标准流程直接跑通，不需要任何绕行手段**。
>
> 本项目早期用 5.1.1.1 时踩过一个坑（量化把权重负半轴钳成 0，输出恒定垃圾），
> 排查过程和绕行脚本保留在[附录 B](#附录-b权重被钳成-0早期版本踩过的坑) ——
> **但那个坑的真正原因是配置项写错，不是工具版本**，详见附录 B 的 2×2 实测表。

> 转换产物（`.omc` + `SubGraph_0.weight`）有数 GB，**不进本仓库**，需要自己转。

---

## 0. 流程总览

```
HF 检查点 (model.safetensors)
  │
  │  ① dopt 三阶段量化   在 GPU 机器上跑（本机是 RTX 3080 Ti）
  ▼
fake_quant_weight.pth  +  dopt_config.json
  │
  │  ② 导出             官方示例 npu_tuned_export/export_model_single_qwen2.py
  ▼
ONNX  +  外置的 embedding_weights / embedding_dequant_scale
  │
  │  ③ OMG 转换         官方 tools_omg（大 MatMul 由它内部自动分块）
  ▼
<name>.omc  +  SubGraph_0.weight
  │
  │  ④ 装配模型目录      executor.json / context.json / tokenizer.json / embedding
  ▼
可推理的模型目录 → cann-llm
```

全流程只需官方工具，**不涉及任何自定义脚本改写计算图**。

**实测耗时**（Qwen2.5-1.5B，RTX 3080 Ti）：

| 步骤 | 耗时 |
|---|---|
| 量化 stage1 | ~2.8 min |
| 量化 stage2 | ~20 s |
| 量化 stage3 | ~2 min |
| 导出 ONNX | ~2.4 min |
| OMG 转换 | ~3 min |

---

## 1. 前置条件

| 项 | 说明 |
|---|---|
| **转换机** | Ubuntu x86_64（官方推荐）。OMG 是 x86_64 二进制；aarch64 上要用 qemu 模拟，见[附录 A](#附录-a在-aarch64-设备上直接转换) |
| **GPU** | 量化需要 CUDA 设备（官方 `run.sh` 里 `DEVICE=cuda`）。CPU 能否跑未验证 |
| **DDK 工具** | DDK 工具包 + kirinx90 平台插件包 —— **下载地址见下面** |
| **华为示例代码** | [`cannkit_samplecode_lm_engine_cpp`](https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp) —— 量化与导出脚本都在里面 |
| **HF 检查点** | 官方文档列出的受支持模型之一，例如 `Qwen2.5-1.5B-Instruct`（safetensors 格式） |
| **Python** | 3.10（本项目用 `venv310`，torch 2.4.0+cu121），需要 `onnx` / `onnxruntime` / `numpy` / `safetensors` |
| **磁盘** | 至少 20 GB（ONNX + 权重 + 中间产物） |

### 长任务一定要挂 `tmux`（或者 `nohup`）

量化、导出、OMG 都是**几十分钟到几小时**的任务。如果你是 SSH 到转换机上跑，
**连接一断（网络抖动、笔记本休眠、终端关掉），进程就会被一起杀掉** ——
往往跑了一半才发现白跑，而且中间产物处于半成品状态，很难判断能不能复用。

```bash
# 起一个后台会话（detached），断线不影响
tmux new-session -d -s build "bash build.sh > build.log 2>&1"

# 随时回来看进度
tmux ls                       # 有哪些会话
tmux attach -t build          # 接回去看
tail -f build.log             # 或者直接看日志
```

`tmux` 不在的话用 `nohup … &` 也能扛断线，但没有"接回去看现场"的能力。

> 本项目的 `scripts/model-conversion/build_model.py` 会打印进度，建议把它的输出
> 重定向到日志文件，这样即使挂了会话也能事后查。

### DDK 工具从哪下载

**→ [开发准备（CANN Kit）](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations)**

该页有 Tools 下载表格，列出当前版本的 **DDK 工具包**（含 `tools_dopt`、
`tools_omg`、`tools_ascendc`、`platform`）与各平台插件包（`kirinx90` /
`kirin9020` / `kirin9030`），每项附 SHA256 校验码。

> **插件包版本必须与 DDK 工具包一致**（都在同一张表里）。本项目平台是 Kirin X90，
> 选 **`kirinx90`**。

相关页面：

| 页面 | 用途 |
|---|---|
| [开发准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) | **工具下载**、版本匹配、SHA256 |
| [环境准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-usage-environmental-preparation) | 量化流程、受支持模型列表与下载链接、`config.yaml` / `run.sh` 模板、目录结构 |
| [三段式量化步骤](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-three-stage-quantification) | dopt 量化的三个阶段 |

**本文实测环境**（先核对 SHA256 再解压）：

```
DDK-tools-next-6.1.1.0.zip        87d7e3f186ad5c527a9385cea555559ea53c63b87dc483820523bcf7bf6f87e5  (264 MB)
kirinx90-plugin-next-6.1.1.0.zip  0657efdddd2267949e83af2a382603b523d30b258d72e077a6975eb87d4f10b1  ( 26 MB)
```

解压与安装插件：

```bash
mkdir -p ~/ddk && cd ~/ddk
unzip -q DDK-tools-next-6.1.1.0.zip
unzip -q kirinx90-plugin-next-6.1.1.0.zip

# 插件包解压出 image/ddk_platform_plugin/kirinx90，要放进 tools/platform/
mv DDK-tools-next-6.1.1.0/tools .
cp -r image/ddk_platform_plugin/kirinx90 tools/platform/

# 注意：omg 是包装脚本，解压后可能没有执行权限
chmod +x tools/tools_omg/omg tools/tools_omg/master/omg

python3 -m venv venv310 && source venv310/bin/activate
pip install onnx onnxruntime numpy safetensors pyyaml torch --index-url ...   # 按你的 CUDA 版本装 torch
```

装好后的目录结构：

```
tools/
├── platform/
│   └── kirinx90/          # 平台插件（从插件包拷进来）
├── tools_dopt/            # 量化：dopt_pytorch_py3 / dopt_onnx_py3 / dopt_tf_py3
├── tools_omg/             # ONNX → .omc
└── tools_ascendc/         # 算子编译（含 set_ascendc_env.sh）
```

> **6.1.1.0 的一处改动**：量化入口从 `dopt/dopt_llm/` 改名成了
> **`dopt/dopt_lm/`**。如果你沿用旧版脚本，会报 `No module named dopt.dopt_llm`。

---

## 2. dopt 三阶段量化

### 2.1 准备 `config.yaml`

放在工作目录（例如 `quant/`）下。**逐字照官方文档的模板改**：

```yaml
kd:
  enable: False                 # false 走 PTQ，不做蒸馏
  loss: mse
  micro_batch_size: 2
  gradient_accumulation_steps: 4
  weight_decay: 0.0
  warmup_steps: 10
  num_epochs: 3
  learning_rate: 1.0e-4
  eval_step: 1
  logging_step: 50
  lr_scheduler_type: cosine
  trainable_keys:
    - quant_alpha
    - norm
  no_split_module_classes:
    - Qwen2DecoderLayer
    - Qwen3DecoderLayer
    - GlmDecoderLayer
    - LlamaDecoderLayer
dataset:
  train_files: wikitext2       # 或你自己的 dataset.json
  train_samples: 256
  ptq_samples: 128
extra_training_config:
  fp16: True
cutoff_len: 128
num_samples: 64
quant_param_2: False           # ★★★ 见下
embedding_separate: True
lm_head_size:
```

> [!WARNING]
> **`quant_param_2` 必须按目标平台设置，写错会导致模型输出恒定垃圾。**
>
> * **kirinx90 → `False`**
> * kirin9020 → `True`
>
> 官方文档就是这么写的（`quant_param_2: False // kirinx90默认false，kirin9020平台默认为true`）。
>
> 本项目踩过这个坑：早期配置里写成了 `True`，量化产物里**所有权重的负半轴被钳成 0**，
> 模型在 NPU 上能跑但输出恒定重复垃圾。完整的实测对照与排查方法见
> [附录 B](#附录-b权重被钳成-0早期版本踩过的坑)。

### 2.2 准备 `run.sh`

```bash
#!/bin/bash
set -o pipefail
cd "$(dirname "$0")"

QLIBS=/path/to/ddk/tools/tools_dopt/dopt_pytorch_py3
export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=0
export PYTHONPATH=${QLIBS}:$PYTHONPATH
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

ROOT=.
testcase='output_dir'
mkdir -p ${ROOT}/${testcase}/train_output

model_path='/path/to/Qwen2.5-1.5B-Instruct'
dopt_config=./${testcase}/dopt_config.json
EXTRA=""
[ -f "$dopt_config" ] && EXTRA="--dopt-config $dopt_config"

# 6.1.1.0：入口是 dopt_lm/opt_main.py（旧版是 dopt_llm，且要另一个 wrapper）
python -u ${QLIBS}/dopt/dopt_lm/opt_main.py \
    --model-path $model_path \
    --optimize-config ${ROOT}/config.yaml \
    --quant-stage $1 \
    --group-size 128 --w-bits 4 --act-bits 16 --block-size 128 \
    $EXTRA \
    --output-dir ${ROOT}/${testcase}/train_output 2>&1 | tee ${ROOT}/${testcase}/train_output/logs-$1.log
```

> 首次运行若 `dopt_config.json` 不存在，`opt_main.py` 会**先生成一份然后退出**
> （提示 `generate plugin quang config please set quant strategy firstly`）。
> 检查/调整那份配置里的 `quant_strategy`，再重跑即可。

### 2.3 跑三个阶段

```bash
./run.sh stage1 && ./run.sh stage2 && ./run.sh stage3
```

产物在 `output_dir/train_output/`：

| 文件 | 说明 |
|---|---|
| `trained_quant_weight.pth` | stage1 产出 |
| `fake_quant_weight.pth` | stage3 产出，**导出时用它** |
| `quant_params_file` | stage3 产出，量化参数 |
| `logs-stage*.log` | 各阶段日志 |

### 2.4 先验一下权重没有被钳位（建议做，很便宜）

```python
import torch
sd = torch.load('output_dir/train_output/fake_quant_weight.pth', map_location='cpu', weights_only=False)
for k in ('state_dict','model','module'):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict): sd = sd[k]; break
tot = neg = n = allpos = 0
for k, v in sd.items():
    if not torch.is_tensor(v) or v.numel() < 10000 or not k.endswith('.weight'): continue
    t = v.numel(); g = int((v < 0).sum().item())
    tot += t; neg += g; n += 1
    allpos += (g == 0)
print(f"权重张量 {n} 个, 负值占比 {neg/max(tot,1)*100:.2f}%, 全非负 {allpos} 个")
```

**正常结果**：负值占比 ≈ **43%~44%**，全非负张量 **0 个**。

若看到负值占比只有 **13% 左右、且几乎全部张量无负值**，说明被钳位了 ——
回 2.1 检查 `quant_param_2`。

---

## 3. 导出 ONNX

### 3.1 准备 `model_info_target.yaml`

```yaml
embedding_config:
  embedding_separate: True       # embedding 表单独导出成文件
  embedding_as_fp16: False
  mul_twice: False
no_gemm: True
mock_as_s16: False

model_arch: qwen2                # 模型架构（换模型时改这里）
hf_model_path: /path/to/Qwen2.5-1.5B-Instruct
config_file: /path/to/quant/output_dir/dopt_config.json
quant_pth:   /path/to/quant/output_dir/train_output/fake_quant_weight.pth
output_dir:  /path/to/quant/onnx_out

onnx_output_model_name: qwen2_1p5b_w4
onnx_opset: 12
batch: 1
kv_cache_max_len: 2048           # KV 缓存长度，决定能塞多长的上下文
layers: 28
seq_len:
  - 64                           # prefill 每轮喂的 token 数
```

> `config_file` / `quant_pth` / `output_dir` **建议写绝对路径**。相对路径的解析基准
> 不是 yaml 所在目录。

### 3.2 跑导出

```bash
cd /path/to/cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export
source /path/to/venv310/bin/activate
python export_model_single_qwen2.py /path/to/model_info_target.yaml
```

输出目录名会被自动加后缀（`onnx_out` → `onnx_out_embedding_out_no_output_pos/`），里面有：

- `<name>.onnx` —— 主图
- `<name>.pb` —— 外置权重（ONNX 的 external data）
- `<name>_64_2048.embedding_weights` —— int8 量化后的 embedding 表
- `<name>_64_2048.embedding_dequant_scale` —— 反量化 scale

**后两个文件后面要复制进模型目录**，记下路径。

---

## 4. OMG 转换（ONNX → .omc）

```bash
#!/bin/bash
set +u
OMG=/path/to/ddk/tools/tools_omg
ASC=/path/to/ddk/tools/tools_ascendc
MODEL=/path/to/quant/onnx_out_embedding_out_no_output_pos/qwen2_1p5b_w4.onnx
OUT=/path/to/quant/om_out/qwen

export PATH=$ASC/bisheng/bin:$ASC/package:$PATH
source $ASC/set_ascendc_env.sh >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=$OMG/master/lib64:/path/to/ddk/tools/platform/kirinx90/lib64:${LD_LIBRARY_PATH:-}
export SOC_VERSION=kirinx90
export PYTHONPATH=$OMG/../platform/kirinx90/ops/impl:${PYTHONPATH:-}
export TMPDIR=/some/writable/dir          # /tmp 只读时必需
mkdir -p "$TMPDIR" "$(dirname $OUT)"

cd $OMG
./omg --model $MODEL --framework 5 --output $OUT \
  --input_shape="input_embed:1,-1,1536;attention_mask:1,1,-1,2048;position_ids:1,-1;past_key_in0:2048,2,1,128;...;new_kv_cache_pos:-1;embed_scales:1,-1,1" \
  --dynamic_dims="1,1,1,1,1;64,64,64,64,64" \
  --input_type="past_key_in0:FP16;past_value_in0:FP16;..." \
  --output_type="lm_logits:FP32;past_key0:FP16;past_value0:FP16;..." \
  --weight_data_type FP16 \
  --save_weights_as_external_data=true \
  --platform=kirinx90 \
  --target=omc
```

`--input_shape` / `--input_type` / `--output_type` 是三个**按层数展开的超长字符串**
（28 层时分别约 1654 / 1099 / 946 字符）。手写容易错，用
[`scripts/model-conversion/omg_convert.py`](../scripts/model-conversion/omg_convert.py)
按 `--layers / --kv-len / --hidden` 自动生成：

```bash
python3 scripts/model-conversion/omg_convert.py \
    --onnx /path/to/quant/onnx_out_embedding_out_no_output_pos/qwen2_1p5b_w4.onnx \
    --out  /path/to/quant/om_out/qwen \
    --layers 28 --kv-len 2048 --hidden 1536 \
    --weight-data-type FP16 --dry-run      # 先看一眼命令
```

**关键环境变量**：`SOC_VERSION=kirinx90`、`PYTHONPATH` 指向
`platform/kirinx90/ops/impl`、`LD_LIBRARY_PATH` 含 `tools_omg/master/lib64`
（缺了会报 `RmsNorm ... infershape func failed`）。

**成功标志**：

```
I/OMG_TOOL  main.cpp main(24)::"OMG generate offline model success."
OMG_EXIT=0
```

产出 `<OUT>/<name>.omc` + `<OUT>/SubGraph_0.weight`。

> **大 MatMul 不用管**：日志里会看到 `mlp.down_proj_0` / `q_proj_1` 这类名字 ——
> OMG 自己按 K 把大矩阵拆成了分块。**不需要**手工切图。
> （早期版本会直接报 `Node ... type MatMul don't support!`，见附录 B。）

---

## 5. 装配模型目录

### 5.1 目录结构

```
my-model/
├── executor.json                                  # 引擎配置
├── context.json                                   # 生成/采样配置
├── tokenizer.json                                 # 从 HF 检查点复制
├── <name>.omc                                     # 第 4 节产物
├── SubGraph_0.weight                              # 第 4 节产物
├── <name>_64_2048.embedding_weights               # 第 3 节产物
└── <name>_64_2048.embedding_dequant_scale         # 第 3 节产物
```

### 5.2 `executor.json`

```json
{
  "version": 1,
  "engine_type": "autoregressive",
  "llm_config": {
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "kv_cache_max_len": 2048,
    "sliding_window_len": 0,
    "max_position_embeddings": 32768,
    "num_attention_kv_heads": 2,
    "num_attention_head_dims": 128,
    "num_hidden_layers": 28,
    "prefill_len": 64,
    "decode_len": 1,
    "vocab_size": 151936,
    "vocab_real_size": 151936,
    "use_output_pos": false,
    "max_io_tokens": 4096,
    "hidden_size": 1536,
    "embedding_weights": "qwen2_1p5b_w4_64_2048.embedding_weights",
    "embedding_dequant_scale": "qwen2_1p5b_w4_64_2048.embedding_dequant_scale",
    "embedding_input_type": "int8"
  },
  "tokenizer": { "type": "qwen", "path": "tokenizer.json" },
  "autoregressive": { "model_path": "qwen2_1p5b_w4.omc", "weight_path": "./" }
}
```

里面的数字要和你的模型对上（`num_hidden_layers` / `hidden_size` / `vocab_size` /
`kv_cache_max_len`），`embedding_*` 两个文件名要和实际文件一致。

### 5.3 `context.json`

生成参数。`max_gen_tokens` 与 `stop_sequence` 每次请求都会被覆盖，这里给的是默认值：

```json
{
  "version": 1,
  "engine_type": "autoregressive",
  "generate_options": {
    "callback_freq": 1,
    "max_gen_tokens": 128,
    "stop_sequence": ["<|im_end|>"],
    "init_token_len": 0
  },
  "sampler": {
    "do_sample": true,
    "seed": 99,
    "top-k": 20,
    "top-p": 0.95,
    "temperature": 0.7,
    "repetition_penalty": 1.1
  }
}
```

> `callback_freq: 1` 才会**每生成一个 token 回调一次**（逐字流式的关键）。

### 5.4 `tokenizer.json`

直接从 HF 检查点复制。

---

## 6. 验证

分层验证，从下往上：

```bash
cd /path/to/cann-llm

# ① 能加载吗
python3 -m cann_llm.cli.chat --list-backends

# ② 单个 prompt 能不能出正确结果（贪心，便于判断）
PYTHONPATH=src python3 -m cann_llm.cli.chat \
    -d /path/to/my-model -p "The capital of France is" --temp 0

# ③ 长一点，看是否通顺（不只是短答案对）
PYTHONPATH=src python3 -m cann_llm.cli.chat \
    -d /path/to/my-model -p "Explain what a large language model is in three sentences." \
    --temp 0 --maxtok 80

# ④ 是不是逐字流式
PYTHONPATH=src python3 scripts/stream_check.py -d /path/to/my-model

# ⑤ 交互式跑一轮多轮对话
PYTHONPATH=src python3 -m cann_llm.cli.chat -d /path/to/my-model
```

**本项目实测的期望结果**（`temp=0`）：

| 输入 | 输出 |
|---|---|
| `The capital of France is` | `Paris` |
| `1+1=` | `2` |
| `The sun rises in the` | `east.` |
| 80 token 的长生成 | 通顺连贯的英文段落 |

若输出是 `imentaryimentary…` 这类恒定重复，**回第 2.4 节**检查权重是否被钳位。

---

## 7. 排错

| 现象 | 多半是 |
|---|---|
| 输出恒定垃圾、与 prompt 无关 | 权重被钳位 → 第 2.1 节的 `quant_param_2`；见[附录 B](#附录-b权重被钳成-0早期版本踩过的坑) |
| `No module named dopt.dopt_llm` | 6.1.1.0 改名成了 `dopt.dopt_lm`，改脚本 |
| `./omg: 权限不够` | `chmod +x tools/tools_omg/omg tools/tools_omg/master/omg` |
| `RmsNorm ... infershape func failed` | `LD_LIBRARY_PATH` 里缺 `tools_omg/master/lib64` |
| `Node ... type MatMul don't support!` | 旧版 OMG 不支持大 K 的 MatMul → 升级到 6.1.1.0，或见附录 B |
| ONNX 导出报找不到 `.pb` | 外置权重路径问题；跑检查脚本时先 `cd` 到 onnx 所在目录 |
| 引擎加载就崩 | `executor.json` 里的层数/隐藏维/词表大小与实际不符 |
| 输出正常但上下文一长就变垃圾 | 超过 KV 缓存长度（`kv_cache_max_len`），见 [cann-engine-notes 第 9 节](cann-engine-notes.md) |

---

## 附录 A：在 aarch64 设备上直接转换

不借助第二台机器，用 `qemu-user` 在设备上跑 x86_64 的 OMG。已实测可行，有三个坑：

```bash
export GLIBC_TUNABLES=glibc.pthread.rseq=0    # 沙箱挡 rseq → SIGSYS
export TMPDIR=/writable/dir                    # /tmp 只读

# ① qemu 要显式指定内核版本，否则 glibc 报 "kernel too old"
#    （设备上有 LD_PRELOAD shim 让 uname 报 Linux，glibc 会拿它去校验）
# ② 必须显式调用 x86_64 的 ld.so —— 二进制的 PT_INTERP 指向
#    /tmp/ld-linux-x86-64-2.35.so.2，而 /tmp 只读
qemu-x86_64-static -r "$(uname -r)" \
    "$OMG/master/x86_64-pc-linux-gnu-6.3.0/ld-linux-x86-64.so.2" \
    --library-path "$OMG/master/lib64:$OMG/master/x86_64-pc-linux-gnu-6.3.0:$OMG/platform/kirinx90/lib64" \
    ./master/omg <Args...>
```

> 量化（`tools_dopt`）需要 CUDA，这一步在设备上做不了 —— 只能在有 GPU 的 x86_64
> 机器上完成后把 ONNX 传过来。

---

## 附录 B：权重被钳成 0（早期版本踩过的坑）

> **结论先行**：这不是工具版本的 bug，而是**配置项 `quant_param_2` 写错**。
> 用 6.1.1.0 但把 `quant_param_2` 写成 `True`，**一样会钳位**。

### B.1 症状

模型能在 NPU 上跑、引擎返回成功、有输出，但内容是**恒定重复的垃圾**
（例如 `imentaryimentaryimentary…`），与 prompt 内容、长度都无关。
CPU 上用同一份 HF 检查点跑 llama.cpp 则完全正常。

### B.2 根因：2×2 实测

同一台机器、同一份 HF 检查点、同一个 `dopt_config.json`，只改两个变量：

| DDK 版本 | `quant_param_2` | `fake_quant_weight.pth` 负值占比 | 全非负的权重张量 |
|---|---|---|---|
| 5.1.1.1 | **True** | 13.17% | **196 / 198 (99%)** ✗ |
| 5.1.1.1 | **False** | 43.98% | **0 / 198 (0%)** ✓ |
| 6.1.1.0 | **True** | 13.17% | **196 / 198 (99%)** ✗ |
| 6.1.1.0 | **False** | 43.98% | **0 / 198 (0%)** ✓ |

两两数字**逐位相同** —— 决定因素是 `quant_param_2`，与版本无关。
未量化的 HF 原权重负值占比是 43.24%，所以 43.98% 才是正常值。

**正确取值**：kirinx90 → `False`；kirin9020 → `True`。官方文档写得很清楚，
照抄就不会踩。

### B.3 怎么判断自己中招了

先跑 [2.4 节](#24-先验一下权重没有被钳位建议做很便宜)那段检查。若要追到 ONNX 层：

```bash
cd scripts/model-conversion
python quant/negcheck.py <fake_quant_weight.pth>        # 简化版
python3 confirm_relu.py <导出的 onnx>                    # 与 HF 逐权重比对
```

**钳位的特征信号**：

```
corr(ONNX 权重, HF 权重)        ≈ 0.79 ~ 0.84
corr(ONNX 权重, ReLU(HF 权重))  ≈ 0.98        ← 这个特征最直接
ONNX 权重负值占比 0.00%   vs   HF 的 50%
```

即：权重不是被量化误差弄坏的，而是**负半轴被整个钳掉**（等价于对权重做了一次 ReLU）。

### B.4 绕行手段（已不推荐，仅作参考）

**首选修法是改 `quant_param_2` 重跑量化** —— 不用绕。

如果因为某些原因必须在钳位产物上继续（例如拿不到能重跑的机器），历史上用过两条
绕行，脚本都收在 [`scripts/model-conversion/`](../scripts/model-conversion/)：

1. **`rebuild_weights.py`** —— 从 HF 检查点把 ONNX 里每个权重原样重建回去
   （量化等于被绕开，后面用 `--weight_data_type FP16` 不量化）。
   脚本处理三类映射：具名 initializer、MatMul 节点权重（转置）、`lm_head`。

2. **`split_downproj_fixed.py`** —— 旧版 OMG 不支持 K=8960 的 `MatMul`，
   需要沿 K 切块再相加。**6.1.1.0 的 OMG 会自己分块，不需要这个。**

配套的验证脚本：`patch_ort.py`（给 `ScatterND` 的 indices 插 `Cast`，ORT 才能加载
CANN 导出的图）+ `ort_check.py`（在 ORT 里跑一遍，判断是"模型本身错"还是"仅 NPU 侧错"）。

**注意**：这两条绕行都会改写计算图，其数值正确性无法由工具链自身保证 ——
所以能用官方路径就别用它们。

---

## 附录 C：换一个别的模型

链路本身与模型无关，需要改的只有：

1. `config.yaml` 里的 `no_split_module_classes` 与 `quant_param_2`（**按平台**）；
2. `model_info_target.yaml` 里的 `model_arch` / `hf_model_path` / `layers` /
   `kv_cache_max_len` / `onnx_output_model_name`；
3. `executor.json` 里对应的数字与 `tokenizer` 类型；
4. OMG 的 `--layers` / `--kv-len` / `--hidden`。

**前提是华为的导出脚本支持该架构。** 官方目前列出受支持的模型（见
[环境准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-usage-environmental-preparation)，
页面上直接给了下载链接）：

| 模型 | 备注 |
|---|---|
| Qwen2.5-1.5B | **本项目实测用的就是这个** |
| DeepSeek-R1-Distill-Qwen-1.5B | |
| GLM-1.5B | |
| Qwen2.5-7B-Instruct | 参数量大得多，本设备未必跑得动 |
| Qwen3-8B | 同上 |

不在这个列表里的架构，这条链路基本走不通 —— 得先确认华为的导出脚本支持它。

---

## 附录 D：转 Qwen3 系列的额外坑（实测记录）

Qwen2.5-1.5B 那条链路是**完全跑通并验证过**的（见第 6 节）。
换到 **Qwen3 系列**（本文用的是 `Qwen3-4B-Instruct-2507`）时会额外踩到 4 个坑 ——
都是官方示例代码本身的问题，下面按踩到的顺序列出，脚本收在
[`scripts/model-conversion/`](../scripts/model-conversion/)。

> **状态（已跑通）**：Qwen3-4B-Instruct-2507 已完整转换并在 NPU 上验证通过。
> 实测：`The capital of France is` → `Paris.`、`What is the capital of Japan?` →
> `The capital of Japan is Tokyo.`、80 token 长文本通顺、真流式逐 token；
> 速度 2.5~4.1 tok/s（同设备上 Qwen2.5-1.5B 是 13.1 tok/s）。
>
> 关键约束见坑 4：**导出必须用 FP32**，因此转换机需要 **≥ 64 GB 内存**
> （实测在 62 GB 的机器上通过，31 GB 的机器会确定性 OOM）。

### 坑 1 —— `export_model_single_qwen3.py` 的 dopt import 路径是错的

```python
from dopt.do_opt import optimize_model_gemm2matmul     # ✗ 两个 DDK 版本都没有这个路径
```

5.1.1.1 里是 `dopt/dopt_llm/do_opt.so`，6.1.1.0 里是 `dopt/dopt_lm/do_opt.so`，
**都没有顶层的 `dopt.do_opt`**，而且这两处的函数名也不含 `optimize_model_gemm2matmul`。

真正的定义在**示例代码自带的 `npu_tuned_export/do_opt.py`** 里 ——
隔壁的 `export_model_single_qwen2.py:87` 就是按 `from do_opt import ...` 写的（所以
1.5B 那条路没碰上这个问题）。

**修法**：把 `from dopt.do_opt import` 改成 `from do_opt import`（4 处）。

### 坑 2 —— 独立 embedding 导出的整段代码被注释掉了

Qwen2.5 的脚本会调用 `process_embedding_weights(...)` 产出
`<name>_<seq>_<kv>.embedding_weights` / `.embedding_dequant_scale`
（引擎要在图外算 `input_embed`，`executor.json` 也引用这两个文件）。

Qwen3 的脚本把这一整段注释掉了，结果导出的 ONNX 有 `input_embed` 输入却没有
embedding 文件，模型目录装配不起来。

**修法**：`patch_qwen3_embedding.py` 按 qwen2 的写法恢复这段。

### 坑 3 —— 新版 tokenizer.json 的 merges 格式引擎不认 ★

新版 HF（Qwen3 等）把 BPE merges 存成「数组的数组」，旧版（Qwen2.5 等）是空格分隔的字符串：

```json
"merges": [ ["Ġ","t"], ["Ġ","a"] ]      // 新版 —— 引擎直接 abort
"merges": [ "Ġ t", "Ġ a" ]              // 旧版 —— 引擎认
```

引擎解析时抛 `nlohmann::json type_error.302: type must be string, but is array`，
然后 **core dumped**，模型根本加载不了。另外新版还多一个 `model.ignore_merges`。

**修法**：`normalize_tokenizer_merges.py` 把每项 `" ".join(pair)` 并去掉
`ignore_merges`（15 万条，秒级完成）。

### 坑 4 —— 导出精度：FP32 装不下，FP16 会破坏 RoPE 融合 ★★

`export_model_single_qwen3.py:88` 把精度**硬编码**成 FP32：

```python
hf_model_device = "cpu"
hf_model_dtype = torch.float32
```

**用 FP32**：4B 的模型 16 GB + `from_pretrained` 转换峰值 + 量化权重 ckpt ~8 GB，
超过本文测试机（31 GB 内存）→ 确定性 OOM（实测可用内存掉到 33 MB）。

**改成 FP16**：导出能过、OMG 能出 `.omc`，但 OMG 日志里出现
**36 层 × 4 = 144 条**：

```
E/AI_NPUCL rope_llm_fusion_pass.cc CheckMul0(222)::mul0 weight size invalid 0 != 1
```

RoPE 融合 pass 完全匹配不上（FP16 会引入额外的 Cast，破坏模式匹配）→
图里留下 kirinx90 执行不了的 RoPE → 引擎能加载模型，但 **`Generate` 恒返回 1**。

**结论（已验证）**：这一步**必须用 FP32**，且转换机需要 **≥ 64 GB 内存**。

实测对照（同一份 ONNX 配置，只改精度）：

| 导出精度 | OMG 里的 RoPE 融合错误 | 引擎 `Generate` | 转换机内存 |
|---|---|---|---|
| FP32 | **0 条** | ✓ 正常出词（`Paris.`） | 需 > 31 GB（62 GB 机器实测通过） |
| FP16 | **144 条**（36 层 × 4） | ✗ 恒返回 1 | 31 GB 够用但没用 |

所以正确做法是**换一台大内存的机器**，而不是降精度。

**同时可做的两个内存优化**（`patch_qwen3_export_mem.py` + 手工一处）：

1. `onnx_utils.process_onnx` 里的 `onnxsim.simplify` 是内存峰值之一，可加环境变量
   `CANN_SKIP_ONNX_SIMPLIFY=1` 跳过；
2. 导出脚本把 `quant_pth` **`torch.load` 了两次**（一次给 `ckpt`、一次内联），
   改成复用同一个 `ckpt` 并在 `load_state_dict` 后 `del ckpt; gc.collect()`。

### 坑 5 —— Qwen3 的 `q_norm` / `k_norm` 是被支持的

顺带确认：Qwen3 在 Q/K 上多出的 `q_norm` / `k_norm`（per-head RMSNorm）
**OMG 能正常处理**（日志里可见 `SetWeightInfo, node: model.layers.N.self_attn.q_norm_3_0`），
不是坑。

### 坑 6 —— 换机器做 OMG 时缺 `/tmp/ld-linux-x86-64-2.35.so.2`

`tools_omg/master/omg` 的 ELF 解释器被指定成了 **`/tmp/ld-linux-x86-64-2.35.so.2`**
（不是标准的 `/lib64/ld-linux-x86-64.so.2`）。这是为设备侧 qemu 场景准备的，
所以在设备/老机器上通常有个 `ln -snf` 建它（`to_omc.sh` 里就有这一步）。

**换到一台干净的机器做 OMG 时，若忘了这个链接**，`master/omg` 会以
`FileNotFoundError: .../master/omg` 的形式失败 —— **报的却是"文件不存在"，
而文件其实在**（内核加载不了 ELF 解释器就是不报解释器缺失）。容易误判。

**修法**：

```bash
ln -snf "$(readlink -f /lib64/ld-linux-x86-64.so.2)" /tmp/ld-linux-x86-64-2.35.so.2
```

若之后报 `libomg.so: cannot open shared object file`，那是 `LD_LIBRARY_PATH`
没设（用本文第 4 节的脚本跑就正常）。

### 与 Qwen2.5-1.5B 的形状对照（生成 OMG 参数时要用）

| | Qwen2.5-1.5B | Qwen3-4B-Instruct-2507 |
|---|---|---|
| `num_hidden_layers` | 28 | 36 |
| `hidden_size` | 1536 | 2560 |
| `num_key_value_heads` | 2 | 8 |
| `head_dim` | 128 | 128 |
| `vocab_size` | 151936 | 151936 |
| `bos_token_id` / `eos_token_id` | 151643 / 151643 | **151643 / 151645** |
| RoPE theta | 1000000 | 5000000 |

`eos_token_id` 不同这点在写 `executor.json` 时容易忽略（Qwen3 的 eos 是 `<|im_end|>`)。
