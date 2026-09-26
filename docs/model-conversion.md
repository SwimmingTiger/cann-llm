# 模型转换：从 HuggingFace 检查点到能上 NPU 的模型

本文记录**实际跑通过**的完整链路，目标是让你能照着把任意一个受支持的模型
转成 `cann-llm` 能加载的模型目录。

> 转换产物（`.omc` + `SubGraph_0.weight`）有数 GB，**不进本仓库**，需要自己转。

---

## 0. 这条链路长什么样

```
HF 检查点 (model.safetensors)
  │
  │  ① 导出       华为官方示例 npu_tuned_export/export_model_single_qwen2.py
  ▼
ONNX  +  外置的 embedding_weights / embedding_dequant_scale
  │
  │  ② ★ 修复权重  rebuild_weights.py  ← 见第 3 节，这步不做就是垃圾输出
  │  ③ 切分大矩阵  split_downproj_fixed.py
  ▼
修复后的 ONNX
  │
  │  ④ OMG 转换    omg_convert.py --weight-data-type FP16
  ▼
<name>.omc  +  SubGraph_0.weight
  │
  │  ⑤ 装配模型目录 executor.json / context.json / tokenizer.json / embedding
  ▼
可推理的模型目录 → cann-llm
```

**每一步都在第 3 节之后有对应脚本**（`scripts/model-conversion/`）。

---

## 1. 前置条件

| 项 | 说明 |
|---|---|
| **转换机** | 建议 Ubuntu x86_64（官方推荐；OMG 是 x86_64 二进制）。aarch64 上需要用 qemu 模拟，见 [附录 A](#附录-a在-aarch64-设备上直接转换) |
| **DDK 工具** | DDK 工具包 + kirinx90 平台插件包 —— **下载地址见下面** |
| **华为示例代码** | [`cannkit_samplecode_lm_engine_cpp`](https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp) —— 量化与导出脚本都在里面 |
| **HF 检查点** | 官方文档列出的受支持模型之一，例如 `Qwen2.5-1.5B-Instruct`（safetensors 格式） |
| **Python** | 3.10，需要 `onnx` / `onnxruntime` / `numpy` / `safetensors` |
| **磁盘** | 至少 20 GB（ONNX + 权重 + 中间产物） |

### DDK 工具从哪下载

**→ [开发准备（CANN Kit）](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations)**

该页有 Tools 下载表格，列出当前版本的 **DDK 工具包**（含 `tools_dopt`、
`tools_omg`、`tools_ascendc`、`platform`）与各平台插件包（`kirinx90` /
`kirin9020` / `kirin9030`），每项附 SHA256 校验码。

> **插件包版本必须与 DDK 工具包一致**（都在同一张表里）。选 **`kirinx90`** ——
> 本项目的目标平台就是 Kirin X90。

相关页面：

| 页面 | 用途 |
|---|---|
| [开发准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) | **工具下载**、版本匹配、SHA256 |
| [环境准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-usage-environmental-preparation) | 量化流程总览、受支持模型列表与下载链接、`config.yaml` / `run.sh` 模板、目录结构 |
| [三段式量化步骤](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-three-stage-quantification) | `dopt` 量化的三个阶段 |
| [LLM 大模型能力开放](https://developer.huawei.com/consumer/cn/doc/HarmonyOS-Guides/cannkit-llm-summary) | CANN Kit LLM 总览 |

> ⚠️ **本文的实测基于 `DDK-tools-next-5.1.1.1` + `kirinx90-plugin-next-5.1.1.1`。**
> 官方页面上的版本会持续更新，新版本不保证第 3～5 节的绕行手段仍然必要或仍然
> 适用（尤其第 3 节的量化钳位问题 —— 那是 `dopt` 的行为，如果新版修了，就
> 不需要重建权重了）。**先按第 3 节的诊断脚本确认自己是否中招，再决定要不要绕。**

拿到两个 zip 之后：

```bash
mkdir -p ~/ddk && cd ~/ddk
# 把 DDK 工具包与 kirinx90 插件包解压到此处
unzip -q DDK-tools-next-*.zip
unzip -q kirinx90-plugin-next-*.zip

python3 -m venv venv310 && source venv310/bin/activate
pip install onnx onnxruntime numpy safetensors pyyaml
```

校验一下下载完整性（SHA256 以上面「开发准备」页为准）：

```bash
sha256sum DDK-tools-next-*.zip kirinx90-plugin-next-*.zip
```

工具目录展开后长这样（**记住这几个路径，后面要用**）：

```
~/ddk/ddk/tools/
├── tools_dopt/       # 量化 + 导出
├── tools_omg/        # ONNX → .omc
├── tools_ascendc/    # 算子编译（含 set_ascendc_env.sh）
└── platform/kirinx90/
```

---

## 2. 导出 ONNX

导出由华为官方示例完成。先准备一份模型描述 YAML：

`model_info_target.yaml`：

```yaml
embedding_config:
  embedding_separate: True       # 把 embedding 表单独导出成文件
  embedding_as_fp16: False
  mul_twice: False
no_gemm: True
mock_as_s16: False

model_arch: qwen2                # 模型架构（换模型时改这里）
hf_model_path: /path/to/Qwen2.5-1.5B-Instruct
config_file: ./output_dir/dopt_config.json
quant_pth:   ./output_dir/train_output/fake_quant_weight.pth
output_dir:  ./onnx_out

onnx_output_model_name: qwen2_1p5b_w4
onnx_opset: 12
batch: 1
kv_cache_max_len: 2048           # KV 缓存长度，决定能塞多长的上下文
layers: 28
seq_len:
  - 64                           # prefill 每轮喂的 token 数
```

然后跑导出：

```bash
cd ~/ddk/cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export

source ~/ddk/venv310/bin/activate
python export_model_single_qwen2.py /path/to/model_info_target.yaml
```

产出（在 `output_dir` 下）：

- `qwen2_1p5b_w4.onnx` —— 主图
- `*_64_2048.embedding_weights` —— int8 量化后的 embedding 表
- `*_64_2048.embedding_dequant_scale` —— 反量化 scale

> **这两个 embedding 文件后面要复制进模型目录**，先记下路径。

导出前会先跑 `dopt` 量化（`run.sh stage1/2/3`），产出
`fake_quant_weight.pth`。**下一节要处理的就是它带来的问题。**

---

## 3. ★ 修复权重 —— 不做这步输出全是垃圾

> 这是整条链路里最坑的一步，跳过它模型能跑但输出毫无意义。详细排查过程见
> [cann-engine-notes 第 8 节](cann-engine-notes.md)。

### 问题是什么

`dopt` 的 W4 量化（`quant_strategy: "Quant_act_weight_eco"`）导出的
`fake_quant_weight.pth`，**把所有权重的负半轴钳成了 0**。

实测：871 个大张量里有 792 个**完全没有负值**；ONNX 权重负值占比 0.00%
（HF 原权重是 50%）。也就是**一半的权重信息被抹掉了**。

### 怎么判断自己中招了

```bash
cd scripts/model-conversion
python3 confirm_relu.py /path/to/qwen2_1p5b_w4.onnx
```

它会打印每个权重与 HF 原权重、以及 `ReLU(HF 权重)` 的相关系数。特征结果：

```
corr(ONNX, HF)        ≈ 0.79 ~ 0.84
corr(ONNX, ReLU(HF))  ≈ 0.98          ← 就是这个特征
ONNX 负值占比 0.00%   vs   HF 50%
```

### 修复

从 HF 检查点把每个权重原样重建回去：

```bash
python3 rebuild_weights.py \
    /path/to/qwen2_1p5b_w4.onnx \
    /path/to/repaired.onnx \
    /path/to/Qwen2.5-1.5B-Instruct          # 也可用环境变量 HF_MODEL
```

脚本处理三类权重：

| 类型 | 映射 |
|---|---|
| 具名 initializer | `model.model.layers.N.X` → `model.layers.N.X` |
| MatMul 节点权重 | 节点名 `model.layers.N.X` → `model.layers.N.X.weight`（转置） |
| `lm_head` | 节点名 `lm_head` → `lm_head.weight`（转置，tied） |

重建后权重是**未量化**的 float32。反正后面用 `--weight_data_type FP16`
转换，量化这一步等于绕开了 —— 也就把坏掉的 fake-quant 从链路里彻底去掉。

### 验证修复成功（强烈建议做）

修复后的 ONNX 先用 ONNXRuntime 跑一遍，确认模型本身是对的：

```bash
# ORT 要求 ScatterND 的 indices 是 int64，CANN 导出的是 int32，先打补丁
python3 patch_ort.py /path/to/repaired.onnx /path/to/repaired_ort.onnx

# 跑数值检查
ONNX_MODEL=/path/to/repaired_ort.onnx \
EMB_WEIGHTS=/path/to/xxx.embedding_weights \
EMB_SCALES=/path/to/xxx.embedding_dequant_scale \
python3 ort_check.py
```

**判定标准**：输入 `"The capital of France is"`，若 ORT 给出的下一个 token 是
`12095`（`" Paris"`），说明 ONNX + 权重是对的。

- **ORT 正确、NPU 垃圾** → 问题在 OMG/引擎侧，继续往下查
- **ORT 也是垃圾** → 图本身就不对，别浪费时间在 NPU 上

---

## 4. 切分过大的矩阵

OMG 不支持 `K` 过大（如 8960）的 `MatMul`，会直接报：

```
Node model.layers.0.mlp.down_proj type MatMul don't support!
```

把 `down_proj` 沿 K 切块再相加：

```bash
python3 split_downproj_fixed.py \
    /path/to/repaired.onnx \
    /path/to/repaired_split.onnx \
    28 \        # 层数，可省略（默认 28，或用 NLAYERS）
    2048        # 每块 K 大小，可省略（默认 2048，或用 K_CHUNK）
```

> **为什么用 `_fixed` 版本**：原始脚本把新节点 `append` 到图末尾，破坏了拓扑序
> （layer 0 的子图排到了 2005/2117 号节点）；而且当 `K ≤ chunk` 只切出一块时，
> 原输出名永远不会被产生（悬空引用）。这个版本就地插入并跳过单块情况。
> OMG 内部会重排节点，所以顺序本身不影响结果，但悬空引用会让 IR 生成失败。

---

## 5. OMG 转换（ONNX → .omc）

```bash
python3 omg_convert.py \
    --onnx /path/to/repaired_split.onnx \
    --out  /path/to/om_out/model \
    --layers 28 --kv-len 2048 --hidden 1536 \
    --weight-data-type FP16
```

先加 `--dry-run` 看一眼生成的命令再执行。脚本会按层数/KV 长度/隐藏维自动拼出
`--input_shape` / `--input_type` / `--output_type`（官方示例里这些是写死的超长
字符串，换模型就没法用）。

**关键环境变量**（脚本已设置，手工跑时需要）：

```bash
export SOC_VERSION=kirinx90
export PYTHONPATH=$OMG_DIR/../platform/kirinx90/ops/impl:$PYTHONPATH
export TMPDIR=/some/writable/dir        # /tmp 只读时必需
```

**务必注意 `--compress_conf` 与 `--weight_data_type` 二选一：**

| | 说明 |
|---|---|
| `--compress_conf <dopt 参数>` | 走量化。但**切分过 down_proj 后不能再用** —— 改名后的节点找不到量化参数，会报 `Node:...down_proj has quant params, but not in the graph` |
| `--weight_data_type FP16` | 不量化，权重按 FP16 存。**本项目推荐的路线**（配合第 3 节的权重重建） |

产出：

```
om_out/model.omc              主图（几百 KB ~ 几 MB）
om_out/SubGraph_0.weight      权重（数 GB）
```

### 在 aarch64 设备上直接转换

如果不想开第二台机器，可以用 qemu 在设备上跑 x86_64 的 OMG。见
[附录 A](#附录-a在-aarch64-设备上直接转换)。

---

## 6. 装配模型目录

把转换产物和配置文件放成一个目录。`cann-llm` 加载的就是这个目录。

### 目录结构

```
my-model/
├── executor.json                                  # 引擎配置（下面详述）
├── context.json                                   # 生成/采样配置
├── tokenizer.json                                 # Qwen tokenizer
├── <name>.omc                                     # 第 5 节的产物
├── SubGraph_0.weight                              # 第 5 节的产物
├── <name>_64_2048.embedding_weights               # 第 2 节导出时产出的
└── <name>_64_2048.embedding_dequant_scale         # 同上
```

**embedding 那两个文件直接复用第 2 节导出时的产物**，名字要和
`executor.json` 里写的一致。

### `executor.json`

`autoregressive.model_path` 指向 `.omc`，`weight_path` 指权重目录：

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
  "autoregressive": {
    "model_path": "rebuilt.omc",
    "weight_path": "./"
  }
}
```

里面的数字要和你的模型对上（`num_hidden_layers` / `hidden_size` / `vocab_size` /
`kv_cache_max_len`），否则引擎行为会很奇怪或直接崩。

### `context.json`

生成参数。`max_gen_tokens` 与 `stop_sequence` 是**每次请求都会被覆盖**的
（见 `backends/cann.py`），这里给的是默认值：

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

> `callback_freq` 设成 `1` 才会**每生成一个 token 回调一次**（逐字流式的关键）。
> 设成 2 则是每两个 token 回调一次。

### `tokenizer.json`

用 HF 检查点里的 `tokenizer.json`（就是 Qwen 的 BPE 词表）直接复制过去。

---

## 7. 验证

### 分层验证，从下往上

```bash
cd /path/to/cann-llm

# ① 能加载吗
python3 -m cann_llm.cli.chat --list-backends

# ② 单个 prompt 能不能出正确结果
PYTHONPATH=src python3 -m cann_llm.cli.chat \
    -d /path/to/my-model -p "The capital of France is" --temp 0

# ③ 是不是逐字流式（而不是最后一次性吐出）
PYTHONPATH=src python3 scripts/stream_check.py -d /path/to/my-model

# ④ 交互式跑一轮多轮对话
PYTHONPATH=src python3 -m cann_llm.cli.chat -d /path/to/my-model
```

`--temp 0`（贪心）时 `"The capital of France is"` 应该给出 `Paris` 开头。
若输出是 `imentaryimentary…` 这类恒定重复，**回到第 3 节** —— 权重是坏的。

---

## 8. 排错

| 现象 | 多半是 |
|---|---|
| 输出恒定垃圾、与 prompt 无关 | 权重被量化破坏 → 第 3 节 |
| 输出重复同一个词 | 同上，或采样参数设成了贪心但权重坏 |
| `Node ... type MatMul don't support!` | 大矩阵没切 → 第 4 节 |
| `has quant params, but not in the graph` | 切分后还在用 `--compress_conf` → 改用 `--weight_data_type FP16` |
| `RmsNorm ... infershape func failed` | `LD_LIBRARY_PATH` 里缺 `tools_omg/master/lib64` |
| `FATAL: kernel too old`（qemu 下） | 需要 `-r <内核版本>`，见附录 A |
| ORT 报 `ScatterND` 类型错 | 先跑 `patch_ort.py` |
| 引擎加载就崩 | `executor.json` 里的层数/隐藏维/词表大小与实际不符 |
| 输出正常但上下文一长就变垃圾 | 超过 KV 缓存长度（`kv_cache_max_len`），见 [cann-engine-notes 第 9 节](cann-engine-notes.md) |

---

## 附录 A：在 aarch64 设备上直接转换

不借助第二台机器，用 `qemu-user` 在设备上跑 x86_64 的 OMG。已经实测可行，
但有三个坑：

```bash
export GLIBC_TUNABLES=glibc.pthread.rseq=0    # 沙箱挡 rseq → SIGSYS
export TMPDIR=/writable/dir                    # /tmp 只读

# ① qemu 要显式指定内核版本，否则 glibc 报 "kernel too old"
#    （设备上有 LD_PRELOAD shim 让 uname 报 Linux，glibc 会拿它去校验）
# ② 必须显式调用 x86_64 的 ld.so —— 二进制的 PT_INTERP 指向
#    /tmp/ld-linux-x86-64-2.35.so.2，而 /tmp 只读
qemu-x86_64-static -r "$(uname -r)" \
    "$OMG/master/x86_64-pc-linux-gnu-6.3.0/ld-linux-x86-64.so.2" \
    --library-path "$OMG/master/lib64:$OMG/master/x86_64-pc-linux-gnu-6.3.0:$OMG/../platform/kirinx90/lib64" \
    ./master/omg <Args...>
```

完整的、可对照的脚本参考 `scripts/model-conversion/` 与设备上
`~/work/llm/ddk-tools/omg-convert.sh`。

---

## 附录 B：换一个别的模型

链路本身与模型无关，需要改的只有：

1. `model_info_target.yaml` 里的 `model_arch` / `hf_model_path` / `layers` /
   `kv_cache_max_len`；
2. `rebuild_weights.py` 里的权重名映射（`_hf_name()`）—— 目前按 Qwen2 的命名
   规则写（`model.layers.N.X`）；
3. `executor.json` 里对应的数字与 `tokenizer` 类型；
4. OMG 的 `--hidden` / `--layers` / `--kv-len`。

**前提是华为的导出脚本支持该架构**（`npu_tuned_export` 里支持的 `model_arch`
有限）。官方目前列出受支持的模型（见 [环境准备](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-llm-usage-environmental-preparation)，
页面上直接给了下载链接）：

| 模型 | 备注 |
|---|---|
| Qwen2.5-1.5B | **本项目实测用的就是这个** |
| DeepSeek-R1-Distill-Qwen-1.5B | |
| GLM-1.5B | |
| Qwen2.5-7B-Instruct | 参数量大得多，本设备未必跑得动 |
| Qwen3-8B | 同上 |

不在这个列表里的架构，这条链路基本走不通 —— 得先确认华为的导出脚本支持它。
