# 离线模型 + NNRt：让 NPU 跑"引擎装不下"的模型

> 这是一条**与 [模型转换](model-conversion.md) 完全不同的路**。
> 那条是把模型转成引擎要的 `.omc`；这条是**把厂商离线模型包成 MindSpore Lite 的 `.ms`，
> 再由 NNRt 在设备上直接加载执行** —— 中间**不经过华为的 LLM 引擎**。

本文记录的是**我们实际跑通过**的一条链路，供你参考尝试。

> **术语**：本文里的**转换机**指装了华为 DDK 的 Linux 机器（我们用的是 x86_64），
> **设备**指鸿蒙设备本身。转换在转换机上做，产物拷到设备上跑。

> [!IMPORTANT]
> **什么时候需要这条路**：华为 LLM 引擎要求模型是「逐层一对 K/V + 每层 `past_key_value`
> 进出」的结构。像 Gemma 4 这类新架构（每层不同配置、K=V、K/V 跨层共享、每层额外输入通路）
> **引擎根本装不下**，图导得出来、OMG 也转得动，照样进不去。
>
> 而**离线模型对 MindSpore Lite 是个黑盒** —— 输入输出张量由扩展配置声明，模型内部结构
> 上层完全不解析，真正算它的是硬件厂商的工具链。**所以这条路不受任何结构约束。**

> [!NOTE]
> **验证程度**：我们用一个 `y = Add(x1, x2)` 的极小图**端到端验证通过**
> （转换成功 + 设备上 NNRt 后端推理结果正确）。**真实 LLM 尚未在这条路上验证过**，
> 请把它当作"路是通的、模型自己试"。

---

## 0. 流程总览

```
原始模型 (ONNX 等)
  │ OMG（DDK 自带）                      --platform=kirinx90
  ▼
厂商离线模型 (.om / .omc)
  │ converter_lite --fmk=THIRDPARTY      ★ 关键是这个参数
  │ + [third_party_model] 扩展配置
  ▼
MindSpore Lite 模型 (.ms)
  │ 设备上 libmindspore_lite_ndk.so + NNRt 后端
  ▼
推理结果
```

三个环节各自的前提：

| 环节 | 前提 |
|---|---|
| OMG | 华为 DDK（含对应平台的 platform 插件） |
| `converter_lite --fmk=THIRDPARTY` | **必须源码编译**（公开的预编译包里没有这个功能） |
| 设备侧推理 | 系统自带 `libmindspore_lite_ndk.so`（鸿蒙设备通常自带） |

---

## 1. 为什么必须自己编转换器

公开的预编译包（例如 `mindspore-lite-2.7.0-linux-x64.tar.gz`）**不支持**这条路：

* 它的 `--fmk` 只认 `TF | TFLITE | CAFFE | MINDIR | ONNX | OM | PYTORCH | MSLITE`
  —— **没有 `THIRDPARTY`**；
* 它内部的 `--fmk=OM` 走的是 **Ascend(ACL) 的自定义算子**路径
  （源码里写死 `"ACL_om_data"`），在麒麟上跑不了；
* 全仓库 grep `third_party_model` **零命中**。

而 OpenHarmony 用的那份 MindSpore 源码里有：

```
https://gitcode.com/openharmony/third_party_mindspore.git

mindspore-src/source/mindspore-lite/tools/converter/parser/third_party/third_party_model_parser.cc
mindspore-src/source/mindspore-lite/tools/converter/converter_lite/converter_flags.cc
    → {"THIRDPARTY", kFmkTypeThirdParty}
mindspore-src/source/mindspore-lite/test/ut/test_data/third_party_model.cfg   ← 官方配置样例
```

> 判断你手上的转换器支不支持，一条命令即可：
> `converter_lite --fmk=THIRDPARTY --modelFile=/nonexistent --outputFile=/tmp/x`
> 若报错里出现 `third_party_param_parser`，说明支持（报"文件不存在"是对的，
> 因为这一步只是让参数校验通过）。

---

## 2. 编译带 THIRDPARTY 的 converter_lite（容器里做）

> **别在宿主机上直接编。** 这份代码（2022 年前后）对工具链很挑，我们用 Arch 的最新
> 工具链一路报错；**换 Debian 12 容器后一次通过**。宿主机要避开的坑：
>
> | 宿主机现象 | 容器里为什么没有 |
> |---|---|
> | CMake 4.x：`Compatibility with CMake < 3.5 has been removed` | Debian 12 是 CMake 3.25 |
> | gcc 16 编老代码报一堆错 | Debian 12 是 gcc 12 |
> | jpeg 等依赖装进 `lib64/`，CMake 却找 `lib/` | Debian 的 install libdir 惯例不同 |
>
> 另外 `mindspore-lite/cmake/compile_link_option.cmake` 里有 `-Werror`，
> 会把 gcc 12 的**假阳性**警告当错误（我们踩到的是
> `-Wfree-nonheap-object` 在 `Tensor::shape()` 内联时的误报）。
> 官方自己也有绕过的先例（`src/litert/cxx_api/kernel_executor/CMakeLists.txt` 里
> `string(REPLACE "-Werror" "" ...)`），照做即可。

```bash
# ① 取源码
git clone --depth 1 https://gitcode.com/openharmony/third_party_mindspore.git
cd third_party_mindspore

# ② 起一个常驻构建容器（只装一次工具链，之后都用 docker exec）
docker run -d --name mslite-dev \
    -v "$PWD":/src -w /src/mindspore-src/source \
    debian:12 bash -c 'tail -f /dev/null'

docker exec mslite-dev bash -c '
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake python3 python3-dev python3-numpy python3-yaml \
        python-is-python3 git ca-certificates curl patch unzip wget
    git config --global --add safe.directory "*"'
```

> 那几个 Python 包不是可选项：`gencode` 步骤要用 `yaml`
> （缺了会以 `ModuleNotFoundError: No module named '\''yaml'\''` 失败），
> 而 Debian 没有 `python` 命令，脚本里回退用的就是 `python`，
> 所以要装 `python-is-python3`。

```bash
# ③ 把 -Werror 换成 -Wno-error
docker exec mslite-dev bash -c '
    F=/src/mindspore-src/source/mindspore-lite/cmake/compile_link_option.cmake
    cp -n $F ${F}.bak && sed -i "s/ -Werror/ -Wno-error/g" $F'

# ④ 配置 + 编译（不用官方 build.sh：它每次会删掉 build/ 重下依赖）
docker exec mslite-dev bash -lc '
    set -e
    S=/src/mindspore-src/source
    mkdir -p $S/build && cd $S/build
    cmake -DCMAKE_BUILD_TYPE=Release -DVERSION_STR=2.7.0 -DENABLE_ASAN=off \
          -DCMAKE_INSTALL_PREFIX=$S/output/tmp -DPLATFORM_X86_64=on \
          -DMSLITE_ENABLE_TRAIN=off \
          $S/mindspore-lite/
    echo $(git -C /src rev-parse HEAD) > .commit_id
    cmake --build . --target install -- -j8'
```

两点说明：

* `.commit_id` 是 `install` 阶段要的文件，平时由官方 `build.sh` 生成。
  直接跑 cmake 时要自己补，否则 install 会在最后一步失败
  （`file INSTALL cannot find ".../build/.commit_id"`）—— **编译其实已经成功了**。
* 第一次配置会下载并编译 protobuf / flatbuffer / glibc 等一堆依赖（几十分钟）。
  **只要不删 `build/`，以后重编只走 `cmake --build . --target install -- -j8`，不会再下依赖。**

产物：

```
<source>/output/tmp/mindspore-lite-2.7.0-linux-x64/tools/converter/converter/converter_lite
```

---

## 3. 厂商离线模型 → `.ms`

### 3.1 先得到离线模型

用 DDK 自带的 `omg`。注意平台名要和你的设备对上：

```bash
export LD_LIBRARY_PATH=<ddk>/tools/tools_ascendc/package/lib64:<ddk>/tools/tools_ascendc/package/lib64/plugin:<ddk>/tools/tools_omg/master/lib64
omg --model=model.onnx --framework=5 --output=model --platform=kirinx90
# → model.om   （kirinx90 平台的产物后缀是 .om）
```

> 平台插件在 `<ddk>/tools/platform/<平台名>/`。只有插件里有的平台名才可用；
> 用错会报 `platform/<名字> not exists! Please install platform plugin first!`。

### 3.2 写扩展配置（**格式很容易搞错**）

离线模型是黑盒，**输入输出张量必须手写**在这个配置文件里：

```ini
[third_party_model]
input_names=in_0;in_1
input_dtypes=float32;float32
input_shapes=1,4;1,4
output_names=out_0
output_dtypes=float32
output_shapes=1,4
```

> [!WARNING]
> **三个必须记住的点**（我们全踩过）：
>
> 1. **节名是 `[third_party_model]`**，不是 `[om_converter]`；
> 2. **多个条目用分号 `;` 分隔**，不是冒号；形状内部才用逗号 `,`；
> 3. **类型名是小写**（`float32` / `float16` / `int32` …），不是大写。
>
> `[om_converter]` + 冒号 + 大写 是**另一条路**（`--fmk=OM`，走 Ascend ACL）的格式，
> 两者互不通用。最可靠的参照是官方样例
> `mindspore-lite/test/ut/test_data/third_party_model.cfg`。

### 3.3 转换

```bash
B=<source>/output/tmp/mindspore-lite-2.7.0-linux-x64
export LD_LIBRARY_PATH=$B/tools/converter/lib:$B/runtime/lib:$B/runtime/third_party/glog

$B/tools/converter/converter/converter_lite \
    --fmk=THIRDPARTY \
    --modelFile=model.om \
    --outputFile=model_ms \
    --configFile=third_party.cfg
# → model_ms.ms
```

成功时输出 `CONVERT RESULT SUCCESS:0`（"这条配置已被保存，将被覆盖"之类的
WARNING 是无害的）。

> `.ms` 可能是以 **root** 身份生成的（如果你的转换在容器里做），
> 拷出来前记得 `chmod 644`，否则宿主机读不到。

---

## 4. 设备侧：用 NNRt 后端跑 `.ms`

系统自带 `libmindspore_lite_ndk.so` 与配套头文件（OpenHarmony SDK 的
`native/sysroot/usr/include/mindspore/`）。关键枚举：

```c
OH_AI_DEVICETYPE_NNRT = 60      /* types.h */
OH_AI_MODELTYPE_MINDIR = 0      /* .ms 就是 MindIR */
```

最小示例见 [`examples/mslite-nnrt/`](../examples/mslite-nnrt/)：

```bash
cc -O1 -I<sysroot>/usr/include mslite_run.c -o mslite_run \
   -L/system/lib64/ndk -lmindspore_lite_ndk
LD_LIBRARY_PATH=/system/lib64/ndk ./mslite_run model_ms.ms nnrt
```

我们那次实测输出：

```
后端: NNRT (devicetype=60)
OH_AI_ModelBuildFromFile -> 0 (SUCCESS)
输入 2 个 / 输出 1 个
  in[0] x1       dtype=43 shape=[1,4] 元素=4
  in[1] x2       dtype=43 shape=[1,4] 元素=4
OH_AI_ModelPredict -> 0 (SUCCESS)
  out[0] y        shape=[1,4] → 2.000 2.000 2.000 2.000   （期望 2.0 ✓）
```

---

## 5. 一个很有说服力的交叉验证

同一个 `.ms`，换后端结果完全不同：

| 后端 | Build | Predict | 输出 |
|---|---|---|---|
| **CPU** | SUCCESS | SUCCESS | `0.000 0.000 0.000 0.000` ✗ |
| **NNRt** | SUCCESS | SUCCESS | `2.000 2.000 2.000 2.000` ✓ |

**CPU 后端出 0 不是 bug，而正好印证了官方那句「离线模型仅支持在 NNRt 后端推理」**：
离线模型被包成一个 **custom 算子**交给底层，CPU 侧没有它的实现，输出就是未初始化的；
真正算它的是 NPU 侧的厂商实现。

> 反过来说：**如果你在 NNRt 后端拿到了正确数值，就说明整条链（转换 → 包成 `.ms` →
> 厂商算子 → NPU）真的通了。**

---

## 6. 带权重的模型也验证过 ✓

上面那个 `Add` 没有权重，容易让人怀疑"是不是只对无权重的小图有效"。我们另外做了一次
**带权重**的验证：`y = Gelu(x @ W1 + b1) @ W2`（三个权重作为 ONNX initializer 内嵌）。

```
ONNX（权重内嵌）→ OMG → mlp_w.om（4.2 KB）→ converter_lite → mlp_w.ms（4.8 KB）
设备上 NNRt 后端：
  Build -> 0 (SUCCESS)
  NPU :  -0.0794   0.0216   0.0823  -0.1355 …
  参考:  -0.0795   0.0216   0.0823  -0.1355 …
  最大偏差 0.000250 · 16 个元素全部在 1e-2 内 ✓
```

**结论：只要模型能被 OMG 编成 `.om`，就能通过这条路在 NPU 上跑起来**（含权重）。
配套示例见 [`examples/mslite-nnrt/mlp_run.c`](../examples/mslite-nnrt/mlp_run.c)。

> ⚠️ **一个容易走错的方向**：我们也试过直接把**厂商给 LLM 引擎用的 `.omc`**
> （`qwen15b_repo.omc` 那种）拿来包成 `.ms`，**转换能成功**（`CONVERT RESULT SUCCESS:0`），
> 但设备上 `Build -> -1` 失败。
>
> 原因不难理解：`third_party_model_parser.cc` 是把 **`.omc` 文件的内容整块**塞进 `.ms` 的
> 一个 tensor 里（`ReadFile` + `memcpy_s`），而厂商 LLM 的 `.omc` **只有图、权重是外挂的**
> （旁边 3 GB 的 `SubGraph_0.weight`），并且它本来就是设计给 **LLM 引擎**加载的
> （引擎负责 KV 管理与权重分页）。
>
> **所以正确做法不是去包厂商的 LLM 产物，而是把我们自己的图（含我们自己设计的 KV 接口）
> 交给 OMG 编成 `.om`** —— 这样权重随图走，也不需要迁就引擎的接口。

---

## 7. 当作 cann-llm 的后端用（`-b nnrt`）

这条路已经接成 cann-llm 的第三个后端：

```sh
cann-llm-chat --list-backends          # 后端: cann, hiai, nnrt
cann-llm-chat -b nnrt -d <放 .ms 的目录>
```

* 后端实现：`src/cann_llm/backends/nnrt.py`（`ctypes` 直调 `libmindspore_lite_ndk.so`，
  与 `cann` / `hiai` 后端同一套协议，上层零改动）；
* 模型目录里放**一个** `.ms`（有多个会报错，要求明确）；`.ms` 依赖的权重文件要放在同一目录；
* `CANN_LLM_MSLITE_LIB` 可换库路径，`CANN_LLM_MSLITE_DEVICE=cpu` 可切 CPU 对照，
  `CANN_LLM_MSLITE_MODEL` 可直接指定 `.ms` 文件。

### 用 `start_chat.sh` 跑（要带三个参数）

```sh
scripts/start_chat.sh -b nnrt -d <放 .ms 的目录> -t plain -s '' \
    -p="<逗号分隔的输入数据>" --max-tokens 1
```

实测输出（模型是文档 §6 那个 `Gelu(x@W1+b1)@W2`）：

```
bot> y=-0.0794067,0.021637,0.0822754,-0.135498,-0.467773,-0.113281,…
参考  y=-0.0795441,0.0216283,0.0823334,-0.1355471,-0.4680230,-0.1131856,…
（与 C 版示例、以及直接调后端的结果一致）
```

**除了 `-b nnrt`，还有三个参数必须给**，原因都是"聊天的默认行为会把 prompt 改掉"：

| 参数 | 为什么必须 |
|---|---|
| `-t plain` | 默认 `chatml` 模板会把 prompt 包成 `<|im_start|>system…`，后端会报"输入里有非数字" |
| `-s ''` | 默认系统提示词（`You are Qwen, …`）同样会被拼进去；清空才拿得到原始输入 |
| `-p="…"` **带等号** | 输入以 `-`（负数）开头，argparse 会把它当选项 ⇒ 必须用 `-p=值` 的写法 |

交互模式下也一样：`scripts/start_chat.sh -b nnrt -d <目录> -t plain -s ''`，
然后每行敲一串逗号分隔的浮点数，就是一次前向。

> **它现在还不是 LLM 后端**：`generate()` 跑一次前向，prompt 里给第一个浮点输入的数据
> （逗号分隔），输出以同样格式返回 —— 用来验证链路。要跑真正的 LLM，得把带 KV 接口的图
> 交给 OMG 编成 `.om`（那才是这条路真正的用武之地）。

实测（NNRt 后端跑 `Gelu(x@W1+b1)@W2`，与 CPU 参考对比）：

```
后端: -0.07941, 0.02164, 0.08228, -0.13550 …
参考: -0.07954, 0.02163, 0.08233, -0.13555 …
最大偏差 0.000250 · 超差 0/16  ✓
```

> 后端里特意**不调用** `OH_AI_ContextDestroy` / `OH_AI_ModelDestroy` ——
> 它们会让进程在退出阶段 core dump（结果已经算对了，崩在析构/卸载）。
> 这与 `cann` 后端不调用 `Context_Destroy` 是同一类处理。

---


---

## 8. 把**真实 LLM** 放上去：一个关键前提

我们拿 Qwen2.5-0.5B 做过一轮 PoC（接口由我们设计：`input_ids[1,16] → logits[1,16,151936]`，
**完全不含 KV**）。结论是：**链路没问题，但模型侧必须用「NPU 亲和」的图**。

### 8.1 直接用 HuggingFace 导出的原生 ONNX 不行

用 `torch.onnx.export`（fp16 · opset 12 · `onnxsim.simplify`）导出的图，OMG 会在
**算子兼容性检查**这一步拦下：

```
E model_compatibility_check.cpp CheckOpSupported:
  "Node /m/model/layers.0/self_attn/Unsqueeze_2 type ExpandDims don't support!"
E general_model_compiler.cpp: "check ir model compatibility failed"
```

**NPU 侧不支持 `ExpandDims` / `BroadcastTo` 这类算子**，而 HF 的原生图里就带着它们。
这不是图画错了 —— 这正是硬件厂商要提供**定制导出脚本**的原因。

### 8.2 用厂商的 NPU 亲和导出器（推荐）

华为的 CANN LLM 示例里有针对每类模型的导出器（内含把不支持的算子消掉/融合的 wrapper）：

```
https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp

CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/
  export_model_single_qwen2.py / _qwen3.py / _glm.py
  npu_tuned_model/{qwen2,qwen3,glm}/   ← ★ 这些 wrapper 就是"NPU 亲和的模型结构"
  onnx_utils.py · do_opt.py            ← 图优化（GEMM→MatMul 等）
  model_info_target.yaml               ← 配置模板
```

跑法（配置里改路径即可）：

```sh
python export_model_single_qwen2.py <你的 model_info.yaml>
```

它 30 秒左右就能导完一个 0.5B，产物**没有** `ExpandDims`/`BroadcastTo`：

```
opset 12 · 1744 节点 · 52 输入 / 49 输出（input_ids/attention_mask/position_ids/
past_key_in0..N/past_value_in0..N/new_kv_cache_pos → lm_logits/past_key0..N/...）
算子：Mul Reshape Add MatMul Transpose Slice Concat ReduceMean Sqrt Div ScatterND Softmax
```

配置里几个要点：

| 配置项 | 说明 |
|---|---|
| `model_arch` | `qwen2` / `qwen3` / `zhipu`（对应上面三套 wrapper） |
| `no_gemm: True` | 把 GEMM 拆成 MatMul（NPU 亲和） |
| `onnx_opset: 12` | 厂商实测值 |
| `layers` / `seq_len` / `kv_cache_max_len` | 要**对上你的模型**（0.5B 是 24 层） |
| `quant_pth` | 量化权重；**为空时代码里虽然允许，但后面的流程仍会要 embedding 的量化 scale** |
| `embedding_config.embedding_in_omc` | 这个为 `True` 时不需要外部传 `embed_scales`（我们踩过 `assert embed_scale is not None`） |

> Python 依赖：厂商这版 wrapper 需要 **transformers ≥ 4.48**
> （用到 `FlashAttentionKwargs`），torch 用 2.5.x 一档即可。

### 8.3 OMG 参数：别手写，从 ONNX 里读

厂商的参数长这样（`input_embed` 是 embedding 分离后的名字，我们自己的图可能叫 `input_ids`）：

```
--input_shape="input_embed:1,-1,1536;attention_mask:1,1,-1,2048;position_ids:1,-1;
               past_key_in0:2048,2,1,128;…;new_kv_cache_pos:-1;embed_scales:1,-1,1"
--dynamic_dims="1,1,1,1,1;64,64,64,64,64"
--input_type="input_embed:INT8;attention_mask:FP32;position_ids:INT32;past_key_in0:FP32;…"
--output_type="lm_logits:FP32;past_key0:FP16;past_value0:FP16;…"
--weight_data_type FP16 --save_weights_as_external_data=true
--platform=kirinx90 --target=omc
```

**最省事的做法：直接从 ONNX 的 `graph.input` 读每个张量的 dtype 与形状来生成**
（项目里的 [`scripts/model-conversion/omg_convert.py`](../scripts/model-conversion/omg_convert.py)
就是这么做的，可以直接借用）。我们手写时踩了三个坑：

1. **输入类型必须与 ONNX 里的真实 dtype 一致** ——
   `Gather` 的 indices（来自 `input_ids`）若是 `INT64`，OMG 直接拒绝：
   `Verify failed, Input[1] DataType INT64 is wrong` ⇒ ONNX 里把 `input_ids` 改成 **INT32**；
2. **`new_kv_cache_pos` 在 `--input_shape` 里要写 `-1`（动态）**，且
   **不要**把它放进 `--input_type`（放了会报 `not supported type:INT32`）；
3. **`--dynamic_dims` 的每组维数要和图里动态维的个数一致**
   （厂商的图有 5 个动态维因为带 `embed_scales`，没有它就只有 4 个）。

### 8.4 还有一个坑：KV 写入的几何要对上

如果图里序列长度是**静态**的（例如我们为简化设了
`embedding_config.embedding_separate: False`，图直接吃 `input_ids[1,64]`），
OMG 会在 KV 写入那步失败：

```
E ScatterNdUpdateVerify: "The dim value should be same,
   but now is indices.dim[0]:1, update.dim[0]:64"
```

厂商流程里序列维是**动态**的（`input_embed:1,-1,hidden`），ScatterND 的几何才对得上。
**所以要用厂商那套配置（`embedding_separate: True`）**，而不是自己简化。

> 一句话：**这条路的"最后一公里"不是 NNRt，而是把模型图做成 NPU 亲和的形态** ——
> 那部分华为已经把工具和 wrapper 都给全了，照它的流程走即可。


---

## 9. ★ 完整走通：真实 LLM 在 NPU 上跑起来（Qwen2.5-0.5B 实测）

这一节是**已跑通的完整配方**，每一步都在 x570 + MateBook Pro 上实测过。

```
HF 权重 → dopt 三阶段量化 → 厂商导出器(NPU 亲和) → OMG → .omc
        → converter_lite(--fmk=THIRDPARTY) → .ms → 设备 NNRt 加载并推理 ✓
```

### 9.1 量化（dopt，用 GPU）

```sh
QLIBS=<ddk>/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$QLIBS:$PYTHONPATH DEVICE=cuda CUDA_VISIBLE_DEVICES=0
python -u $QLIBS/dopt/dopt_lm/opt_main.py --model-path <HF> \
    --optimize-config ./config.yaml --quant-stage $1 \
    --group-size 128 --w-bits 4 --act-bits 16 --block-size 128 \
    --dopt-config ./output_dir/dopt_config.json --output-dir ./output_dir/train_output
```

* `config.yaml` 里 **`quant_param_2: False`**（kirinx90 ✓；写错会把权重负半轴钳成 0）；
* 首次运行会生成 `dopt_config.json` 后退出 ⇒ 用
  `scripts/model-conversion/set_quant_strategy.py` 填入量化策略，再跑 `stage1/2/3`；
* 跑完用 `check_quant_clamp.py` 验证：**负值占比应 ≈43~45%、全非负张量 0 个**；
* 0.5B 三阶段合计 **~2 分钟**（4B 要几十分钟）。

### 9.2 导出（厂商 NPU 亲和导出器 + 量化权重）

配置（`quant_pth` / `config_file` 指向 9.1 的产物，**`embedding_separate: True`**）：

```yaml
embedding_config: {embedding_separate: True, embedding_as_fp16: False, mul_twice: False}
no_gemm: True
model_arch: qwen2
config_file: <quant>/output_dir/dopt_config.json
quant_pth:   <quant>/output_dir/train_output/fake_quant_weight.pth
onnx_opset: 12
batch: 1
kv_cache_max_len: 2048
layers: 24            # ← 要对上你的模型
seq_len: [1]          # ★★ 见下
```

```sh
python export_model_single_qwen2.py <你的 yaml>
```

> ★★ **`seq_len` 必须让 KV 写入的几何自洽**。OMG 会校验 `ScatterND`：
> `indices.dim[0]` 与 `update.dim[0]` 必须相等。
> 用 `seq_len: [64]` 时我们一直撞
> `ScatterNdUpdateVerify: indices.dim[0]:1, update.dim[0]:64`；
> **改成 `seq_len: [1]`（decode 步形态）后 OMG 一次通过**。

### 9.3 OMG

参数**从 ONNX 的 `graph.input` 读**（dtype 映射用
`{1:FP32, 2:UINT8, 3:INT8, 6:INT32, 7:INT64, 10:FP16}` —— 注意 **3 才是 INT8**，
写成 2 会得到 `UINT8`），并**全静态形状**：

```sh
omg --model qwen05_w4.onnx --framework 5 --output qwen05s1 \
    --input_shape="input_embed:1,1,896;attention_mask:1,1,1,2048;position_ids:1,1;
                   past_key_in0:2048,2,1,64;…;new_kv_cache_pos:1" \
    --input_type="input_embed:INT8;attention_mask:FP32;position_ids:INT32;past_key_*:FP32;…" \
    --output_type="lm_logits:FP32;past_key0:FP32;…" \
    --weight_data_type FP16 --save_weights_as_external_data=true \
    --platform=kirinx90 --target=omc
```

产物：`qwen05s1/qwen05s1.omc`（1.3 MB）+ `qwen05s1/SubGraph_0.weight`（**951 MB**）。
日志出现 `OMG generate offline model success.` 即成功。

> 动态形状（`input_embed:1,-1,896` + `--dynamic_dims`）我们**没试通** ——
> 静态 + `seq_len=1` 这条路是通的。

### 9.4 转 `.ms` 并在设备上加载

`[third_party_model]` 扩展配置按 ONNX 的 53 输入 / 49 输出逐条生成
（dtype 用小写 `float32/int8/int32/float16`），然后：

```sh
converter_lite --fmk=THIRDPARTY --modelFile=qwen05s1.omc \
               --outputFile=qwen05s1 --configFile=llm05.cfg
# CONVERT RESULT SUCCESS:0 → qwen05s1.ms (1.3 MB)
```

`.ms` 与 `SubGraph_0.weight` **放同一目录**，设备上：

```
OH_AI_ModelBuildFromFile -> 0 (SUCCESS)      ★ 53 输入 / 49 输出
  in[0] input_embed     INT8  [1,1,896]
  in[1] attention_mask  FP32  [1,1,1,2048]
  in[2] position_ids    INT32 [1,1]
  in[3+] past_key/value_in0..23  FP32 [2048,2,1,64]
Predict ✓（输出了完整的 151936 个 logits）
```

**⇒ 一个真实的 LLM，用自己的图、自己的接口、自己的转换链路，在麒麟 NPU 上跑起来了** ——
全程没有用到华为的 LLM 引擎。


---

## 10. 调试这个平台上的原生进程（lldb）

### 10.1 `lldb-server` 从哪来：CodeArts IDE

> ⚠️ **先纠正一个常见错误说法**：鸿蒙 7 **没有系统自带的 `lldb` / `lldb-server`**，
> 所以不要写成"系统自带的 `/data/service/hnp/bin/lldb-server` 不行"——
> 那个路径本身就不成立。

在当前身份下会 `ptrace failed: Permission denied` 的是
**DevBox、Harmonybrew 和 OHOS-SDK** 提供的 `lldb` / `lldb-server`（应用沙箱禁 ptrace）。

**可行方案**：应用商店里的 **CodeArts IDE**（`com.huawei.codearts`；
注意它与白名单一节提到的 `com.huawei.codearts.agent` 是**两个应用**）
自带一个只依赖 musl libc 的自包含 `huawei-debug-lldb-server`。
它躺在 CodeArts IDE 自己的沙箱里，需要在 **CodeArts IDE 的终端**里拷出来：

```console
$ mkdir -p ~/.local/bin
$ cp /data/storage/el2/base/files/huawei-debug-lldb-server ~/.local/bin/
```

之后用 `~/.local/bin/huawei-debug-lldb-server` 就能正常拉起进程被 lldb 调试。

### 10.2 ★ 只有它能 launch

本机 `~/.harmonybrew/bin/lldb` 直接 `lldb -- ./prog args` **会失败**：

```
error: 'A' packet returned an error: 8
```

`'A'` 包是设置运行参数用的 —— 与「是否带参数」无关（不带参数同样报），
**是这个平台上客户端自己拉起进程这条路不通**。必须由 `huawei-debug-lldb-server`
先拉起、再让 lldb 连上去：

```sh
# ① 服务端（它会 launch 程序，并等你连）
LD_LIBRARY_PATH=/system/lib64/ndk ~/.local/bin/huawei-debug-lldb-server \
    gdbserver 127.0.0.1:<空闲端口> -- <可执行文件>

# ② 客户端
LD_LIBRARY_PATH=/system/lib64/ndk ~/.harmonybrew/bin/lldb --batch \
  -o "gdb-remote 127.0.0.1:<空闲端口>" \
  -o "breakpoint set -n OH_AI_ModelPredict" \
  -o "continue" -o "bt 8" -o "detach"
```

两个坑：**端口**别用常见的（12345 被占会报 `Address in use`）；
**参数要写死在程序里** —— 客户端侧的 `settings set target.run-args` 同样走 `'A'` 包 ✗。

> **查找这份 server 时别只试一个位置。** `hvm-cli` 的 `scripts/hwdbg.sh` 现在会：
> ① 优先用 `HVM_LLDB_SERVER`；② 否则把 `~/.local/bin` 与
> `/data/storage/el2/base/files` **追加到 `PATH` 末尾**（追加而非前插，免得盖掉
> 用户已有的同名程序）再按名字找；③ 都没有才提示从 CodeArts IDE 里拷。

### 10.3 符号都在，可以逐层下断点

`libmindspore-lite.so` / `libmindspore_lite_ndk.so` 都带符号：

```
mindspore::ModelImpl::Predict(...)          mindspore::Model::Predict(...)
mindspore::lite::LiteSession::GetPredictions()
mindspore::Status::IsOk() / StatusCode()    CustomPredictInferShape
```

实测一次定位（Qwen2.5-0.5B 图 `Build=0 / Predict=-1`）：

```
OH_AI_ModelPredict                                → 返回 -1
  +1588: bl mindspore::Status::IsOk() const
  +1600: b.ne  +2784        ← ★ Status 不 OK ⇒ 跳错误路径 ⇒ -1 ★
        └ mindspore::ModelImpl::Predict  ← ★ 它执行完了（推理跑过了）★
⇒ 失败不在推理本身，而在【Predict 返回的 Status 不是 OK】⇒ 包装层拒绝把结果交出去
```

`Status` 的布局（反汇编 `Status::operator bool` 得来）：

```asm
ldr x8, [x0]        ; Status 内部指针
cbz x8, +24         ; 空 ⇒ OK(true)
ldr w8, [x8]        ; ★ 状态码在这里 ★
cmp w8, #0x0
```

⇒ 在断点处读 `*(int*)*(void**)$x0` 就能拿到确切错误码，再对 MindSpore 的
`StatusCode` 枚举即可知道是哪一类失败。

### 10.4 实测结果：NNRt 只回了「通用失败 -1」

按上面读出来了（Qwen2.5-0.5B 的图，FP16 与 W4 两个版本都是这个结果）：

```
frame #1: libmindspore_lite_ndk.so`OH_AI_ModelPredict + 1592   ← 就是那个 IsOk 检查
状态码 = 0xffffffff  ( = -1 )
```

**这不是 MindSpore 的标准 `StatusCode` 枚举值**（那些是有编号的），
而是 **NNRt 后端自己构造的一个"未分类通用失败"** —— 和 `OH_AI_ModelPredict`
最终返回的 -1 一模一样。

⇒ 所以：**失败源头在 NNRt 侧，而且它没给出更细的原因** ⇒ 这也解释了
为什么 `hilog` 里抓不到任何 MindSpore/NNRt 的行（§9 那次尝试）。

⇒ 排查只能从**输入/图**这一侧反推。结合前面所有证据，最强的嫌疑仍然是：
**`.omc` 依赖外挂权重 `SubGraph_0.weight`，而通用「第三方离线模型」路径
没有 `weightDir` 这样的通道告诉驱动权重在哪**（厂商的 LLM 引擎是靠
`api_config.json` 里的 `weightDir` 传的）——
⇒ 验证法：OMG 时**不加** `--save_weights_as_external_data`，让权重内嵌进 `.omc`。

## 11. 已知问题

* **进程退出阶段会 core dump**（`exit code 139`）。推理**结果已经正确**，
  崩在析构/动态库卸载阶段，是独立问题，尚未定位。
* 只在**单算子极小图**上验证过。真实 LLM（Transformer 级图）尚未在这条路上试过 ——
  需要厂商工具链能把该图编成 `.om`，且 `.om` 体量与设备内存匹配。
* `converter_lite --help` 打印的 `--fmk` 列表**不含 THIRDPARTY**（那段帮助文本没更新），
  以实际能否接受该参数为准。
