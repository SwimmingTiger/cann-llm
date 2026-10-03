# 维护者笔记

给**改这个项目的人**看的：环境怪癖、平台限制、踩过的坑。
面向使用者的用法一律不写在这里，写在该功能自己的文档里 ——
比如 `offline-model-nnrt.md`。

## ★★ 基准测试的铁律：同时只能有【一个】推理进程 ★★

**这块 NPU 没有能力并行跑多个推理。** 只要同时存在第 2 个推理进程：

* 两个都会变慢 ⇒ **所有数据都是垃圾**，不能拿来比较任何东西；
* 每个进程要把段图（合计约 10 GB）连同缓存装进内存 ⇒ **2~3 个进程就能把 32 GB 撑爆并进 swap** ✗。

**2026-10-04 的实测反例**：维护者一边跑"布局 A/B"、一边用户自己在跑 CLI，
结果内存和 swap 全部被打满，**双方数据都不可用**（用户当场指出）。

⇒ 因此：

1. 跑任何对照实验**之前**，先确认没有别的推理在跑 —— **自己挂的后台任务也算**；
2. 所有对照组**串行**执行，**绝不允许**用 `nohup` / `&` 并发跑推理；
3. 每组跑完确认 `free -m` 已回落，再跑下一组；
4. 宁可**少测几组**，也不要并发 —— 并发出来的数字比不测更糟（会导出错误结论）。

> 这条规则是**用户明确要求**的，不是建议。

## 本机调试器（lldb）的可用性

* 鸿蒙 7 **不带**系统自带的 `lldb` / `lldb-server`。
* `/data/service/hnp/bin/lldb-server` **是存在的**，但它是 **DevBox 装的**，
  不是系统自带；它与 Harmonybrew、OHOS-SDK 提供的那几份一样，
  在当前身份下会 `ptrace failed: Permission denied`（应用沙箱禁 ptrace）。
* 唯一实测可用的：应用商店 **CodeArts IDE**（`com.huawei.codearts`，
  注意与白名单里的 `com.huawei.codearts.agent` 是**两个应用**）自带那份
  只依赖 musl libc 的自包含 `huawei-debug-lldb-server`。它躺在 IDE 自己的沙箱里，
  必须在 **CodeArts IDE 的终端**里拷出来（源文件在 `/data/storage/el2/base/files/`）。
  ```
  $ mkdir -p ~/.local/bin
  $ cp /data/storage/el2/base/files/huawei-debug-lldb-server ~/.local/bin/
  ```
* 本机 lldb 客户端直接 `lldb -- <prog>` 会报
  `error: 'A' packet returned an error: 8` —— 这是**平台限制**，
  与"是否带参数"无关（不带参数也一样）。必须由它 `gdbserver` 先拉起进程，
  再用 lldb `gdb-remote` 接上去（`src/cann_llm/lldb_launch.py` 已经封装）。
* 查找这份 server 时**别只试一个位置**：`CANN_LLM_LLDB_SERVER` 是显式覆盖点；
  没有它时把 `~/.local/bin` 与 `/data/storage/el2/base/files` **追加**到 `PATH`
  末尾（追加而非前插，免得盖掉用户已有的同名程序）再按名字找。

### 想调试 Python 的话

* `/data/service/hnp/` 里那份 Python（`/data/service/hnp/bin/python3`）**无法被 lldb 调试**。
* 把它**拷出来并重新签名**之后也**不行** —— cann-llm 的推理跑不起来，
  会报内存分配相关的错误：

  ```
  MemoryError:
  ```

  ⇒ **拷出来这条路也不通，别在上面花时间。**
* ⇒ 想调试 Python，**得自己编译一份不使用 `libmusl_compat` 垫片的 Python**。
  原因：引擎 `libcann_llm_engine.so` / `libhiai_llm_engine.so` 是按 musl 编译的，
  而 glibc 构建的 Python 靠 `libmusl_compat` 垫片运行，
  把 musl 版引擎加载进来就会崩（README「依赖与构建」里那段"不要用 glibc 构建的
  Python"说的就是这件事）—— 换个壳（拷贝 / 重新签名）解决不了。


---

# int8 探索记录（结论：本设备不可行，原因在**设备侧**）

> 目标：让 Gemma 4 以 int8 推理，看能否提速 / 减小体积。
> **结论先说**：★本设备的 NNRt 后端**无法**跑 int8，原因是设备固件明确声明
> "hiai foundation not support extension config" ✗ —— **不在我们的代码** ✓。

## 1. 四条路线与实测结果

| 路线 | 能否产出 int8 | 能否上 NNRt |
|---|---|---|
| ① HF 上找现成 int8 ONNX | ✗ **不存在** —— 只有 fp32 / fp16 / **4-bit `MatMulNBits`** / q2f16 | — |
| ② ONNX 量化算子 → OMG | ✗ OMG **完全不认识**量化算子 | ✗ |
| ③ `converter_lite --fmk=ONNX` + 内置量化器 | ✓ 1188 → **298 MB** | ✗ 仅 CPU 可跑，NNRt `Build -1` |
| ④ **dopt + OMG + `compress_conf`**（厂商正路） | ✓ **298 MB**，OMG 成功 | ✗ `Build -1`；运行时传配置**被静默忽略** |

**①的取证**：HF 上 gemma-4 的导出只有 `f32 / f16 / q4 / NF4 / q2f16`；
名字带 `int8` 的那个是 **MNN 格式**（`.mnn`，非 ONNX）；
`onnx-community/...-qat-mobile-ONNX` 实际是 **q2f16**（不是 int8）。
⇒ ★不要相信仓库名里的 "quantized"/"int8" ✗，一律用 `onnx.load` 读 `initializer.data_type` 与算子★。

**②的取证**：`onnxruntime.quantization.quantize_dynamic` 产出的真 int8（`DynamicQuantizeLinear` +
`MatMulInteger`）交给 OMG ⇒ 失败。OMG 自带的 `check_result.json` 显示
`fail` 数 **恰好等于**这两类算子的个数（74 + 50 = 124 ✓）。
⇒ ★在 OMG 全部库里搜 8 个 int8 算子名（`MatMulInteger` / `DynamicQuantizeLinear` /
`QuantizeLinear` / `DequantizeLinear` / `QLinearMatMul` / `QLinearConv` / `ConvInteger` /
`QuantizeBias`）⇒ **0 个库命中**★ ✗。

**③的取证**：`converter_lite` 自带量化器（`tools/converter/quantizer/`，含
`dynamic_quant.cfg` / `full_quant.cfg` / `fixed_bit_weight_quant.cfg` 等现成配置 ✓）。
**注意**：第三方模型要**把两段配置合到同一个 `--configFile`**（`[third_party_model]` +
`[common_quant_param]`），否则报 `third_party_param_parser` 的错 ✗。
产出的 `.ms` 在 **CPU 后端**能 `Build 0 · Predict 0` ✓，但换 NNRt 设备即 `Build -1` ✗。

## 2. ★失败的确切原因（hilog + lldb 证据链）★

```
CheckNPUPrefix# strncmp: 0 failed, device_name: HIAI_F…          设备名匹配成功
★CANN: hiai foundation not support extension config ✗★           ← ★★根因★★
AI_FMK: HIAI_HCL_ModelBuilder_BuildV2 failed ✗
NNRt:   OH_NNCompilation_Build failed ✗
[nnrt_delegate.cc:772]  InitNNCompilation# Build NNCompilation failed, ret: 1
[nnrt_delegate.cc:262]  BuildOfflineModel# Init NNCompilation failed
[scheduler.cc:542]      ReplaceDelegateKernels# Delegate prepare kernels failed
[lite_session.cc:616]   CompileGraph# Schedule kernels failed: -1
[model_c.cc:259]        OH_AI_ModelBuildFromFile# Common error code
```

**lldb 侧的确认**：断点 `HMS_HiAIOptions_SetQuantConfig` **命中**（落在
`libhiai_foundation.so`）⇒ 量化配置**确实传到了 hiai foundation** ✓，
**是它自己声明不支持扩展配置** ✗。

**自洽性检查**（解释了全部现象 ✓）：
* fp32 段 + 运行时传量化配置 ⇒ 输出**比特级相同**、速度无差别 ✗
  ⇒ 配置被 hiai foundation **静默忽略**（它不支持扩展配置）✓
* `OH_AI_DeviceInfoAddExtension` 返回 **0（成功）** ✓ ⇒ 只是**注册**成功，
  真正消费它的 hiai foundation 不支持 ✗
* int8 在 **CPU 后端**能跑 ✓ ⇒ CPU 走 MS-Lite 自己的 int8 kernel，不经 hiai foundation ✓
* **fp16** 能走得更深（图 P 甚至能跑）⇒ 因为 fp16 用**专用 API** `EnableFloat16` ✗，不走扩展项 ✓

## 3. ★厂商未文档化的格式（可复用资产）★

### 3.1 dopt 校准数据 `.bin` 的格式

来源：`tools_dopt/dopt_pytorch_py3/demo/quant8-8/notrain/bin_data_preprocessing.py` ✓

```
magic = 510（4 维张量）：
    write(<magic 4B 小端>) + write(每维 shape 4B 小端 × 4) + write(tensor.tobytes())
magic = 610（非 4 维张量）：
    write(<magic 4B 小端>) + write(<rank 4B 小端>) + write(每维 shape 4B × rank) + write(tensor.tobytes())
```

★ 之前一直报 `Invalid bin file!` 就是因为给的是**无头裸数据** ✗。

### 3.2 `cal_conf` 的 `.prototxt` 写法

```
strategy: 'Quant_INT8-8'
device: USE_CPU
preprocess_parameter:          # ★每个输入一个块★（不是重复的 input_file_path 键 ✗）
{
    input_type: BINARY         # ★必须显式写 BINARY★（默认/IMAGE 会拒 bin ✗）
    input_file_path: "<绝对路径>/xxx.bin"    # ★必须绝对路径★（相对会以 dopt 工作目录为基准 ✗）
}
...（每个输入重复一个 preprocess_parameter 块）
```

* `PreProcessParameter` 只有 5 个字段：`input_type` / `image_format` / `input_file_path` /
  `mean_value` / `standard_deviation`（用 protobuf 反射 dump 出来的 ✓）
* 报错 `Input nodes number does not match preprocess parameters!` ⇒ 块数与输入数不符
* 报 `No such file or directory: <dopt 工作目录>/calib/…` ⇒ 用了相对路径

### 3.3 dopt 的调用

```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  <venv310>/bin/python dopt_so.py --mode 0 --framework 5 \
    --model <量化前 onnx> --cal_conf <config.prototxt> \
    --input_shape "<name:d0,d1,…;name2:…>" --out_nodes "<输出名>" \
    --output <量化后 onnx> --compress_conf <给 OMG 的压缩配置>
```

* ★dopt 的 `.so` 是 python3.10 编的★（否则报 `undefined symbol: _PyUnicode_Ready`）
* ★protobuf 太新会报 `Descriptors cannot be created directly`★
  ⇒ 设 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` ✓（或降级到 protobuf ≤3.20）
* 产物：**量化后的 ONNX 体积不变**（仍是浮点 ✓），真正的 int8 信息在
  **`compress_conf`** 里，由 OMG 在转换时应用 ✓

### 3.4 OMG 侧与运行时

```
omg … --compress_conf <compress_conf>        # OMG 有这个选项 ✓
runtime: OH_AI_DeviceInfoAddExtension(dev, "QuantConfigData" 或 "QuantBuffer", <数据>, <长度>)
```

（源码：`nnrt/extension_options_parser.cc` 的 `kQuantConfigData` / `kQuantBuffer`；
`nnrt_delegate.cc:744` 的 `HMS_HiAIOptions_SetQuantConfig`）
—— 机制**齐全** ✓，但**设备端不支持** ✗（见 §2）。

## 4. ★★调试方法：hilog（本项目首次使用，极有效）★★

NNRt / MindSpore-Lite 的 `MS_LOG` **不走 stderr** ✗，而是写 **hilog** ✓。
⇒ ★任何 NNRt 侧的 `Build -1`，直接读 hilog 就能看到**确切原因**，不用猜★：

```sh
hilog -x | grep -aiE "MS_LITE|NNRt|CANN|AI_FMK|hiai"
```

本次正是靠它一眼看到 `hiai foundation not support extension config` ✗ ——
比逐个试参数快得多 ✓。**建议以后所有 NNRt 失败都先做这一步** ✓。

配合 **lldb**（`huawei-debug-lldb-server` + `lldb -o "gdb-remote …"`，见前文）可以确认
「某个 API 到底被调用了没、参数对不对」✓ —— 这次就确认了
`HMS_HiAIOptions_SetQuantConfig` **确实被调用** ✓，从而把责任定位到设备侧 ✅。

## 5. 与 fp16 的同型结论

| | fp16 | int8 |
|---|---|---|
| API 映射 | ✓ 有（`TypeId 30 → OH_NN_FLOAT16`） | ✓ 有（`TypeId 32 → OH_NN_INT8`） |
| 工具链 | ✓ OMG / converter 都能转 | ✓ dopt + OMG + converter 全通 |
| 开关机制 | ✓ 专用 API `EnableFloat16` | ✓ 扩展项 `QuantConfigData` |
| **设备端结果** | ✗ 逐图：图 P 可，段图与 lm 失败 | ✗ **扩展配置不被支持 ⇒ 无法编译 int8** |
| 归属 | **设备/固件侧** ✓ | **设备/固件侧** ✓ |

⇒ **两者同型：通路俱全，设备端能力不足** ✓ —— 不是我们的代码问题 ✓。
