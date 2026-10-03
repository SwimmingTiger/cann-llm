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

---

# ★NPU 能否跑自定义 kernel / int8 的最终归因（逆向证据）★

材料：固件解包里的 `system/lib64/platformsdk/libneural_network_runtime_ext.so`（0.1 MB）
+ `system/lib64/libhiai_*.z.so`（proxy）+ `libhiai_llm_engine.so`，
用 idalib（`~/rev/nnrt_ext.so`）反编译 ✓。

## 1. `HIAIDevice`（麒麟 NPU 后端）的真实接口

```
★ 全部围绕 ★mindspore::lite::LiteGraph★★：
    PrepareModel(shared_ptr<const LiteGraph>, const ModelConfig&)
    BuildLiteGraph(shared_ptr<hiai::HiaiExecutor>, shared_ptr<const LiteGraph>)
    PrepareOfflineModel(...) · GetOfflineModelFromLiteGraph(...)
    PrepareModelFromModelCache(...) · GetSupportedOperation(shared_ptr<const LiteGraph>)
    AllocateDeviceBufferForOfflineModel · CopyOfflineModelToDevice · ConvertShape
    AllocateTensorBuffer · ReleaseBuffer · ReadOpVersion
    IsFloat16PrecisionSupported · IsPrioritySupported · IsModelCacheSupported
    IsDynamicInputSupported · IsPerformanceModeSupported · IsSupportNpu
⇒ ★算子集由 hiai 决定；没有任何"自定义算子"入口★ ✗
```

## 2. 三条关键证据

```
★★ ① `GetSupportedOperation` 是【空实现】：反编译结果就是 `return 0;` ★★
     ⇒ 这个后端【不暴露"支持哪些算子"】✗
     ⇒ ★这正好解释了为什么用 `OH_NNModel_GetAvailableOperations` 得到"算子数=0"✓★

★★ ② 能力查询族里【只有 fp16，没有 int8】✗★★
     有：IsFloat16PrecisionSupported / IsDynamicInputSupported / IsPerformanceModeSupported …
     ★没有：IsInt8Supported / IsQuantSupported 之类 ✗★
     ⇒ ★"int8" 根本不是这个后端的能力维度★ ✓

★★ ③ 有 `TransDataType`，且会【拒绝不支持的 dtype】✗：
     字符串：`TransDataType failed, data type is unsupported.`
             `AllocateTensorBuffer failed, transform data type failed.`
     ⇒ ★dtype 支持集合是【固定枚举】；不在集合内即报此错★ ✓
     （分支被优化得很难逐条读，但"存在固定白名单"这一点是确定的 ✓）

★ ④ 底层 hiai 在固件里只有【IPC 代理】：
     libhiai_nn_proxy_1.0/1.1/2.0/2.1.z.so（7–8 KB）· libhiai_single_op_proxy_*.z.so
     · libhiai_aiv_proxy_*.so · libhiai_infra_proxy_*.so · libhiai_llm_engine.so（3.2 MB）
     ⇒ ★真正的 NPU 实现在【驱动/固件】侧，用户态拿不到★ ✗
```

## 3. 结论（三条路都堵死）

```
★ 通过 NNRt  → ✗ 后端只认 LiteGraph，算子集固定；连算子支持查询都是空实现
★ 通过 Ascend C / asc-devkit → ✗ 需要 CANN 运行时 + NPU 驱动接口（npu-smi / CANN 包），
                                 设备上【全部缺失】；HarmonyOS SDK 里也没有 CANN Kit 工具链
★ 通过 hiai  → ✗ 只有 IPC 代理，实现在固件/驱动侧
⇒ ★★ 本设备上【无法】运行自定义 NPU kernel ★★ ✗
```

## 4. int8 的三层归因（最终）

| 层面 | 状态 | 依据 |
|---|---|---|
| 硬件 / 指令集 | ★支持 int8 / uint8★ ✓ | asc-devkit 官方类型表（Kirin X90 含 `int8_t`/`uint8_t`）· X90∩int8 API 150/300 |
| 算子层（Ascend C） | ★有量化 API★ ✓ | `adv_api/quantization/`（AntiQuantize/AscendAntiQuant）· Matmul SetQuant*/SetAntiQuant*/SetDequantType |
| **端侧运行时（NNRt / HIAIDevice / hiai）** | ★**没有 int8 维度**★ ✗ | 能力查询无 int8 ✗ · dtype 固定白名单 ✗ · 模型转换链无量化途径 ✗ · int8 图 Build FAILED ✗ |

⇒ **硬件能算 int8，但端侧运行时不给这条通路；而"自己写 kernel"这条路也在本设备上不可行** ✓

## 5. `npu.img` 的真实身份（更正 + 四层图景）

★ 先更正一处我自己的错误 ✗：曾据 `strings npu.img | head -18`（恰好全是证书头）
断言"签名加密读不出"—— **错**。实际有 **1804 行可读内容** ✓。

```
★ npu.img（171 KB）= NPU 的 ★LiteOS 固件镜像★（3 级证书只是外层包裹 ✓）
  证据：LOS_CoreDumpInit · LOS_TaskBackTrace · OsAppTaskCreate · OsMemFreeNode ·
        OSAL_interrupt_* · HalIrqMask · Swt_Task · IdleCore000（LOS = LiteOS ✓）
        CREATE_TASK_CMD_SQCQ / DESTROY_STREAM / CREATE_HEART_BEAT_SQCQ /
        CREATE_PROFILE_CMD_SQCQ / HalClockStart（任务·命令队列·心跳·性能采集 ✓）
        Environment call from M/S/U-mode（特权级异常模型 ✓）
        dev_type/dev_state · engine_handle_task · schd task（设备与任务调度 ✓）
★ 精度词汇统计：int8 / INT8 / uint8 / UINT8 / quant / Quant / FP16 / fp16
   ⇒ ★全部 0 次★ ✗ —— 因为 ★精度不是固件的概念★，固件只负责调度与执行 ✓
```

### 四层图景（每层都有可复核出处 ✓）

| 层 | 载体 | int8 | 证据 |
|---|---|---|---|
| ① 固件内核 | `npu.img`（LiteOS） | 无此概念 ✗ | 调度/命令队列 ✓，精度字符串 0 次 ✓ |
| ② 运行时 + 算子实现 | `libai_npucore_*.so` | ★有★ ✓ | DDK `elementary.so`：`"dataType should be DT_FLOAT16/DT_FLOAT/DT_INT8"` · `TransFilterConvForInt8` · `UpdateBias_WeightInt8_Gen` ✓ |
| ③ 模型编译入口 | `HIAIDevice` / hiai | ★不暴露★ ✗ | 只认 LiteGraph · `GetSupportedOperation` 返回 0 · 能力查询只有 fp16 · int8 图 Build FAILED ✓ |
| ④ 硬件/指令集 | Kirin X90（DAV_3510） | ★支持★ ✓ | asc-devkit 类型表 · X90∩int8 API 150/300 ✓ |

★ 结论：**硬件与算子层都支持 int8；卡点在③模型编译入口（厂商栈），不在我们的代码** ✓
★ 教训：**`head` 截断 + 单一现象 ⇒ 不要下"读不出/不可行"这类全称结论** ✓

---

# ★在线构图 int8 为何失败：逆向到最深一层的完整证据★

材料：设备上真实加载的
`/vendor/lib64/passthrough/indirect/libai_fmk_graph_optimizer.so`（1.48 MB）、
`libai_fmk_hcl_model_runtime_impl.so`、`libai_infra_log.so`、`libhiai.so`、
`libhiai_adapter.so`（均在固件解包中取得同样副本）；
工具：lldb（设备侧断点 + 读寄存器/栈字符串）、idalib（反编译）。

## 1. 失败的精确位置（lldb 实测，进程停在断点处）

```
调用栈（自下而上）：
  uint8_test`main
   → OH_NNCompilation_Build + 888
     → NNCompiler::Build → OnlineBuild → NormalBuild
       → HIAIDevice::PrepareModel → BuildLiteGraph
         → libhiai_adapter.so(+560,+1256) → libhiai.so
           → HIAI_MR_ModelBuilder_Build → HIAI_HCL_ModelBuilder_BuildV2
             → HCL_ModelBuilder_Build → HclModelBuilderImpl::BuildModel
               → BuildForStandardModel → GeneralModelCompiler::Compile → BeforeCompile
                 → ge::ModelOptimizer::Optimize → InferShapeOptimize
                   → ★ge::IrInferShapeOptimizer::InferShape ⇒ 报错 ⇒ 失败★
```

```
★ 最深一层：ge::IrInferShapeOptimizer::InferShape(ge::InferContext&, ge::ComputeGraph&)
  库：libai_fmk_graph_optimizer.so  ·  偏移 +704  ·  行号 272
★ 实际错误串（从 AI_Log_Print 的 x2 读出）：
    "%s %s(%d)::"[op:%s type:%s] ★Infershape failed, %s★""
    "%s %s(%d)::"[op:%s type:%s] Verify failed, %s""
★ 两个 %s 实参（x6/x7 指向栈字符串）读出：★"Add:0"★
  ⇒ 完整即：[op:Add type:0] Infershape failed, …
★ 另有旁证：hiai::ModelTypeUtil::GetModelType 认的 magic = 1146047817 = 0x44504948 = "HIPD"
  我们的 int8/uint8 图不匹配 ⇒ 判为 type=7（未知），函数本身仍返回 0（它不报错）
```

## 2. InferShape 的 dtype 规则：★逐算子硬编码，没有统一表★

全库仅出现两个 GE dtype 名称字符串（`FLOAT`、`UINT8`），其余以数字硬编码。

| 算子 | 允许 dtype | 出处 |
|---|---|---|
| ★**Add**★ | ★float 或 int32★ | 字符串 `"Data type of add OP must be float or int32."` |
| FloorDiv / Range / StridedSlice | float 或 int32 | 同上系列 |
| Greater / Maximum / Clip | float 或 int32_t | 同上系列 |
| Sqrt / Permute | 仅 DT_FLOAT | `"The input fo sqrt only support DT_FLOAT"` |
| Pack / Tile | float 或 int32 或 bool | `"Data type of Pack OP must be float or int32 or bool"` |
| ScatterNd 类 | float / int32 / bool / ★DT_UINT8★ | `"valueDataType must be float or int32 or bool or DT_UINT8."` |
| Slice 的 begin/size | 必须 DT_INT32 | `"not matched DT_INT32"` |
| Shape 常量 | int32 或 int64 | — |
| MaxPool argmax / Size | int32 或 int64 | — |
| MatMul+α / LayerNorm ε | 必须 DT_FLOAT | — |

⇒ **int8/uint8 在通用算子上几乎一律不支持；只有极少数算子显式允许 UINT8** ✓

## 3. ★GE 期望的"量化算子形态"：四个硬要素★

反编译 `SetDtypeAttr` / `SetScaleOffsetAndDtypeAttr`（`general_ir_quantize_saver.cpp`）得知：
量化**不是**"把张量 dtype 设成 int8"，而是**给算子挂一组量化属性**：

```c
ge::AttrUtils::SetInt(opDesc, "dst_type", opParam->dtype);      // SetDtypeAttr
SetScaleAndOffsetAttr(...);                                     // 先设 scale/offset
ge::AttrUtils::SetInt(opDesc, "dtype", opParam->dtype);         // SetScaleOffsetAndDtypeAttr
```

**要素 ①：每输入一个量化类型**（属性名）
`x_quant_type` · `x1_quant_type` · `x2_quant_type` · `w_quant_type` · `filter_quant_type` ·
`quantType` · `has84QuantType`
（校验：`param["quantType"] is not in valid range` · `Get quant type fail` ·
 `Input quant index is illegal:%u` · `quant is oneside quant`）

**要素 ②：每张量一对 scale/offset**
`x_quant_scale` · `x1_quant_scale` · `scale_data_value`/`offset_data_value` ·
`scale_weight_value`/`offset_weight_value` · `scale_from_blob` · `scale_weight_mode`
（校验：`Get weight quant params fail, node:%s` · `Quantize scale is zero.`）

**要素 ③：图内必须插入量化节点，且上游要有 fakequant/dynamicQuant 节点**
`Quantize` / `Dequantize` / `AntiQuantize` / `Requantize` / `DynamicQuantize` ·
`Creator_Quantize_Kernel` · `GetQuantizeOpName`
属性：`fakequantNode` · `dynamicQuantNode` · `quantizeNode` · `dequantNode` ·
`quantizeOpDesc` / `antiQuantizeOpDesc`
（校验：`Insert antiquantize node after Data node fail.` ·
 `param["fakequantNode"] must not be null.` · `param["dynamicQuantNode"] must not be null.`）
⇒ **即 GE 期待的是 QAT 训练 / OMG 量化流程产出的图** ✓

**要素 ④：必须带一个"量化配置 blob"**
`Load quantize config fail.` · ★`Quant config buffer is empty.`★ ·
`ParseOpQuantizeConfig` / `UpdateQuantizeConfig` / `Parse quantize config failed.`
⇒ ★这正是 dopt 那条路产出的 `compress_D.json`（= compress_conf）★

**版本化路径**：`SetQuantizeInfosV1` / `SetQuantizeInfosV2` ·
`CheckQuantizeInfosV2` · `CheckNeedCompatibleQuantV2` · `DequantizeOldIR` · `QuantizeV2`/`DequantizeV2`
**融合限制**：`Not support QuantConv+Bn.` · `Not support QuantConv+Scale.`

## 4. 结论：两条 int8 路径在同一处断掉

```
★ 我们自己手搭（在线构图 + 张量量化参数）：
    只有 int8/uint8 张量 + SetTensorQuantParams
    ⇒ 缺 quant_type 属性 ✗ 缺 scale/offset 属性 ✗ 缺 Quant/Dequant 节点 ✗ 缺 quant config blob ✗
    ⇒ GE 眼里就是"一堆 int8 张量的普通 Add" ⇒ Add 只认 float/int32 ⇒ InferShape 直接拒 ✓
★ dopt 路径：★形态是对的★（compress_conf 就是要素④的 blob ✓）
    但它必须经【扩展配置】送入 ⇒ 设备侧 hiai foundation 拒绝扩展配置 ✗
⇒ ★两条路都断在"量化配置 blob 送不进去 / 形态凑不齐"★★
```

## 6. 尝试结果与最后一环：hiai 适配层支持的量化能力

**尝试**：用 NNRt 的量化专用算子 `OH_NN_OPS_QUANT_DTYPE_CAST`（= 52）
拼出 GE 期望的量化形态。踩到两个坑后★构图层全通★：

| 坑 | 设备库原话 | 正确做法 |
|---|---|---|
| 参数张量用 `OH_NN_INT32` | `"SetSrcT failed, the src_t should be type OH_NN_INT64"` | ★必须 `OH_NN_INT64`★ |
| 参数张量 type 用 `OH_NN_TENSOR` | — | 专属 type：`OH_NN_QUANT_DTYPE_CAST_SRC_T=72` / `_DST_T=73` / `_AXIS=126` |

满足后 6 个 dtype 变体（U8→F32 / I8→F32 / F32→U8 / F32→I8 / 无 axis / F32→F32）
`AddOperation` 与 `Finish` **全部 SUCCESS** ✓

**但 Build 仍 FAILED**，hilog 原文：

```
E NNRt_HiAIAdapter: ★Unrecognized node type 113 for QuantDTypeCast:0.★
E NNRt_HiAIAdapter: Exec Op Convert failed.
E NNRt_HiAIAdapter: Convert CNode to hiai op failed.
E NNRt_HiAIAdapter: Create Op Convert failed for node type 113 for QuantDTypeCast.
E NNRt_HiAIAdapter(via MindIROpConvertFactory): CreateOpConvert: Not supported Type: QuantDTypeCast
```

**适配层实际支持什么**（`libhiai_adapter.so` 字符串为证 ✓）：

```
✓ 反量化：★DequantData★（逐张量）· ★DequantPerChannelData★（逐通道）
    （"DequantData failed, Quant params is empty." / "DequantPerChannelData failed, …"）
✓ 反量化卷积权重："Dequant conv weight tensor enter." / "Dequant weight failed!"
✓ Cast："Convert cast node failed, datatype %d of input/output is not supported."
✗ ★没有 QuantDTypeCast（node type 113）的转换器★ ⇒ Unrecognized node type
★ 其它限制："arithmetic op not supported in IR." · "Not support NPU." ·
   "Dst data type fp16 is not support now!" · "currently do not support scale with actType other than relu."
```

**⇒ 卡点在 `MindIROpConvertFactory::CreateOpConvert` 的查找表里没有 113** ✓

## 7. int8 归因的最终全图（逐层都有出处）

| 层 | int8 状态 | 证据 |
|---|---|---|
| ① 硬件/指令集 | ★支持★ ✓ | asc-devkit 类型表；X90∩int8 API 150/300 |
| ② NPU 算子实现 | ★有 int8 代码路径★ ✓ | `libai_npucore_elementary.so`：`DT_FLOAT16/DT_FLOAT/DT_INT8`、`TransFilterConvForInt8` |
| ③ 固件 `npu.img` | 无精度概念（只管调度） | LiteOS；int8/FP16 字符串 0 次 |
| ④ GE 通用算子 InferShape | ★只认 float/int32★ ✗ | `"Data type of add OP must be float or int32."`；`[op:Add type:0] Infershape failed` |
| ⑤ GE 量化形态 | 需四要素 + 配置 blob | `quant_type`/`scale`·`offset` 属性、Quant/Dequant 节点、`"Quant config buffer is empty."` |
| ⑥ 配置 blob 通道 | ★被设备拒绝★ ✗ | `"hiai foundation not support extension config"` |
| ⑦ NNRt 量化算子接口 | ★有★ ✓ | `QUANT_DTYPE_CAST=52` + `NN_QuantParam` 系列；构图层全通 |
| ⑧ hiai 适配层 | ★无 113 的转换器★ ✗ | `Unrecognized node type 113 for QuantDTypeCast` |

⇒ **每层都"差一点点"：硬件有、算子实现有、NNRt 接口有；
   但 GE 通用算子不收 int8，且 hiai 适配层没有量化算子转换器** ✓
