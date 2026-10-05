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
用 idalib（`<workdir>/nnrt_ext.so`）反编译 ✓。

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

## 8. ★重要修正：`hiai foundation not support extension config` 是 warning，不是致命点★

**起因**：用户追问"那不用 extension config 呢"。
排查发现：`AddExtensionConfig` 只被 `scripts/model-conversion/int8/probe_quant.py`（一个探针）调用；
正式的 `.ms` 是 OMG 直接产出的，**量化配置已烘焙在模型里** ✓。

**做法**：`probe_quant.py <int8_dopt.ms> - QuantConfigData 60`
（第二个参数 `-` ⇒ 不传 compress_conf ✓，即只用模型里烘焙的配置）

**结果**：Build 仍失败，但失败链**完全不同**，且**深得多**：

```
W/E CANN: hiai foundation not support extension config.        ← ★仍出现，但只是 warning★
         （这次是 MS-Lite 的 NNRt delegate 自己传的 extensions，并非探针）
E AI_FMK: general_compiled_model.cpp operator()(797)
         ::★"compatibleHelper CheckCompatibility failed by cl CPUCL"★   ← ★真正的致命点★
E AI_INFRA: general_model_recompiler.cpp Recompile(178)
         ::★"HcsCompiledModelPreLoadProcess(options, generalCompiledModel) == ge::SUCCESS" "false"★
E AI_FMK: hcl_model_builder_impl.cpp BuildForHcsModel(121)::"modelRecompiler Recompile failed!"
E AI_FMK: hcl_model_builder_impl.cpp BuildModel(570)::"BuildModelByHcl failed"
E AI_INFRA: hcl_model_builder.cpp StaticShapeBuildModel(112)::"ret == SUCCESS" "false"
E CANN: build failed → NNRt: OH_NNCompilation_Build failed
E MS_LITE: [nnrt_delegate.cc:772] InitNNCompilation# Build NNCompilation failed, ret: 1
E MS_LITE: [lite_session.cc:616] CompileGraph# Schedule kernels failed: -1
```

**修正的两点**：

```
✗ 旧判断："hiai foundation not support extension config" 是 int8 失败的根因
✓ 新事实：它只是 warning 级；编译【继续往下走了】——致命点是
         ★"compatibleHelper CheckCompatibility failed by cl CPUCL"★
✗ 旧判断：烘焙配置的 .ms 在 NNRt 层就被拒
✓ 新事实：它一路走到了 ★HCL 的模型重建（Recompile / BuildForHcsModel）★★，
         比"在线构图 int8"（卡在 hiai 适配层）深得多
```

**其它观察**：`CANN: nn proxy 2.1` · `AI_NPUCL: CreateProcess call rtProcessCreate` ·
`MS_LITE: "cpu's architecture is unknown."` · `CheckNPUPrefix: device_name: HIAI_F…`

**另外，离线模型文件那条路（`OH_NNCompilation_ConstructWithOfflineModelFile`）确认不可用**：
hilog 原话 `[NNBackend] CreateCompiler failed, ★only support build NN model and NN model cache★`
⇒ 该后端只支持"在线构图"与"模型缓存"，**不支持离线模型文件** ✓
（这解释了此前 README 中"NNRt 接受文件但编译失败"的现象 ✓）

**下一步**：查 `CheckCompatibility by cl CPUCL` 为何失败 ——
这个检查针对的是模型的**算子/精度兼容性**，是 int8 通路最后一个已知关卡 ✓

## 9. ★★ 找到了：`CheckCompatibility failed by cl CPUCL` 的原因 = 量化类型白名单 ★★

**逆向目标**：`libai_fmk_hcl_model_runtime_impl.so`
（`/vendor/lib64/passthrough/indirect/`，2.24 MB；`general_compiled_model.cpp` 与
 `compatibleHelper CheckCompatibility failed by cl %s` 都在此库 ✓）

**调用链**：`GeneralCompiledModel::CheckCompatibility(vector<string>&)` 用 `GraphListWalker`
遍历图上所有节点，对每个节点调 `OpKernelStoreManager::GetCompatibleHelper(CL名)`；
CL 名为 `CPUCL` 时取不到 helper ⇒ 打印 `compatibleHelper CheckCompatibility failed by cl CPUCL`。

**决定性的函数**：`hiai::IsCompatibleQuantType(ge::DataType, ge::DataType)` @0xe7e5c
反编译即：静态初始化一个 `vector<QuantDataType>`（每条 8 字节 = 两个 int32），
然后**线性查找配对**；在表里 ⇒ true，不在 ⇒ false。

**表地址 `unk_52D34`，长度 0x78 = 120 字节 = 15 条**（用 ida_bytes 读出）：

| # | 配对 | 含义 |
|---|---|---|
| 0 | DT_FLOAT ⇄ DT_INT8 | W8A32 |
| 1 | DT_FLOAT16 ⇄ DT_INT8 | W8A16 |
| 2 | DT_INT32 ⇄ DT_INT8 | — |
| 3 | DT_RESOURCE ⇄ DT_RESOURCE | 资源 |
| 4 | DT_INT32 ⇄ DT_QUINT16 | — |
| 5 | DT_UINT8 ⇄ DT_VARIANT | — |
| 6 | DT_UINT8 ⇄ DT_INT4 | 低比特 |
| 7 | DT_UINT8 ⇄ DT_UINT1 | 低比特 |
| 8 | DT_INT8 ⇄ DT_INT4 | 低比特 |
| 9 | DT_INT8 ⇄ DT_VARIANT | — |
| 10 | DT_INT8 ⇄ DT_UINT1 | 低比特 |
| 11–14 | (36=HiFloat8) ⇄ VARIANT / INT4 / UINT1 / (30) | 新型低精度 |

**★ 关键结论 ★**

```
✗ 表里【没有 (DT_INT8, DT_INT8)】，也【没有 (DT_UINT8, DT_UINT8)】
  ⇒ "int8 × int8"（纯 int8 计算）不在白名单 ⇒ CheckCompatibility 必失败
  ⇒ 我们此前所有"全 INT8 / 全 UINT8 张量"的探针都落在这里 ✓
✓ 白名单里合法的量化形态只有两类：
  ① 浮点 × INT8：DT_FLOAT16 ⇄ DT_INT8（W8A16）· DT_FLOAT ⇄ DT_INT8（W8A32）
     · DT_INT32 ⇄ DT_INT8
  ② 低比特：UINT8/INT8 × INT4/UINT1
⇒ ★设备真正接受的量化 = 【激活保持浮点、权重用 INT8】（weight-only / W8A16·W8A32）★
```

**这也解释了 HF 上 gemma-4 的 int8 模型为何是 "weight-only INT8"** ——
weight-only 正是这台设备白名单里的形态 ✓（"纯 int8 计算"不是）。

**下一步**：按白名单构造 **FP16 激活 × INT8 权重** 的图/模型再试 ✓

### 9.1 原始证据与可复现步骤（供复核）

**目标库**：`/vendor/lib64/passthrough/indirect/libai_fmk_hcl_model_runtime_impl.so`
（2 236 552 字节；设备上真实加载的路径可用 `/proc/<pid>/maps` 确认）

**关键符号（`nm -D` / IDA 均可取）**：

```
hiai::GeneralCompiledModel::CheckCompatibility(std::vector<std::string>&)   @0x83764
hiai::OpKernelStoreManager::GetCompatibleHelper(std::string const&)         @0x20d448
hiai::IsCompatibleQuantType(ge::DataType, ge::DataType)                     @0xe7e5c
ge::ModelCompatibilityCheck::CheckIRGraphCompatibility(...)                 @0x118c70
ge::ModelCompatibilityCheck::GetIRGraphSupportResult(...)                   @0x1195c0
hiai::IRTransformer::IsCompatible(...)                                      @0x101880
```
相关字符串：`"compatibleHelper CheckCompatibility failed by cl %s"`（0x2e565）·
`"compatibleHelper" "null, return FAIL."`（0x3e6c8）· `"get npu cl compatibleHelper fail!"`（0x502f6）

**白名单表原始字节**（表头 `unk_52D34`，120 字节；小端 int32 成对）：

```
06000000 02000000   → (6, 2)   … 见下方逐条列表
00000000 02000000   01000000 02000000   04000000 02000000
16000000 16000000   04000000 15000000   06000000 19000000
06000000 1a000000   06000000 1b000000   02000000 1a000000
02000000 19000000   02000000 1b000000   24000000 19000000
24000000 1a000000   24000000 1b000000   24000000 1e000000
```
（注：上表首行 `06000000 02000000` 属相邻数据，实际 15 条自 `(0,2)` 起，见 §9 的列表）

**IDA 复现步骤**：
```python
idapro.open_database(P, run_auto_analysis=True)
ida_hexrays.decompile(0xE7E5C)          # IsCompatibleQuantType：静态表 + 线性查找
ida_bytes.get_bytes(0x52D34, 0x78)      # 15 条 int32 配对
```

**配对枚举依据**：GE `ge::DataType` 序号
`DT_FLOAT=0 · DT_FLOAT16=1 · DT_INT8=2 · DT_INT16=3 · DT_INT32=4 · DT_INT64=5 ·
 DT_UINT8=6 · DT_UINT16=7 · DT_UINT32=8 · DT_UINT64=9 · DT_DOUBLE=10 · DT_BOOL=11 ·
 DT_STRING=12 · DT_DUAL_SUB_INT8=13 · DT_DUAL_SUB_UINT8=14 · DT_COMPLEX64=15 ·
 DT_COMPLEX128=16 · DT_QINT8=17 · DT_QINT16=18 · DT_QINT32=19 · DT_QUINT8=20 ·
 DT_QUINT16=21 · DT_RESOURCE=22 · DT_STRING_REF=23 · DT_DUAL=24 · DT_VARIANT=25 ·
 DT_INT4=26 · DT_UINT1=27 · DT_INT2=28 · DT_UINT2=29`
（序号 36 未在上述常见枚举中，疑似 `DT_HIFLOAT8` 等新低精度类型 ✓）

**判定语义**：`IsCompatibleQuantType(A,B)` = “(A,B) 是否在白名单中”，
**成对**比较（顺序敏感），不在表中即返回 false ✓

**旁证（同库其它相关符号）**：
`hiai::V100CompiledModel::CheckCompatibility` · `RefreshCompatibilityStatus` ·
`HIAI_HCL_BuiltModel_CheckCompatibility_Impl` · `hiai::EnumShapeCompiledModel::CheckCompatibility` ·
`hiai::MultishapeCompiledModel::CheckCompatibility` ·
`V100ModelConverter::OptimizeCpuClRomSubGraph`（CPUCL 子图优化）·
`UpdateDataType4CPUCLSubGraph` ·
`HIAI_HCL_BuiltModel_CheckCompatibility_Impl`（对外 C 接口）
⇒ CPUCL = **CPU Compute Library**，实现库为 `libcpucl_itf.so` / `libcpucl_rom.so` ✓

## 10. ★★ 找到了 GE 侧 MatMul 的 dtype 白名单（决定"合法量化形态"）★★

**起因**：按 §9 的 `IsCompatibleQuantType` 白名单试 `(DT_FLOAT16, DT_INT8)`（W8A16），仍然失败。
hilog 给出确切原因：

```
E AI_FMK: math_op_infershapes.cpp VerifyMatMulInputsDataType(765)
  ::"Node:Matmul:0 Verify inputs data type:1 and filter data type:2 fail, not support currently."
E AI_FMK: ir_infer_shape_optimizer.cpp RunInferShape(296)::"Infershape for [op:Matmul:0 type:MatMul] failed."
E HIAI_DDK_MSG: hiai_model_runtime_repo.c ModelRuntimeRepo_TryBuild(154)::"no runtime support the Model."
```
（`inputs data type:1` = DT_FLOAT16，`filter data type:2` = DT_INT8）

**逆向 `libgraph.so` 的 `VerifyMatMulInputsDataType`**（反编译）：
它建两个 `set<ge::DataType>` 分别调 `ge::InferUtil::VerifyInputDataType(node, 0/1, set)`：

```
① 输入张量白名单：@unk_2A270，28 字节 = 7 个 DataType
     DT_FLOAT(0) · DT_UINT16(7) · DT_UINT8(6) · DT_INT32(4) · DT_INT8(2) ·
     DT_RESOURCE(22) · DT_FLOAT6_E2M3(36)
   ★ 注意：★没有 DT_FLOAT16(1)★
② 权重白名单：@unk_2A28C，40 字节 = 10 个 DataType
     DT_FLOAT(0) · DT_INT8(2) · DT_RESOURCE(22) · DT_VARIANT(25) · DT_INT4(26) ·
     DT_COMPLEX32(30) · DT_UINT1(27) · DT_UINT16(7) · DT_UINT8(6) · DT_INT32(4)
```

**⇒ 合法的量化 MatMul 只有 `(DT_FLOAT, DT_INT8)`（W8A32）** ✓
—— 与 §9 的 `IsCompatibleQuantType` 白名单第 0 条完全一致，两张表互相印证 ✓

## 11. ★★ W8A32 实测：越过两道检查，卡在 QuantizeOptimizer ★★

`MatMul(x: FP32, w: INT8) -> y: FP32`，w 挂 `SetTensorQuantParams`，实测 hilog：

```
✓ VerifyMatMulInputsDataType 【本次不再报错】 ⇒ dtype 校验通过
✓ CheckCompatibility          【本次不再报错】 ⇒ 量化类型白名单通过
✗ AI_NPUCL: quantize_optimizer.cc QuantizeOptimizer(28)::"QuantizeOptimizer Fail!"
✗ AI_NPUCL: sub_graph_optimizer.cc QuantizeOptimizer(200)::"graph[SubGraph_0] quantize optimizer fail."
✗ AI_FMK: model_optimizer.cpp GraphPreGraphSaveOptimize(320)::"optimizer_presave in cl NPUCL failed !"
⇒ HIAI_DDK_MSG: "no runtime support the Model." → NNRt Build failed
```
（CL 名从 CPUCL 变为 **NPUCL** ⇒ 该图已被分给 NPU 计算库 ✓）

**⇒ 闭环结论**：

```
✓ 白名单合法形态 (DT_FLOAT × DT_INT8) 能过 dtype 校验 + 量化兼容性检查
✗ 但卡在 QuantizeOptimizer —— 它要求 §3 的【四个量化要素】：
    quant_type 属性 · scale/offset 属性 · Quant/Dequant 节点 · ★量化配置 blob★
⇒ 而配置 blob 只能经【扩展配置】送入 ✗（设备侧 hiai foundation 拒绝）
⇒ ★★ 与 §8 的结论闭环：量化形态与配置 blob 两条路在本设备都不通 ★★
```

**新增探针**：`w8a16.c`（4 变体，含 MATMUL 参数要求）· `w8a32.c`（白名单合法形态单测）
**MATMUL 参数要求（NNRt 源码 `ops/matmul_builder.cpp` 原文）**：
`TransposeA/TransposeB` 必须 **OH_NN_BOOL 标量**；`ActivationType` 必须 **OH_NN_INT8 标量** ✓

## 12. ★★ `.ms` 路线的正确组合：converter 的 `quant_type=WEIGHT_QUANT` ★★

**问题**：既然已知合法形态是 `(浮点激活, INT8 权重)`（§10 §11），能否产出这样的 `.ms`？

**排查**：

```
✗ dopt 路径：只有一种组合
  tools_dopt/dopt_tf_py3/dopt/notrain_tensorflow/quant_utils/tf_quant_int8_8_utils.so
  tools_dopt/dopt_tf_py3/dopt/notrain_tensorflow/quant_utils/tf_quant_int8_8_hp.so
  ⇒ 库名即组合：★只有 int8_8★（激活 8bit × 权重 8bit）⇒ 必然落在 GE 白名单之外
  · dopt_onnx_py3 甚至没有 strategy/ 目录（tf/pytorch 版才有）
  · dopt.so 符号：StrategyManager / create_strategy / StrategyImport
    ⇒ strategy 由名字 import 对应策略模块，没有对应模块就无法换组合
✓ converter_lite 自带量化器：模板在 MindSpore Lite 源码树
  mindspore-lite/tools/converter/quantizer/config/
    dynamic_quant.cfg · fixed_bit_weight_quant.cfg · full_quant.cfg · mixed_bit_weight_quant.cfg
```

**★ 关键：`quant_type` 的三个取值（模板原文注释）★**

| 取值 | 含义 | 是否合法 |
|---|---|---|
| ★`WEIGHT_QUANT`★ | ★只量化权重（激活保持浮点）★ | ★✓ 即 W8A16/W8A32，白名单第 0/1 条★ |
| `FULL_QUANT` | 激活与权重都量化 | ✗ int8×int8，白名单之外 |
| `DYNAMIC_QUANT` | 动态量化 | 待测 |

`fixed_bit_weight_quant.cfg` 原文：

```
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16] ...
bit_num=8
min_quant_weight_size=0
min_quant_weight_channel=16
```

**⇒ 结论**：
之前 converter 路线失败的原因之一是用了 `FULL_QUANT`（int8×int8）✗；
改用 ★`quant_type=WEIGHT_QUANT`★ 才能产出 GE/hiai 白名单认可的形态 ✓
（且与 HF 上 "weight-only INT8" gemma-4 模型同构 ✓）

**下一步**：用 `WEIGHT_QUANT` 产 `.ms`，走 MS-Lite + NNRt 实测
（预期可越过 §10 的 dtype 校验与 §9 的兼容性检查；能否过 `QuantizeOptimizer` 待验 ✓）

## 13. ★★ WEIGHT_QUANT 实测：converter 直转 ONNX 的 `.ms` 缺 third-party 标记 ★★

**做法**：用 `converter_lite --fmk=ONNX` + `quant_type=WEIGHT_QUANT` 量化一个 BERT 小模型
（`tiny-test.onnx`：input_ids/attention_mask/token_type_ids → last_hidden_state）。

```
✓ 转换成功：CONVERT RESULT SUCCESS:0
✓ 产出 226,656 字节（输入 ONNX 449,599 ⇒ 压到约一半 ⇒ 权重确实被量化 ✓）
✗ 设备侧 Build -1，hilog：
    E MS_LITE: [nnrt_delegate.cc:237] BuildOfflineModel# ★not third party model★
```

**排查**：给 cfg 补上 `[third_party_model]` 段（按官方示例格式，见
`mindspore-lite/test/ut/test_data/third_party_model.cfg`：`input_names` / `input_dtypes` /
`input_shapes` / `output_names` / `output_dtypes` / `output_shapes` / `extended_parameters`），
填入该 BERT 的真实 I/O 后重转：

```
✓ CONVERT RESULT SUCCESS:0
✗ 产出仍是 ★226,656 字节（与不写该段时完全相同）★
✗ 设备侧仍报 ★not third party model★
```

**结论**：

```
★ `[third_party_model]` 段★只在 `--fmk=THIRDPARTY` 时生效★ ✗
  · 对 `--fmk=ONNX` 直转，无论 cfg 怎么写都不会打上 third-party 标记
    （两次 .ms 字节数完全相同即为证据）
⇒ 要让 `.ms` 带 third-party 标记，★必须经过 OMG★：
    ONNX → OMG → .omc → converter --fmk=THIRDPARTY → .ms → MS-Lite + NNRt ✓
  （这正是本仓库 gemma4 流水线的既有路径 ✓）
⇒ 而量化组合要在 OMG 那一层决定：
  · OMG 的 `--compress_conf` 来自 dopt ⇒ ★只有 int8-8★（不在白名单 ✗）
  · 或先离线把权重量化成 int8、再让 OMG 透传 ⇒ 需要自行实现 ✓
```

**顺带确认的工具位置（开发机）**：
`<MSLITE_BUILD>/tools/converter/converter/converter_lite`
`<MSLITE_SRC_BUILD>/tools/converter/converter_lite/converter_lite`
量化配置模板：`mindspore-lite/tools/converter/quantizer/config/` ✓

**本机无 `onnx` 模块**，读 ONNX 的 I/O 可用手写 protobuf 扫描（`ModelProto.graph`(7) →
`GraphProto.input`(11)/`output`(12) → `ValueInfoProto.name`(1)）✓

## 14. ★★ 走通 OMG 那一步；`--fmk=THIRDPARTY` 需要容器内的 converter ★★

**目标**：产出【带 third-party 标记】且【weight-only int8】的 `.ms`。

**① 手写最小 ONNX（无需 onnx 模块）** ✓

本机无 `onnx`/`torch`/`numpy`，改用直接写 protobuf 字节：

```
ModelProto{ ir_version(1)=8 · graph(7) · opset_import(8) }
GraphProto{ node(1)=MatMul · initializer(5)=w[4,4] float32 · input(11)=x · output(12)=y }
OperatorSetIdProto{ version(2)=13 }     ← ★易错点：version 是字段 2★
```
⇒ 产出 175 字节的 `mm.onnx`（`y[1,4] = MatMul(x[1,4], w[4,4])`）✓
（第一次写成 `field 1 = 13` ⇒ OMG 报
 `onnx_parser.cpp:579 "unsupported opset version 0, need to be in [7, 19)"` ✓）

**② OMG 成功** ✓

```
DDK=$HOME/ddk
LD_LIBRARY_PATH=$DDK/tools/tools_omg/master/lib64:$DDK/tools/platform/kirinx90/lib64
$DDK/tools/tools_omg/omg --model mm.onnx --framework 5 --output mm \
  --input_shape "x:1,4" --out_nodes "y:0" --platform=kirinx90 --target=omc
⇒ "OMG generate offline model success." ✓  产出 mm.omc = 19,145 字节 ✓
```

**③ `converter_lite --fmk=THIRDPARTY` 卡在工具版本** ✗

| converter | 结果 |
|---|---|
| `<MSLITE_BUILD>/tools/converter/converter/converter_lite`（打包版） | `Flags Init failed. Ret: -600`（help 里 `--fmk` 取值无 THIRDPARTY） |
| `<MSLITE_SRC_BUILD>/tools/converter/converter_lite/converter_lite`（源码构建版） | 接受该取值，但内部按 **MSLITE** 处理 ⇒ `converter.cc:1185 "When fmk is set to MSLITE, only support micronization."` ⇒ `Fail to support` |

**根因**：`scripts/model-conversion/gemma4/gemma4_batch.sh` 第 4 行

```
B=${MSLITE_BUILD:-<MSLITE_BUILD>}
                      ^^^ ★容器内路径★（注释原文：容器内路径，可用 MSLITE_BUILD 覆盖）
```
⇒ ★经 OMG 的 `--fmk=THIRDPARTY` 转换必须在【CANN 容器】里做★
（开发机 上这两个 converter 都不是那一版 ✓）

**⇒ 下一步**：在 CANN 容器内（或找到等价版本）跑
`converter_lite --fmk=THIRDPARTY --modelFile=mm.omc --outputFile=wq7_mm --configFile=mm.cfg`
（cfg 用 `quant_type=WEIGHT_QUANT` + `[third_party_model]` 段 ✓）

## 15. ★（标题已更正，结论见 §20）★ THIRDPARTY 链路的完整配方

> ★更正（2026-10）：本节原标题为"int8 首次在 NPU 上跑通"，
> 经 §16–§20 的进一步核查，★该路径产出的模型【不是 int8】★ ✗。
> 准确的结论是：★该链路能产出"设备可加载并 Predict 成功"的 `.ms`（对外接口为 fp16），
> 但不是 int8★ ✓。完整证据链与结论见 §20。以下配方本身仍然有效（用于打通链路）。

**结果**：`Build 0 · Predict -> 0` ✓，且 hilog 证明在 NPU 上执行：

```
W AI_NPUCL: npu_graph_executor_om.cc Init(115)::"load model succ: modelName=default_ndk modelId=64"
W AI_NPUCL: npu_graph_executor_client.cc Init(406)::"client executor id = 65536"
W AI_NPUCL: npu_graph_executor_service_init.cc GraphExecutorInit(129)::"load model finish, pid: 9484, client id: 65536, server id: 828"
E AI_NPUCL: npu_graph_executor_om.cc EnableIfuPrelod(1752)::"smDesc is null, kernelInfo.stubName = ★executor_batchmatmul_cube★"
W hiaiserver/RUNTIME: rpc_request_service.cpp BindCurTidToMidBigCore(55)::"Bind Core Success, tid:9497"
```

**关键结论：合法形态 = `(FP32 激活 × INT8 权重)` = W8A32**
（与 §9 的 `IsCompatibleQuantType` 第 0 条、§10 的 `VerifyMatMulInputsDataType` 两张表一致 ✓；
 纯 int8×int8 两者都不接受 ✗）

### 完整配方（四步）

```sh
# ① 准备 ONNX（W8A32 的目标形态；本仓库 mm.onnx 为 175 字节手写样例：
#    y[1,4] = MatMul(x[1,4], w[4,4])，注意 opset_import 的 version 是字段 2）

# ② OMG ⇒ .omc
DDK=$HOME/ddk
LD_LIBRARY_PATH=$DDK/tools/tools_omg/master/lib64:$DDK/tools/platform/kirinx90/lib64 \
  $DDK/tools/tools_omg/omg --model mm.onnx --framework 5 --output mm \
  --input_shape "x:1,4" --out_nodes "y:0" --platform=kirinx90 --target=omc
#   ⇒ "OMG generate offline model success."  mm.omc

# ③ ★在 厂商转换环境 容器内★ converter（--fmk=THIRDPARTY 只有容器内那版支持）
#    mm.cfg: [common_quant_param] quant_type=WEIGHT_QUANT / bit_num=8 / ...
#            [third_party_model]  input_names/input_dtypes/input_shapes/
#                                 output_names/output_dtypes/output_shapes
copy mm.omc 厂商转换环境:/tmp/ && copy mm.cfg 厂商转换环境:/tmp/
在厂商转换环境中执行 '
  B=<MSLITE_BUILD>
  cd /tmp && $B/tools/converter/converter/converter_lite --fmk=THIRDPARTY \
    --modelFile=mm.omc --outputFile=wq7_mm --configFile=mm.cfg'
#   ⇒ CONVERT RESULT SUCCESS:0   wq7_mm.ms（比 .omc 略大，量化信息已写入）

# ④ 设备：MS-Lite + NNRt（不传扩展配置，量化信息已在模型里）
python3.14 scripts/model-conversion/int8/probe_quant.py wq7_mm.ms - QuantConfigData 60
#   ⇒ ★Build 0 · Predict -> 0 ✓★
```

### 三个必须同时满足的条件（少一个都不行）

| 条件 | 原因 | 错则报 |
|---|---|---|
| ★量化组合 = WEIGHT_QUANT★（激活浮点 × 权重 int8） | GE/hiai 白名单只收 `(FLOAT/FP16, INT8)` | `VerifyMatMulInputsDataType … fail` / `Infershape failed` |
| ★必须带 third-party 标记★ | MS-Lite 的 NNRt delegate 只处理 third-party 模型 | `nnrt_delegate.cc:237 "not third party model"` |
| ★`--fmk=THIRDPARTY` 需容器内 converter★ | 开发机 上两版都不支持该取值 | `Flags Init failed Ret:-600` / `only support micronization` |

**不需要扩展配置** ✓：量化信息随模型一起下发（`AddExtensionConfig` 那条路设备端不支持 ✗，
但那不是必经之路 ✓）

## 17. 用调试器确认"模型被解析成了什么"（方法与实测数据）

**动机**：§15 曾据"体积 + 字节搜索"间接推断该 `.ms` 不是 int8。**间接证据不足以定性** ✗ ——
改为用调试器直接观察解析过程与结果。

**调试姿势**（仅用 `python3.14` 运行探针 ✓）：

```
lldb-server gdbserver <host>:<port> -- <python3.14> <probe_quant.py> <model.ms> - …
断点（符号取自 /system/lib64/platformsdk/libmindspore-lite.so）：
  · mindspore::lite::LiteModel::ConstructModel(char const*, unsigned long, bool)
  · mindspore::lite::LiteModel::PrepareInnerTensors()
  · mindspore::lite::LiteModel::CheckQuantAllInit(flatbuffers::Vector<QuantParam> const*)
  · mindspore::lite::Tensor::Tensor(mindspore::TypeId, std::vector<int>, Format const&, Category)
```

**实测 1：模型缓冲区被确认** ✓

```
LiteModel::ConstructModel 命中，参数：
  x0 = LiteModel this
  x1 = 模型缓冲区地址
  x2 = ★19712★  ← 与 wq7_mm.ms 的字节数完全一致
用 lldb 把 x1 处 19712 字节 dump 出来（memory read --force --binary --outfile …）：
  ★与 wq7_mm.ms 逐字节完全一致★ ✓（说明调试器读到的就是被解析的原始数据）
```

**实测 2：缓冲区头与标记** ✓

```
头 8 字节：24 00 00 00 4d 53 4c 32   ⇒ "★MSL2★"（MS-Lite v2）
可读串：★subgraph_0_third_party★ · ★ThirdPartyModel★ ·
        IMOD · ge_default · x:0 · Node_Output:0 · attr_* · NCHW
⇒ ★该模型确实带 third-party 标记★ ✓（与 Build 0 · Predict 0 相符）
```

**实测 3：张量创建时的 TypeId** ✓

`Tensor::Tensor(TypeId, …)` 命中 5 次，`x1`（TypeId）为：

```
0x2b = 43 (x4)   ·   0x25 = 37 (x1)
```
`TypeId` 枚举（`mindspore/core/include/mindapi/base/type_id.h`，自 0 顺序计数）：

```
30 kNumberTypeBegin · 31 Bool · 32 Int · ★33 Int8★ · 34 Int16 · 35 Int32 · 36 Int64 ·
37 UInt · ★38 UInt8★ · 39 UInt16 · 40 UInt32 · 41 UInt64 · 42 Float ·
★43 Float16★ · 44 Float32 · 45 Float64 · 46 BFloat16 …
⇒ 43 = kNumberTypeFloat16，37 = kNumberTypeUInt ⇒ 本次创建的张量为 float16 / uint ✓
```

**记录边界**（避免再次过度推断 ✗）：

```
· 5 次命中说明的是【这些张量创建时】的类型；权重张量若在别处创建或延后创建则未被覆盖
· CheckQuantAllInit 未命中，可能只是该分支未走到，不能据此断言"模型无量化信息"
· 因此本条【只记录事实与方法】，不对"该 .ms 是否为 int8"下结论
```

## 18. ★★★★★ 用官方 schema 解析 `.ms`：它是【薄包装】，真模型在 `ThirdPartyModel` blob 里 ★★★★★

**方法**：直接用 MS-Lite 源码里的官方 schema 编译一个解析器
（`mindspore-lite/schema/model.fbs` + `model_generated.h` + `flatbuffers` 头）：

```
root_type ★MetaGraph★ · file_identifier ★"MSL2"★（与模型头字节 4d 53 4c 32 一致 ✓）
MetaGraph: name · version · fmkType · inputIndex · outputIndex · mempoolSize ·
           nodes · ★allTensors()★ · ★subGraph()★ · obfuscate · encrypt · obfMetaData · decryptTable
SubGraph: name · inputIndices · outputIndices · nodeIndices · tensorIndices   ← 只有索引
Tensor:   nodeType · ★dataType★ · dims · format · offset · ★data★ · ★quantParams★ ·
          quantClusters · name · enableHuffmanCode · ★weightQuantCompressType★ · externalData
QuantParam: scale · zeroPoint · min · max · narrowRange · ★numBits★ · ★inited★ · varCorr · meanCorr · ★dstDtype★
WeightQuantCompressType: ★NONE=0 · INDEXING=1 · SPARSE=2 · FSE=3 · BITPACKING=4 · FSE_INT=5 · FSE_INFER=6★
```

**解析 `wq7_mm.ms`（19,712 字节）结果**：

```
version = MindSpore Lite 2.7.0     fmkType = 6 (THIRDPARTY)
subGraph[0] name = ★subgraph_0_third_party★   tensors = 3   nodes = 1

allTensors: 3 个
 [0] y                 dataType = 43 (Float16)  dims = [1,4]       data = 0      compress = NONE  quant = (none)
 [1] x                 dataType = 43 (Float16)  dims = [1,4]       data = 0      compress = NONE  quant = (none)
 [2] ★ThirdPartyModel★ dataType = 37 (UInt)     dims = [★19145★]   data = ★19145★ compress = NONE  quant = (none)
```

**★ 关键结论 ★**

```
★ `ThirdPartyModel` 张量是一个【不透明的字节 blob】，长度 ★19145★
  —— 与 `mm.omc` 的字节数【完全一致】✓
⇒ ★该 `.ms` 只是把 `.omc` 原样包了一层★ ✗
  · MS-Lite 自身【不解析】其中的内容，整体作为 data 交给下游（NNRt → hiai）
  · x / y 是 float16，只是对外接口签名
  · `.ms` 层的 quantParams 为空、compress=NONE —— 因为这一层本来就没有量化信息
⇒ ★量化信息与权重都在【`ThirdPartyModel` 那个 blob（即 .omc）里】★
```

**⇒ 修正此前的提问层次**：

```
✗ "`.ms` 里是不是 int8" —— 问错了层（.ms 只是壳）
✓ 正确的问题：「`.omc` 里的权重是什么类型、量化参数是什么」
   —— 需要解析 .omc 才能回答
```

**顺带记录（避免再次误判）**：

```
· WeightQuantCompressType 有 7 种取值 ⇒ 权重可能是位打包/稀疏/熵编码存储，
  ★因此"在模型里搜原始浮点权重的字节"这种判据【本身就是无效的】★
· 本节只记录解析工具与解析结果，不对 int8 是否生效下结论
```

## 19. 解析 `.omc`：容器格式 + 内嵌 protobuf 片段

`.ms` 里的 `ThirdPartyModel` blob 就是 `.omc`（字节数一致 ✓），因此真正要看的是 `.omc`。

**格式探测**：

```
头 8 字节：49 4d 4f 44 00 01 00 00  = "★IMOD★" + 版本/长度字段
整体不是单个 protobuf（protoc --decode_raw 报 "Failed to parse input"；
自写扫描器在偏移 0..200 均无法完整解析到文件尾）
⇒ ★它是容器 + 内嵌的 protobuf 片段★
```

**内嵌片段的 protobuf 结构（可读属性名 + 紧邻的取值字段）**：

```
形如： 0a <len> "<属性名>"  12 02 <字段号> <值>
例：   0a 09 "src_dtype"     12 02 18 00        ⇒ src_dtype = 0
```

**在 `mm.omc` 中读到的 dtype 属性**：

| 偏移 | 属性 | 值 | 解释 |
|---|---|---|---|
| 0x326 | `src_dtype` | 0 | DT_FLOAT |
| 0xb64 | `src_dtype` | 1 | DT_FLOAT16（上下文 `SubGraph_0:0`） |
| 0x34e | `dst_dtype` | 1 | DT_FLOAT16 |
| 0xbdf | `dst_dtype` | 0 | DT_FLOAT |

**⇒ 即该图里的 `Cast` 节点是 FLOAT ⇄ FLOAT16 转换；在这些属性中未见 int8** ✓

**记录边界**（避免再次越界）：

```
· 读到的是 ★Cast 节点的属性★，不是 ★权重张量本身的数据类型★ ✗
· `graphop_weight_offset` 显示权重另有存放位置（需进一步解析容器目录）
· `.omc` 是容器 ⇒ 要完整读出权重类型/量化参数，需要先解出它的目录结构
⇒ 本节只记录格式探测结果与已读到的属性值
```

**环境**（可复现）：目标机上 `protoc` 与 python `protobuf` 均可用 ✓；
但 `.omc` 不是单一 protobuf，故通用解码器不适用，需按容器格式解析 ✓

## 20. ★结论（已按 §21 限定范围）：`--fmk=THIRDPARTY` + `WEIGHT_QUANT` 路径没有产出 int8

> ★范围限定（2026-10）：本节结论**只对 `--fmk=THIRDPARTY` 路径成立**。
> 经 §21 解析原生 `.ms` 发现：★`--fmk=ONNX` + `WEIGHT_QUANT` 确实产出了 int8 权重★ ✓。
> 不要据此推断"本设备无法做 int8" ✗。

**结论**：该路径产出的 `.ms` 中，权重与计算均为 float16/float32，**不存在 int8 量化权重** ✗。

**三条互相独立的证据**：

```
★ 证据 ①：`.ms` 只是薄包装，真模型 = `ThirdPartyModel` blob ★
   · 用 MS-Lite 官方 flatbuffer schema 解析 wq7_mm.ms：
       subGraph[0] = subgraph_0_third_party, fmkType = 6 (THIRDPARTY)
       allTensors = 3 个：
         y  dataType = 43 (Float16)  dims=[1,4]     data=0
         x  dataType = 43 (Float16)  dims=[1,4]     data=0
         ThirdPartyModel  dataType = 37 (UInt)  dims=[19145]  data=19145
       三者 quantParams 均为空、weightQuantCompressType = NONE
   · ThirdPartyModel 的长度 19145 与 mm.omc 的字节数完全一致
     ⇒ MS-Lite 不解析其内容，整体交给下游（NNRt → hiai）
     ⇒ ★要判断量化，必须看 .omc★

★ 证据 ②：`.omc` 内部的 dtype 属性只有浮点 ★
   mm.omc 内嵌 protobuf 片段中读到的全部 dtype 属性：
     src_dtype = 0 (DT_FLOAT)     @0x326
     dst_dtype = 1 (DT_FLOAT16)   @0x34e
     src_dtype = 1 (DT_FLOAT16)   @0xb64  （上下文 SubGraph_0:0）
     dst_dtype = 0 (DT_FLOAT)     @0xbdf
   ⇒ 仅有 FLOAT / FLOAT16，无 int8 ✓

★ 证据 ③：调试器实测张量创建时的类型 ★
   断在 mindspore::lite::Tensor::Tensor(TypeId, std::vector<int>, Format const&, Category)
   命中 5 次，TypeId 为：0x2b = 43 (kNumberTypeFloat16) ×4 · 0x25 = 37 (kNumberTypeUInt) ×1
   TypeId 枚举中 Int8 = 33、UInt8 = 38 ⇒ ★未出现任何 int8/uint8★ ✓
   （另：调试器 dump 出的模型缓冲区与 wq7_mm.ms 逐字节一致，
     确认观察对象无误 ✓）
```

**⇒ 因此对 §15 的更正**：

```
✗ 原表述："int8 首次在 NPU 上跑通"
✓ 更正为：THIRDPARTY 链路能产出「设备可加载并 Predict 成功」的 .ms
          （Build 0 · Predict 0 ✓，对外接口 x/y 为 float16），
          ★但该模型不是 int8★ ✗
```

**⇒ 与既有逆向结论一致（互不矛盾）**：

```
· §9  hiai 的量化类型白名单只收 (FLOAT/FLOAT16, INT8)，不收 int8×int8 ✓
· §10 GE 的 MatMul dtype 白名单：输入无 FLOAT16、合法组合为 (FLOAT, INT8) ✓
· §12 converter 的 quant_type 有 WEIGHT_QUANT / FULL_QUANT / DYNAMIC_QUANT 三种取值 ✓
· 而 `--fmk=ONNX` + WEIGHT_QUANT 路径【确实会压缩体积】（449,599 → 226,656，约 50%）✓
  —— 那一条才是真正做了量化的路径，但它产出的 .ms 不带 third-party 标记 ✗
⇒ 两条路径的需求（真量化 vs third-party 标记）目前仍未同时满足 ✓
```

**未完成的部分**：`.omc` 的容器目录尚未解开，
因此"权重段自身的 dtype 与量化参数"尚未直接读出；
但证据 ①②③ 已足以判定该 `.ms` 不含 int8 ✓

## 21. ★★★★★ 重大更正：原生 `.ms`（`--fmk=ONNX` + `WEIGHT_QUANT`）里【确实有 int8 权重】★★★★★

§20 的结论只对 `--fmk=THIRDPARTY` 成立。用官方 schema 解析**原生** `.ms` 后，
发现另一条路径【确实产出了 int8 量化权重】✓。

**解析 `wq_tiny.ms`（226,656 字节；`--fmk=ONNX` + `quant_type=WEIGHT_QUANT`）**：

```
version = MindSpore Lite 2.7.0     fmkType = 2 (ONNX)
subGraph[0] name = subgraph_0_main_graph      tensors = 446   nodes = 255

带量化参数的权重张量（★关键★）：
 [12] embeddings.position_embeddings.weight    dataType=32  dims=[512,32]   data=★16384★
      quant[{s=0.000333553 zp=29  numBits=★8★} ×32]
 [15] embeddings.token_type_embeddings.weight  dataType=32  dims=[16,32]    data=★512★
      quant[{s=0.000221831 zp=42  numBits=★8★} ×16]
 [19] embeddings.word_embeddings.weight        dataType=32  dims=[1124,32]  data=★35968★
      quant[{s=3.92157e-13 zp=★-128★ numBits=★8★} ×1124]
 [27] onnx::MatMul_669                         dataType=32  dims=[32,32]    data=★1024★
      quant[{s=0.000405379 zp=29  numBits=★8★} ×32]

非量化张量（对照）：
 [24] embeddings.LayerNorm.weight              dataType=★43 (Float16)★
 [25] embeddings.LayerNorm.bias                dataType=★43 (Float16)★
```

**★ 判定 int8 的三条依据 ★**

```
① `quantParams` 非空 ⇒ 每个权重都有 scale / zeroPoint / ★numBits = 8★ ✓
② `data` 字节数 = 【元素个数 × 1 字节】：
     512×32 = 16384 ✓   16×32 = 512 ✓   1124×32 = 35968 ✓   32×32 = 1024 ✓
     ⇒ ★每权重 1 字节 ⇒ int8 量化权重★ ✓
③ 量化张量 dataType = 32（Int），非量化张量为 43（Float16）—— 两类明确区分 ✓
   （`quantParams.dstDtype` = 32，即反量化回 Int 域由下游按 scale/zeroPoint 处理）
```

**⇒ 因此两条路径的准确结论**：

| 路径 | 是否产出 int8 | third-party 标记 |
|---|---|---|
| `--fmk=THIRDPARTY` + `WEIGHT_QUANT` | ✗ 无（fp16 包装，见 §20） | ✓ 有 |
| ★`--fmk=ONNX` + `WEIGHT_QUANT`★ | ★✓ 有真正的 int8 权重★ | ✗ 无 |

**⇒ 之前 §20 的表述范围过宽，现更正为"仅 THIRDPARTY 路径无 int8"** ✓

**⇒ 这也印证了另一条线索**：使用 `BuildOfflineModel`（third-party 专用通道）去加载原生 `.ms`
本就不是它的加载方式（`nnrt_delegate.cc:237 "not third party model"` ✓）；
原生 `.ms` 有它自己的加载路径 ✓

## 22. ★★★★★ 原生 int8 模型的正确加载方式：设备 id 必须指向 NPU ★★★★★

**问题**：原生 int8 `.ms`（`--fmk=ONNX` + `WEIGHT_QUANT`）在 NNRt 下也 `Build -1`，
且 hilog 显示它走了 `BuildOfflineModel` 并报 `not third party model` ✓。

**根因（源码逐层追出）**：

```
NNRTDelegate::Build()
  ├─ is_kirin_online = IsKirinNPUWithOnlineInference()   // 前缀 "NPU_"
  │    ⇒ 真 ⇒ BuildKirinNPUModel()      ← ★原生模型应走这条★
  ├─ is_kirin_offline = IsKirinNPUWithOfflineInference() // 前缀 "HIAI_F"
  │    ⇒ 真 ⇒ BuildOfflineModel()       // 要求 IsCustomModel() = 单节点 Custom
  └─ 都不匹配 ⇒ 直接 kSuccess（不建 kernel）

bool CheckNPUPrefix(prefix) {
  auto device_id = nnrt_device_info_.device_id_;
  NNDeviceGetName(device_id, &device_name);
  return strncmp(prefix, device_name, prefix.size()) == 0;
}

// NNRtDeviceInfo 默认值（inner_context.h:79）
struct NNRtDeviceInfo { size_t device_id_ = ★0★; ... };
```

```
⇒ ★因为 `device_id_` 默认 0，`NNDeviceGetName(0,…)` 拿到的名字不是 "NPU_…"，
  于是 online=false、offline 也判为真路径但 `IsCustomModel()`=false ⇒ 报 not third party model★ ✗
```

**赋值链（device_id 从 API 一路到 delegate）**：

```
converters.cc:185  AddNNRtDevice(inner_context, ★nnrt_device_info->GetDeviceID()★, …)
converters.cc:89   device_info.nnrt_device_info_.device_id_ = device_id;
lite_session.cc:700 delegate_ = make_shared<NNRTDelegate>(iter->device_info_.nnrt_device_info_);
```

**⇒ 正确做法：把 device id 设成 NNRt 枚举出来的 NPU 设备** ✓

设备侧 `libmindspore_lite_ndk.so` 导出的相关 API（权威清单）：

```
★ OH_AI_DeviceInfoSetDeviceId ★        ← 设置设备 id
★ OH_AI_GetDeviceIdFromNNRTDeviceDesc ★ ← 由 NNRt 设备描述符取得 MS-Lite 认的 id
OH_AI_DeviceInfoCreate / Destroy / GetDeviceId / GetDeviceType /
Set/GetEnableFP16 / Set/GetFrequency / Set/GetPerformanceMode / Set/GetPriority /
Set/GetProvider / Set/GetProviderDevice / AddExtension
（共 18 个）
```

**⇒ 因此原生 int8 `.ms` 的加载方式应为**：

```
① OH_NNDevice_GetAllDevicesID(...)            // NNRt 枚举，取 NPU 设备
② OH_AI_GetDeviceIdFromNNRTDeviceDesc(...)    // 转成 MS-Lite 的设备 id
③ OH_AI_DeviceInfoSetDeviceId(dev, id)        // 写入 DeviceInfo
④ OH_AI_ContextAddDeviceInfo(context, dev)    // 之后即走 BuildKirinNPUModel（在线）
```

⇒ 写入正确 id 后 `CheckNPUPrefix("NPU_")` 应匹配，
⇒ 原生模型不再走 `BuildOfflineModel`，`not third party model` 也不会再出现 ✓

## 23. ★★★★★ 里程碑：device_id 指向 NPU 后，原生模型走上【在线通道】★★★★★

按 §22 的思路实现后（挑选 device_name 以 `NPU_` 开头的 NNRt 设备，取其 id 写入 DeviceInfo），
hilog 首次出现 **`BuildKirinNPUModel`** —— 即原生模型不再被当成 third-party ✗。

**正确的加载序列（实测通过 ✓）**：

```python
ctx = OH_AI_ContextCreate()

# ① 枚举 NNRt 设备（注意：返回指针，只有一个出参）
num   = c_size_t(0)
descs = OH_AI_GetAllNNRTDeviceDescs(byref(num))          # ★ NNRTDeviceDesc *  (size_t *num) ★
                                                          #   不是 (NNRTDeviceDesc**, size_t*) ✗
# 实测返回 2 个设备：
#   [0] name = NPU_ohos.boot.hardware.KirinX90_v2_0   id = 5337627887595434492
#   [1] name = HIAI_F                                 id = 8987859593747354028
for i in range(num.value):
    d   = OH_AI_GetElementOfNNRTDeviceDescs(descs, i)
    nm  = OH_AI_GetNameFromNNRTDeviceDesc(d)
    if nm.startswith(b"NPU_"):                            # ★ 选“在线推理”设备 ★
        dev_id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(d)
        break

# ② 写进 DeviceInfo
dev = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT)       # 60
OH_AI_DeviceInfoSetDeviceId(dev, dev_id)                  # ★★ 关键一步 ★★
OH_AI_DestroyAllNNRTDeviceDescs(...)
OH_AI_ContextAddDeviceInfo(ctx, dev)

# ③ 构建模型（★第三个参数必须为 0★；写 1 会 "Read model file failed"）
OH_AI_ModelBuildFromFile(model, model_path, ★0★, ctx)
```

**两处必须注意的细节（都踩过 ✓）**：

| 项 | 错误写法 | 正确写法 | 错误现象 |
|---|---|---|---|
| `OH_AI_GetAllNNRTDeviceDescs` | `(NNRTDeviceDesc**, size_t*)` | ★`(size_t*)` 返回指针★ | 返回乱码、count=0 |
| `OH_AI_ModelBuildFromFile` 第 3 参 | `1` | ★`0`★ | `lite_session.cc:2126 Read model file failed` |
| DeviceInfo | 只 `Create(60)`，不设 id | ★`SetDeviceId(dev, NNRt 设备 id)`★ | `device_id_=0` ⇒ 落到 offline ⇒ `not third party model` |

**改动后的实测日志（关键行）**：

```
E NNRt_HiAIAdapter: BuildImpl from lite graph failed, failed to parse from lite graph.
E NNRt: [NNCompiler] Build failed, fail to build model online.
E NNRt: OH_NNCompilation_Build failed, fail to build compilation.
E MS_LITE: [nnrt_delegate.cc:772] InitNNCompilation# Build NNCompilation failed, ret: 1
E MS_LITE: [nnrt_delegate.cc:308] CreateFullModelKernel# Init NNCompilation failed
E MS_LITE: [nnrt_delegate.cc:220] ★BuildKirinNPUModel★# Create full model kernel failed
```

**⇒ 结论（本步）**：

```
✓ device_id 修正【确实有效】：模型走上 BuildKirinNPUModel（在线通道）
  —— 不再出现 "not third party model" ✓
✗ 新的卡点在更后一层：hiai 适配层解析 LiteGraph 失败
  （`BuildImpl from lite graph failed, failed to parse from lite graph`）
  —— 与 §6 中 QUANT_DTYPE_CAST 的 `Unrecognized node type 113` 属同族问题
⇒ 下一步：查 hiai 适配层解析该 LiteGraph 时具体在哪一步失败
```

## 24. 在线通道的新卡点精确定位：hiai 适配层转换 `/embeddings/Gather` 失败

§23 之后走上 `BuildKirinNPUModel`（在线），新的失败点已在 hilog 中定位到**具体节点**：

```
E NNRt_HiAIAdapter: gather op x_input invalid.
E NNRt_HiAIAdapter: gather op axis param invalid.
E NNRt_HiAIAdapter: Convert failed for node /embeddings/Gather.
E NNRt_HiAIAdapter: Exec Op Convert failed.
E NNRt_HiAIAdapter: Convert CNode to hiai op failed.
E NNRt_HiAIAdapter: BuildImpl from lite graph failed, failed to parse from lite graph.
```

**反编译适配层的 Gather 参数解析（`libhiai_adapter.so`，sub_54594）**：

```c
MindIR_Tensor_GetData(&v13, (*a1)[2], a2);     // 第 3 个输入（axis）的数据
if (v13 == v14) { LOG("gather op axis param invalid"); return 1; }        // ① axis 数据为空

DataType = MindIR_Tensor_GetDataType((*a1)[2], …);
if (DataType == 35)      v7 = *(int64 *)v13;   // Int32
else if (DataType == 34) v7 = *(int32 *)v13;   // Int16
else LOG("Gather op: dataType %d not support.", DataType);                // ② 其它 dtype
*a2 = v7;

MindIR_Tensor_GetDims(&v11, **a1, …);          // 第 1 个输入（x_input）的 dims
if (v11 == v12) { LOG("gather op x_input invalid."); return 1; }          // ③ ★dims 为空★
if (*a2 >= (v12 - v11) / 4) { LOG("gather op axis param invalid, axis = %ld"); return 1; }  // ④ axis 越界
return 0;
```

**⇒ 判定**：

```
★ 失败发生在【③ x_input 的 dims 为空】或【④ axis 越界】★
  —— 都是形状/参数问题，★与 int8 权重无关★ ✓
★ dtype 这一关是过的：Gather 的 axis 支持 Int16(34)/Int32(35)，
  而模型中相关张量正是 34 (Int16) ✓
★ 适配层有专门的 `hiai::mindir::GatherOpConvert` ⇒ 它【认识】Gather，
  只是该校验未通过 ✓
```

**⇒ 下一步**：
定位 `/embeddings/Gather` 这个节点在 LiteGraph 里的实际输入张量
（哪个是 x_input、其 dims 为何为空），以及 axis 的实际取值 ✓

## 25. ★★★ 在线通道失败根因：模型输入为【动态形状】⇒ Shape 派生张量 dims 为空 ★★★

用官方 schema 解析原生 `.ms` 的**节点级**信息（`MetaGraph::nodes()` +
`CNode::inputIndex()/outputIndex()/quantType()`），得到确切数据：

```
节点 [1] /embeddings/Gather              value_type=69  quantType=7
   输入[0] idx=0  /embeddings/Shape        dt=34(Int16)  ★dims=[]★   data=0
   输入[1] idx=3  /embeddings/Constant     dt=34(Int16)  dims=[]     data=4
   输入[2] idx=4  /embeddings/Gather_axis  dt=34(Int16)  dims=[1]    data=4
   输出[0] idx=2  /embeddings/Gather       dt=34(Int16)  dims=[]     data=0

节点 [4] /embeddings/position_embeddings/Gather   （★同类但正常的 Gather★）
   输入[0] idx=12 embeddings.position_embeddings.weight  dt=★32(Int)★ dims=[512,32] data=16384 ★quant★
   输入[1] idx=6  /embeddings/Slice           dt=34  dims=[]
   输入[2] idx=13 …Gather_axis                dt=34  dims=[1]
   输出[0] idx=11 /embeddings/position_embeddings/Gather  dt=43(Float16) dims=[]

节点 [5] token_type_embeddings/Gather  ⇒ 输入[0] = …weight dt=32 ★quant★ ✓
节点 [6] word_embeddings/Gather        ⇒ 输入[0] = …weight dt=32 ★quant★ ✓
```

**⇒ 结论**：

```
✗ 失败【不是】dtype 问题（相关张量都是 Int16(34)/Int(32)，均在适配层接受范围内 ✓）
✗ 失败【不是】int8 权重问题（带 ★quant★ 的兄弟 Gather 节点本身结构正常 ✓）
✓ 失败在 ★节点 [1] 的 x_input 是 `/embeddings/Shape`★，而它的 ★dims 为空★ ✗
  ⇒ 适配层 `MindIR_Tensor_GetDims(x_input)` 返回空 ⇒ 立即报 "gather op x_input invalid." ✓
✓ 而 Shape 张量没有静态 dims 的根因：★模型输入 `input_ids` 的 dims = [-1,-1]（动态形状）★
```

**⇒ 修法：把输入固定成静态 shape** ✓

```
converter_lite --fmk=ONNX --modelFile=<onnx> --outputFile=<out> \
  --configFile=<WEIGHT_QUANT cfg> \
  --inputShape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128"
```
（另：`quantType=7` 出现在所有 Gather 节点上，含义待查，但不影响上述判定）

## 26. ★★★★★ 成功：静态形状 + 在线通道 ⇒ Build 0，模型在 NPU 上加载成功 ★★★★★

**修法（两步，都已验证）**：

```
① 转换时固定输入形状（否则 hiai 适配层 Gather 校验失败，见 §25）：
   converter_lite --fmk=ONNX --modelFile=<onnx> --outputFile=<out> \
     --configFile=<WEIGHT_QUANT cfg> \
     --inputShape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128"
   产出 181,144 字节（动态形状版是 226,656 —— 静态更小，图也更紧凑：219 张量/130 节点 对 446/255）

② 加载时必须让 device_id 指向 NPU（否则落到 offline 分支，见 §22/§23）：
   OH_AI_GetAllNNRTDeviceDescs → 选 name 以 "NPU_" 开头的设备 →
   OH_AI_GetDeviceIdFromNNRTDeviceDesc → OH_AI_DeviceInfoSetDeviceId →
   OH_AI_ContextAddDeviceInfo → OH_AI_ModelBuildFromFile(model, path, 0, ctx)
```

**实测结果**：

```
★ OH_AI_ModelBuildFromFile → 0（成功）★

hilog（NPU 侧）：
  W AI_NPUCL: npu_graph_executor_om.cc Init(115)::"load model succ: modelName=default_test modelId=53"
  W AI_NPUCL: npu_graph_executor_service_init.cc GraphExecutorInit(129)
      ::"load model finish, pid: 22659, client id: 65536, server id: 838"
  E AI_NPUCL: npu_graph_executor_om.cc EnableIfuPrelod(1752)::"smDesc is null, kernelInfo.stubName = <内核名>"
     ⇒ 报出的 NPU 内核（部分）：
        executor_batchmatmul_cube_cutm
        executor_batchmatmul_cube_two_tensor_cutk_perf
        executor_real_div_scalar_half_nd
        executor_add_ch_broadcast_half_nd
        executor_softmax_nd_w_tiling0_fp16
        executor_permute_0213_single_col_loop_twobyte
        executor_layernorm_last_dim_nd
        executor_gelu_half_nd
        executor_add_eltwise_half_nd
```

**模型结构复核**（官方 schema）：

```
input_ids            dt=34(Int16)  dims=★[1,128]★   （静态 ✓）
embeddings.*.weight  dt=★32(Int)★  dims=[…]  data=元素数×1  ★quant[{scale, zp, bits=8}]★ ✓
/embeddings/*/Gather 输出 dt=43(Float16) dims=★[1,128,32]★ （静态 ✓）
```

**⇒ 结论**：

```
✓ Build 成功（rc=0）
✓ 模型在 NPU 上加载成功（npu_graph_executor: load model succ）
✓ 图内含 ★真 int8 量化权重★（dt=Int + numBits=8 + 每权重 1 字节）
△ 计算内核名多为 *_half_* / *_fp16 ⇒ 计算以 fp16 为主，权重为 int8
  ⇒ 属 ★weight-only 量化（W8A16/W8A32）★ 形态，与 §9/§10 白名单一致
⇒ 待完成：推理（Predict）与输出核对

## 27. ★★★ 两类模型的区分与设备选择：实测四组合矩阵（纠正 §22/§23 的说法）★★★

**问题**：能否区分 third-party 与原生模型，并各用对应的加载方式？

**判据（确定性 ✓）**：

```
third-party 模型：`.ms` 中【恰好 1 个节点，且该节点类型为 Custom】
                  —— 即 NNRTDelegate::IsCustomModel() 的判据
                  也等价于：fmkType = 6 (THIRDPARTY) / 存在 ThirdPartyModel 张量
原生模型：        普通多节点图；fmkType = 2 (ONNX) 等
```

**实测四组合矩阵**（同一设备，逐个单独运行 ✓）：

| 模型 | 设备前缀 | Build | Predict | NPU 侧模型名 |
|---|---|---|---|---|
| third-party `wq7_mm.ms` | `HIAI_F` | ★0★ | ★0★ | `default_ndk` |
| third-party `wq7_mm.ms` | `NPU_` | ★0★ | ★0★ | `default_ge_default` |
| 原生 `wq_static.ms` | `NPU_` | ★0★ | ★0★ | `default_test` |
| 原生 `wq_static.ms` | `HIAI_F` | ✗ -1 | ✗ -2 | — |

**⇒ 关键结论（与 §22/§23 的表述不同，需以此为准）**：

```
✗ "两类模型必须各用对应设备" —— 不准确
✓ 实际情况：
   · ★原生模型【只能】用 "NPU_"（在线）★；用 "HIAI_F" 必然失败（not third party model）
   · ★third-party 模型【两种都能用】★：
       "HIAI_F" ⇒ 走 BuildOfflineModel，NPU 侧模型名 default_ndk
       "NPU_"   ⇒ 走 BuildKirinNPUModel，NPU 侧模型名 default_ge_default
                  （InitNNCompilation 打印 "current device name prefix is not HIAI_F"，
                    随即按在线方式构图）
⇒ ★★ 因此最简通用的做法是：一律选 "NPU_"（在线通道）—— 对两类模型都成立 ★★
   "HIAI_F" 只对 third-party 有效，是它能用的子集
```

**⇒ 实用分辨办法**（需要时可用）：

```
· 解析 `.ms`：节点数 == 1 且类型为 Custom ⇒ third-party
· 或读 fmkType：6 = THIRDPARTY · 2 = ONNX（原生）
· 或看 NPU 侧模型名：default_ndk（离线）· default_ge_default / default_test（在线）
```

## 28. ★★★★★ gemma4 段模型以 int8 权重在 NPU 上跑通 ★★★★★

**输入**：`g4seg0/seg.onnx`（552 MB，fp32；I/O 已是静态形状：
`hidden[1,4,1536]`、`mask3[1,4,4]`、`cos_sl[4,256]`、`sin_sl[4,256]`、`per_layer_0..3[1,4,256]` ⇒ `hidden_out[1,4,1536]`）

**配置**（两段都要 ✓，缺 `[third_party_model]` 会报 `Parse third party param failed`）：

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
bit_num=8
min_quant_weight_size=0
min_quant_weight_channel=1

[third_party_model]              ; 与 c.cfg 相同的 I/O 描述
input_names=hidden;mask3;cos_sl;sin_sl;per_layer_0;per_layer_1;per_layer_2;per_layer_3
input_dtypes=float32;float32;float32;float32;float32;float32;float32;float32
input_shapes=1,4,1536;1,4,4;4,256;4,256;1,4,256;1,4,256;1,4,256;1,4,256
output_names=hidden_out
output_dtypes=float32
output_shapes=1,4,1536
```

**转换**（`--fmk=ONNX` + 上述配置）：

```
CONVERT RESULT SUCCESS:0     产出 147,128,016 字节（147 MB）
对照：同一段的 THIRDPARTY/fp16 版 .ms 为 579,566,040 字节（579 MB）⇒ 缩小 ★3.9×★
```

**官方 schema 验证（int8 验收 ✓）**：

```
fmkType = 2 (ONNX)   655 个张量 / 394 个节点
★ 37 个张量带 quantParams（numBits = 8）★

权重形态（每权重 1 字节 ⇒ 真 int8）：
  onnx::MatMul_870   dt=32(Int)  dims=[1536,256]   data=393216  = 1536×256×1  quant[bits=8]
  onnx::MatMul_852   dt=32(Int)  dims=[1536,2048]  data=3145728 = 1536×2048×1 quant[bits=8]
  onnx::MatMul_859   dt=32(Int)  dims=[1536,256]   data=393216  quant[bits=8]
  onnx::MatMul_858   dt=32(Int)  dims=[256,256]    data=361     ★compress=SPARSE★ quant[bits=8]
⇒ 另发现部分权重带 ★SPARSE 稀疏压缩★（压缩类型见 §18 的 7 种取值）
```

**设备实测（一次一个推理进程 ✓）**：

```
OH_AI_ModelBuildFromFile → ★0★
 输入数 = 8：24576 + 64 + 4096×6 字节（与模型 I/O 完全对应 ✓）
OH_AI_ModelPredict       → ★0★
 输出 1 个：24576 字节（1×4×1536×4，fp32 ✓）
```

**⇒ 结论**：

```
✓ gemma4 段模型以 int8 权重（weight-only）在 NPU 上 Build 0 · Predict 0
✓ 权重量化确实发生（每权重 1 字节 + numBits=8，共 37 个量化张量）
✓ 使用 §27 的结论：一律选 "NPU_"（在线通道）
✓ 关键前提：输入形状必须是静态的（该段 ONNX 本身就是静态 ✓）
```

## 29. ★★★★★ gemma4 全部 9 段以 int8 权重在 NPU 上跑通（goal 验收完成）★★★★★

**配方（对每个段相同）**：

```
① 用现成的段 ONNX（`g4segN/seg.onnx`，I/O 已是静态形状）
② 配置两段都要：
     [common_quant_param]  quant_type=WEIGHT_QUANT · bit_num=8
                           min_quant_weight_size=0 · min_quant_weight_channel=1
     [third_party_model]   该段的 I/O 描述（取自导出的 c.cfg）
   —— 缺 [third_party_model] 会报 Parse third party param failed
③ converter_lite --fmk=ONNX --modelFile=seg.onnx --outputFile=<out> --configFile=<cfg>
④ 加载：device 选 "NPU_"（在线通道，见 §27）⇒ OH_AI_ModelBuildFromFile(..., 0, ctx)
```

**转换结果（9/9 成功）**：

| 段 | ONNX | int8 `.ms` | 缩小 |
|---|---|---|---|
| seg0 | 552 MB | ★140 MB★ | 3.9× |
| seg4 | 580 MB | ★147 MB★ | 3.9× |
| seg8 | 580 MB | ★147 MB★ | 3.9× |
| seg12 | 685 MB | ★173 MB★ | 4.0× |
| seg16 | 997 MB | ★252 MB★ | 4.0× |
| seg20 | 972 MB | ★246 MB★ | 4.0× |
| seg24 | 997 MB | ★252 MB★ | 4.0× |
| seg28 | 997 MB | ★252 MB★ | 4.0× |
| seg32 | 754 MB | ★191 MB★ | 3.9× |

**官方 schema 验收（每段都含真 int8 权重 ✓）**：

| 段 | 张量 | 节点 | 量化张量 | 例：权重张量（dataType=32 Int，每权重 1 字节） |
|---|---|---|---|---|
| seg0 | 655 | 394 | ★37★ | [1536,256] data=393216 = 1536×256×1 |
| seg4 | 659 | 396 | ★38★ | [1536,512] data=786432 = 1536×512×1 |
| seg8 | 660 | 396 | ★38★ | [1536,256] data=393216 |
| seg12 | 627 | 377 | ★36★ | [1536,256] data=393216 |
| seg16 | 534 | 320 | ★30★ | [1536,2048] data=3145728 |
| seg20 | 526 | 318 | ★29★ | [1536,2048] data=3145728 |
| seg24 | 534 | 320 | ★30★ | [1536,4096] data=6291456 = 1536×4096×1 |
| seg28 | 534 | 320 | ★30★ | [1536,2048] data=3145728 |
| seg32 | 405 | 241 | ★23★ | [1536,2048] data=3145728 |

（另有 SPARSE 稀疏压缩张量，每段 1~2 个）

**设备实测（逐个单独运行，一次一个推理进程 ✓）**：

```
9/9 全部：OH_AI_ModelBuildFromFile → 0 且 OH_AI_ModelPredict → 0
seg0 详情：输入 8 个（24576 + 64 + 4096×6 字节，与模型 I/O 一致）
           输出 1 个 24576 字节（1×4×1536×4 fp32）
NPU 侧 hilog：npu_graph_executor_om.cc "load model succ: modelName=… modelId=…"
              GraphExecutorInit "load model finish, pid, client id: 65536, server id: 838"
```

**⇒ 结论**：

```
✓ gemma4 全部 9 段均以 int8 权重（weight-only）在设备上 Build 0 · Predict 0
✓ 每段都经官方 schema 确认含真 int8 量化权重（dataType=Int + numBits=8 + 每权重 1 字节）
✓ 加载一律走 "NPU_"（在线通道），不需要为模型类型切换设备
△ 已知待办：hilog 有 88 次 "… is not supported in npucl"（BatchMatMul/ReduceMean 等），
  说明相当一部分算子未落在 NPU 上，需另行核实其实际执行者
```

## 30. ★★★★★ 更粗的切法实验：段 0-11（12 层）跑通；含 KV 槽的段卡在适配层 Set Data ★★★★★

**目的**：验证"OMG 退出新路线后，是否还能用更少的段"（旧 §2 把「OMG 单张量 ≤ INT_MAX」
列为"必须分段"的头号理由）。

**做法**：用 `--fmk=ONNX + WEIGHT_QUANT`（不带 OMG）导出并按 12/12/11 层切三段。

**结果**：

| 段 | 层数 | KV 模式 | ONNX | 转换 | int8 `.ms` | 量化张量 | 设备 Build/Predict |
|---|---|---|---|---|---|---|---|
| 0-11 | 12 | none | 1.79 GB | ★SUCCESS★ | 435 MB | ★110★ | ★0 / 0 ✓★ |
| 12-23 | 12 | ★out★（store+shared） | 2.78 GB | ★SUCCESS★ | 672 MB | ★92★ | ✗ -1 / -2 |
| 24-34 | 11 | ★in★（shared） | 2.88 GB | ★SUCCESS★ | 696 MB | ★79★ | ✗ -1 / -2 |

**结论 1：OMG 的 INT_MAX 限制确实随 OMG 一起消失** ✓

```
旧理由「OMG 单张量 ≤ INT_MAX」针对的是 `embed_tokens_per_layer` = [262144, 8960] = 2.35e9 元素 ✗
但那张量【根本不在图里】—— 它是【输入】`per_layer_N`，由主机侧切片后喂进来 ✓
⇒ 所以它从未进入 OMG 或 converter 的图 ⇒ 该限制对本路线不成立 ✓
⇒ 实测印证：12 层/段（图 1.79 GB）在三段里都【转换成功】✓
```

**结论 2：但含 KV 共享槽的段在设备侧失败，原因【不是】旧文档担忧的"特定 KV"** ✗

```
失败链（hilog）：
  ① nnrt_delegate.cc:765 InitNNCompilation# hiai_foundation is not nullptr,
     current device name prefix is not HIAI_F        ⇒ ★确实进了在线通道★ ✓
  ② ★NNRt_HiAIAdapter: [nodict]★★Set Data Fail.★★★  ⇒ ★★就是这里★★
  ③ BuildImpl from lite graph failed, failed to parse from lite graph.
  ④ nnrt_delegate.cc:772 InitNNCompilation# Build NNCompilation failed, ret: 1
  ⑤ BuildKirinNPUModel# Create full model kernel failed

对照成功段（g4s0_wq.ms，4 层）：没有 Set Data Fail；两边都有 libhiai_ir_infershape 加载失败
（该库在系统里确实不存在，errno=2）⇒ ★与本失败无关★ ✓
```

**反编译该报错点（`libhiai_adapter.so` sub_67CBC）**：

```c
v7 = *(int *)(a4 + 48);      // isMerged
v8 = *(u64 *)(a4 + 24);      // offset
sz = *(u64 *)(a4 + 56);      // size
if (v7 == 2) { rc = memcpy_s(baseAddr + v8, ★bufSize - v8★, src, sz); ge::Tensor::SetData(…); if (rc) LOG("Set Data Fail."); }
if (v7 == 1) { if (memcpy_s(baseAddr + v8, bufSize - v8, src, sz)) LOG("Set Data Fail."); }
if (v7 == 0) { ge::Tensor::SetData(…); }        // 不拷贝
else LOG("Parameter isMerged is invalid.");
```

```
⇒ `Set Data Fail` 的真实含义：★`memcpy_s` 失败★
  —— 即「目标缓冲剩余空间（bufSize − offset）不足以容纳 size」或参数非法 ✓
⇒ 这是 ★hiai 适配层自身的缓冲/偏移计算★ 与 KV 共享槽（输入/输出同名、in-place）形态的冲突 ✗
  ★不是★ "OMG 高层数只吃特定 KV" 那件事 ✗（那条在新路线下没被触发 ✓）
```

**⇒ 净结论**：

```
✓ 更粗的切法【部分可行】：不含 KV 共享槽的段（0-11，12 层）完全跑通（int8 · Build 0 · Predict 0）
✗ 含 KV 共享槽的段（12-23 out / 24-34 in）在 hiai 适配层 Set Data 处失败 ✗
   ⇒ 若要合并这些段，需要先解决适配层的缓冲/偏移与 in-place 语义冲突
   ⇒ 可行方向：把 KV 槽改成显式独立输入/输出（不让同名张量同时进出一张图）
★ 附：>2 GB 的 ONNX 导出会切成 external data（壳 + 数百个数据文件），
  送进转换环境时必须【整目录一起送】，只送 seg.onnx 会失败 ✗（本次踩过 ✓）
```

## 31. ★★★★★ 更粗切法的真正瓶颈：适配层的【模型大小】上限（≈435~672 MB 之间），与 KV 无关 ★★★★★

§30 猜测含 KV 槽的段失败是"输入输出同名别名"所致。**该猜测被源码与实测同时否掉** ✗：

```
★ gemma4_export_seg.py 本来就规避了别名：
    · 第 38 行注释：「KV 槽只在需要的段出现（避免"输入即输出"的别名，OMG 会报 cannot find output tensor）」
    · 第 83 行：MODE=="out" 时输出名是 ★sk_out/sv_out/fk_out/fv_out★（与输入名不同 ✓）
★ 实测也印证：g4seg12（4 层 · 173 MB）输出有 sk_out/sv_out/fk_out/fv_out（★它确实写槽✓★）
  而它 Build 0 · Predict 0 ✓；seg16/20/24/28/32（读槽）也全部成功 ✓
```

**决定性对照（同一套配方，仅切法不同）**：

| 段 | `.ms` | 写槽(13/14) | 读槽(≥15) | KV 模式 | 设备 Build/Predict |
|---|---|---|---|---|---|
| seg0 | 140 MB | no | no | none | ★0 / 0★ |
| seg4 / seg8 | 147 MB | no | no | none | ★0 / 0★ |
| **seg12** | **173 MB** | ★yes★ | ★yes★ | **out** | ★**0 / 0**★ |
| seg16 / seg20 | 252 / 246 MB | no | yes | in | ★0 / 0★ |
| seg24 / seg28 | 252 / 252 MB | no | yes | in | ★0 / 0★ |
| seg32 | 191 MB | no | yes | in | ★0 / 0★ |
| **L0-11** | **435 MB** | no | no | none | ★**0 / 0**★ |
| L12-23 | **672 MB** | yes | yes | out | ✗ -1 / -2 |
| L24-34 | **696 MB** | no | yes | in | ✗ -1 / -2 |
| L0-23 | **1107 MB** | yes | yes | out | ✗ -1 / -2 |

**⇒ 结论**：

```
✓ 分界线【完全】由模型大小决定：≤435 MB 全过（含写槽/读槽/无槽三种形态），≥672 MB 全不过
✗ 与 KV 共享槽（写槽/读槽/别名）【无关】—— 这是 §30 猜测的否定
⇒ ★适配层存在一个约 (435, 672] MB 的【模型大小上限】★，
  失败点即 §30 反编译出的 `Set Data Fail`
  = `memcpy_s(baseAddr + offset, bufSize - offset, src, size)` 失败
```

**⇒ 对"能否用更少的段"的最终回答**：

```
✗ 旧文档 §2 的头号理由（OMG 单张量 ≤ INT_MAX）在新路线下【不成立】（见 §30 结论 1）✓
✗ 旧文档 §5 的"OMG 高层数只吃特定 KV"也未出现（见 §30 结论 2）✓
✓ 但出现了【新的、与 OMG 无关的】限制：适配层的模型大小上限（≈435~672 MB）✗
⇒ 这个上限【等价地】取代了原来的分段压力：
  · 9 段（每段 4 层，140~264 MB）在限内 ✓ —— 现役可用
  · 3 段（每段 11~12 层，672~696 MB）超限 ✗
  · 每段层数应控制在【约 7 层以内（≈300 MB 以下）】比较安全 ✓
★ 结论：可以比 9 段少，但不能少到 3 段；要确定精确边界需在 435~672 MB 之间做二分
```

**附（顺带纠正一处我自己的操作失误）**：

```
第一次 12 层实验的输出名用了"层数"当键（c12 = 12 层 ✗），
后续批量的输出名用了"起始层号"当键（c${START} = c12 = 起始 12 ✗）⇒ ★两者撞名★ ✗
容器 cp 到主机时把前者【覆盖】了 ✓（不可恢复，但可 1 分钟重导重转 ✓）
现改为按层区间命名：★L0-11_wq.ms / L0-23_wq.ms★ ✓
（重导 L0-11 得到 455,682,136 字节、110 个量化张量，与首次【逐位一致】⇒ 可复现 ✓）
```

---

## 32. ★★★★★ 上限的真正机制（调试器实测 + 直接探测）：适配层为整段权重声明约 4×模型大小的缓冲，securec `memcpy_s` 在 destMax ≥ 2 GiB 时一律返回 ERANGE ★★★★★

§31 用"模型大小"括出了 `(435, 672] MB` 的**经验**边界，但没给机制。本节把机制**测出来**（不是推的）：

```
★真正限制不是"模型大小"这个数字，而是适配层的 32 位缓冲上限：
  bufSize(适配层为整段权重一次性声明的缓冲) ≈ 3.94 × .ms 文件大小
  而 securec 的 memcpy_s 规定 destMax > 0x7fffffff(2 GiB - 1) 就【直接拒绝】
  ⇒ .ms 文件 ≤ ~520 MiB 才可能过；按层数说就是【每段约 ≤ 7 层】★
```

### 32.1 现场（lldb 附加到真实失败的进程）

```
失败样本：L0-23_wq.ms（1,161,204,624 B，含真 int8 权重）
断点：libhiai_adapter.so 基址 + 0x67d68
      （即失败日志 "Set Data Fail." 所在函数 1181/sub_67CBC 里的 memcpy_s 调用点）
```

调用点反汇编：

```
<+168>: add x0, x0, x8      ; dest    = 缓冲基址 + offset
<+172>: sub x1, x1, x8      ; destMax = bufSize - offset        ← 断在这
<+176>: mov x2, x19         ; src
<+180>: mov x3, x21         ; count
<+184>: bl  memcpy_s
<+188>: mov w22, w0         ; 保存返回码
<+208>: cbz w22, <+252>     ; 返回 0 才算成功
<+212>: ...                 ; 否则走 "Set Data Fail."
```

断点处寄存器（第一次命中，就是失败的那次）：

```
x0  = 0x59d5201b40                     dest
x1  = 0x110bfc874 = 4,575,984,244      ★destMax ≈ 4.26 GiB★
x8  = 0                                offset
x19 = 0x568dce3478                     src
x21 = 4                                count —— ★只有 4 字节★
```

再在失败分支（`基址+0x67d90`）下断，命中的返回码：

```
w22 = 0x22 = 34 = ERANGE
```

⇒ **失败与"正在拷贝的那个张量"毫无关系**：`memcpy_s` 在动任何字节之前，就先否决了**声明的 `destMax`**。

### 32.2 这块缓冲是谁定的

```
调用链：HIAIDevice::BuildLiteGraph（libneural_network_runtime_ext.so）
        → 适配层 sub_2579 → sub_2574 → sub_2019 → sub_1197 → sub_1182 → sub_1181（memcpy 处）
```

`sub_2019` 里的关键三条：

```
ldr  x1, [x20, #0x108]        ; ★size = *(x20+0x108)：一个【预先算好】的字段★
ge::Buffer::Resize(size)
ge::Buffer::MutableData()     ; → base
循环遍历张量（stride 0x40）：
    sub_1197(name, base, *(x20+0x108), tensor)      ; bufSize 原样往下传
```

现场读该结构：`*(x20+0x108) = 0x110bfc874`，紧邻的 `*(x20+0x100) = 0xe0c`（3596，形如条目数）。
⇒ **整段模型的权重被塞进【一整块】缓冲，一次 `Resize` 分配**；每个张量只是在这块缓冲里按 `offset` 续着写。

### 32.3 为什么是 2 GiB：直接把阈值的边界二分出来

`memcpy_s` 可以从 `libhiai_adapter.so` 的依赖链上 dlsym 到，所以不必猜实现——直接二分它的上限（返回 `34` 视为被拒）：

```python
import ctypes
h = ctypes.CDLL("<libhiai_adapter.so 的绝对路径>")   # dlsym 会沿依赖链找到 memcpy_s
f = h.memcpy_s; f.restype = ctypes.c_int
f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
dst = ctypes.create_string_buffer(64)
src = ctypes.create_string_buffer(b"abcd", 16)
for n in (0x7ffffffe, 0x7fffffff, 0x80000000, 0x110bfc874):
    print(hex(n), f(dst, n, src, 4))
```

实测输出：

```
destMax = 0x7fffffff (2,147,483,647)   -> 0    （接受）
destMax = 0x80000000 (2,147,483,648)   -> 34   （ERANGE）
二分结果：最大可接受 = 0x7fffffff，第一个被拒 = 0x80000000
```

⇒ **阈值正好是 2³¹ − 1**，即 securec `memcpy_s` 的 `destMax > SECUREC_MEM_MAX_LEN(0x7fffffff)` 检查。
这是**32 位有符号数的硬上限**，与 RAM / NPU / 显存 / KV **统统无关**（也再次印证 §31 的"与 KV 无关"）。

### 32.4 换算成工程规则

```
实测比例（单点）：bufSize / 文件大小 = 4,575,984,244 / 1,161,204,624 = 3.9407
⇒ 适配层按【约 4 字节/元素】给权重计数
   —— ★int8 的 1 字节存储【并不能】减少它声明的缓冲★（花在量化上的存储收益在这条路径上没了）
⇒ 能过的文件大小上界 ≈ 0x7fffffff / 3.9407 ≈ 545,000,000 B ≈ 520 MiB
```

与 §31 的实测完全对得上：

| 段 | `.ms` 大小 | 外推 bufSize | 设备实测 |
|---|---|---|---|
| L0-11 | 455,682,136 B | ≈ 1.68 GiB（限内 ✓） | ★Build 0 · Predict 0★ |
| L12-23 | 705,524,856 B | ≈ 2.59 GiB（超限 ✗） | ✗ -1 / -2 |
| L0-23 | 1,161,204,624 B | 4.26 GiB（★就是实测字段值★） | ✗ -1 / -2 |

**结论（可直接当规则用）**：

```
· 每段 .ms 控制在 ★≤ 300 MB★（≈ ≤ 7 层）—— 与 §31 的建议一致，安全侧留足余量
· 理论上界 ≈ 520 MiB，但别贴着走（比例目前是单点外推）
· 这个上限【无法从外部绕过】：系统库只读、实现专有
  ⇒ 唯一手段仍然是分段；"能不能用更少的段"的答案就是"受这条 32 位上限约束"
```

### 32.5 仍未验证的部分（如实标注，别当结论用）

```
✗ 精确公式 "bufSize = 4 × 元素数"【未验证】：只验证了 (a) 单点比例 3.94、
  (b) 边界落在 (455 MB, 705 MB]。第二点（如 g4s0_wq.ms 147 MB，外推 ≈ 580 MB）还没测
✗ "为什么是 4 字节/元素"未知（GE 图按 fp32 建 desc？还是别的途径）
  ⇒ 这一条决定了【用 fp16 导出 ONNX 能否把上界抬高一倍】——★值得单独一试★
```

### 32.6 复现步骤

**前置条件（漏了就白折腾一小时）**：`LD_LIBRARY_PATH` 必须含 `ndk` **和** `platformsdk` 两个目录，
否则加载 `libmindspore_lite_ndk.so` 会在 musl 的 `dlopen` 里★段错误★ ✗，看起来像"环境/ABI 坏了" ✓
（机制与实测见 `docs/gemma4-on-npu.md` §8.6；`launcher.py` 的 `ENGINE_LIB_DIRS` 已固化这条 ✓）。

```
export LD_LIBRARY_PATH=/system/lib64/ndk:/system/lib64/platformsdk
# ★注意：探测脚本【不需要】调试器，普通进程直接跑即可★（过去"必须挂 gdbserver"的印象是缺上面那行导致的 ✓）

# 终端 A：前台起调试服务（阻塞，别用 nohup/后台）
LD_LIBRARY_PATH=/system/lib64/ndk:/system/lib64/platformsdk \
  huawei-debug-lldb-server gdbserver 127.0.0.1:40021 -- <python3.14> <加载 .ms 的临时探测脚本> 60
# 终端 B：
lldb -o "gdb-remote 127.0.0.1:40021"
(lldb) image list -o -f libhiai_adapter.so     # 取基址，例如 0x...6c0000
(lldb) breakpoint set -a <基址+0x67d68>        # memcpy_s 调用点
(lldb) continue
(lldb) register read x0 x1 x8 x19 x20 x21      # ★x1 = bufSize★
(lldb) frame select 3 ; register read x20      # 缓冲大小字段所在结构
(lldb) memory read -s 8 -f x -c 12 <x20+0xf0>  # +0x108 = bufSize
(lldb) breakpoint set -a <基址+0x67d90>        # 失败分支
(lldb) continue ; register read w22            # ★34 = ERANGE★
```

（加载脚本是临时探针，不入库；每次设备侧加载都用同一个 Python 解释器，见 §22。）

### 32.7 ★这块缓冲属于哪类内存？（实测：普通进程堆，不是 NPU 可见内存）+ "是否禁止单次分配 >2 GB" ★

三个问题分开回答，都有实测：

**① 它是什么**

```
适配层 ge::Buffer::Resize(*(x20+0x108))  →  ge::Buffer::MutableData()  →  base
= ★GE 的主机侧缓冲★：把整段模型的所有权重【摊平成一块连续内存】，
  源数据是 .ms 文件内容在进程内的副本（MindIR_Tensor_GetData 给的指针）
它【不是】NPU 可见内存 / 设备内存 —— 交给设备是【之后】HIAI 那一步的事 ✓
```

**② 普通内存 vs NPU 可见内存 ⇒ 实测是【普通进程内存】✓**

在进程内部轮询 `/proc/self/maps`，记录全过程出现过的 ≥256 MiB 映射：

```
005d99001000-005f43801000  6824.0 MiB rw-p [anon:native_heap:jemalloc]
005f44801000-0060c4801000  6144.0 MiB rw-p [anon:native_heap:jemalloc]
005bf9401000-005cf2e01000  3994.0 MiB rw-p [anon:native_heap:jemalloc]
005ac5664000-005b45389000  2045.1 MiB ---p [anon:cfi_shadow:musl]
……（全部大映射的名字只有 native_heap:jemalloc 与 cfi_shadow:musl 两种）
```

⇒ **一条设备/驱动映射都没有**（没有 `dma_heap` / `ion` / `dma-buf` / `/dev/*`）✓
结合 lldb 现场：`dest = 0x59d5201b40`、`src = 0x568dce3478` 都落在这些 jemalloc 匿名区间内 ✓
⇒ **结论：普通匿名内存（jemalloc 堆）**，NPU 可见/设备内存是后面另一步 ✓

**③ 华为禁止单次分配 2 GB 以上内存吗？⇒ 不禁 ✗**

```
· Resize(0x110bfc874 = 4.26 GiB) 本身【成功了】✓：
  MutableData() 返回了非空指针，而且该指针确实被当作 dest 使用
  （否则根本走不到 memcpy_s 的参数检查那一行）✓
· 独立实测（设备上、同一个解释器）：匿名分配并在【每一页】写入
      2 GiB + 4 KiB ✓   2.5 GiB ✓   ★4.26 GiB ✓★   ★8 GiB ✓★
  （MemTotal 32 GiB、ulimit -v unlimited）
⇒ OS / 分配器 / 驱动都【不】禁止 >2 GB 的单次分配 ✗
```

**唯一的 2 GiB 限制在 securec `memcpy_s` 的 API 契约里**：`destMax > 0x7fffffff` 直接 `ERANGE`
（§32.3 二分实测：`0x7fffffff` 收、`0x80000000` 拒）。

```
⇒ 准确说法是：★内存分配得出来，但"把 4.26 GB 当 destMax 传给 memcpy_s"这个调用被安全函数拒绝★
   失败发生在【参数校验】阶段——一个字节都还没拷 ✓（§32.1 现场：count 只有 4 字节 ✓）
```

### 32.8 ★为什么会有这条限制、目的是什么、有没有文档★ —— 出处是 securec 源码本身

**设备上实现这个限制的是华为的 securec（libboundscheck，木兰 PSL v2 开源）**：
设备侧导出实测 `/system/lib64/chipset-pub-sdk/libsec_shared.z.so` →
`memcpy_s / memset_s / memmove_s / strcpy_s / vsnprintf_s` ✓

**① 有文档，而且就写在头文件里**（`include/securectype.h` 文件头 Notes）：

```
 * Notes: User can change the value of SECUREC_STRING_MAX_LEN and SECUREC_MEM_MAX_LEN
 *        macro to meet their special need, but ★The maximum value should not exceed 2G★.
```

对应的宏定义与编译期兜底（同一文件）：

```c
/* Define the max length of the string */
#ifndef SECUREC_STRING_MAX_LEN
#define SECUREC_STRING_MAX_LEN 0x7fffffffUL
#endif
/* Add SECUREC_MEM_MAX_LEN for memcpy and memmove */
#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL          /* ← 就是 §32.3 二分到的那个值 ✓ */
#endif
#if SECUREC_STRING_MAX_LEN > 0x7fffffffUL
#error "max string is 2G"
#endif
```

⇒ 这是**白纸黑字的设计约束**（"不得超过 2G"），而且**允许使用者按需改这个宏** ✗
—— 但系统库是预编译的专有二进制，我们改不了 ✓

**② 我们命中的分支与源码逐字对应**（`src/memcpy_s.c`）：

```c
SECUREC_INLINE errno_t SecMemcpyError(void *dest, size_t destMax, const void *src, size_t count)
{
    if (destMax == 0 || destMax > SECUREC_MEM_MAX_LEN) {
        SECUREC_ERROR_INVALID_RANGE("memcpy_s");
        return ERANGE;                 /* ★34★ —— 正是 §32.1 读到的 w22 ✓ */
    }
    ...
    if (count > destMax) {
        (void)SECUREC_MEMSET_FUNC_OPT(dest, 0, destMax);   /* ← 注意这里用 destMax 去复位 */
        ...
        return ERANGE_AND_RESET;
    }
```

参数快检宏（先在这里就被拦下）：

```c
#define SECUREC_MEMCPY_PARAM_OK(dest, destMax, src, count) (SECUREC_LIKELY((count) <= (destMax) && \
    (dest) != NULL && (src) != NULL && ★(destMax) <= SECUREC_MEM_MAX_LEN★ && \
    (count) > 0 && SECUREC_MEMORY_NO_OVERLAP((dest), (src), (count))))
```

**③ 目的是什么（从代码可确证的动机）**

```
· destMax 是 Annex K 语义里"调用方【声明】的目标缓冲容量"，抓的是缓冲区溢出
  —— 它约束的是【声明值】，不是真实可用内存 ✗（所以 Resize 出 4.26 GB 与它无关 ✓）
· 错误分支里会 `memset(dest, 0, destMax)` 做复位 ⇒ ★库必须先确认 destMax 本身可信★，
  否则一个垃圾 destMax 会直接变成"memset 几个 GB"的灾难 ✗
· 封顶 INT_MAX 让 32/64 位行为一致、比较与指针加法在 32 位下不溢出
  （同文件里 "Limited format input and output width, use signed integer" 是同一取向 ✓）
⇒ 本质是 ★"明显不可能正确的参数"上的 fail-fast★：>2 GB 的声明值几乎必然是调用方 bug ✓
```

**④ 华为自己也被这条卡过 —— 源码里有现成的例外注释**（很能说明"它是参数合理性检查、不是内存检查"）：

```c
#if defined(SECUREC_COMPATIBLE_WIN_FORMAT)
    /*
     * The fread API in windows will call memcpy_s and pass 0xffffffff to destMax.
     * To avoid the failure of fread, we don't check desMax limit.
     */
#define SECUREC_MEMCPY_PARAM_OK(...)   /* ★Windows 兼容构建里干脆不查这条★ */
```

⇒ 官方认可"这条限制会误伤正常调用"，但**例外只给 Windows 兼容构建**；
我们设备上走的是非 Windows 分支，照旧强制 ✗

**⑤ 为什么 hilog 里什么都没有**：错误处理器宏在 release 构建里是**空宏**
（`securecutil.h`：`/* Default handler is none */`），只有 `_DEBUG` +
`SECUREC_ERROR_HANDLER_BY_ASSERT/PRINTF/FILE_LOG` 才会输出 ✓
⇒ 所以只能靠返回码定位（§32.1 的 `w22 = 34`）✓ ✗ 日志指望不上

**出处**（开源仓库 `openharmony/third_party_bounds_checking_function`）：

```
include/securectype.h   —— "The maximum value should not exceed 2G" + 宏定义 + #error
include/securec.h       —— API 原型与错误码（ERANGE 34 / ERANGE_AND_RESET 162 …）
src/memcpy_s.c          —— SecMemcpyError() 与 SECUREC_MEMCPY_PARAM_OK()
src/securecutil.h       —— 错误处理器宏（release 下为空）
```

**⑥ 对我们的意义**：这条限制在**用户态 API 契约**里，绕不过（系统库只读、专有）
⇒ 只能分段（§32.4：每段 .ms ≤ ≈520 MiB，工程上 ≤300 MB ✓）。
真正"可修"的点在**调用方**：适配层拿整块 4.26 GB 当 `destMax`，而每次只拷 4 字节 ✗
—— 若它按"本次可用窗口"声明就不会撞上限（但那是专有实现，我们改不了 ✓）。

---

## 33. ★★★★★ "有没有办法绕过 2 GB" —— 第一道能绕（已实测），第二道在同一规则上且 securec 是静态链入，第三道还有 protobuf 的 2 GB ★★★★★

§32 得出"每段 .ms ≲ 500 MB"的天花板。本节回答"能不能绕"：**能绕第一道，但绕不过后两道**。

### 33.1 第一道确实能绕：`LD_PRELOAD` 插桩 `memcpy_s`（实测通过 ✓）

**可行性依据**：`libhiai_adapter.so` 对 `memcpy_s` 是**动态导入**
（`llvm-nm -D` 显示 `U memcpy_s`；§32.1 反汇编里也是 `bl … ; symbol stub for: memcpy_s`）
⇒ 预加载对象在全局符号作用域里排在前面，可以顶掉它 ✓

**工具**：本机就有 aarch64-ohos 交叉工具链（`clang 21` + `aarch64-linux-ohos` sysroot）✓

**插桩 so 的核心**（完整版在临时目录，不入库）：

```c
int memcpy_s(void *dest, size_t destMax, const void *src, size_t count)
{
    if (dest == NULL || src == NULL) return 22;   /* EINVAL */
    if (count > destMax)             return 34;   /* ERANGE */
    /* ★只去掉这一条：destMax <= SECUREC_MEM_MAX_LEN★，不再因 >2GB 拒绝 */
    if (count) (void)memmove(dest, src, count);
    return 0;
}
```

编译与运行：

```sh
clang --target=aarch64-linux-ohos --sysroot=<ohos-sysroot> -shared -fPIC -O2 -o libwqfix.so wqfix.c
export LD_LIBRARY_PATH=/system/lib64/ndk:/system/lib64/platformsdk
LD_PRELOAD=<libuname.so>:<libwqfix.so> python3.14 <加载脚本> <model.ms> 60
```

**实测结果（决定性）**：

| 样本 | 超 2 GB 的 `memcpy_s` 调用 | 结果 |
|---|---|---|
| L0-23（1107 MB，staging 4.26 GB） | ★688 次全部放行★（首条正是 `destMax=4575971444 count=4`） | 权重真的拷进去了（`count` 最大 75,497,472） |
| L12-23（672 MB，staging ≈2.59 GB） | ★154 次全部放行★ | 同上 |
| 小模型 g4s0（147 MB） | 无 | ★Build 0 · Predict 0★（插桩无副作用 ✓） |

⇒ 第一道（§32 的 `Set Data Fail.`）**确实被绕过了** ✓ —— 但它不是唯一的坎 ✗

**两个实现要点**（下次复现别踩）：
· 故意【不】复刻 securec 的 reset 语义（失败时 `memset(dest, 0, destMax)`）——
  那恰恰是拿到一个垃圾 `destMax` 时最危险的动作 ✓
· 要用 `memmove` 而不是 `memcpy`（原函数有重叠检查，我们直接给出更安全的行为）✓

### 33.2 第二道绕不过：`libhiai_ir.so` 里【静态链入】的 securec

**真实报错**（用插桩接管 `OH_LOG_Print` 才拿到；还要先剥掉 hilog 的
`%{public}` / `%{private}` 隐私标记，否则标准 `vsnprintf` 解析不了、消息是空的 ✗）：

```
[F] AI_INFRA: model.cpp BuildWeightMergedModelBuffer(206)::
    "memcpy_s(reinterpret_cast<void*>(basePtr + offset), totalBufferSize - offset,
              weightBuffer.GetData(), weightBuffer.GetSize()) == EOK"   "false, return false."
[F] AI_INFRA: model.cpp SaveFullModel(248)::"ret"  "false, return FAIL."
[F] NNRt_HiAIAdapter: BuildImpl from lite graph failed, failed to serialize IR model.
[F] NNRt_HiAIAdapter: Build from lite graph failed.
```

**为什么插桩拦不到它**（结构性证据，不是猜）：

```
libhiai_ir.so       动态符号表里【没有】memcpy_s（llvm-nm -D 无输出）
                    DT_NEEDED = libhilog_ndk.z.so / libc++_shared.so / libc.so
                    ★也不依赖 libsec_shared★ ⇒ securec 被【静态链进去】了 ✗
libhiai_adapter.so  动态符号表里【有】U memcpy_s ⇒ 所以能被 33.1 顶掉 ✓
```

⇒ `LD_PRELOAD` 对静态链接的 `memcpy_s` **无效** ✗
⇒ 而且它在**同一条规则**上：`destMax = totalBufferSize - offset`，
   即 IR 的"合并权重缓冲"一旦 > 2 GB 就返回 ERANGE ✓

**对照实验（两个超限样本 + 一个未超限样本）**：

| 段 | `.ms` | staging 缓冲 | 第一道 | 第二道（IR 合并权重缓冲） |
|---|---|---|---|---|
| L0-11 | 435 MB | ≈1.68 GB | 本来就过 ✓ | ★Build 0 · Predict 0★ ✓ |
| L12-23 | 672 MB | ≈2.59 GB | 插桩放行 ✓ | ✗ `BuildWeightMergedModelBuffer` 断言失败 |
| L0-23 | 1107 MB | 4.26 GB | 插桩放行 ✓（688 次） | ✗ 同上 |

### 33.3 第三道（即使第二道也绕过）：protobuf 的 2 GB 硬限制

`libhiai_ir.so` 里就带着 **protobuf 自己的报错串**：

```
 exceeded maximum protobuf size of 2GB:
```

并且它导出的是 protobuf 风格的序列化入口：

```
ge::GraphSerializer::SerializeTo / UnSerialize
ge::NodeSerializer::SerializeTo / SaveSubGraphs / SaveEdge
SerializeModelToModelDef / SerializeModelDefToBuffer / SerializeGraphDefToBuffer
```

⇒ IR（`ModelDef` / `GraphDef`，权重内联）最终要经 protobuf 序列化，
   而 protobuf 的序列化长度是 **int32 设计**，> INT_MAX 时 protobuf 自己就拒绝 ✓
⇒ 这是**库的设计限制**，任何符号插桩都解决不了 ✗
（★此条为强推断，未实测★：要实测就得先绕过 33.2 那道，代价大且大概率只是把
 失败点往后挪到 protobuf 这一层。）

### 33.4 结论

```
✓ 第一道：能绕（LD_PRELOAD 插桩 memcpy_s，实测 688/154 次放行）
✗ 第二道：绕不过 —— libhiai_ir.so 的 securec 是静态链入的，符号插桩够不着（有结构性证据）
✗ 第三道：原理上绕不过 —— protobuf 的 2 GB 是 int32 设计限制
⇒ 唯一的"理论绕过"是运行时改内存（mprotect + 打补丁 / 调试器改返回码），
  既不可交付、也很可能继续撞第三道 ✗
★ 结论不变：分段是唯一实际路线；有效天花板 = 每段权重/IR 缓冲 ≤ 2 GB
  ⇒ 每段 .ms ≲ 500 MB（≈4× 关系），工程上 ≤300 MB（≈≤7 层）✓
```

**顺带记录两条可复用技巧**：

```
· 想抓适配层自己的日志：插桩 OH_LOG_Print（默认 hilog 缓冲里看不到它），
  并且必须先剥掉 %{public}/%{private} 标记，否则消息是空的 ✗
· 想证明"某库的 securec 是静态还是动态"：看它的动态符号表有没有 memcpy_s
  （U=动态导入⇒可插桩 / 没有⇒静态链入⇒插不进去）✓ 这比读源码还快 ✓
```

---

## 34. ★★★★★ 用调试器在运行时打掉两处检查 ⇒ 672 MB 的段（staging 2.59 GB）直接跑通；1107 MB 的失败点前移到 DDK ★★★★★

§33 的结论是"第一道能绕、后两道绕不过"。**§33.3（protobuf 那条）被本节实验否掉了** ✗ ——
在运行时把两处 `cbnz` 改成 `nop` 之后，**672 MB 的段 Build 0 · Predict 0**，
protobuf 的 2 GB **根本没触发**。

### 34.1 改哪两条指令

静态定位（`llvm-objdump` 反汇编 + 从断言串反推调用点）：

```
① libhiai_ir.so 里【静态链入】的 memcpy_s（不是导出符号）：
     调用点：ge::Model::SaveFullModel(ge::Buffer&) 内 0x44a68（4 参数 + 返回值判 0）
     函数体：0xc16e4
       c16e4: cbz  x3, fail          ; count==0
       c16e8: lsr  x8, x1, #31       ; ★x1 = destMax★
       c16ec: cbnz x8, fail          ; ★destMax ≥ 2GB ⇒ 失败（就是这条）★
       c16f0: cbz  x2, fail
       c16f4: cbz  x0, fail
       c16f8: cmp  x3, x1 / b.hi fail ; count > destMax（保留）
   ⇒ 编译器把 `destMax > 0x7fffffff` 优化成 `destMax >> 31` 测试，
     所以按常量 0x7fffffff 搜是搜不到的 ✗（这是个坑）

② libsec_shared.z.so 的导出 memcpy_s（第一道，适配层用的那个）：
     符号地址 0x3e34，同样是 `lsr x9, x8, #31` ⇒ `cbnz x9` 在 ★0x3e50★
```

补丁就是一条 `nop`（`1f 20 03 d5`）：

```
memory read  -s 1 -f x -c 4 <base_ir + 0xc16ec>     # 原字节 e8 02 00 b5（cbnz x8）
memory write -s 1 <base_ir + 0xc16ec> 0x1f 0x20 0x03 0xd5
memory read  -s 1 -f x -c 4 <base_sec + 0x3e50>    # 原字节 09 03 00 b5（cbnz x9）
memory write -s 1 <base_sec + 0x3e50> 0x1f 0x20 0x03 0xd5
```

**可行性要点**（都实测过）：
```
· 文本段是 r-xp，但 ptrace 仍可写 ⇒ lldb 的 memory write 直接成功 ✓
· 补丁不影响 BTI/CFI（改的是函数体中间，不是入口）✓
· 保留 count > destMax / NULL / 重叠检查，只拿掉"destMax ≥ 2GB"这一条 ✓
· 基址用 image list -o -f <lib> 取（每次运行 ASLR 不同，必须重取）✓
```

### 34.2 实测结果

| 样本 | `.ms` | staging | 打了补丁之后 |
|---|---|---|---|
| **c12（层 12-23）** | **672 MB** | **≈2.59 GB** | ★**Build 0 · Predict 0**★（输入 18 / 输出 5，全部成功）|
| L0-23（层 0-23） | 1107 MB | 4.26 GB | ✗ **失败点前移**（见 34.3）|

⇒ 只改内存里的 8 个字节，**一个原本 Build -1 的大段就在 NPU 上跑通了** ✓
（没有改任何系统文件、没有落盘、没有用 memcpy_s 插桩 ✓）

### 34.3 1107 MB 的段：失败点前移到 DDK 的 runtime 选择

```
[F] HIAI_DDK_MSG: hiai_model_runtime_repo.c ModelRuntimeRepo_TryBuild(154)::"no runtime support the Model."
[F] HIAI_DDK_MSG: model_builder_impl.cpp BuildModel(34)::"build model failed."
[F] NNRt_HiAIAdapter: HiaiExecutorImpl::model build fail !.
[F] NNRt_HiAIAdapter: Build from lite graph failed.
```

```
★关键：IR 序列化【过了】—— libhiai_ir.so 里那句 protobuf
  "exceeded maximum protobuf size of 2GB" ★没有出现★ ✗
⇒ §33.3 那条"protobuf 会拦住"是【错的】：IR 里的权重并没有以 >2GB 的
  单个 protobuf 消息形式出现（至少这个尺寸下不是）✗
```

**新的边界**：介于 **2.59 GB（过）** 与 **4.26 GB（不过）** 之间，
而且 `0xFFFFFFFF = 4.294 GB` 正好落在这个区间里 ⇒ **怀疑某处按 32 位截断**
（4,575,984,244 − 2³² = 281,016,948，模型在 DDK 眼里就是个"对不上"的模型）
⇒ 但这【只是怀疑，未验证】：DDK（`hiai_model_runtime_repo.c`）是闭源的，
   要坐实得再造一个 staging 刚好 3.8~4.2 GB 的段来二分 ✗

### 34.4 意义与边界（别把研究结论当交付方案）

```
✓ 结论 1：那两道 2 GB 检查【不是】物理上限，是可以绕的（§33.1 的插桩、本节的运行时 nop 都行）
✓ 结论 2：绕掉之后，天花板至少抬到 ★2.59 GB staging（672 MB 段）★以上
          ⇒ 现役 9 段（≤264 MB）与 L0-11（435 MB）之外，
            ★L12-23（672 MB）这种也能跑★，即"更少的段"在技术上成立 ✓
✗ 结论 3：但到 ~4 GB 量级会撞上 DDK 的 runtime 选择（"no runtime support the Model"），
          这一层是闭源 DDK，不可控 ✗
★ 交付口径【不变】：不碰系统库时，每段 .ms ≲ 500 MB（§32.4）；
  运行时补丁属【研究/验证手段】，不是一个可以随 app 发布的方案 ✗
```

**如果要让 app 侧也用上（未做，仅供参考）**：可以不用调试器——在 `LD_PRELOAD` 的
构造函数里 `dl_iterate_phdr` 找到 `libhiai_ir.so` 基址、`mprotect` 那一页、
写 `nop`、再还原保护位 ✓。但那属于"进程内自打补丁"，风险与合规都要另行评估 ✗

---

## 35. ★★★★★ 追"runtime 选择失败"：定位到 /vendor 的 HCL 运行时；四个 securec 检查全打掉后仍失败 ⇒ 拒绝原因在 HCL 自己 ★★★★★

§34 把 1107 MB 段的失败点推到了 `"no runtime support the Model."`。本节把它继续往下钉。

### 35.1 定位手法：给日志钩子加"帧指针链"

`OH_LOG_Print` 的 `__builtin_return_address(0)` 只能给到**日志助手自己**的返回地址
（`libhiai_ir.so+0xc1698`），拿不到真正的调用者 ✗。所以钩子里再加一段**帧指针链**回放
（编译时加 `-fno-omit-frame-pointer`），一次就拿到了完整路径 ✓：

```
[F] HIAI_DDK_MSG: hiai_model_runtime_repo.c ModelRuntimeRepo_TryBuild(154)::
    "no runtime support the Model."
   stack: <libhiai_ir.so+0xc1698>   ← 日志助手
          <libhiai.so+0x60200>      ← ★真正打日志的地方★
          <libhiai.so+0x5d530>      ← 调用它的函数
          <libhiai.so+0x32e40> <libhiai.so+0x32d68>
```

### 35.2 判定逻辑（libhiai.so 反汇编）

```
0x6014c: 函数入口
0x6017c: x24 = 0                       ; i = 0
0x6018c: loop:
0x60194:   bl 0x60238                  ; 取第 i 个 runtime（按名字/版本 strstr+strncmp 选）
0x601ac:   bl 0x60db4                  ; TryBuildOne(runtime, ...)
0x601b4:   cbnz x0, 0x60200            ; 成功 ⇒ 返回
0x601b8:   i++; cmp i, #3; b.ne loop   ; ★最多试 3 个 runtime★
0x601fc:   bl AI_Log_Print             ; 全失败 ⇒ "no runtime support the Model."
```

`0x60db4`（TryBuildOne）核心就三条：

```
60ddc: ldr  x8, [x4, #0xe0]     ; runtime->fn（或 +0x8）
60df4: blr  x8                  ; ★调用该 runtime 的建模型接口★
60df8: cbz  w0, 成功            ; w0 != 0 ⇒ 这个 runtime 不接受
```

### 35.3 运行时是谁：`/vendor/lib64/passthrough/` 的 HCL 运行时

在 `0x60df4`（`blr x8`）下断，读 `x8` 得到：

```
x8 = libai_fmk_hcl_model_runtime.so`HIAI_HCL_ModelBuilder_BuildV2
     （实现在同目录 indirect/libai_fmk_hcl_model_runtime_impl.so）
```

薄库的 `BuildV2` 只是包一层 trace，真体转发到 `HIAI_HCL_ModelBuilder_Build`；
它导入 `AI_Log_Print`（所以它的日志本可被钩子抓到 ✓）与 **`memset_s`**（securec ✓）。

### 35.4 把【四个】securec 2 GB 检查全打掉 —— 仍然失败 ✗

| 位置 | 指令 | 补丁 |
|---|---|---|
| `libhiai_ir.so+0xc16ec` | `cbnz x8`（静态 memcpy_s） | `nop` |
| `libsec_shared.z.so+0x3e50` | `cbnz x9`（memcpy_s） | `nop` |
| `libsec_shared.z.so+0x51dc` | `cbnz x9`（**memset_s**） | `nop` |
| `libsec_shared.z.so+0x5138` | `b.hs`（**memmove_s**） | ★不是 nop★：它是**反极性**（跳向正常路径），要改成无条件 `b 0x5150` = `06 00 00 14` |

⇒ 四条全打掉、逐条回读确认之后，**L0-23 依然 `rc=-1`，日志依旧是 `no runtime support`** ✗
⇒ **HCL 的拒绝与 securec 的 2 GB 限制无关** ✓（这一条把嫌疑彻底排除了 ✓）

### 35.5 嫌疑落在 HCL 自己的判定上

`libai_fmk_hcl_model_runtime_impl.so` 里有一批"限制类"断言字符串，最像的两条：

```
%s %s(%d)::"(buf.st_size <= MAX_FILE_SIZE_LIMIT)"        "false, return %s."
%s %s(%d)::"Current platform:%s does not support."
```

⇒ 下一步：在 `HIAI_HCL_ModelBuilder_Build`（薄库里有符号 ✓）里下断、追它哪一步返回非 0，
   把那处判定/比较打掉 ✓（`MAX_FILE_SIZE_LIMIT` 的具体值还没取出来，静态没搜到引用点 ✗）

### 35.6 一个重要的边界认识

```
· "选择循环"本身当然可以 patch（0x6014c / 0x60df4 都是普通指令）
  但那样【没有意义】：x19 是"建好的模型句柄"，强制让它"成功"只会把 NULL 往下传 ✗
· 真正要解决的是 HCL builder 内部的拒绝判定 —— 它在我们不可控的 /vendor 库里 ✓
· 且即便打通：4.26 GB 已越过 2³²，模型内部若真有 32 位长度（§34.3 的怀疑），
  后面可能出现【静默的错误结果】而不是干脆的失败 —— 这类风险要单独评估 ✗
★ 本节把"到底卡在哪"钉到了具体函数与具体字符串；是否继续往里打，是一个
  "收益 vs 依赖 /vendor 库行为"的判断 ✗
```

**顺带记一个补丁极性坑**：securec 里 `memcpy_s`/`memset_s` 的 2 GB 检查是
`cbnz` 跳向【错误】路径（所以 `nop` 掉即可 ✓），而 `memmove_s` 是
`b.hs` 跳向【正常】路径（`nop` 反而会掉进 ERANGE ✗）——
所以要改成无条件跳转 `b 正常路径` ✓。

---

## 36. ★★★★★ "no runtime support" 的对照实验：环境/装载完全一致，差别只在运行时对模型的裁决 ★★★★★

§35 把失败点定位到 `/vendor` 的 HCL 运行时。本节做**成功样本 vs 失败样本**的逐行对照，
并排除掉"环境/装载有问题"这条岔路。

### 36.1 关键技巧：默认被级别过滤掉的日志，看不到真相

`AI_Log_Print` 在打日志前会先问 `OH_LOG_IsLoggable(domain, tag, level)`，
**不允许就直接丢弃、连 `OH_LOG_Print` 都不调** ✗ ⇒ 我们的钩子完全看不见这些消息 ✗。

把 `OH_LOG_IsLoggable` 也顶掉（恒返回 1）之后，整条 DDK 路径就现形了 ✓：

```c
int OH_LOG_IsLoggable(unsigned int domain, const char *tag, int level) { return 1; }
```

（顺带给日志量封顶，避免刷爆磁盘。）

### 36.2 两次运行的前置步骤【逐行一致】

```
[W] model_manager_impl.cpp CreateModelManager(530)::"DDK pipe is HCL."
[W] hiai_plugin_version.c HIAI_MR_GetVersion(104)::"version is 800.636.120.010"
[W] model.cpp BuildWeightMergedModelBuffer(209)::"Build success!"          ← 补丁有效
[I] model_type_util.cpp GetModelType(45)::"model type: 7"
[I] hiai_model_runtime.c HIAI_ModelRuntime_LoadSo(324)::
    "dlopen libhiai_hcl_model_runtime.so fail: No such file or directory."  ← ★成功样本里也有★
[I] hiai_model_runtime.c HIAI_ModelRuntime_LoadFromAllSymbols(215)::"...success."
```

★注意★：那条 `dlopen libhiai_hcl_model_runtime.so fail` **在成功样本里同样出现** ✓
（HCL 存根的真实文件名是 `libai_fmk_hcl_model_runtime.so`，DDK 先试一个旧名，
失败后走 `LoadFromAllSymbols` 成功）⇒ **它不是故障，是正常的回退路径** ✗
（§35.3 曾把它当嫌疑，这里更正 ✓）

### 36.3 差别只在这一步

| 样本 | `.ms` | staging | 后续日志 |
|---|---|---|---|
| c12（层 12-23） | 672 MB | ≈2.59 GB | `model_builder_impl.cpp BuildModel(42)::"build model success."` ✓ |
| L0-23（层 0-23） | 1107 MB | 4.26 GB | `ModelRuntimeRepo_TryBuild(154)::"no runtime support the Model."` ✗<br>`model_builder_impl.cpp BuildModel(34)::"build model failed."` ✗ |

⇒ **环境、装载、权重合并缓冲、模型类型全都一样**；唯一变量是模型尺寸，
   而裁决结果不同 ⇒ 卡点确实在"该 runtime 能不能吃下这个模型" ✓

### 36.4 运行时是谁、怎么被调的（ptrace 实测）

```
libhiai.so 0x6014c（ModelRuntimeRepo_TryBuild）：i=0..2 逐个 runtime
   0x60db4 TryBuildOne → runtime+0xe0/+0x8 的函数指针
   0x60df4 blr x8
实测 x8 = libai_fmk_hcl_model_runtime.so`HIAI_HCL_ModelBuilder_BuildV2（/vendor/lib64/passthrough/）
```

在该次运行里另外两个断点**没有命中** ✓：
```
breakpoint HIAI_HCL_ModelBuilder_Build        （薄库导出，0x10448）  ✗ 未命中
breakpoint HIAI_HCL_ModelBuilder_Build_Impl   （实现库 0x12d7dc）    ✗ 未命中
```
⇒ 薄库的 `BuildV2` 直接走**内部派发器**（0x1051c，它 `dlopen/dlsym` 实现库、
并读 `GetParameter@1.0`），而不是走导出的 `Build`；
  实现库里真正被调的是 **`HIAI_HCL_ModelBuilder_BuildV2_Impl`（0x12de7c）**，
  不是我们先前反汇编的 `Build_Impl`（那是 V1 入口，所以它的入口日志没出现 ✓）✗

### 36.5 当前结论与下一步

```
✓ 已排除：securec 的 2 GB（四个检查全打掉仍失败 ✓）
✓ 已排除：环境/装载（成功样本有完全相同的启动序列 ✓）
✓ 已确认：卡点在 HCL 运行时对模型的裁决，边界在 (2.59, 4.26] GB staging 之间 ✓
★ 首要嫌疑仍是 §34.3 的 32 位截断：4.26 GB > 2³² = 4.295 GB，
  4,575,984,244 − 2³² = 281,016,948 —— 若某处按 32 位存长度，
  模型在运行时眼里就是"对不上"的 ✗（★仍未验证★）
⇒ 下一步二选一：
  (a) 在 HIAI_HCL_ModelBuilder_BuildV2_Impl（0x12de7c）下断，追它哪一步返回非 0；
  (b) 造一个 staging 落在 3.0~4.2 GB 的段（例如层 0-17 左右）来二分 ——
      若 ≤4.295 GB 能过、>4.295 GB 不过，就是 2³² 截断，且【每段 .ms 上限约 1 GB】
      （比不碰系统库时的 ~500 MB 高一倍 ✓）
```

---

## 37. ★★★★★ 对照实验（ptrace 实测）：同一个 runtime、同一个函数，仅因模型尺寸返回 0 / 1 ★★★★★

§36 把卡点收敛到 HCL 运行时。本节用 ptrace 在 `libhiai.so+0x60df4`（`blr x8`）与
`+0x60df8`（调用返回处）分别下断，两个模型各测一遍 —— **结论完全确定** ✓

### 37.1 三个 runtime 里只有一个有效

断点命中次数（`breakpoint list` 的 hit count，比盯屏可靠 ✓）：

```
libhiai.so+0x60df4（blr x8）  hit count = 1      ← L0-23 那次运行
libhiai.so+0x60df8（调用后）  hit count = 1
```

而 `ModelRuntimeRepo_TryBuild` 的循环是 **i=0..2 共 3 次** ✓
⇒ 另两个 runtime 的对象是 **NULL**（它们的 `LoadSo` 就是 §36 那条裸名 dlopen 失败 ✗），
   `0x60dd4 cbz x4, fail` 直接把它们判为"不支持"，**连函数都不会调** ✓
⇒ 所以模型只能由这**唯一**的 runtime 建 ✓

### 37.2 同一个 runtime、同一个函数，两个模型两个结果

| 样本 | `.ms` | staging | `x8`（被调函数） | **`w0`（返回值）** | 结果 |
|---|---|---|---|---|---|
| c12（层 12-23） | 672 MB | ≈2.59 GB | `libai_fmk_hcl_model_runtime.so`★`HIAI_HCL_ModelBuilder_BuildV2`★ | **0x00000000** ✓ | Build 0 · Predict 0 |
| L0-23（层 0-23） | 1107 MB | 4.26 GB | **同一个** | **0x00000001** ✗ | Build -1 |

⇒ **限制就在 `HIAI_HCL_ModelBuilder_BuildV2` 这条链里，而且是尺寸相关的** ✓
（不是"环境/装载/权限"，也不是 securec —— 那些都已排除）

### 37.3 薄库里的派发（反汇编 0x1051c）

```
10544: cbz  x3, 0x105c8        ; 参数缺 → 打日志(行22) → return 1
10560: bl   0xe0e8             ; 取配置/选项
10564: bl   0x15804            ; ★拿到实现函数指针 x22（内部走 dlopen/dlsym）★
1057c: cbz  x22, 0x10638       ; 指针为空 → 打日志(行26) → return 5
10590: blr  x22                ; ★调用真正的实现★
10594: cbnz w0, 0x105f8        ; 实现返回非 0 ⇒ 原样上抛（★本次 w0=1 就是从这里来的★）
10598: ldur x20,[x29,#-8]; cbz x20, 0x1066c   ; 产出句柄为空 → 打日志(行29) → return 1
105a0-105c0: 分配 16 字节包装 → return 0
```

**关键推理**：薄库自己所有失败路径**都会 `AI_Log_Print`**（行 22/26/29），
而这次运行里**薄库一条日志都没打** + `w0=1` ✗
⇒ 这个 1 是 **`blr x22` 那个实现函数**返回的 ✓
⇒ 真正的限制在**实现函数内部**，且它是**静默返回**（不打日志）✗

### 37.4 下一步（唯一还没到的那一层）

```
在 薄库基址 + 0x10590（blr x22）下断 → 读 x22，确认实现函数属于哪个 .so/符号
  → 再在那个实现函数里找"尺寸相关的静默 return"
  → 找到那处比较（大概率是一个上限常量或 32 位截断点），把限制本身改掉
```

★仍未验证的核心怀疑★（§34.3 提出）：
```
L0-23 的 staging = 4.26 GB > 2³² = 4.295 GB，而 c12 的 2.59 GB 在 2³² 以内；
4,575,984,244 − 2³² = 281,016,948
⇒ 若实现里某处按 32 位存长度，模型在它眼里就是"对不上"的 ⇒ 静默拒绝 ✓ 与现象吻合
⇒ 要坐实只需造一个 staging 落在 3.0~4.2 GB 的段（约层 0-17）做二分：
   能过 ⇒ 就是 2³²，且【每段 .ms 上限约 1 GB】（比不碰系统库的 ~500 MB 高一倍 ✓）
   不过 ⇒ 上限更低，继续在实现函数里找那处比较
```

---

## 38. ★★★★★ 确切限制找到了：UnifiedModel 头里的长度字段是【32 位】，模型 > 4 GiB 必被截断 ★★★★★

§37 把失败收敛到 `HIAI_HCL_ModelBuilder_BuildV2_Impl`。本节把它钉到**一条比较指令**上，
并**实测出被截断的数值**——不是猜的。

### 38.1 完整调用链（ptrace 逐层下断实测）

```
libhiai.so : ModelRuntimeRepo_TryBuild → TryBuildOne → blr x8
  x8  = libai_fmk_hcl_model_runtime.so : HIAI_HCL_ModelBuilder_BuildV2
        └─ GetBuildSymbol(...) + 0x74 : blr x22
             x22 = ★libai_fmk_hcl_model_runtime_impl.so : HIAI_HCL_ModelBuilder_BuildV2_Impl★
                   ├─ CheckInputParam(name, data, size, out)   → 返回 1（通过 ✓）
                   ├─ hiai::UnifiedModel::UnifiedModel(obj, data, size, 0)
                   └─ ldrb w8,[obj+0x130]; cbnz w8 → 继续
                      ★实测 w8 = 0 ⇒ 判为"非法" ⇒ 直接返回 1 ⇒ "no runtime support"★
```

`UnifiedModel` 的合法性就写在这一个字节上（impl 0xa27b0）：

```
a27d8: bl   0xa27f4          ; 校验子函数：0 = 合法、非 0 = 非法
a27dc: cbnz w0, a27e8        ; 非 0 ⇒ 跳过
a27e0: mov  w8,#1; strb w8,[x19,#0x130]   ; ★合法才写 1★（obj+0x130 正是 [sp+0x4b8]）
```

### 38.2 那条比较：头里的长度字段是 u32

校验体再进一层（impl 0xa288c），关键就四条：

```
a28b4: ldr  x9,  [x0, #0x8]      ; x9 = size（64 位）
a28d8: ldr  w21, [x8, #0x4c]     ; ★头里偏移 0x4c 的长度字段，32 位★
a28dc: add  x10, x21, #0x100     ; x10 = (u32)字段 + 0x100
a28e0: cmp  x10, x9
a28e4: b.ne 0xa2a00              ; ★不等 ⇒ 报错 ⇒ 校验返回非 0 ⇒ 非法★
```

### 38.3 实测数值（决定性）

在 `BuildV2_Impl` 里读出 `UnifiedModel` 对象的 `(data, size)` 与模型头：

```
obj->data = 0x0000005b56201ac0
obj->size = 0x0000000110c9bbc8 = 4,576,119,752 ≈ 4.262 GiB     ★ > 2³² = 4,294,967,296 ★
magic @ data      = 0x444f4d49                                  （"MODI"，合法 ✓）
head  @ data+0x4c = 0x10c9bac8                                  ← 【32 位字段】
size − 0x100      = 0x110c9bac8
0x110c9bac8 & 0xFFFFFFFF = ★0x10c9bac8★  ← 与头里的字段【逐位一致】⇒ 就是截断值 ✓✓
```

⇒ **结论（确切的限制）**：

```
★模型头里的总长度字段是 32 位（u32）★：
   校验要求 (u32)head[0x4c] + 0x100 == size(64 位)
   一旦模型 > 4 GiB，写入方只能存下低 32 位 ⇒ 必然不等 ⇒ UnifiedModel 判非法
   ⇒ BuildV2_Impl 静默返回 1 ⇒ ModelRuntimeRepo_TryBuild 打 "no runtime support the Model."
★这是【格式/序列化限制】，不是可以"改掉"的策略限制★
```

### 38.4 那能不能把这处比较也 patch 掉？

能改（`a28e4: b.ne` 写成 `nop` 即可），但**没有意义** ✗：

```
· 该字段随后被【存进对象】并用在下游：a2940: str w21,[x19,#0x5c]
  ⇒ 下游拿到的是 0x10C9BAC8（281 MB），而真实模型是 4,576,119,752（4.26 GiB）
  ⇒ 尺寸/偏移全错 ⇒ 大概率崩溃或【静默的错误结果】，而不是"跑通" ✗
· 也就是说：模型本身是【自相矛盾的】，不是被某条策略拦下的 ✓
```

### 38.5 真正的天花板（把三条限制合起来）

| 层次 | 限制 | 来源 | 可绕过？ |
|---|---|---|---|
| ① 权重缓冲拷贝 | `destMax ≤ 0x7fffffff`（2 GiB−1） | securec `memcpy_s`×2 + `memset_s` + `memmove_s` | ✓ 可（§32–34：插桩或 ptrace 改 4 条） |
| ② 同上（另一份代码） | 同① | `libhiai_ir.so` 里静态链入的 securec | ✓ 可（ptrace 改 `cbnz`） |
| **③ 模型头长度字段** | **u32 ⇒ 模型必须 < 4 GiB** | `UnifiedModel` 头格式（impl 0xa28d8-e4） | ✗ **不可**（改了只会让自相矛盾的模型继续往下走） |

⇒ **实际天花板 = 送给运行时的 IR/UnifiedModel 必须 < 4 GiB**。
   而 IR 的大小 ≈ staging 缓冲 + 头（≈ 3.94×`.ms` + 0x100，因为权重按 fp32 展开）：
```
   IR ≈ 3.94071 × |.ms| + 135,508   （135,508 = 实测 IR 4,576,119,752 − staging 4,575,984,244）

   ★不修改系统库★  约束是 securec：staging ≤ 0x7fffffff = 2,147,483,647
       ⇒ |.ms| ≤ 2,147,483,647 / 3.94071 ≈ 544,900,000 B ≈ ★545 MB（≈520 MiB）★
       实测夹逼：435 MB（455,682,136 B）过 ✓；672 MB（705,524,856 B）挂 ✗

   ★ptrace 改掉那四条 securec 检查后★  约束上移到模型头 u32：IR ≤ 2³²−1
       ⇒ |.ms| ≤ (4,294,967,295 − 135,508) / 3.94071 ≈ 1,089,872,000 B ≈ ★1.09 GB（≈1.02 GiB）★
       实测夹逼：672 MB 过 ✓；1107 MB（1,161,204,624 B）挂 ✗
       （两锚点正好把预测的 1.09 GB 夹在中间 ✓）

   ⇒ 比值来自同一对实测（IR / staging / 文件大小），两个锚点都与预测一致 ✓
   实测锚点：c12 = 672 MB → IR ≈ 2.59 GB ✓ 通过；L0-23 = 1107 MB → IR 4.576 GB ✗ 失败
```

### 38.6 一个仍未解释的现象（如实记录）

实现库 `libai_fmk_hcl_model_runtime_impl.so` 的一堆日志（入口行 198 "start to buildV2 model by hcl"、
以及上面那条校验失败的分支）**在我们的日志钩子里一条都没出现** ✗
（日志里只有 `NNRt_HiAIAdapter` 与 `HIAI_DDK_MSG` 两种 tag）。
已知：该库**动态导入** `AI_Log_Print`（`U`），而全进程只有 `libhiai_ir.so` 定义它（`T`）；
`AI_Log_Print` 内部会调 `OH_LOG_IsLoggable`（我们已强制返回 1）与 `OH_LOG_Print`（已挂钩）✓
⇒ 说明这些消息在 `AI_Log_Print` 内部还有一道我们没接上的过滤（例如按 tag/域的白名单），
   ★但本节的所有结论都不依赖日志——全部由寄存器/内存实测得到 ✓★

---

## 39. ★★★★★ 实测：三段切法的两个大段能在 NPU 上【依次真正运行】，KV 槽跨段传递成功 ★★★★★

§38 算出"打完四条 securec 补丁后每段 `.ms` ≲ 1.09 GB"，于是三段切法
（A=0-11 435 MB · B=12-23 672 MB · C=24-34 696 MB）在尺寸上都进得来。本节做实测。

### 39.1 三段各自的可装载性（都有设备实测）

| 段 | `.ms` | IR（推算） | 实测 |
|---|---|---|---|
| A = 层 0-11 | 435 MB | ≈1.80 GB | ★不打补丁就 Build 0 · Predict 0★（§31 表，早先实测）✓ |
| B = 层 12-23 | 672 MB | ≈2.59 GB | 打补丁后 ★Build 0 · Predict 0★（§34/§37 实测，输入18/输出5）✓ |
| C = 层 24-34 | 730 MB | ≈2.68 GB | 打补丁后 ★Build 0 · Predict 0★（输入21/输出1）✓ |

### 39.2 顺序串联实测（覆盖全部 35 层）

`L0-11_wq.ms` 在临时目录里已被清掉 ✗，所以用现成的分片把 A 的**层区间**覆盖掉，
跑了一条 5 段链路（脚本 `g4_3seg_run.py`：按名字把上一段输出喂给下一段输入，其余清零）：

```
段0 g4s0_wq.ms  (层 0-3)    Build=0 Predict=0  输入 8(按名传递 0) 输出 1  hidden_out:24576
段1 s4_wq.ms    (层 4-7)    Build=0 Predict=0  输入10(按名传递 1) 输出 1  hidden_out:24576
段2 s8_wq.ms    (层 8-11)   Build=0 Predict=0  输入10(按名传递 1) 输出 1  hidden_out:24576
段3 c12_wq.ms   (层 12-23)  Build=0 Predict=0  输入18(按名传递 1) 输出 5  hidden_out:24576,
                                                                        sk_out:4096, sv_out:4096,
                                                                        fk_out:8192, fv_out:8192
段4 c24_wq.ms   (层 24-34)  Build=0 Predict=0  输入21(按名传递 5) 输出 1  hidden_out:24576
★ 全部 5 段顺序执行完成（每段 Build/Predict 均为 0）★
```

**最有价值的一条**：段 4（C，层 24-34）的 21 个输入里有 **5 个是按名从段 3（B）的输出接上的**
（`hidden_out` + `sk_out/sv_out/fk_out/fv_out`）✓
⇒ ★跨段 KV 槽传递在 NPU 上真的走通了★（这原本是最容易出问题的一环 ✓）

另外：每段用完即 `ModelDestroy`（5 段同时驻留会吃光 NPU 内存 ✗），链路能跑完说明释放也正常 ✓

### 39.3 结论与边界（别过度解读）

```
✓ 三段切法在"能不能运行"这一层【成立】：
  · B、C 这两个需要补丁的大段，建得起来、跑得起来，KV 还能跨段传 ✓
  · A 的尺寸更小（435 MB），本来就在不补丁的限内（§31 已实测）✓
✗ 但这不是"字面的三文件链路"：A 的单文件（L0-11）已被清掉，
  本次是用 0-3 / 4-7 / 8-11 三个子模型覆盖它的层区间 ✓（要重导才能跑字面三段）
✗ 这是【冒烟测试】：除 hidden 与 KV 槽外，其它输入（mask3 / cos / sin 等）都是清零的，
  没有与 HF 对拍 ⇒ ★只证明"跑得起来"，不证明"算得对"★
★ 补丁路线（ptrace 改 4 条 securec 检查）仍属验证手段，不是可交付方案 ✗
```

**遗留的两件事**（要完整跑通三段还需要）：
```
① 重导 L0-11_wq.ms（455,682,136 B，与首次逐位一致 ✓ 见 §31 附注）→ 才能跑字面的三文件链路
② 重造主机侧输入（hidden/mask3/per_layer_*/cos_*/sin_* + ref_ids.txt）→ 才能做数值对拍
   （临时目录里的 io 产物已清掉：无 ref_ids.txt、无 seg 目录 ✗）
```

---

## 40. ★★★★ `--large-mem`：把 §32–§38 的补丁做成一条命令（含三个 lldb 实测坑）★★★★

§32–§38 把「2 GiB 上限」查清并验证了 patch 可行；本节把它做成发布包里的功能：
`scripts/start_chat.sh … --large-mem` / `scripts/start_server.sh … --large-mem`。

### 40.1 组成

```
src/cann_llm/large_mem.py        宿主侧：补丁表（单一来源）+ argv 构造 + 用户提示
src/cann_llm/large_mem_lldb.py   lldb 侧：装断点、改内存、等退出、记退出码
scripts/large_mem_run.sh         编排：gdbserver 起进程 → lldb 批处理接入 → 传退出码
launcher.py / launcher_server.py 接线 --large-mem（与 --lldb 同一套，优先于 --lldb）
scripts/start_{chat,server}.sh   用法说明；并把 bin/ lib/ python3/ 接到 PATH / LD_LIBRARY_PATH 开头
```

### 40.2 补丁表（★反汇编逐条核对过的真实字节★）

| 模块 | 偏移 | 原指令 | 补丁 | 说明 |
|---|---|---|---|---|
| `libhiai_ir.so` | `0xC16EC` | `e8 02 00 b5` | `1f 20 03 d5` | `cbnz x8`，x8 = destMax>>31 ⇒ 静态链入的 memcpy_s（★符号表把它归在 `AI_Log_Print` 名下★，因为它是本地符号） |
| `libsec_shared.z.so` | `0x3E50` | `09 03 00 b5` | `1f 20 03 d5` | `memcpy_s+0x1C`：`lsr x9,x8,#31` → `cbnz x9` |
| `libsec_shared.z.so` | `0x51DC` | ★`89 01 00 b5`★ | `1f 20 03 d5` | `memset_s+0x20`：同族上限门，**但立即数与 memcpy_s 不同** |
| `libsec_shared.z.so` | `0x5138` | `c2 00 00 54` | ★`06 00 00 14`★ | `memmove_s+0x1C`：`b.hs 0x5150` 是跳向【正常】路径 ⇒ **nop 会掉进 ERANGE，必须改成无条件跳转** |

★**踩过的坑**★：`memset_s` 那处最初照抄了 `memcpy_s` 的 `09 03 00 b5`，
运行期被"原字节不符"拦下（实际 `89 01 00 b5`）—— 这正说明**补丁前必须读回校验**：
同为 `cbnz x9`，跳转距离不同，编码就不同 ✓

### 40.3 实测（设备端）

```
① 672 MB 段(c12) 不带补丁：  ⑥ ModelBuildFromFile rc=-1  ✗
② 672 MB 段(c12) 带 --large-mem：
     [large-mem] ✓ libhiai_ir.so      +0xC16EC  e8 02 00 b5 → nop 已就位
     [large-mem] ✓ libsec_shared.z.so +0x03E50  09 03 00 b5 → nop 已就位
     [large-mem] ✓ libsec_shared.z.so +0x051DC  89 01 00 b5 → nop 已就位
     [large-mem] ✓ libsec_shared.z.so +0x05138  c2 00 00 54 → 06 00 00 14 已就位
     [large-mem] 补丁结果：成功 4 / 跳过 0（共 4 处）
     ⑥ rc=0 ✓   ⑧ ModelPredict rc=0（输出 5 个）✓
③ 退出码传递：被调试进程 exit(7) ⇒ 编排脚本返回 7 ✓；exit(0) ⇒ 0 ✓
```

### 40.4 ★三个 lldb 实测坑（都在实现时踩到）★

```
① 批处理模式下，"python 实现的命令"会【截断】后续 -o ✗
   lldb --batch -o A -o B -o "process continue" …
   一旦执行了 `script …` 或 `command script add` 注册的命令，
   后面的 -o 就不再执行 ⇒ `process continue` 根本没跑、进程一直停着 ✗
   （表现：日志停在装断点那行，然后 lldb 直接退出、返回 0）
   ⇒ 解法：把「装断点 + 放行 + 等结束 + 记退出码」全塞进【一条】python 命令
     （large_mem_run）✓

② 一条命令内要继续跑进程，必须用 SBListener 泵事件 ✗→✓
   SBProcess.Continue() 是异步的；紧接着轮询 GetState() 拿不到状态更新
   （事件没被泵）✗ ⇒ 用 SBListener.WaitForEvent() 等
   eBroadcastBitStateChanged，状态才会动 ✓

③ ★本机 /tmp 是【只读】的★
   退出码最初想用 mktemp 落地 ⇒ 直接失败 ✗
   ⇒ 改放仓库 .run/（已 gitignore，launcher 本来也用这个目录放 pid/state）✓

④ stdout 必须直通终端
   取退出码若去捕获 lldb 的 stdout，对话就变成"跑完才出字" ✗
   ⇒ 退出码走【状态文件】(CANN_LLM_LARGE_MEM_STATUS_FILE)，stdout 不拦 ✓
```

### 40.5 发布包布局（与本节配套）

```
bin/      lldb、huawei-debug-lldb-server            （3.6 MB）
lib/      liblldb.so + 8 个依赖                     （99 MB；★必须进 LD_LIBRARY_PATH★）
python3/  bin/python3（推理用，3.14）+ lib/（含 lldb 内嵌 python 3.11 的 stdlib）（422 MB）
          └─ lib/python3.11/lldb/_lldb.so → ../../../../lib/liblldb.so
             ★符号链接★：它与 liblldb.so 是同一份文件（前 64KB md5 一致），链接省 91 MB ✓
```

`liblldb.so` 的 RUNPATH 是 `$ORIGIN/../lib:$ORIGIN/../python3/lib`，且内嵌 python 按
`$ORIGIN/../python3` 推算 `PYTHONHOME` ⇒ **`python3/` 这个名字和位置不能改**，
`lib/python3.11/`（lldb 用）与 `lib/python3.14/`（推理用）各占一个版本子目录 ✓

### 40.6 落地后修掉的两个实测 bug（真机跑 `--large-mem` 才暴露）

```
① ★SBListener.WaitForEvent 的秒数是 uint32_t（整数），传 float 直接 TypeError★
     listener.WaitForEvent(1.0, event)
     → TypeError: in method 'SBListener_WaitForEvent', argument 2 of type 'uint32_t'
   为什么单测/短测没抓到：**它只在"进程活得够久"时才走到那行** ——
   短命探针（1~2 秒就退出）还没轮到就结束了 ⇒ 长驻的对话/服务进程必踩 ✗
   ⇒ 改成 WaitForEvent(1, event)；并在 tests/test_large_mem.py 里加**静态检查**
     （ast 扫源码，断言该参数是 int 字面量）—— 这类 SWIG 类型坑用假 lldb 也照不出来 ✓

② ★nnrt 后端下 libhiai_ir.so 是【第一次 BuildFromFile 之后】才加载的★
   （实测日志：`✗ 没找到已加载的 libhiai_ir.so` ⇒ 那处补丁被跳过 ✗）
   而 hiai/NDK 路径下它在首次 Build 时就已经在了 ✓ —— 两条路径加载时机不同。
   ⇒ 两处改动：
     · 模块没加载【不算错误】，只记 "pending"，断点先别禁用；
     · run 的等待循环里同时监听 SBTarget.eBroadcastBitModulesLoaded，
       新模块一出现就立刻补（幂等，只有真补上才打印）✓
   实测（用户的 5 段 gemma4 nnrt 模型）：
     第 1 次 Build：libsec_shared 三处 ✓（libhiai_ir 等到模块加载）
     第 2 次 Build：libhiai_ir ✓ + 另外三处"= 已补丁（跳过）" ⇒ ★4/4 全部生效★
     模型照常输出 bot> 2 ✓
   ★注意★：nnrt 下真正卡住大段的是**动态导入的 libsec_shared.memcpy_s**
     （适配层搬权重那条），它在首次 Build 前就打好 ✓；
     libhiai_ir 里那份静态 memcpy_s 属 IR/hiai 管线，晚一点补不影响大段单跑 ✓
     （c12 672 MB 单跑实测：首次 Build 即 4/4、rc=0 ✓）
```

### 40.7 ★★★★★ 交互式对话在 `--large-mem` 下卡死的真因与修法（实测）★★★★★

```
现象：./scripts/start_chat.sh -d … --large-mem      （交互式，不带 -p）
      you> 1+1=        ← 敲进去被回显了，但【没有任何推理输出】，一直卡着 ✗
      （同一模型的 -p "1+1=" 单轮却是好的 ⇒ 说明补丁本身没问题 ✓）
```

#### 真因：`gdbserver -- prog` 会给被调试进程**另开一个 pty**

```
被调试进程 fd0/fd1 → /dev/pts/11   ← 不是当前终端 ✗
lldb        fd0/fd1 → /dev/pts/7   ← 用户的终端被它握着
```
程序的**输出**由 lldb 转发到终端，所以看起来"能跑" ✓；但**键盘输入永远不会转发进那个私有 pty** ✗
⇒ 任何读键盘的程序（对话 REPL）必然卡死 ✗。（服务端/`-p` 单轮不读键盘，所以看不出 ✗）

#### 修法：改成 **attach**（程序自己启动、保留自己的终端），补丁打完就地放行

```
1. 驱动自己把程序起起来（终端是它的 ✓），并用 -c 包一层让它【启动即自停】SIGSTOP
   —— 确定性，无竞态（等 /proc/<pid>/stat 第 3 列变成 T 再继续）✓
2. gdbserver --native-regs 127.0.0.1:PORT --attach <pid>
3. lldb --batch
     -o "gdb-remote …"
     -o "process handle SIGSTOP -s false -p false"   ← ★见下★
     -o "command script import …"                    ← 断点在 import 时就装上 ✓
     -o "process continue" </dev/null                ← ★stdin 让开，别抢键盘★
4. 断点回调里打补丁、返回 False（继续）—— ★不要 detach★（见下）
```

#### 三个实测踩出来的关键点

```
① ★自停用的 SIGSTOP 是"待处理信号"，会在 continue 后【再投递一次】★
    现象：continue → 立刻又停在 SIGSTOP（frame 还在 kill 里）⇒
          批处理结束 ⇒ lldb 退出时把停着的进程【SIGKILL】✗（实测 "Signal 9"、rc=137）
    ⇒ 必须先 `process handle SIGSTOP -s false -p false`（不因它停、也不传递）✓

② ★不要在断点回调里 proc.Detach()★
    实测 lldb 会报
      error: Failed to resume process: process not in stopped state after
             synchronous resume: detached.
    之后程序【活着却卡住不再往下跑】✗
    ⇒ 补丁打完就 return False 继续跑；让"输入不被抢"靠 ③ 解决，而不是靠 detach ✓

③ ★lldb 的 stdin 必须让开（</dev/null）★
    程序用的是自己的终端 ✓，但 lldb 也握着同一个 tty ⇒ 用户敲的字会被
    lldb 的命令解释器抢走 ✗ ⇒ 把 lldb 的 stdin 指到 /dev/null 即可 ✓
    （batch 模式下 lldb 只在 -o 之间才碰 stdin，所以这样最省事 ✓）

④ ★POSIX 冷知识★：非交互 shell 里 `cmd &` 的 **stdin 默认被指到 /dev/null** ✗
    （POSIX 对"异步列表"的规定）⇒ 必须显式重定向一次：
      exec 3<&0 … "$PY" … 0<&3 &
    ★显式重定向优先于那条默认规则★ ✓（不写的话程序读到 EOF，直接 EOFError ✗）
```

#### 实测结果（用户那条命令，逐字）

```
$ ./scripts/start_chat.sh -d ../models/gemma4_5seg_s128 --large-mem
[large-mem] 附着到 pid=… （程序已自停，等打补丁）…
[large-mem] 命中 OH_AI_ModelBuildFromFile —— 开始打补丁
[large-mem] ✓ libsec_shared.z.so   +0x03E50 / +0x051DC / +0x05138  已就位
[large-mem] 补丁结果：成功 3 / 跳过 0 / 等模块加载 1（共 4 处）
[large-mem] 还有 1 处要等模块加载 —— 下次 Build 时自动补 ✓
cann-llm 0.1.0 · gemma4_5seg_s128 · nnrt · 加载 2.3s
you> 1+1=
[large-mem] 命中 OH_AI_ModelBuildFromFile —— 开始打补丁
[large-mem] ✓ libhiai_ir.so        +0xC16EC  e8 02 00 b5 → nop 已就位
[large-mem] = libsec_shared.z.so   三处「已补丁（跳过）」
[large-mem] 补丁结果：成功 4 / 跳过 0 / 等模块加载 0（共 4 处）
[large-mem] 补丁已就绪；程序带着自己的终端继续 ✓
bot> 2                                        ← ★交互式回答出来了★
```
★注意★：这个模型的段是**懒加载**的 —— 第一次推理时才第一次 `BuildFromFile`
⇒ 补丁正好在**该次 build 的入口**打上，赶在适配层搬权重之前 ✓
（`-p "1+1="` 单轮同样通过 ✓）

### 40.8 ★★★★★ runner「报到—等放行」：把补丁时机从"撞运气"变成"确定性"（并堵掉 §40.6 的短板）★★★★★

§40.7 的接法靠"程序启动即自停"，只能保证**启动**那一刻的同步；而 §40.6 记着一处短板：

```
nnrt 路径下 libhiai_ir.so 是【第一次 build 中途】才加载的
⇒ import 时补不到它 ⇒ 只能等第二次 build 才补上（单段超大模型就赶不上 ✗）
```

改成 **runner 主动报到**（应用侧配合，代码就是我们自己的 ✓）：

```
src/cann_llm/large_mem.py        rendezvous() / preload_targets()
src/cann_llm/backends/nnrt.py          build 之前调 rendezvous()
src/cann_llm/backends/gemma4_runner.py build 之前调 rendezvous()
```

#### 握手协议（一个环境变量给两边用）

```
CANN_LLM_LARGE_MEM_RENDEZVOUS = <放行文件路径>      ← 驱动 export，app 与 lldb 都继承 ✓
CANN_LLM_LARGE_MEM_WAIT       = <等待上限秒，默认 30>（调试器不来就自己走 ✗ 绝不永久卡住）

runner（第一次 build 之前）：
  ① ★先 dlopen 补丁目标的库★（libhiai_ir.so / libsec_shared.z.so）
     —— 这样那 4 处补丁**这次就都在**，不用等第二次 build ✓
  ② 等放行文件出现（已存在 ⇒ 立刻走 ✓；超时 ⇒ 打印警告后自己走 ✓）
lldb（`command script import` 时，进程此刻是停着的）：
  ③ 装断点（安全网）+ 就地补一遍 + **创建放行文件** ⇒ 取消 runner 的等待 ✓
  ④ `process continue`（原生命令）→ 程序带着自己的终端跑完 ✓
```

#### 三种时序都成立（这就是"确定性"的含义）

```
· lldb 先在：import 补 3/4（libhiai_ir 还没加载）→ 放行 → runner 看到标记直接走
             → 第一次 build 命中断点 ⇒ ★在该次 build 入口补上第 4 处★ ✓
· runner 先在：它已经预加载好 4 个模块 ⇒ import 时一次补满 4/4 ✓ → 放行 ✓
· 调试器一直没来：runner 等 WAIT 秒后自己走（这次没补丁），程序不卡死 ✓
```

★关键点★：`libhiai_ir` 那一处现在**总能在该次 build 的入口**补上 ——
因为 runner 在 build 前把它 dlopen 了，而断点正是在 build 入口触发的 ✓
（实测两次运行的日志都收敛到 `补丁结果：成功 4 / 跳过 0 / 等模块加载 0`）

#### 实测（用户那条交互式命令，逐字）

```
$ ./scripts/start_chat.sh -d ../models/gemma4_5seg_s128 --large-mem
[large-mem] 附着到 pid=36070 …
(lldb) gdb-remote 127.0.0.1:5092
[large-mem] 断点已装 ✓
[large-mem] ✓ libsec_shared.z.so   +0x03E50 / +0x051DC / +0x05138 已就位
[large-mem] 补丁结果：成功 3 / 跳过 0 / 等模块加载 1（共 4 处）
[large-mem] 已放行 runner ✓（它可以继续做第一次 build 了）
(lldb) process continue
cann-llm 0.1.0 · gemma4_5seg_s128 · nnrt · 加载 3.3s
you> 1+1=
[large-mem] 调试器已就位（放行标记已存在），直接继续 ✓        ← runner 报到
[large-mem] 命中 OH_AI_ModelBuildFromFile —— 开始打补丁
[large-mem] ✓ libhiai_ir.so +0xC16EC e8 02 00 b5 → nop 已就位  ← ★第 4 处赶在这次 build 里★
[large-mem] 补丁结果：成功 4 / 跳过 0 / 等模块加载 0（共 4 处）
[large-mem] 补丁已就绪；程序带着自己的终端继续 ✓
bot> 2                                                        ← ★交互式回答★
you>
```
`-p "1+1="` 单轮同样通过 ✓（rc=0）

### 40.9 服务端（`start_server.sh`）接入 `--large-mem`：外加两个实测修出来的毛病

```
scripts/start_server.sh -d <模型目录> --large-mem          # 前台
scripts/start_server.sh -d <模型目录> --large-mem -B       # ★后台也可以★（补丁日志进 log/）
```
服务端不读键盘 ⇒ §40.7 的"私有 pty"问题在它身上根本不存在 ✓；
`--large-mem` 的报到发生在**第一次真正 build**（也就是第一个请求进来时），
那时放行标记早就写好了 ⇒ runner 直接过 ✓，补丁在该次 build 入口落地 ✓（实测 4/4）。

#### 修出来的毛病 ①：服务端**不分后端**硬查 OMC 包的文件 ✗

```
run_server 里原本一律要求  executor.json / context.json / *.omc / SubGraph_*.weight
⇒ 那些文件只有官方 OMC 包才有 ⇒ hiai 自打包、nnrt 的分段 .ms 目录**根本起不来** ✗
（chat 路径 run_chat 从来不做这个校验 ✓）
```
⇒ 改成**与 run_chat 完全一致**：找到目录就打印，模型文件够不够**交给后端自己报**
（各后端要什么不一样，启动器猜不准；后端报的错也更准确 ✓）

#### 修出来的毛病 ②：`--stop` 收不干净（留孤儿，还占着 NPU ✗）

```
现象：--stop 等满 10 秒 → SIGKILL → 看着"已停止"，但 app/gdbserver/lldb 还活着 ✗
两个原因（都实测）：
· do_stop 只对**组长 pid** 发信号，而 --large-mem 下组长是**编排脚本**，
  它下面挂着被调试的服务 + gdbserver + lldb ⇒ 只杀组长就留下一串孤儿 ✗
· 而且它只等**组长**退出就收工 ⇒ 组里还有活进程也不管 ✗
⇒ 改成：按**进程组**收发信号（os.killpg），并等**整个组**消失
  （`killpg(pgid, 0)` 返回成功即"组里还有活的"），超时才整组 SIGKILL ✓
· 另外：被 ptrace 跟踪的进程收到 TERM 会**停在 ptrace-stop 里**，
  信号只是挂着不生效 ✗ ⇒ 编排脚本的 trap 改成：先摘调试器 → TERM 程序并重试 3 秒
  → 还不走就 KILL ✓
实测：--stop 从「10 秒 + SIGKILL + 留孤儿」变成 ★0.65 秒、0 残留★ ✓
```

## 41. 三段切法"能不能真聊天"的结论：★现有产物都不行，必须按 S=128 重导★

问题：有了 `--large-mem`（每段 ≲1.09 GB，§38）之后，gemma 是不是可以只分 3 段了？

### 41.1 尺寸上：成立 ✓（但要看量化与否，比例不同）

```
· int8（--fmk=ONNX + WEIGHT_QUANT）导出：IR ≈ 3.94 × .ms（权重按 fp32 展开）
    ⇒ 每段 .ms 必须 ≲ 1.09 GB（§38 实测）
· fp16（整图 fp16）导出：IR ≈ 1 × .ms（不展开）
    ⇒ 每段 .ms 可以 ≲ 4 GiB(minus 头)
  ★这解释了为什么 5 段 fp16 模型（mseg0 = 1.19 GB）不打补丁也能跑 ✓★
```
三段（12/12/11 层）的 int8 产物：435 / 672 / 696 MB ⇒ 都在 1.09 GB 内 ✓ 尺寸上成立 ✓。

### 41.2 实测上：**现有三段产物都不能聊天** ✗

```
· x570 ~/L0-11_wq.ms / c12_wq.ms / c24_wq.ms（int8，三段齐全 ✓）
  实测各段 inputs：hidden=24576 B、mask3=64 B、cos_sl=4096 B
    ⇒ ★全是 S=4 的"测试尺寸"图★ ✗（24576 = 4×1536×4）
  聊天要 S≥prompt（模板 ~66 token ⇒ 走 S=128）⇒ 结构上就不匹配 ✗
  表现：ValueError: 输入 hidden 大小不符: 786432 vs 24576（dtype=43, 6144 元素）
        （★我一开始把两个数字读反了 ✗★：786432 是"我喂的"、24576 是"模型要的"）

· x570 ~/g4m3/（fp16 三段，S 对 ✓）
  mseg0/seg.ms = 1,795,846,564 B ✓ 真段
  mseg12/seg.ms = 912,696 B ✗   mseg24/seg.ms = 773,600 B ✗  ← 失败残留
  ⇒ 只有第 0 段，另两段根本没导出来 ✗

· 本机 models/gemma4_3seg_s128（我拼的）已改名 models/gemma4_3seg_S4_probe
  —— 明确标出它是 S=4 探针目录，别误当聊天模型 ✗
```

**结论**：要"真正可聊天的三段目录"，必须**按 S=128 重导**（x570 上工具齐全：
`~/work/llm/.tmp/g4_export.py`、`g4_exp_lm.py`、`g4_exp_p.py` 那套 ✓）。
路线上两条都行：重导 **int8** 三段（IR 1.7~2.9 GB ✓ 需 `--large-mem` ✓），
或补齐 **fp16** 的 12/24 段（IR ≈ .ms ✓）。

### 41.3 这一轮顺带修掉的三个真问题（都在 runner 里，已实测）

```
① ★合并切法不能再硬编码★：MERGED 分支原本写死 SEG_STARTS=(0,8,16,24,30)（5 段切法 ✗）
   ⇒ 改成从目录名推断：mseg0 / mseg12 / mseg24 …
     层数 = 相邻起点之差、最后一段到 35 ✓
   实测：三段目录解析出 SEG_STARTS=(0,12,24)、SEG_NO={0:12,12:12,24:11} ✓

② ★KV 槽改成数据驱动★：原本 `if st >= 16: feeds.update(kv)`（只对 5 段成立 ✗）
   ⇒ 改成 `if kv:`（谁声明谁用；多喂用不到的键无害、少喂会 KeyError ✗）
   依据（实测 inputs 数）：c12 = 6+12 = 18（★不声明槽★，槽由它自己的 13/14 层产生 ✓）
                            c24 = 6+11+4 = 21（★声明 4 个槽★）

③ ★★device id 必须"按模型"决定★★（这是本轮挖出的大坑）
   · int8 段图：不设 device id ⇒ Build 直接 -1 ✗；显式设了才 0 ✓
   · fp16 图（graphP / 5 段那批）：设了 device id 反而 Predict 失败 ✗
   · 而且"枚举 NNRt 设备"这个动作本身有副作用：一上来就
     OH_AI_GetAllNNRTDeviceDescs，会把后续 graphP 的 Predict 弄坏 ✗
   ⇒ 解法：**懒枚举 + 按模型重试** —— 先按默认建，失败才枚举 id 并重建 ✓
     （实测：5 段回归恢复 bot> 2 ✓；int8 段图日志出现
       "[nnrt] mseg0_s128 需要显式 device id ⇒ 已用它建成 ✓"）
```

## 42. ★★★★★ 实测：三段 int8 的 OOM 到底吃在哪（建完 2 段就停，量出来的）★★★★★

背景：三段切法（0-11 / 12-23 / 24-34）在**加载第三段时把设备打爆**（32 GB 内存耗尽、
swap 暴涨、低内存告警 ⇒ 重启 ✗）。为定位，临时在 runner 里加了个 env 开关
（`CANN_LLM_DEBUG_HOLD_AFTER`：建满 N 个段图后自报内存账目并暂停，**量完即撤** ✓），
分别量 int8 三段与 fp16 五段"建完 2 段"时的占用 ✓。

### 42.1 数字

| 模型（建完 **2 段**后） | 两段 `.ms` 合计 | **VmRSS** | **VmSwap** | 大映射（≥256 MB） | 放大 |
|---|---|---|---|---|---|
| **5 段 fp16**（已知能跑 ✓） | 2.52 GB | **7.43 GB** | **0** ✓ | 4608+3328+672+520+480+278+160 MB | ≈ **2.9×** |
| **3 段 int8**（爆掉的 ✗） | 1.14 GB | **11.42 GB** | **10.5 GB** ✗ | ★9216+8056+6720+4184+3584 MB★ | ≈ **19×** ✗ |

系统同时刻：`MemAvailable` 掉到 **2.3 GB** ✗、`SwapFree` 从 50 GB 掉到 38 GB ✗
（= 十几 GB 已换出 ✓，与用户看到的"swap 极速上涨"一致 ✓）。

### 42.2 这些内存是【谁的】

```
· 我们这边没有复制 .ms ✗：_Mslite 只把【路径】交给 OH_AI_ModelBuildFromFile ✓
· weights/ 那几个大文件是按行 seek/read ✓（embed_tokens_per_layer.f16 4.70 GB 只读一行 512 B ✓，
  走 page cache、可回收 ✓，RssFile 只有 0.24 GB ✓）⇒ ★mmap 这条建议我们已经等价做到了★ ✓
· 大块全是 [anon:native_heap:jemalloc] —— 即【进程内、DDK/适配层申请的堆】✗
  （§38 抓到的那个 3994 MiB 就是同一类 ✓）＋ smaps_rollup 里 Pss_HHP 4.3 GB（huge pages ✓）
· ★建完不释放★：这些映射一直活到进程结束 ✗（swap 里都留着 ✓）
```

⇒ **int8 路线上限极低**：三段 `1.89 GB × ~19 ≈ 36 GB` ✗ ⇒ 必然 OOM ✓（实测吻合 ✓）。
而 int8 那两个 **9.2 / 8.1 GB** 的映射，尺寸与 `per_layer` 表（262144×8960×4 B = **9.39 GB**）
高度吻合 —— 强烈怀疑**这条路把 PLE 表在主机内存里具化了** ✗（fp16 那条路没有：PLE 是输入 ✓）。

### 42.3 由此得到的方案判据（都基于实测）

| 方案 | 预计常驻 | 结论 |
|---|---|---|
| **三段 int8**（converter WEIGHT_QUANT ✗） | ~36 GB ✗ | **不可行** ✗（除非消掉那 9.4 GB 的来源）|
| **三段 fp16**（同样导出、去掉量化 cfg ✓） | 两段 7.43 GB ⇒ 三段 ≈ **10~11 GB** ✓ | **可行** ✓（远低于 32 GB ✓）；`.ms` 合计 ≈ 3.7 GB ✓ |
| **五段 fp16**（现状 ✓） | 7.46 GB `.ms`，**已验证能跑** ✓ | 零风险 ✓ |
| **dopt + OMG + hiai**（Qwen3-8B 那条 ✓） | 8B 只要 **5.1 GB** ✓ | 最省 ✓，但要把 runner 从 `.ms`/NDK 换到 hiai/omc ✗ |

### 42.4 唯一可能"释放"的杠杆（待验证）

```
runner 现在【从不】调用 OH_AI_ModelDestroy（docstring 明说"不 destroy" ✓）
⇒ 段图建完就常驻到进程结束 ✗
值得一试：每段用完 destroy，看那几块 jemalloc 映射是否回落 ✓
  · 回落 ⇒ 说明能"用完即卸"⇒ 三段 int8 也许都能跑 ✓（只是每段重建要付 Build 时间 ✗）
  · 不回落 ⇒ 那些缓冲是 DDK 全局的，只能靠换路线 ✓
```

## 43. 三段（12/12/11 层）的三条路线实测 —— ★其中路线③的结论已被 §46 推翻，务必先读 §46★

> [!IMPORTANT]
> **本节路线③（OMG）当时的结论是错的 ✗，已在 [§46](#46-修正-43-的一个错误结论omg-没有模型大小上限大模型走的是外置权重形态) 更正。**
> 当时把「OMG 产出的 `omc` 只有 916 KB」误判成「静默截断 / 空壳」✗ ——
> 实际那是 **图与外置权重分离** ✓（`models/Qwen3-8B` 的 `SubGraph_0.weight` **4.39 GB** + `omc.omc` 6.8 MB 就是同一形态 ✓）。
> ⇒ **OMG 没有模型大小上限** ✗（8B 都能转，哪来的 2 GB 上限 ✗）。
> 12 层段能不能转，取决于**外置权重形态是否与转换器对齐** ✓（开关是 OMG 的 `--weight_merge` ✓，待验证）。
> 本节路线 ① ② 的结论**仍然成立** ✓（int8 直转 19× 主机内存 ✗；fp16 I/O 段图设备 Build -1 ✗）。

目标：用 `--large-mem` 打掉 2 GiB 限制后，把 gemma 从 5/9 段减到 **3 段**（12/12/11 层）。
结论（更正后）：路线 ① ② 堵死 ✗；**路线 ③ 未被证伪，待续** ✓。

### 43.1 路线①：`converter_lite --fmk=ONNX + WEIGHT_QUANT`（int8）★能建，但内存爆★

```
产物：435 / 705 / 730 MB（S=128 ✓ 原生 ✓ Build 0 · Predict 0 ✓ 单段验证通过）
★但主机内存放大 ~19×★（§42 实测：建完 2 段 = VmRSS 11.42 GB + swap 10.5 GB ✗）
  两个 9.2 GB / 8.1 GB 的 jemalloc 映射，尺寸与 per_layer 表（262144×8960×4 = 9.39 GB）
  高度吻合 ⇒ 怀疑这条路把 PLE 表在主机内存里具化了 ✗
⇒ 三段预计 ~36 GB ✗ ⇒ 加载第三段必 OOM ✓（与实测现象一致 ✓）
```

### 43.2 路线②：`DTYPE=fp16` 导出 + `converter_lite --fmk=ONNX`（fp16 I/O）★能转，设备拒收★

```
产物：898 MB / 1.39 GB / 1.44 GB（合计 3.73 GB ✓，转换仅 4 分钟 ✓）
★但设备上 Build rc=-1 ✗（输入数 0）★
⇒ 与文档早先的记录完全一致（gemma4-on-npu.md：段图 真实 I/O dtype=FP16 ⇒ ✗ Build 返回非 0）
⇒ ★段图必须 fp32 I/O★ —— 这条是硬约束 ✓
```

### 43.3 路线③：OMG（`--weight_data_type FP16` ⇒ fp16 权重 + fp32 I/O）★小段能过；12 层段的结论见 §46★

```
s0（12 层，fp32 ONNX 1.79 GB）⇒ OMG ok=1 ✓ converter SUCCESS ✓ seg.ms 1.80 GB ✓ 真能跑 ✓
s12（12 层，fp32 ONNX 2.78 GB）⇒ OMG ok=1 ✓ 但 ★q/seg.omc 只有 916 KB★
s24（11 层，2.88 GB）         ⇒ 776 KB
★当时的（错误 ✗）解读★：以为 OMG "静默截断" ⇒ 断定它只吃 ≤8 层
★更正（§46）★：
  · OMG ★没有大小上限★ ✗；它在大模型上把权重【外置】成独立文件 ✓
    （官方形态叫 SubGraph_0.weight ✓ —— Qwen3-8B 的 4.39 GB 就是它 ✓）
  · 916 KB 的 omc 只是【图】✓ 不是空壳 ✗
  · 真正的问题是：OMG 散落在 cwd 的逐张量外置权重（onnx__MatMul_*）
    ★转换器（converter_lite --fmk=THIRDPARTY）不吃★ ✗ ⇒ seg.ms 只有 918 KB ✗
    ⇒ 控制开关是 OMG 的 `--weight_merge`（"weight data will be merged in IR model" ✓）
  · 另外：torch 导出的大 ONNX 自己也会外置数据 ✓
    （12 层的 seg.onnx 只有 218 KB + 152 个 onnx__MatMul_* ✓，命名规则 onnx::MatMul_2081 → onnx__MatMul_2081 ✓）
    ⇒ 搬迁/复用时★必须整目录带上★，只搬 .onnx 会让 OMG 报 ParseOriginONNX2IrGraph FAIL ✗
```

### 43.4 还试过一个"组合拳"：fp16 小 ONNX + OMG 显式声明 FP32 I/O ✗

```
动机：ONNX 小（1.39 GB ✓ 过得了 OMG）+ 声明 FP32 I/O（避开 43.2 的硬约束）
实测：★OMG ok=0 ✗★（converter 随之失败）⇒ OMG 不接受"网络 fp16 但 I/O 声明 fp32"
```

### 43.5 结论与出路（表格中 OMG 一行已按 §46 更正）

| 路线 | 能建？ | 内存 | 结论 |
|---|---|---|---|
| int8（ONNX 直转） | ✓ | ~19× ✗ | 三段必 OOM ✗（**此结论不变** ✓） |
| fp16 I/O（ONNX 直转） | ✗ Build -1 | — | 段图必须 fp32 I/O ✗（**此结论不变** ✓） |
| OMG（fp16 权重 + fp32 I/O） | ✓ 小段已通 ✓ | ~2.9× ✓ | ★12 层段**未被证伪**★ ✓ —— 卡在"外置权重形态未与转换器对齐" ✓（§46）；`--weight_merge` 待验证 |
| dopt + OMG + **hiai**（Qwen3-8B 那条 ✓） | ✓（8B 已证 ✓） | 8B 仅 5.1 GB ✓ | 最省内存的路 ✓；官方流程产出的就是转换器友好的 `SubGraph_0.weight` 形态 ✓ |

**短期建议**：维持 5 段（7.46 GB `.ms`、~2.9×、**已验证能跑** ✓）—— 在 §46 的 `--weight_merge` 验证通过前，三段仍不可用 ✓。
**要更少的段 / 更小的常驻**：① 先把 `--weight_merge` 配对（10 分钟级 ✓，能成就回到"三段 + `--large-mem`"验证 ✓）；② 或走官方 **dopt(W4) + OMG + hiai**（§42 表最后一格 ✓）。

## 44. 本机"能跑多大的模型"——实测能力边界（Qwen3.8 之类的问题）

### 44.1 已实测能跑的

```
Qwen3-8B（models/Qwen3-8B，5.1 GB，hiai 路）
  ★实测：in 66 tok · out 171 tok · prefill 307 ms · decode 12.8 tok/s ✓（回答正确 ✓）★
gemma4 五段（.ms 7.46 GB，nnrt + fp16 权重/fp32 I/O）✓
```

### 44.2 实测的失败边界

```
gemma4 三段 int8：建完 2 段 = VmRSS 11.42 GB + swap 10.5 GB ⇒ 第三段触发 OOM Killer ✗（§42）
⇒ 本机【实际可用】的量级 ≈ 5~8 GB（模型常驻），
   到 ~11 GB 就已经贴着 32 GB 内存的边 ✗
```

### 44.3 Qwen3.8 系列的尺寸账（HuggingFace 实测）

| 模型 | bf16 权重 | 参数 | W4 后权重 | 本机 |
|---|---|---|---|---|
| **Qwen3.8-27B** | 55.6 GB | ≈27.8 B | **≈13.9 GB** ✗ | ✗ 跑不动（权重就超线 ✗） |
| Qwen3.8-Flash-Next | 360 GB | ≈180 B（MoE） | ≈90 GB ✗ | ✗ |
| Qwen3.8-2.4T-A95B | — | 2.4 T（MoE） | — | ✗ |
| 社区 35B-A3B 蒸馏 | — | 35 B（MoE） | ≈17.5 GB ✗ | ✗（MoE 全部专家都要常驻 ✗） |
| 社区 2.6B 蒸馏 | — | 2.6 B | ≈1.3 GB ✓ | 尺寸可以 ✓，但**只有 GGUF 形态** ✗ |

**关键点**：MoE 的"激活参数小"✗ 救不了内存（**所有专家都要常驻** ✗）；
而小尺寸蒸馏**只有 GGUF** ✗（llama.cpp 格式），hiai/NPU 引擎吃的是 omc/`.ms` ✗ ——
要转必须先有 **safetensors 基座** ✓ 再走 dopt → OMG → hiai ✓（且转换链对结构极敏感 ✗）。

### 44.4 结论

```
· 本机能力边界 ≈ ★8B 级 W4（5 GB 左右）跑得舒服；13B 级 W4（≈7 GB）勉强可能★
· Qwen3.8-27B 及以上 ✗ 不是选型问题，是内存的硬账（W4 权重 13.9 GB > 本机 ~11 GB 的实测线）
· 想让更大的模型上机：① 13B 级 + dopt/hiai（有风险 ✗）；② 等官方出小尺寸 omc 包 ✓
· 现成可用：Qwen3-8B（5.1 GB ✓ 12.8 tok/s）、qwen25_coder_7b（4.3 GB）、qwen15b_e2e（3.2 GB）
```

## 45. GGUF → ONNX：★可复现的配方★（实测 LFM2.5-2.6B 成功），但转 NPU 可行性低

### 45.1 配方（全部实测通过 ✓）

```bash
# 环境：x570 的 venv-msllm（transformers 4.57.6 + gguf ≥0.19 + torch 2.14.1 + onnxruntime）
pip install -U gguf            # ★0.17.1 太老：MODEL_ARCH 里没有 LFM2 ✗；0.19.0 有 LFM2/LFM2MOE ✓
pip install onnxruntime onnxscript

# ① 直接读 GGUF（★不需要手写 HF↔GGUF 映射表★ —— transformers 自带）
m = Lfm2ForCausalLM.from_pretrained(dir, gguf_file="model-bf16.gguf")   # ✓ 266 张量自动反量化+映射
# ② ★必须用新版 dynamo 导出器★
torch.onnx.export(m, (ids, pos), "x.onnx", dynamo=True, opset_version=18, ...)
#    旧 TorchScript 导出器会 ★RuntimeError: unordered_map::at★ ✗（卡在 masking_utils 的 vmap 路径）
#    且要绕开 attention_mask（用一个只暴露 input_ids/position_ids 的 wrapper ✓）
# ③ onnxruntime 数值校验 ✓
```
**实测结果**：误差 **8.58e-05** ✓（数值一致）；图 2.93 MB + 外置权重 **10.79 GB**（fp32 = 2.7B×4B ✓）；
1417 节点 / 35 种算子 ✓。想变一半大小：`m.half()` 后再导 ✓。

### 45.2 但转 NPU 的判定：★低可行性★

```
该 ONNX 需要的算子          DDK 平台库（libai_npucore_fusionengine_internal.so）
  Conv×22（★Conv1D★）   →   Conv2D ✓ / Conv1D ✗（需 OMG 转换，未见支持）
  ★IsNaN×8★             →   ✗ 缺
  ★CumSum×1★            →   ✗ 缺
  ★GatherND×1★          →   ✗ 缺
  Where ✓ Expand ✓ Range ✓ Slice ✓ Pad ✓ Softmax ✓ MatMul ✓
⇒ 三类算子缺失 ✗ + 动态形状（Slice×81 · Shape×10 · Range×1）
  ＋ 10.8 GB（fp32）/ 5.4 GB（fp16）的规模 —— ★但注意：规模不是"OMG 上限"问题✗★
     （OMG 对大模型走外置权重 ✓，见 §46 ✓）；真正的规模障碍是
     ① OMG 散落的外置权重要与转换器对齐（`--weight_merge` ✓ 待验证）
     ② 必须分段才能上 NPU（LFM2 每层还有 conv 状态要做成显式 I/O ✓）
```

### 45.3 结论

```
· GGUF → ONNX ★技术上完全可行★ ✓（配方见 45.1，半小时级）
· 但"GGUF→ONNX→NPU"这条路对本例（LFM2.5 混合 conv/attention 架构）★不现实★ ✗
· 只有 GGUF 的模型，最现实的本机跑法仍是 ★llama.cpp 直接跑★ ✓
  （本机已编译：CPU 正确 ✓；参考速率 Qwen2.5-1.5B Q4_K_M = 1.62 tok/s ⇒ 2.6B ≈ 1 tok/s ✗）
· 另注：这类"Qwen3.8 蒸馏"多半不是 Qwen3.8 架构 ✗（本例基座是 LFM2.5 ✓，标签 lfm2.5 ✓）
```

## 46. ★修正 §43 的一个错误结论★：OMG 没有"模型大小上限"——大模型走的是【外置权重】形态

§43 里我写"OMG 对 ≥~2 GB 的 ONNX 会静默截断 ✗" —— **这是错的** ✗。
证据（用户一问点破 ✓）：`models/Qwen3-8B` 的 `SubGraph_0.weight` 有 **4.39 GB** ✓，
而它正是 OMG 那条链转出来的 ✓ —— 8B 都转得出，哪来的 2 GB 上限 ✗。

### 46.1 真相

```
OMG 的输出有两种形态：
  · 小模型：权重【内联】进 .omc ✓（omc 文件本身就很大）
  · 大模型：权重【外置】成独立文件 ✓，.omc 只剩图（可以只有几百 KB）
    - Qwen3-8B：SubGraph_0.weight 4.39 GB + omc.omc 6.8 MB ✓
    - 我那次 12 层段：q/seg.omc 916 KB + ★152 个 onnx__MatMul_* 外置文件★ ✓
⇒ ★"omc 只有 916 KB" 不是失败，是"图与外置权重分离"★ ✓（我误判成空壳 ✗）
```

同理，**torch 导出的大 ONNX 也会把权重外置** ✓：
``` 
g4_export.py 的 12 层导出留下 ★seg.onnx 218 KB + 152 个 onnx__* 文件★ ✓
（命名规则：onnx::MatMul_2081 → onnx__MatMul_2081 ✓）
⇒ 只搬 seg.onnx 会让 OMG 报 ParseOriginONNX2IrGraph FAIL ✗（我踩过 ✓）
```

### 46.2 我真正卡住的地方（与"大小"无关）

```
① use 外置数据：必须把 .onnx 与它的 onnx__* 外置文件【一起】用 ✓（OMG 按 onnx 所在目录找 ✓）
② OMG 产物 → converter_lite --fmk=THIRDPARTY：转换器要的是【聚合权重】形态
   （官方产物就叫 SubGraph_0.weight ✓，`g4_all16.sh` 的 conv() 正是这么拷的 ✓）
   而 OMG 在 cwd 里散落的逐张量 onnx__* 形态 ✗ 转换器不吃 ✗ ⇒ 我得到的 seg.ms 只有 918 KB ✗
   ★控制这个的开关是 OMG 的 `--weight_merge`★（help: "weight data will be merged in IR model" ✓）
   —— 我试了两次都还没配对（一次参数出错 ✗、一次外置数据没带全 ✗），★待续★
```

### 46.3 结论修正

> 📌 这条知识已同步进转换文档：[model-conversion.md §0.1](model-conversion.md)
> （面向"要转换模型的人"写了两条常见误判：看 omc 小就以为失败 ✗、搬迁 ONNX 漏掉外置数据 ✗）

```
· ✗ 旧结论："OMG 有 ~2 GB 上限 ⇒ 12 层段转不出"
· ✓ 新结论："OMG 对 12 层段没问题 ⇒ 只是【外置权重形态】要与转换器对齐"
  ⇒ ★"三段（12/12/11）"在 OMG 路上并未被证伪★ —— 之前 §43 的否定只对
    "int8/ONNX 直转（19× 主机内存 ✗）"与"fp16 I/O（设备 Build -1 ✗）"这两条成立 ✓
· 最靠谱的对齐方式：照【官方 dopt 流程】（它产出的就是 converter 友好的
  SubGraph_0.weight 形态 ✓ —— Qwen3-8B 就是这么来的 ✓）
  dopt 量化 → 官方导出 → OMG(--compress_conf) → SubGraph_0.weight → converter_lite ✓
```

## 47. ★★★★★ 官方 dopt(W4) + OMG + hiai 链路：端到端跑通 ★★★★★

从 HF 检查点出发，**我们自己**产出了一个能在设备 NPU（hiai 引擎）上聊天的 W4 模型包 ✓。

```
模型：Qwen/Qwen2.5-1.5B-Instruct（W4 / group 128 / act 16）
实测：bot> 2 ✓
      '用一句中文说明你是什么模型。' ⇒ "我是一个由阿里云开发的文本生成模型。" ✓
      [in 74 tok · out 12 tok · prefill 603 ms · decode 6.0 tok/s] ✓
产物：SubGraph_0.weight 3104 MB + qwen2_1p5b_w4.omc 3.0 MB
      + embedding_weights 233 MB(int8) + embedding_dequant_scale 0.6 MB + tokenizer 7 MB = 3.2 GB
      ★与 models/Qwen3-8B 完全同形态★ ✓ ⇒ 再次印证 §46「大模型 = 图 + 外置权重」✓
```

### 47.1 完整配方（转换机 = RTX 3080 Ti）

```bash
# 0) ★先看显存★：dopt 必须 CUDA。实测踩坑：GPU 被别的进程占用
#    （当时是游戏占 9.4 GB / 12 GB）⇒ stage1 直接 CUDA OOM ✗
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 1) 官方示例仓库（量化模板 + 导出脚本都在里面）
git clone --depth 1 https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp.git
#   → CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/export_model_single_{qwen2,qwen3,glm}.py
#   → 同目录还有 model_info_target.yaml 模板 ✓

# 2) HF 检查点（1.5B ≈ 3.1 GB，转换机可直连 HF ✓）
#    config.json · model.safetensors · tokenizer.json · tokenizer_config.json · vocab.json · merges.txt

# 3) dopt 三阶段（DDK 里：tools/tools_dopt/dopt_pytorch_py3/dopt/dopt_lm/opt_main.py）
#    config.yaml 关键项：★quant_param_2: False★（kirinx90；写 True ⇒ 权重负半轴被钳 0 ⇒ 输出恒定垃圾 ✗）
#    首跑会"先生成 dopt_config.json 再退出" ✓（31100 B）
#    再用仓库脚本填量化策略：
#      scripts/model-conversion/set_quant_strategy.py output_dir/dopt_config.json
#      ⇒ ★196 个 Quant_act_weight_eco + 2 个 float★（28 层 × 7 个 Linear = 196 ✓ 完全对上）
./run.sh stage1 && ./run.sh stage2 && ./run.sh stage3
#    实测耗时（1.5B/3080Ti）：stage1 2m39s · stage2 20s · stage3 2m24s ✓
#    产物：fake_quant_weight.pth 3087.6 MB + quant_params_file 581.6 MB
#    ★验权重没被钳位★：check_quant_clamp.py ⇒ 198 张量 · 负值 43.98% · 全非负 0 个(0%) ✓ 正常

# 4) 导出 + onnxsim + OMG + 装配：一条命令（仓库脚本 build_model.py）
python3 scripts/model-conversion/qwen/build_model.py \
    --hf-model <HF 目录> --quant-pth <fake_quant_weight.pth> --dopt-config <dopt_config.json> \
    --export-dir <npu_tuned_export> --omg-dir <tools_omg> --asc-dir <tools_ascendc> \
    --name qwen2_1p5b_w4 --workdir <工作目录> --kv-len 2048
#    ★它保证 KV 长度三处一致★：yaml / OMG --input_shape / executor.json ⇒ 实测都是 2048 ✓
#    （KV 是编译期属性：改了必须重走导出+OMG ✓）

# 5) 设备侧补两个文件（转换机上没有仓库 src/）
PYTHONPATH=src python3 -m cann_llm.modelpkg <模型目录>
#    ⇒ 写 api_config.json + <model>.json（采样参数 / 停止符 / chat template；
#       <model>.json 必须与 .omc 同名 ✓）

# 6) 跑
scripts/start_chat.sh -d <模型目录> -p '1+1='
```

### 47.2 意义与边界

```
· ★我们从此能自己产出"hiai 能吃的 W4 模型包"★ ✓（不必依赖官方发布的模型包 ✓）
· W4 打包 + 外置权重 = 目前最省内存的形态 ✓（Qwen3-8B 5.1 GB 同族 ✓）
· 边界：官方流程只支持 qwen2 / qwen3 / glm 架构 ✗ ⇒ gemma4 走不了这条 ✗
  （gemma4 若要"少段/省内存"，只能回到 §§41–46 那几条自建路线 ✓）
· 想给设备加新模型（架构在支持列表内、尺寸在本机能力边界内 §44）现在有可复现路径 ✓
```

## 48. ★★★★★ 在本机运行 Qwen3.8（真·qwen3_5 架构）★— llama.cpp CPU 路线跑通 ★★★★★

**目标**：运行 **Qwen3.8**（新家族，不是 "Qwen3 8B" ✗）。蒸馏/量化不限、后端不限。
**结果**：✓ 跑通了 —— 用 `empero-ai/Qwen3.8-2B-Distill`（Q4_K_M，1.31 GB）在**本机 llama.cpp（CPU）**上：

```
问：用一句中文说明你是什么模型。
答：我是 Qwen3.5，一个基于 Qwen3 架构的超大规模语言模型。   ← 家族内部命名即 Qwen3.5 ✓
问：中国的首都是哪里？只回答城市名。    答：北京 ✓
速度：prompt 3.57 t/s · decode ★1.70 t/s★（偏慢 ✗，但正确 ✓）
```

### 48.1 关键事实：这些"Qwen3.8 蒸馏"就是【qwen3_5 新架构】本身

```
empero-ai/Qwen3.8-{2B,4B,9B}-Distill（有 ★safetensors★ ✓，不只是 GGUF ✓）
  config.json: architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5"
  text_config: hidden 2048 / head_dim 256 / attn_output_gate / linear_conv_kernel_dim 4
  ★layer_types = [linear_attention ×3, full_attention] ×6 ⇒ 混合线性注意力（Gated DeltaNet）★
  （标签 qwen3.5 · gated-deltanet ✓；__不需要__自定义建模代码，依赖 transformers 内置 ✓）
safetensors 体积：2B=4.5 GB · 4B=9.3 GB · 9B=19.3 GB
GGUF（2B）：Q4_K_M 1.31 GB · Q5 1.45 · Q6 1.61 · Q8 2.08 · BF16 3.90
★注意★：官方 Qwen3.8 家族只有 27B / Flash-Next(~180B MoE) / 2.4T-A95B ✗
        ⇒ 2B/4B/9B 这些是【社区蒸馏】✓（架构忠实于 qwen3_5 ✓）
```

### 48.2 本机 llama.cpp 路线（已跑通 ✓）

```bash
# ① 本机可直连 HF ✓，直接下 GGUF
curl -L -o models/qwen38_2b_q4/Qwen3.8-2B-Q4_K_M.gguf \
  https://huggingface.co/empero-ai/Qwen3.8-2B-Distill-GGUF/resolve/main/Qwen3.8-2B-Q4_K_M.gguf

# ② ★libomp shim★（不补会满屏 "symbol not found: __kmpc_*" ✗）
cd llama.cpp && mkdir -p libshim
ln -sf <ohos-sdk>/llvm/lib/aarch64-linux-ohos/libomp.so libshim/libomp.so
export LD_LIBRARY_PATH="$PWD/libshim:$PWD/build/bin"

# ③ 跑（本机 llama.cpp 是 2026-09-20 的 b23efaa ✓，★已支持该架构★
#    源码里叫 ★qwen35★ ✓ —— 用 "qwen3_5" 去 grep 会误判成"不支持" ✗）
./build/bin/llama-bench -m <gguf> -ngl 0 -t 16 -p 32 -n 16 -r 1     # qwen35 2B Q4_K ✓ 被识别
./build/bin/llama-server -m <gguf> -ngl 0 -t 16 -c 2048 --port 8123  # 起服务，curl 聊天 ✓
```

### 48.3 ★Vulkan(GPU) 依然算错 ✗★（再次确认 README 的结论）

```
CPU    （-ngl 0） : '北京' ✓   1.72 tok/s
Vulkan （-ngl 99）: 乱码/空 ✗  4.01 tok/s（快 2.3×，但结果不可用 ✗）
```
⇒ Maleoon 916 的 Vulkan 计算着色器在这个驱动上仍然不正确 ✗（README 已记 ✓，本次在 qwen35 架构上复验 ✓）。

### 48.4 后续提速的三条可能（按性价比）

```
① llama.cpp CPU 调优 ✓ 便宜：README 记"CPU 后端比同类 ARM 核慢约一个数量级"✗
   ⇒ 可能缺平台最优 kernel（i8mm/dotprod/SVE）⇒ 试带 GGML_CPU_ARM_ARCH 的重新编译
   预期：3~10× ⇒ 5~17 tok/s，那就"可用"了 ✓
② NPU（hiai/cann/nnrt）✓ 慢工：需要
   ① transformers 先升级到认识 Qwen3_5 的版本 ✗（本地 4.57.6/4.51.0 都【没有】Qwen3_5 类 ✓）
   ② 自己导出 ONNX，③ 清点算子 —— ★线性注意力（gated delta rule / conv1d / 状态传递）大概率
      不在 DDK 平台库里 ✗★（对照 §45 LFM2 的遭遇：IsNaN/CumSum/GatherND 缺 + Conv1D 无 ✓）
   预期：若通，10~20 tok/s 级 ✓（参照 Qwen3-8B 的 12.8 ✓）；但成功率低 ✗
③ 官方 dopt+OMG+hiai ✗ 不通：官方流程只支持 qwen2/qwen3/glm ✗（§47）⇒ qwen3_5 走不了 ✓
```

## 49. ★★★★★ Qwen3.8（`qwen3_5`）上 NPU 的判决：现成工具链不通 ✗（缺 4 个关键算子）★★★★★

路线：自己导出 ONNX（transformers 5.18 已认 `Qwen3_5` ✓）→ 喂 OMG/converter_lite → hiai/nnrt。

### 49.1 关键路径都打通了 ✓

```
· transformers 升级到 5.18.0 后有 Qwen3_5Config/Qwen3_5ForCausalLM/Qwen3_5Model ✓
· empero-ai/Qwen3.8-2B-Distill（safetensors 4.5 GB）★AutoModelForCausalLM 载入成功★ ✓
    → 类 Qwen3_5ForCausalLM，1.88 B 参数；text_config: 24 层 / hidden 2048 / 8 头 / head_dim 256 / vocab 248320
· 导出 ONNX：必须用 dynamo 导出器 ✓，且★要包一层只返回 logits★
    （transformers 5.x 的输出里带 DynamicCache ⇒ dynamo 报 "not a known type" ✗）
· 架构核心 = causal_conv1d + ★chunk_gated_delta_rule★（Gated DeltaNet 的分块递推）
    —— 转换机上两者都退回参考 PyTorch 实现（没装 causal-conv1d / flash-linear-attention）✓ 不影响导出 ✓
```

### 49.2 判决：★4 个算子 DDK 没有 ✗★

用微型同构 config 模型（2 层）秒级导出拿到算子集 ✓，再逐个对 DDK 平台库核验：

```
qwen3_5 的 ONNX：节点 2016 · 27 种算子
  Transpose×397 Slice×382 Unsqueeze×262 Mul×174 Add×147 Gather×135 ReduceSum×130
  ★ScatterElements×128★ ★ScatterND×126★ MatMul×31 Reshape×24 Sqrt×11 Reciprocal×11
  Pad×10 Sigmoid×8 Pow×7 ReduceMean×7 Where×4 Sub×4 Exp×4 Conv×2 Split×2
  Softplus×2 Greater×2 ★CumSum×2★ ★Trilu×2★ Neg×2

DDK 平台库（libai_npucore_*）：
  ✓ 有 23 个
  ★缺 4 个：ScatterElements · ScatterND · CumSum · Trilu★
  —— 而这 4 个正是 Gated DeltaNet 递推的核心 ✗（线性注意力的状态更新）
⇒ ★现成工具链无法把 qwen3_5 转上 NPU★ ✗
```

### 49.3 还有没有救？—— 有，但要重写线性注意力

| 缺的算子 | 能不能 lowering | 方法 |
|---|---|---|
| `Trilu` | ✓ 容易 | 常量三角掩码（`Where` + 常量 mask ✓ 我们在 gemma4 里就用过同类手法）|
| `CumSum` | ✓ 容易 | ★常量下三角矩阵乘法★（cumsum = tril(ones) @ x ✓ —— 与 gemma4 rotate_half 用常量矩阵替代 StridedSlice 是同一招 ✓）|
| `ScatterElements`/`ScatterND` | ✗ 难 | 分块递推里的**状态写入** ✗ ⇒ 要**固定 shape 重写**：把 chunk 内的 scatter 变成静态 `Reshape`+`MatMul`+`Add` ✓（相当于手写一个 NPU 友好的 Gated DeltaNet ✓）|

⇒ **可行性**：不是理论不可能 ✓，而是"**要手写 NPU 版线性注意力**" ✗（参照 gemma4 的 3D 重写经验 ✓，量级 1~3 天 ✗）；
   一旦写成，**所有 qwen3_5 系模型都能复用** ✓（含将来官方的小尺寸包 ✓）。
⇒ **若成功**：2B 在 NPU 上可达 10~20 tok/s 级 ✓（参照 Qwen3-8B 的 12.8 ✓）。

### 49.4 现实建议（三条并行不冲突）

```
① llama.cpp CPU 调优 ✓ 便宜：当前 1.70 tok/s ✗；README 记"比同类 ARM 核慢约一个数量级"✗
   ⇒ 试带 ARM 最优 kernel（i8mm/dotprod/SVE）重编译，预期 3~10× ⇒ 5~17 tok/s ✓ 立刻可用
② 手写 NPU 版 Gated DeltaNet（固定 shape + 常量三角矩阵 + matmul-only）✗ 1~3 天，收益最大 ✓
③ 等官方/DDK 支持 qwen3_5 ✗ 时间不可控
```

## 50. 路线②开工：把 qwen3_5 的线性注意力改写成"NPU 算子友好"版（施工中）

### 50.0 工作机已迁到 hu60tx ✓（x570 让给游戏 ✓）

```
· 已搬迁（x570 → hu60tx 局域网 rsync ✓）：~/q38（Qwen3.8-2B safetensors 4.6G + 脚本 ✓）
  ~/ddk（1.2G ✓）~/mslite（200M，converter_lite ✓）~/tools（转换辅助脚本 ✓）
· hu60tx：x86_64 / 16 核 / 62 GB 内存 / docker 免密 ✓ / uv 0.12.19 ✓ / GPU RTX 2060 6G
· 环境：uv 建 python3.10 ⇒ torch 2.14.1+cpu · transformers 5.18.0 · onnx 1.23.1 · ort 1.23.2 ✓
· ★注意★：x570 的 mslite-dev 容器 = debian:12 + 把 MindSpore Lite 构建目录 bind mount 进 /src ✓
  （200 MB 的 converter_lite 已随 mslite/ 一起搬 ✓，hu60tx 上重建容器即可 ✓）
```

### 50.1 缺口回顾（§49）

```
qwen3_5 的 ONNX 需要 27 种算子，DDK 缺 4 个：
  ★ScatterElements · ScatterND · CumSum · Trilu★（其余 23 个都有 ✓）
```

### 50.2 参考实现的位置（transformers 内置 ✓，纯 PyTorch）

```
~/q38env/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
  · causal_conv1d_fn            (268)  ★短卷积（因果）★
  · ★torch_chunk_gated_delta_rule★ (299)  ★Gated DeltaNet 分块实现★
  · Qwen3_5GatedDeltaNet.forward (549)
  （模型代码里的 masked_scatter 只是图文插入用 ✓ 纯文本路径不涉及 ✓）
  没装 flash-linear-attention / causal-conv1d ⇒ 自动走上面这份参考实现 ✓（对我们有利 ✓）
```

### 50.3 改写方案（不碰 site-packages，导出时 monkey-patch ✓）

| 参考实现里的写法 | 问题 | NPU 友好替换 |
|---|---|---|
| `decay.cumsum(dim=3)` | `CumSum` ✗ 缺 | ★常量下三角矩阵乘法★（`tril(ones) @ decay` ✓ 与 gemma4 用常量矩阵替代 StridedSlice 同一招 ✓）|
| `torch.ones(C,C).triu(1)` / `masked_fill` | `Trilu` ✗ 缺 | ★常量布尔掩码 + `Where`★ ✓（C 固定 ⇒ 编译期常量 ✓）|
| 分块状态更新里的 scatter | `ScatterElements/ScatterND` ✗ 缺 | ★固定 chunk 数 ⇒ `Reshape`/`Concat`/`Pad` 或**预计算常量索引**★ ✓（还要看具体形态 ✓）|
| 三角求逆（若有 `linalg.solve_triangular` ✗）| 无此算子 ✗ | ★UT 变换（固定步数纯 matmul 迭代）★ 或 Neumann 级数截断 ✓ |
| `causal_conv1d_fn`（因果短卷积）| `Conv1D` ✗（只有 Conv2D ✓）| unsqueeze 成 ★Conv2D★ ✓ 或 unfold+matmul ✓ |

**验证方式**：在 hu60tx（CPU ✓）上把改写版与参考实现**逐步对拍**（同一输入、同一 chunk ✓），
误差达标后再导 ONNX、清点算子（目标：只剩那 23 个支持算子 ✓），最后 OMG/converter_lite →
设备单图建图测试 ✓。

## 51. ★★★★★ 路线②实战：qwen3_5 线性注意力改写成 NPU 友好版（目标①②达成，③差临门一脚）★★★★★

工作机 = **hu60tx**（x570 留给游戏 ✓）。工具与改写脚本见
`scripts/model-conversion/qwen38/`（含 README ✓）。

### 51.1 ★目标① 对拍通过★ —— 改写版与参考实现等价

```
函数级（debug_delta.py）：cum_decay / pairwise / ut_system / intra / inv / new_values / k_cumdecay
                         全部 ≤ 5e-07 ✓
整模型（model_parity.py）：logits 最大绝对差 7.7e-04 · 最大相对差 2.9e-05
                         ★argmax 一致率 1.0000★ ✓
补丁命中：chunk 18 次 · conv 18 次（= 18 个线性注意力层 ✓）
```

**踩过的两个坑**：
```
① ★cumsum 的常量矩阵方向★：直觉写 `x @ tril(ones)`，但 `tril(ones)[j,t]=1 ⟺ j≥t`
   ⇒ 算出来是【后缀和】✗（实测正好反了 ✓）⇒ 必须用 `triu(ones)`（前缀和 ✓）
   （第一步就差了 3.4，导致后面全错 ✓）
② ★UT 变换的数值稳定性★：用"平方-乘积恒等式" `(I−L)^-1 = Π(I+L^{2^k})` 数学上精确 ✓，
   但会把条件数平方 ✗ —— 真实模型里 beta≈0.995 的层误差冲到 ★6.6e+16★ ✗
   ⇒ 改成★分块前代★（16×16 小块用平方-乘积 + 块间 matmul ✓）⇒ 降到 ★2.4e-06★ ✓
```

### 51.2 ★目标② 达成★ —— 导出的 ONNX 零缺失算子

```
原始：27 种算子里 DDK 缺 4 种（CumSum/Trilu/ScatterElements/ScatterND ✗）
改写后：Trilu → 常量掩码 ✓；CumSum → 常量 triu 矩阵乘法 ✓；
        Scatter* → 列表+cat（分块扫描）与函数式 M-RoPE 分节 ✓；
        IsNaN → 图级 lowering `Not(Equal(x,x))`（onnx_lower.py ✓）
★最终 25 种算子，DDK 缺失 = 0★ ✓（--tiny 与真实宽度切片都验过 ✓）
```

### 51.3 目标③ 进展：转换成功 ✓，但设备建图仍失败 ✗

```
✓ converter_lite（容器 mslite-dev = debian:12 + libpython3.11 ✓）转换成功：
    真实层型前 4 层（3 线性 + 1 全）· 单输入图（inputs_embeds fp32）⇒ ★.ms = 232 MB ✓ rc=0★
✗ 设备上 OH_AI_ModelBuildFromFile 仍然失败：rc=-1（L4 body ✓）/ rc=-2（微型 6 MB ✓）
    而【同一套转换器】产出的 gemma4 int8 段图当年是 Build 0 ✓（§43.1）
    ⇒ ★不是转换器/框架不兼容，而是 qwen3_5 的【图内容】被 NPU 内核拒收★ ✗
    DDK 日志只有栈没有消息（`[F] libhiai_adapter.so+0x...` ✓）⇒ 静默失败 ✗
```

**一路上踩掉的三个具体坑**（都已修 ✓）：
```
① int64 图输入 ⇒ 设备 Build -1 ✗；int32 图输入 ⇒ converter 报
   "SetMetaGraphInput: input Parameter_1 not found in graph" ✗
   ⇒ ★解法：S 固定 ⇒ 把 position_ids 烘成常量★（`--static-pos` ✓ 单输入图 ✓）
② 词表投影（lm_head 248320×2048 ≈ 2 GB fp32 ✗）与 embedding 表让 ONNX 必然 >2 GB
   ⇒ ★只导 transformer 层★（输入 inputs_embeds / 输出 hidden_states ✓，词表放主机侧 ✓ 与 gemma4 同思路 ✓）
③ ★"position_ids 没被用到"会让转换器丢输入★：前两层都是线性注意力层时
   （线性层不用 RoPE ✓）⇒ 报同一个 SetMetaGraphInput 错 ✗
   ⇒ 导出必须包含至少一个 full_attention 层 ✓（真实层型前 4 层 = 3 线性 + 1 全 ✓ 就正常 ✓）
```

### 51.4 下一步（路线②的收尾）

```
① ★按算子二分图★：把 ONNX 逐块裁剪后转换 + 设备试建，定位到底是哪个算子/结构被拒 ✗
   （怀疑对象：Softplus · Greater/Equal/Not/And · Expand/Split/Squeeze · Sqrt/Reciprocal 等
     —— 我的"字符串比对"判据太弱 ✗：库里出现该字样 ≠ NPU 内核实现了它 ✓）
② 若定位到少数算子 ⇒ 继续 lowering 掉它们（如 Softplus → log1p(exp(x)) ✓ 等）✓
③ 若能建图 ⇒ 把 24 层切成 3~4 段（每段带 position 常量 ✓ + 段间 hidden 传递 ✓）⇒ 设备上跑通前向 ✓
```

## 52. ★★★★★ 目标③ 的真凶：不是算子缺失，是【≥4 维张量】被 OMG 拒收 ★★★★★

### 52.1 两条关键教训（都踩过 ✓）

```
① ★设备只认"经 OMG 编译"的 .ms★
   实测对照（同一条探针 ✓）：
     x570 产的 gemma int8 段（走 OMG + converter --fmk=THIRDPARTY ✓）：Build rc=0 · Predict rc=0 ✓✓
     hu60tx 产的 matmul toy（走 converter --fmk=ONNX 直转 ✗）  ：Build rc=-2 · Predict rc=-2 ✗✗
   ⇒ ★`--fmk=ONNX` 直转出来的 .ms 设备一律拒收✗★（连一个 MatMul 都不行 ✓）
   ⇒ 必须：OMG（--target=omc）→ converter_lite --fmk=THIRDPARTY ✓
     （仓库 `scripts/model-conversion/int8/README.md` 早就写了这一条 ✓ —— 我绕了远路 ✗）
   ★附带坑★：我那个"批量探针"没设 NNRt device id，导致连已知能建的 gemma 段都报 -1/-2 ✗
     （§41 的老坑 ✓）；且同进程连建多个模型不稳 ✗ ⇒ 一律用单进程探针 ✓

② ★OMG 的 pre-check 报告是宝藏★：失败时生成 check_result.json（逐算子 pass/fail ✓）
   用法：grep fail/total，再按 op 的 type 统计 ✓
```

### 52.2 真凶：49 个 `Reshape` 失败 —— 因为它们操作的是 **5 维张量** ✗

```
q35_sp_L4.onnx（真实宽度前 4 层 · 0.92 GB · 单输入图）走 OMG：
  total 893 · pass 844 · ★fail 49★
  失败类型统计：★Reshape ×49（全部是 node_view_*，即 torch 的 .view()）★
  通过的类型：MatMul×219 Add×148 Mul×97 Slice×89 Gather×51 Transpose×35
             Unsqueeze×22 Pow×20 Sqrt×20 Where×15 Concat×15 …（含 5 个普通 Reshape ✓）
⇒ ★不是算子不支持，而是【≥4 维张量】被 NPU-CL 拒收★ ✗
   —— 与 gemma4 当年的结论完全一致（§30：「全部张量 3 维化，NPU-CL 对 ≥4 维支持很差 ✗」✓）
   我们的分块 delta rule 用 5 维张量：[B,H,nc,C,D] 与 [B,H,nc,C,C] ✗
```

### 52.3 修法（照 gemma4 的老办法 ✓）

```
把 (head, chunk) 折进前导维，★全程保持 3 维★ ✓
    [B,H,nc,C,D] → [B*H*nc, C, D]        ✓ 3 维
    [B,H,nc,C,C] → [B*H*nc, C, C]        ✓ 3 维
块间顺序扫描：用固定下标的 Slice 逐块取（chunk 数是编译期常量 ✓ ⇒ 可展开 ✓）
M-RoPE：同样把 (3, bs, pos) 的用法摊平成 3 维 ✓
⇒ 改完重新对拍（目标① 的判据不变 ✓）→ 重新 OMG + THIRDPARTY 转换 → 设备试建 ✓
```

## 53. ★★★★★ 目标③ 的突破：真凶是【dynamo 导出器】—— OMG 的 pre-check 从 55 fail 变 0 fail ★★★★★

### 53.1 决定性对照（同一台机器、同一套工具 ✓）

```
同一个"只有 Reshape"的小图：
  ★旧 TorchScript 导出器（opset 14 / ir 7）★：pre-check total 2 · pass 2 · ★fail 0 ⇒ success★ ✓✓
  ★新 dynamo 导出器（opset 18 / ir 10）★  ：pre-check total 1 · pass 0 · fail 1 ⇒ failed ✗✗
⇒ ★dynamo 导出的 Reshape 节点 OMG 不认✗★（与张量维数无关 ✓ —— 1→3/3→4/4→3/3→2 各种组合都失败 ✗，
  而"通过样例"的形状在最小图里照样失败 ✗）
⇒ 这解释了 gemma4 当年 48 个 Reshape 为何全过 ✓：★当年用的就是旧导出器★ ✓
```

### 53.2 修完之后的成绩

```
① 用【旧导出器】重导 qwen3_5 的 L4 切片（0.92 GB ✓）
② 图级 lowering（onnx_lower.py 扩展 ✓）：
     LessOrEqual     → Not(Greater(a,b))        ✓（旧导出器带进来的 ✗）
     ConstantOfShape → Expand(标量, shape)      ✓（它的 shape 输入不是常量 ⇒ 不能直接物化 ✗）
     IsNaN           → Not(Equal(x,x))          ✓
③ 形状修复 fix_static_shapes()：旧导出器把输出写成 `[0,0,2048]` ✗ ⇒ 用输入形状补齐 ✓
     （并跑 shape inference 补中间张量 ✓）
⇒ ★OMG pre-check：total 2829 · pass 2829 · ★fail 0 ⇒ success★ ✓✓✓
   （此前 dynamo 版本：total 1280 · fail 55 ✗）
```

**顺带修掉的 tracer 坑**：`_solve_unit_lower` 里 `L.shape[-1]` 在旧导出器下是 **Tensor** ✗ ⇒
`bit_length()` 报 AttributeError ✗ ⇒ 改为由调用方传 **Python int**（chunk 大小 ✓）。

### 53.3 仍剩最后一步（下一步做）

```
OMG 的 pre-check 已全过 ✓，但"生成 omc"阶段仍报：
  E: UpdateUserSetNodeNames :: "cannot find output tensor hidden_states"
     （磁盘上的图确实是 hidden_states [1,128,2048] FP32 ✓ —— 名字/形状都对 ✓）
     ⇒ 紧跟着是 ParseFromMemory FAIL ✗ ⇒ ★真正的错是它解析这张图失败★ ✗，
       "找不到输出"只是连带现象 ✓
怀疑点：图级 lowering 的产物（Expand 标量初始化器 / 结构）让 OMG 的 ONNX 解析器不适 ✗
下一步：用 `--mode 1`（model → json）单独验解析 ✓；再逐项排查 lowering 产物 ✓
```

## 54. 目标③ 继续推进：OMG 的"解析失败"根因锁定在 `IsNaN` 的 lowering 上

### 54.1 二分结果（都是微型图，秒级 ✓，逐一变量）

```
变体（在"只含 1 个全注意力层"的微型图上做 ✓）              OMG 结果
  bs_full_raw   不做任何 lowering                        ★解析成功✓★（只报 Pre-check has errors ✓）
  bs_leonly     只改 LessOrEqual  → Not(Greater)          解析成功 ✓
  bs_cosonly    只改 ConstantOfShape → Expand(标量,shape)  解析成功 ✓
  ★bs_isonly    只改 IsNaN → Not(Equal(x,x))★            ★解析失败✗★（cannot find output tensor …）
  bs_isonly2    IsNaN → Expand(False, Shape(x))          ★解析失败✗★
  bs_isonly4    改成"删节点 + 常量输出"（未触发）          解析成功 ✓
```

⇒ ★只要把 `IsNaN` 用【算子】替换，OMG 的解析器就崩✗★（换哪种写法都一样 ✗）；
   而 **不动它 / 删掉它** 都能正常解析 ✓。

### 54.2 本轮顺带确认的两件事

```
· `onnx.checker` 说三张图都【合法】✓ ⇒ 是 OMG 解析器的怪癖 ✗，不是 ONNX 语法问题 ✓
· OMG 的 `--mode 1`（model→json）需要另一套参数 ✗，不适合用来验解析 ✓
· ★线性注意力层那张图解析是【成功】的✓★（它报的是 ascendc 内核路径错 ✗：
    "file path '…/tools_omg/../platform/kirinx90/lib64/…'" ⇒ 属于环境/配置问题 ✗，
    与 §53 的解析问题【不是同一个】✓）⇒ 待修 ✓
```

### 54.3 下一步（已想好的两个方向）

```
① ★换一个"单算子、同形状、同 dtype"的等价写法★：
     IsNaN(x) ≈ Less(x, x)     （非 NaN 时两者都是 False ✓；NaN 时才是 True ✗，
                                 但我们的图全是有限运算 ⇒ 无 NaN ✓ 与 §53 的处置一致 ✓）
     —— 单节点替换，不像 Not(Equal) 那样引入中间张量 ✓
② 或者从源头消除 IsNaN：它来自 transformers 的注意力 mask 代码 ✓
     ⇒ 用"显式 mask + 我们的 patch" 把那条路径换掉 ✓（gemma4 的老办法 ✓）
③ 修 ascendc 路径问题（线性层图的报错 ✗）：检查 PYTHONPATH / --asc-dir / 平台插件目录 ✓
```

## 55. 目标③ 又进一步：OMG 唯一拒绝的算子就是 `IsNaN` ✗，且"改写节点必崩、重接消费者才安全" ✓

### 55.1 干净矩阵（微型图，全部秒级 ✓）

```
m1_raw        不做任何处理                 → 解析成功 ✓；pre-check ★fail 列表里只有 IsNaN 一个★ ✗
                                          （total 220 · pass 219 · ★fail 1 = IsNaN★ ✓）
m4_fix_only   只做形状修复                 → 解析成功 ✓（形状修复无害 ✓）
m2/m3         lowering 之后                → ★解析失败✗★（cannot find output tensor …）
逐个 lowering 单独试（§54）：只有 ★IsNaN 那一条★会让解析崩 ✗（LessOrEqual/ConstantOfShape 都安全 ✓）
换写法也不行 ✗：Not(Equal) / Expand(False,Shape) / ★Less(x,x)（单算子）★ / 删节点+常量 全部崩 ✗
★但★：把整段模式 `Where(IsNaN(x), 0, x)` 重接成 ★`Identity(x)`★ → 解析【成功】✓✓
     （即：不能碰 IsNaN 节点本身，但可以把它【连同消费者一起】换掉 ✓）
```

### 55.2 IsNaN 的来源（图里长什么样）

```
节点名：/b/layers.0/self_attn/IsNaN   ⇒ 在注意力里 ✓
上游：Softmax ← Add ← MatMul ← …（就是 attention 权重 ✓）
下游：Where（`Where(IsNaN(attn), 0, attn)` —— 即 attention 的 NaN 保护 ✓）
· transformers 源码里 grep `isnan` 只命中 import_utils.py（Python 的 if，不会被 trace ✓）
· grep `nan_to_num` 只命中 loss 文件 ✗ ⇒ 不是它 ✓（我把 torch.nan_to_num 恒等化也没用 ✗）
⇒ 具体来源还没查到 ✓；但★不需要查了★：直接在图级把整段模式重接掉即可 ✓
```

### 55.3 下一步（明确）

```
把"重接消费者"的做法★推广到所有 IsNaN★ ✓：
  对每个 `IsNaN(x)` 的消费者 C：
    · C = Where(cond, a, b) 且 cond 是它  ⇒ 输出等价于 b（非 NaN 时 IsNaN=False ⇒ 取 b ✓）
      ⇒ 用 `Identity(b)`（b 是标量则 `Expand(b, Shape(x))` ✓）顶替 C 的输出 ✓
  这样既删掉了 IsNaN ✓，又是【重接】而非【改写节点】⇒ OMG 解析不会崩 ✓
```

## 56. ★★★★★ IsNaN 问题彻底解决 ✓✓ —— OMG 的解析与 pre-check 全过，只剩「ascendc 内核」这一关 ✗

### 56.1 修好的：`Where(IsNaN(x), a, b)` 的重接（用独立脚本验证 ✓）

```
做法：把每个 `Where(IsNaN(x), 0, x)` 整段重接成 `Identity(x)` ✓，并删掉 IsNaN 节点 ✓
   （★重接消费者★而不是★改写 IsNaN 节点★ —— 后者必崩 ✗，§55 已证 ✓）

结果（微型全注意力图）：
  重接 1 处 | 未定义引用 0 | IsNaN 残留 False ✓✓
  ★OMG：解析 ✓ + pre-check ✓ 都过了★ —— 错误信息从
    "cannot find output tensor hidden_states"（解析崩 ✗）
    变成 "ascendc 内核路径无效"（解析成功后的下一阶段 ✗）✓✓
```

（过程中踩的坑：`g.node.remove(n)` 在 protobuf 重建后会报 "x not in container" ✗
 ⇒ 处理必须用【张量名】而不是节点对象、并且整表重建 ✓）

### 56.2 只剩的这一关：某些算子要走 **ascendc** 内核，而该内核不在这个 DDK 版本里 ✗

```
E/ASC ascendc_adaptee.cpp GetKernelbinAddr(95)::
   "file path '…/tools_omg/../platform/kirinx90/lib64/libai_npucore_ascendc_kernel.so' not valid." ✗
W/ASC builtin_ascendc_adaptee.cpp Initialize(70)::
   "kernel binary initialize failed, this store can use JIT only"

★关键★：这个 .so 在【x570 上也不存在】✓（两边平台库都是同样的 17 个文件 ✓）
   而 gemma4 当年 OMG 编译【成功】✓ ⇒ 说明 gemma4 的图【完全走 fusion-engine】✓，
   ★我们的图里有算子被路由到了 ascendc✗★（所以才会去加载那个不存在的内核 ✓）
   另外两条无关的噪声：`libai_npucore_generated.so` 本就不存在（只是警告 ✓）、
   ascendc_config.json 的路径警告（已把配置拷到 OMG 会找的路径 ✓，但没解决 ✓）

★下一步很清楚★：按算子二分，找出【哪些算子会被路由到 ascendc】✗，
   然后像 §52–55 那样把它们 lower 掉 ✓（gemma4 那 12 个算子的路线是通的 ✓）。
   怀疑对象：Softplus · Greater/Equal/Not/And · Expand/Split/Squeeze · Shape · Cos/Sin…
```

### 56.3 当前进度总览（目标① ② 已完成 ✓，③ 只剩算子路由这一关）

```
① 对拍（argmax 1.0000 ✓）          已完成 ✓
② 导出图零缺失算子 ✓（DDK 平台库）  已完成 ✓
③ OMG：解析 ✓ · pre-check ✓ · 内核路由 ✗（本文档 §56.2）→ converter → 设备建图  进行中
```

## 57. ★★★★★ 目标③ 的最后一关：NPU 内核只吃 ≤3 维 —— 拒绝清单一览 ★★★★★

### 57.1 逐算子二分（16 个单算子 toy，旧导出器 ✓，秒级 ✓）

```
15/16 算子单独都能 OMG 编出 omc ✓：
  MatMul · Softplus · Cos/Sin · Sqrt/Reciprocal · Div/Exp/Neg · Greater/Where · Equal/Not
  And · Expand · Split/Concat · Squeeze/Unsqueeze · ReduceSum · Sigmoid/Softmax · Gather/Slice
  · 移位切片版因果卷积 ✓
★只有我那个 constant_matmul toy 失败 ✗★（原因是 toy 自身的结构 ✓，不是算子 ✓）
另外：`ascendc`/`TE_FUSION`/`ascendc_config` 那些 E/W ★每次都有，属于无害噪声★ ✓
   （`libai_npucore_ascendc_kernel.so` 在 x570 上同样不存在 ✓ ⇒ 不是我们独有的问题 ✓）
```

### 57.2 真实图的致命错（顺序推进，逐层剥开 ✓）

```
第 1 层：`Where(IsNaN(x),0,x)` ⇒ 重接（纯重命名 ✓ 连节点都不新增 ✓）
        ⇒ 解析 ✓ + pre-check ✓ 都过了 ✓（§56）
第 2 层：`MatMul(标量, x)`（legacy 把 `q*scaling` 导成 MatMul ✗）
        ⇒ OMG 报 "The value of wDim in x1 should be equal to hDim in x2 … Infershape for MatMul_1 failed" ✗
        ⇒ 改写成 `Mul(x, 标量)` ✓（语义等价 ✓）—— 这一层修掉后，
第 3 层：★NPUCL 的算子级拒绝清单浮出来了★ ✓✓
```

### 57.3 ★真正的最后一关：NPU 内核只吃 ≤3 维★

```
E/AI_NPUCL 统计（真实微型图 nat11）：
  18× strided_slice_get_format.cc IsDimThreeNdCase(): "inputDim.size() 4 dimC 2 dimH 128" ✗
   2× StridedSliceCalcOutDimsNormalize / UpdateStridedSliceWeightsWithMask failed ✗
   2× reshape_check_support.cc: "check reshape dimInfo fail" ✗
   2× expanddims_check_support.cc: "not support input dimCnt >= ?" ✗
   1× PlugIn library :libai_npucore_ascendc.so Initialize failed ✗（非致命噪声 ✓）
⇒ ★Slice / Reshape / ExpandDims 在 4 维上都过不了★ ✗
  —— 与 §30 的老结论一字不差：「NPU-CL 对 ≥4 维支持很差 ✗」
```

**我之前的 3 维化只做了一半** ✗：只改写了**自己写的** delta rule ✓，
而**模型自带**的注意力（`[B,H,S,D]` 的 Q/K/V 切分、RoPE 切片、mask 切片 ✓）仍是 4 维 ✗
（微型"单全注意力层"图里就有 18 处 ✗）。

### 57.4 下一步（方法已知 ✓，就是 gemma4 那一套）

```
把整个 qwen3_5 文本解码器改写成【全程 ≤3 维】✓（等价于当年的 g4_seg3d.py ✓）：
  · (batch, head) 折成一维 ⇒ [B*H, S, D] ✓（我已在 delta rule 里验证过这一招 ✓）
  · attention 里的 Q/K/V 切分、RoPE、mask 全部改成 3 维切片 ✓
  · Reshape/ExpandDims 都要落在 ≤3 维 ✓（或换成 Slice/Concat 组合 ✓）
验收判据不变：① 整模型对拍 argmax 一致 ✓ ② OMG 走到 NPUCL 时【零 E/AI_NPUCL】✓
  ③ converter(THIRDPARTY) → 设备 Build 0 / Predict 0 ✓
```

## 58. 目标③ 收尾工程（1）：★全程 3 维的注意力★已写出，接近对拍

### 58.1 为什么要重写注意力

§57 实测：NPUCL 对 **4 维** 的 Slice/Reshape/ExpandDims 一律拒绝 ✗，而 qwen3_5 的注意力
原本走 `[B,S,H,D] → transpose(1,2) → [B,H,S,D]` ✗ 全程 4 维 ✓（真实图里 18 处 4 维 Slice ✗）。

### 58.2 新文件 `scripts/model-conversion/qwen38/npu_attention.py`（3 维版 ✓）

```
[batch, seq, H*D]  --reshape-->  [B, S*H, D]  --★常量索引 Gather★-->  [B, H*S, D]
                   --reshape-->  [B*H, S, D]      （★等价于 4 维 transpose 的结果✓★）
注意力：q @ kᵀ → [BH,S,S] ✓ → softmax ✓ → @ v → [BH,S,D] ✓ —— 全 3 维 ✓
回来：逆索引 Gather 复原 ✓
```
**关键点**（都是实测踩出来的 ✓）：
```
① ★换序用常量索引的 Gather★：绕开 4 维 transpose/reshape ✓（Gather 单算子 toy 已验证可编 ✓）
② ★q_proj 的输出布局是 [B,S,H,2*D]★ ⇒ 必须【在每个 head 的 2*D 块内部】切 Q/gate ✓
   （直接切整段的前一半会得到完全不同的排布 ✗ —— 第一次就踩了这个 ✓ argmax 只有 0.875）
③ ★partial_rotary_factor = 0.25★：只对 head_dim 的前 64 维做 RoPE ✓，其余原样透传 ✓
   （rope = cat([rot*cos + rotate_half(rot)*sin, pass], -1) ✓ 全 3 维 ✓；与参考实现逐行一致 ✓）
④ ★因果掩码不能漏★：我们导出时传的是全 1 mask ✓，参考实现内部会把它变成因果 mask ✓
   ⇒ 3 维版必须自己加【常量上三角 -inf】✓（加了之后 argmax 0.875 → 0.9583 ✓）
⑤ q_norm/k_norm 是【逐 head】的 ✓ ⇒ 必须在 split 之后套 ✓
```

### 58.3 当前对拍状态（还没过 ✗，但已很近）

```
单层全注意力微型模型（hidden 256 / 4 头 / kv 2 头 / seq 24）：
   最大绝对差 3.047e-02 | 相对 1.689e-02 | ★argmax 一致率 0.9583（23/24）★
   演进：0.8750（漏因果掩码 ✗）→ 0.9583（补上后 ✓）
★下一步★：对 attention 内部【逐步数值对拍】（q / k / v / rope 后 / 概率 / 输出 ✓），
   把剩下那一处差异钉死 ✓（怀疑：GQA 头顺序、或 mask 的具体形式、或 cos/sin 的广播维度）
```

## 59. ★★★★★ 路线校正：不该转 .ms 走 NNRt —— 应该直接用 OMG 的 .omc 走 hiai 引擎 ★★★★★

（用户指出的 ✓：本机既有 hiai 后端，也有现成的 hiai 模型包形态 ✓）

### 59.1 事实（固件镜像为准 ✓，别再凭一次 `find` 下结论 ✗）

```
固件解包：ssh hu60tx:/home/hu60/work/hmos/firmware/unpack_result_010554/system/system/lib64/
  有：★ndk/libhiai_foundation.so★ ✓ · ★libhiai_llm_engine.so★ ✓ · libmindspore_lite_ndk.so ✓
      libhiai_nn_proxy_*.z.so · libhiai_aiv_proxy_*.z.so · libhiai_llm_engine.so …
  没有：libhiai_hcl_model_runtime.so ✗ · libhiai_ir_infershape.so ✗
⇒ ★设备内建的 LLM 通路是 hiai 引擎（libhiai_llm_engine.so）+ 模型包★ ✓，
  而 NNRt 的 `OH_AI_ModelBuildFromFile` 需要 HCL runtime ✗（设备日志原文：
  "dlopen libhiai_hcl_model_runtime.so fail" / "no runtime support the Model" ✗）
⇒ ★我们花了大力气产出的 .omc 本来就是 hiai 的离线模型 ✓，根本不该再转 .ms ✗★
```

### 59.2 应用它需要的东西（全都有了 ✓）

```
① ★图★：OMG 产出的 .omc ✓ —— 本轮已攻克 ✓（rc=0 ✓ 927 MB（4 层）✓ 零缺失算子 ✓）
② ★外置权重★：SubGraph_0.weight ✓（OMG 对大模型自动外置 ✓，§46）
③ ★后端代码★：src/cann_llm/backends/hiai.py ✓（仓库自带 ✓）
   —— 它按【目录】加载 ✓：<name>.omc + <name>.json（同名 ✓）+ api_config.json + tokenizer.json ✓
④ ★可对照的成品包★：models/model_qwen2_1p5b_w4_2048/ ✓
     SubGraph_0.weight 3104 MB · qwen2_1p5b_w4.omc 3.0 MB · <name>.json 1086 B
     · api_config.json · executor.json · context.json
     · <name>_64_2048.embedding_weights 233 MB(int8) + .embedding_dequant_scale
     · tokenizer.json 7 MB      ★与官方 models/Qwen3-8B 完全同形态 ✓★
⑤ ★组装工具★：scripts/import_omc_package.py（官方 OMC 包 → cann-llm 目录 ✓）

### 59.3 本轮另外两个实打实的进展（无论走哪条后端都用得上 ✓）

```
· ★3 维化注意力对拍通过★：npu_attention.py（常量索引 Gather 换序 / per-head 切分 /
  partial RoPE(0.25) / numpy 常量因果掩码）⇒ 单层 rel 1.3e-07 · argmax 1.0000 ✓
  整模型（含 delta rule）argmax 1.0000 ✓（§58/§59）
· ★OMG 通了★：用【旧导出器】+ 三项 lowering + numpy 常量 ⇒ 图不再爆炸
  （Constant 18576→1164 · Shape 10270→29 ✓）⇒ ★rc=0 · omc 927 MB ✓★
  且 converter(THIRDPARTY) 也成功产出 .ms 927 MB ✓（只是 NNRt 这条路走不通 ✗）
```

### 59.4 下一步（转向 hiai 模型包）

```
① 导出+OMG 出【全 24 层】的 onnx→omc ✓
   注意：fp32 全模型 onnx ≈5.5 GB ✗ > OMG 单文件门槛 ✗
   ⇒ 需要先量化（W4/int8 ✓ 仓库有 set_quant_strategy.py / onnx_weights_to_fp16.py ✓）
      —— 这也正是 §47 官方链路（dopt W4）的做法 ✓
② 按 §47 的形态组装目录：omc + SubGraph_0.weight + embedding_weights(+scale) + tokenizer.json
   + <name>.json / api_config.json（照 W4 包改维度 ✓）
③ 用 src/cann_llm/backends/hiai.py 在设备上加载 ⇒ 聊天 ✓（目标③ 直接闭环 ✓）
```

## 60. 转向 hiai 模型包：素材清点 + 官方图接口（本轮摸底）

### 60.1 素材（全部到位 ✓）

```
✓ dopt 工具：~/ddk/tools/tools_dopt/dopt_pytorch_py3/dopt/dopt_lm/opt_main.py（hu60tx ✓）
✓ 官方样例仓库：hu60tx:~/q38/cannkit（gitcode HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp ✓）
    CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export/
      export_model_single_qwen2.py · export_model_single_qwen3.py · export_model_single_glm.py
      onnx_utils.py · model_info_target.yaml
✓ 仓库流水线：scripts/model-conversion/qwen/build_model.py（导出+onnxsim+OMG+装配 一条龙 ✓）
     还支持 --arch / --layers / --hidden / --kv-heads / --head-dim / --vocab-size 等切片参数 ✓
✓ 成品包对照：models/model_qwen2_1p5b_w4_2048/（§47 跑通 6.0 tok/s ✓）
✓ 后端：src/cann_llm/backends/hiai.py（按目录加载 omc + 同名 json + api_config ✓）
✓ 补两个文件的工具：PYTHONPATH=src python3 -m cann_llm.modelpkg <模型目录>
✓ 全部量化+导出+OMG 配方：§47.1（dopt 三阶段 → build_model.py → modelpkg → start_chat.sh）
```

### 60.2 ★官方图接口★（我们的导出必须匹配，摘自 export_model_single_qwen3.py ✓）

```
输入： input_ids            [batch, seq_len]                        int64
       attention_mask       [batch, 1, seq_len, kv_cache_max_len]   dtype
       position_ids         [batch, seq_len]                        int64
       past_key_in{i}       [batch, kv_heads, kv_cache_max_len, head_dim]
       past_value_in{i}     同上
输出： lm_logits            [batch, seq_len, vocab]
       past_key{i} / past_value{i}
  （模型内部先做 embed_tokens ✓，再喂进 decoder ✓）
```

### 60.3 下一步（待做）

```
① 把官方导出脚本适配到 qwen3_5（混合架构：24 层里 18 层线性注意力 + 6 层全注意力 ✗）：
   · 套上我们的 3 维化补丁（npu_gated_delta.py + npu_attention.py ✓ 对拍 argmax 1.0000 ✓）
   · 但两处要补"缓存接口"：全注意力的 past_key/value（4 维 ✓）
     与线性注意力的循环状态（delta rule 的 final_state ✓）
② dopt(W4) 量化 2B 模型（★必须 CUDA★，hu60tx 是 RTX 2060 6 GB ✗ 可能偏紧 ⇒ 备选：
   先在 CPU/切片上验证流程，或改用仓库的 ONNX 级量化 ✓）
③ build_model.py 一条龙 ⇒ omc + SubGraph_0.weight + embedding + 配置 ✓
④ PYTHONPATH=src python3 -m cann_llm.modelpkg ⇒ 补 api_config.json / <name>.json ✓
⑤ scripts/start_chat.sh -d <模型目录> -p '1+1=' ⇒ ★设备上聊天★（目标③ 闭环 ✓）
```

## 61. ★★★★★ 找到直跑 omc 的正路：DDK 的 hiai C++ API（头文件来自官方 HiAIDemo）★★★★★

### 61.1 素材来源（全部是官方仓库 ✓）

```
gitee.com/huawei-hiai-foundation/HiAIDemo   → 已克隆到 hu60tx:~/q38/HiAIDemo ✓
  DDK_Demo/V2/include/                       ← ★DDK 的 hiai C++ 头文件★
    model/built_model.h · model/model_api_export.h
    model_manager/model_manager.h · model_manager_types.h
    tensor/nd_tensor_buffer.h · tensor/nd_tensor_desc.h · tensor/buffer.h
    model_builder/hiai_ir_build.h（离线模型编译 ✓，即 omg 的同类 API）
    base/error_types.h
```

### 61.2 ★API 全貌（这就是"直接加载并运行 .omc"的路 ✓）★

```cpp
namespace hiai {
  // ① 加载已编译模型（.omc）
  std::shared_ptr<IBuiltModel> CreateBuiltModel();
    -> RestoreFromFile(const char* file);          // ★读 omc★ ✓
    -> GetInputTensorDescs() / GetOutputTensorDescs();   // 查 IO 描述 ✓
    -> CheckCompatibility(bool&) / SaveToFile / RestoreFromBuffer …

  // ② 执行
  std::shared_ptr<IModelManager> CreateModelManager();
    -> Init(const ModelInitOptions&, builtModel, listener);
    -> Run(inputs, outputs);                       // std::vector<INDTensorBuffer> ✓
    -> RunAsync(context, inputs, outputs, timeout); -> Cancel(); -> DeInit();
  // 监听：OnRunDone(context, status, outputs) / OnServiceDied()
}
```

### 61.3 为什么这条路对 qwen3_5 特别重要

```
★IO 由我们自己定★ ✓ ⇒ 不必迁就 LLM 引擎的固定接口
  （引擎要求 input_ids/attention_mask/position_ids + 每层 past_key/value ✓，
   而 qwen3_5 是【混合架构】：18 层线性注意力 + 6 层全注意力 ✗
   —— 线性层要传的是【卷积状态 + 递归状态】，引擎没有这个概念 ✗）
⇒ 用 DDK hiai API：我们自己设计状态输入/输出 ✓，解码循环自己写 ✓
```

### 61.4 另外查到的事实

```
· 官方模型矩阵（gitcode.com/openharmony-models）里 ★没有 Qwen3.8/Qwen3.5/3-Next 的包★ ✗
  （Qwen2.5-Coder-7B-Instruct 等有 ✓）⇒ 想要 Qwen3.8 只能自己编译 ✓（正是本目标在做的事 ✓）
· libhiai_foundation.so（NDK，151 个符号 ✓）导出的是另一族 API：
    HMS_HiAIExecutor_* / HMS_HiAIKernelExecutor_*（Buffer 按 fd/addr ✓）
  但它【没有头文件】（DDK 与固件里都查过 ✗）⇒ C++ 的 hiai API 更好用 ✓
```

### 61.5 下一步

```
① 用 DDK 头文件写一个最小 runner（C++ ✓，在设备上用 OHOS clang 编 ✓）
   —— 先拿【已经产出的 4 层 omc（927 MB ✓）】验证：
     RestoreFromFile ✓ → GetInputTensorDescs（看到真实 IO ✓）→ Run ✓
② 跑通后，按"自定状态接口"导出全 24 层图（input_ids/position_ids + 各层状态 → logits + 新状态 ✓）
   ⇒ OMG ⇒ .omc ⇒ 自带解码循环 ⇒ ★设备上聊天★ ✓
```

## 62. DDK hiai C++ runner：编译跑通 ✓，但 `Build/RestoreFromFile` 还没走通

### 62.1 已做出来的东西（`scripts/model-conversion/qwen38/hiai_runner/hiai_runner.cpp` ✓）

```
用官方 HiAIDemo/DDK_Demo/V2 的头文件 + lib64/libhiai.so（aarch64 ✓ 256 KB 薄壳 ✓）写了个 runner：
  .om  → CreateModelBuilder()->Build(opts, name, path, built)      （DDK 的离线模型编译 ✓）
  .omc → CreateBuiltModel()->RestoreFromFile(path)                  （引擎那种已编译模型 ✓）
  打印 GetInputTensorDescs()/GetOutputTensorDescs()（真实 IO ✓）+ CheckCompatibility
  CreateModelManager()->Init(opts, built, nullptr) → Run(inputs, outputs) ✓

设备上用 OHOS clang 编译成功 ✓：
  /data/service/hnp/bin/aarch64-unknown-linux-ohos-clang++ -std=c++11 -O2 -fPIC -Iinclude \
     -DHAVE_PTHREAD -DOHOS … hiai_runner.cpp lib64/libhiai.so -ldl      ⇒ ★42 KB 产物 ✓★
```

### 62.2 ★两条实测教训（很重要 ✓）★

```
① ★设备上不要设 LD_LIBRARY_PATH★ ✗✗
   实测（同一个二进制、同一个目录）：
     设了 LD_LIBRARY_PATH=$B/lib64:$B:/system/lib64/ndk:…  → "Error loading shared library
        libhiai.so / libc++_shared.so" + 一堆 "symbol not found" ✗
     不设任何环境变量                                        → ★正常运行 ✓★
   ⇒ 设它反而把默认搜索路径顶掉 ✗（musl + OHOS 的行为 ✓）—— 这与我们之前跑 python 探针
     的习惯相反 ✓，写脚本时要注意 ✓
② OMG 的 --target 支持 om/omc/tiny/ispnn/security ✓（omg --help ✓）
   ★我们两种都产出来了✓★：--target=omc ⇒ seg.omc 927 MB ✓；--target=om ⇒ seg.om 926 MB ✓
```

### 62.3 当前卡点：`Build` 段错误 / `RestoreFromFile` rc=1

```
· CreateBuiltModel()->RestoreFromFile("<x>.omc")  → rc=1（返回 1 ✓ 不是崩溃 ✓）
    ⇒ .omc 是【LLM 引擎】用的格式 ✓，DDK builder 要的是 .om ✓（语义对得上 ✓）
· CreateModelBuilder()->Build(默认 ModelBuildOptions, "<x>.om") → ★rc=1 ✗★
· ★对照实验★：拿 demo 自带的 540p_544x960.om（20 KB ✓）跑同一个 runner
    ⇒ ★exit=139（段错误）✗★ ⇒ 说明是【我们的用法/初始化不完整】✗，不是我们的 .om 的问题 ✓
ModelBuildOptions 字段（model_builder_types.h ✓）：
    inputTensorDescs · formatMode(USE_NCHW) · precisionMode(FP32) · dynamicShapeConfig
    · modelDeviceConfig · tuningStrategy(OFF) · estimatedOutputSize · quantizeConfig
    ⇒ LLM 显然要改成 FP16 / ND / 指定 device ✓（默认值不适合 ✓）
```

### 62.4 下一步

```
① 把 demo 的 main.cpp 也拿来看（我们只搬了 load_and_run.cpp ✗）——
   看它是否要先做 DDK 初始化（如环境/设备注册 ✓）、以及 Build 的正确选项 ✓
② 读 HiAIDemo 仓库的说明（HMOSNextDemo/*/OHOS_DDK Demo说明.pdf ✓ 之前搜到过）
③ 若 DDK C++ 这条需要 app 沙箱/权限 ✗ ⇒ 回到【LLM 引擎 + 模型包】那条（§47 已跑通 6.0 tok/s ✓），
   它的模型包形态我们全都有 ✓（§60 ✓）
```
