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

## 20. ★★★★★ 结论：`--fmk=THIRDPARTY` + `WEIGHT_QUANT` 路径【没有产出 int8】★★★★★

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
