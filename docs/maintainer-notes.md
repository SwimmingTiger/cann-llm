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
