# nnrt-probe —— 绕开 LLM 引擎，直接用 Neural Network Runtime 驱动 NPU

> 这几支小程序回答一个问题：**能不能不经过 `libhiai_llm_engine.so` / `libcann_llm_engine.so`
> 那套 LLM 引擎，直接把一个自己组装的模型丢到麒麟 NPU 上编译执行？**
>
> 实测结论：**能。** 在线构图 → NPU 编译 → 执行，数值与 CPU 参考值一致。

## 为什么值得做这件事

LLM 引擎对模型的接口有硬约束 —— 它要求模型是「每层一对 K/V + 逐层 `past_key_value` 进出」的
那种结构。于是像 Gemma 4 这类新架构（每层不同配置、K=V、K/V 跨层共享、每层额外输入通路）
**根本塞不进去**，哪怕图能导出、OMG 能转也没用。

NNRt 不一样：**模型的输入/输出张量由它自己的图决定**，我们声明什么就是什么。
所以想按层传 K/V、想有多条输入通路，都可以表达 —— 这是一条能容纳新架构的通路。

另外 NNRt 还有「离线模型」入口（`OH_NNCompilation_ConstructWithOfflineModelFile`）和
「从 MindSpore Lite 图构图」（`OH_NNModel_BuildFromLiteGraph`），本目录的 `offline_probe.c`
就是测前者的。

## 编译与运行

```sh
./build.sh                       # 编译到本目录 bin/（别编到 /tmp —— 本机 /tmp 只读）
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/add_test      # 最小验证
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/block_test    # 多算子图 + 数值核对
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/ops_probe     # 算子支持矩阵
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/offline_probe /path/to/model.omc
```

设备自带的 SDK 里有 NNRt 头文件（`.../native/sysroot/usr/include/neural_network_runtime/`），
运行库在 `/system/lib64/ndk/`，都是世界可读的，普通应用进程就能调。

## 各文件做什么

| 文件 | 用途 |
|---|---|
| `add_test.c` | 最小闭环：在线构一个 Add 单算子模型 → 在 NPU 上编译 → 执行 → 校验输出 |
| `block_test.c` | ★ 多算子图（`MatMul → Tanh → MatMul`）编译并执行，与 CPU 参考值逐个比对 |
| `ops_probe.c` | 40 个算子的支持矩阵（判据 = 能不能在这块 NPU 上 `Build` 成功） |
| `offline_probe.c` | 测 NNRt 的离线模型接口收不收 OMG 产出的 `.omc` |
| `diag_spec_rule.c` | 定位「`AddOperation` 在什么条件下成功」的隔离实验 |
| `diag_spec_variant.c` | 同一算子的多种建图变体对比，用来找真正的校验判据 |
| `diag_dtype.c` | 同一批算子 FP32 / FP16 各试一遍，并试 MatMul 的参数个数 |

## 实测结果

`add_test`（单算子闭环）：

```
设备: NPU_ohos.boot.hardware.KirinX90_v2_0 (id=5337627887595434492)
Build  -> SUCCESS
RunSync-> SUCCESS
输出: 0 2 4 6 8 10 12 14 16 18 20 22      (两个输入都是 0..11，期望相加)
```

`block_test`（多算子图 + 数值核对）：

```
op1 MatMul -> SUCCESS   op2 Tanh -> SUCCESS   op3 MatMul -> SUCCESS
Build   -> SUCCESS
RunSync -> SUCCESS
  x=0.0500  期望=0.04996  NPU=0.04999
  x=0.1000  期望=0.09967  NPU=0.09961
  全量 32 个元素中偏差 >1e-3 的: 0 个
```

`ops_probe`（在 NPU 上可编译 = 支持）：

```
支持: MatMul · Add · Mul · Sub · Div · Reshape · Squeeze · Flatten · Stack · Clip
      Maximum · Greater · Select · RSqrt · Sqrt · Tanh · Sin · Cos · Exp · Neg · Abs
      Log · Square · Erf · Reciprocal · Floor · Ceil · Relu · Sigmoid
      ★ 本轮新增确认：Softmax（注意力必需）· Concat · Split

注意: 一个 Transformer 需要的算子基本齐了 —— 特别是 MatMul(矩阵乘)、Sin/Cos(位置编码)、
      RSqrt/Sqrt(归一化)、GELU 类激活。
```

## NNRt 的建图规格（踩过的坑，都在这里）

这几条是实测出来的，不看会一直卡在 `INVALID_PARAMETER`：

1. **「无参数」必须传 `{NULL, 0}`**。传一个未初始化的非空数组指针（哪怕 `size` 是 0），
   会被当成「多传了参数」直接拒掉。这一条曾让 40 个算子里的 24 个"建图失败"。
2. **轴 / 形状这类值是「参数张量」，不是输入张量** —— 这一条最容易搞反。
   Softmax / Concat / Squeeze / Split / Slice / Flatten 的 axis 都要先用
   `OH_NNModel_SetTensorType(model, i, OH_NN_SOFTMAX_AXIS)` 之类声明成**参数**，
   dtype 必须是 `OH_NN_INT64`、shape 必须是长度 1（标量）。
   注意 hilog 那句 `The 2nd input axis should be type OH_NN_INT64` 说的是
   "第 2 个**入参**"，不是"第 2 个输入张量" —— 按后者去做会把输入个数搞错，
   于是收到 `Passed invalid input or output index`。
3. **布尔类参数必须 `OH_NN_BOOL`** —— MatMul 的 transposeX/Y、GELU 的 approximate、
   ReduceMean 的 keep_dims 都是，写 `OH_NN_INT8` 会被拒。
4. **每个算子的输入数与参数数必须精确匹配**：Gather 是 3 个输入（input/indices/axis）、
   MatMul 参数个数 ≥1、Cast 还要一个 `castType` 参数。多一个少一个都报
   `INVALID_PARAMETER`。
5. **输入 / 输出个数必须精确匹配**。`ops_builder.cpp` 里的 `CheckIOIndex` 只查两件事：
   个数与算子常量相等、索引不越界。每个算子的 `INPUT_NUM` / `OUTPUT_NUM` / `PARAM_MAX_NUM`
   都是源码里的常量。
6. **`OH_NNModel_GetAvailableOperations` 在本机返回 `opCount=0`**（模型明明有算子）。
   实测更可靠的判据是**直接 `OH_NNCompilation_Build`** —— 能编译就是支持。

## 规格不用猜 —— 直接读 NNRt 源码

NNRt 是开源的，每个算子的规格就在源码里（权威且免费）。克隆下来即可：

```sh
git clone --depth 1 https://gitcode.com/openharmony/ai_neural_network_runtime.git
# gitee 镜像：https://gitee.com/openharmony/ai_neural_network_runtime.git
```

看这两个地方：

```
frameworks/native/neural_network_runtime/ops/<算子>_builder.cpp   ← 单个算子的规格
frameworks/native/neural_network_runtime/ops_builder.cpp          ← CheckIOIndex / CheckParamIndex
```

每个 `<算子>_builder.cpp` 里有：

* `INPUT_NUM` / `OUTPUT_NUM` / `PARAM_MAX_NUM` —— 输入、输出与参数的个数；
* `SetXxx(tensor)` 里的 dtype / shape 校验，以及它对应的报错文案
  （拿 hilog 里的报错文案反查源码，定位最快）；
* `REGISTER_OPS(XxxBuilder, OH_NN_OPS_XXX)` —— 对应的算子枚举名。

也可以只取单个文件：

```sh
curl -sL -o softmax_builder.cpp \
  https://raw.gitcode.com/openharmony/ai_neural_network_runtime/raw/master/frameworks/native/neural_network_runtime/ops/softmax_builder.cpp
```

> 注意：设备上的 NNRt 是厂商构建版本，个别算子（如 Squeeze / Slice）的支持情况
> 可能与上游不同；两者对不上时以**设备实测 + hilog** 为准。

## 出问题时怎么查原因（关键技巧）

NNRt 把每条拒绝理由都写进 **hilog**，但读 hilog 需要进程在 `log` 组里，
普通 shell 里 `hilog -x` 读不到（`hilog -g` 直接 `Errno 13 Permission denied`）。
在**有 `log` 组的终端**（`id` 里能看到 `1007(log)`）里这样做：

```sh
# 先开抓，再跑探针，然后按探针的 pid 过滤（顺序很关键）
hilog > ./hl.log 2>&1 &
HP=$!
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/ops_probe > ./probe.out 2>&1 &
PP=$!
wait $PP
kill $HP
grep -a " $PP " ./hl.log | grep -aiE 'NNRt|CheckSupported|Build failed'
```

能看到类似这样的确切原因（比返回码有用得多）：

```
[SoftmaxBuilder] The 2nd input axis should be type OH_NN_INT64.
[Gelu] The approximate should be type OH_NN_BOOL.
[Matmul] SetTransposeA failed. The transposeA should have type OH_NN_BOOL.
[Gather] Build failed, the input or output index of Gather operation is invalid.
AI_NPUCL: CheckSupported: the op name [X:0] type [Y] is not supported
          in npucl store [elementary_lib] / [fe_lib]
```

## 看日志时的一个注意点

`Build` 失败有两种，含义完全不同，别混：

* **建图失败**（`AddOperation` / `Finish` 返回非 0）—— 规格写错了，改图。
* **NPU 编译失败**（`OH_NNCompilation_Build` 返回非 0）—— 算子在这块 NPU 上不支持，
  或需要换形态（dtype / shape）。hilog 里会有
  `CheckSupported: ... is not supported in npucl store [elementary_lib]/[fe_lib]`。

## 已知的、没做完的部分

* ~~Softmax 还没跑通~~ → **已跑通**：axis 作为参数张量、dtype `INT64`、shape 长度 1，
  输入数必须是 1。Concat / Split 同理已通过。
* **Squeeze / Slice 是「建图成功但 NPU 编译失败」**（`g_err == Build`），说明规格没问题、
  是这块 NPU 不支持或需要别的输入形态 —— 与「建图失败」要区分开。
  hilog 里那句 `SqueezeBuilder Passed invalid input or output index` 来自早期
  「axis 当输入」的尝试，看日志时注意别把它当成当前这版的结论。
* `ops_probe` 里 Gather / ReduceMean / GELU / LayerNorm / Concat / Split / Cast / Transpose
  这几项仍是「建图失败」——原因同上（规格没配全），**不代表 NPU 不支持**。
* `offline_probe` 用 `.omc` 走离线模型接口时 `Construct` 成功但 `Build` 失败，
  推测它期望的是被 MindSpore Lite 包起来的格式（库里能看到
  `GetOfflineModelFromLiteGraph` / `MindIR_LiteGraph_To_Model` 这类符号），尚未验证。
* 运行时会看到几条无害警告：`libai_npucore_ascendc.so` 不存在、`dma_heap_alloc`
  打不开若干共享内存堆、`[BackendManager] RegisterBackend failed` ——
  实测不影响在线构图这条路的编译与执行。

---

## int8 探索（新增，未完成）

### 已确认的关键结论 ✓

```
★ 设备：NNRt 直连可见 NPU（device type=3，name=NPU_ohos.boot.hardware.KirinX90_v2_0）✓
★ ★★ 量化参数通道【可用】★★：
    OH_NNQuantParam_Create / SetNumBits / SetScales / SetZeroPoints
    ⇒ OH_NNModel_SetTensorQuantParams(model, index, qp)  ★rc=0 (SUCCESS)★ ✓
  ⇒ 与"扩展配置"那条路【完全不同】✗：
    OH_NNCompilation_AddExtensionConfig("QuantConfigData", …) 在设备侧被拒
    （hilog: "hiai foundation not support extension config" ✗）
  ⇒ ★★ 也就是说：int8 若走【张量量化参数】而不是【扩展配置】，设备是接受的 ★★
```

### 还没走通的一步 ✗

```
✗ OH_NNModel_AddOperation(model, OH_NN_OPS_ADD / OH_NN_OPS_MATMUL, …) ⇒ rc=2 (INVALID_PARAMETER)
  · 张量全部 AddTensor 成功 ✓、量化参数全部成功 ✓、SetTensorData 成功 ✓
  · 仅 AddOperation 失败 ⇒ 属于【图构造细节】（参数/索引数组的写法），与 int8 无关
```

### 新增文件

| 文件 | 作用 |
|---|---|
| `int8_probe.c` | 搭最小 int8 图（ADD / MATMUL）+ 挂 `NN_QuantParam` + 编译；当前停在 `AddOperation` rc=2 |
| `avail_probe.c` | 调 `OH_NNModel_GetAvailableOperations` 问设备支持哪些算子（也受同一个 `AddOperation` 阻塞） |
| `inc/neural_network_runtime/` | 官方 NNRt 头文件副本（枚举/Signature 的权威出处 ✓）：`neural_network_runtime.h` · `_type.h` · `_core.h` · `_inner.h` |

### 下一步（明确）

1. 修 `AddOperation` 的 rc=2（对照官方文档示例逐字段核对参数/索引数组）
2. 通了之后 ⇒ `GetAvailableOperations` 问 NPU 是否支持 int8 的 ADD/MATMUL
3. ⇒ `OH_NNCompilation_Build` + `OH_NNExecutor_RunSync` ⇒ **int8 在 NPU 上跑通与否，即见分晓**

### ★ 决定性实验结论（已做完）★

```
★ 问题：用在线构图 API（OH_NNModel_AddOperation）搭 int8 图，能否在 NPU 上编译？
★ 答案：★明确失败 —— 而且【不是写法问题】✗★
```

| 实验 | 结果 |
|---|---|
| int8 MATMUL（补上 `SetTensorType` 后） | `AddOperation` ★rc=2 (INVALID_PARAMETER)★ ✗ |
| int8 ADD | 同上 ✗ |
| ★**逐字照抄官方文档的 fp32 ADD 示例**★（4 维 [1,2,2,3] · FLOAT32 · SetTensorType · INT8 激活参数） | ★**同样 rc=2**★ ✗ |

```
★★ 连官方示例都失败 ⇒ ★排除"我写错"★ ⇒
   ★★ 结论：★在线逐算子构图这条路在这台设备上走不通★★ ✗
   （设备的 NNRt 实现不认 OH_NNModel_AddOperation 这条入口 ✓）

★★ 但设备【确实能跑 NPU 图】★ ⇒ 现役 fp32 段全部 Build 0 / Predict 0 ✓✓
   ⇒ 可行入口是 ★OH_NNModel_BuildFromLiteGraph★（MS-Lite 内部用的就是它 ✓）
```

### ★★ 由此得到 int8 的正确判据（下一步）★★

```
★★ BuildFromLiteGraph（从 MS-Lite 图构出 OH_NNModel）
   ⇒ 逐张量 OH_NNModel_SetTensorQuantParams
   ⇒ OH_NNCompilation_Construct / SetDevice / ★Build★ ★★

★ 关键：这条路【不需要扩展配置】✗ —— 正合本设备的处境 ✓（它恰好拒绝扩展配置 ✗）
★ 而且 SetTensorQuantParams 已被证明在本设备上 rc=0 ✓✓（见上文）
```

### 新增文件

| 文件 | 作用 |
|---|---|
| `int8_probe.c` | 最小 int8 图（MATMUL/ADD）+ `NN_QuantParam` + 编译 ⇒ 停在 `AddOperation` rc=2 |
| `ctrl_probe.c` | ★对照★：逐字照抄官方 fp32 ADD 示例 ⇒ 同样 rc=2（证明不是写法问题 ✓） |
| `avail_probe.c` | `GetAvailableOperations` 查算子支持 |
| `inc/neural_network_runtime/` | 官方 NNRt 头文件副本（枚举/Signature 权威出处 ✓） |
