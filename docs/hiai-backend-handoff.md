# hiai 后端：进度交接（Handoff）

> 给接手的人（或新会话）：读完这一页就能继续，不必重走弯路。
> 全部结论都是**实测 + 反编译**得来，不是推测。

## 目标

让 `cann-llm` 能驱动 HarmonyOS 内置的 LLM 引擎（`/system/lib64/libhiai_llm_engine.so`），
跑官方 Qwen2.5-Coder-7B 包（`models/qwen25_coder_7b_omc1024`），支持流式与 CLI/HTTP。

## 环境事实（别再试）

| 事实 | 说明 |
|---|---|
| **hilog 读不到** | `hilog` 命令存在但抓到 0 行（权限）→ 引擎自己的报错原文看不到 |
| **服务用 dlsym** | `libhm_model_engine_service.z.so` 同时 NEEDED 了引擎库**又** import `dlopen/dlsym` → 无静态调用点，**xref 追不到调用序列** |
| **`Init_Use_Option` 对入参敏感** | 错误取值会**段错误** → 实验必须**一变体一进程** |
| 编译器 | `/data/service/hnp/bin/aarch64-unknown-linux-ohos-clang`（编 C++ 要加 `-x c++`）|
| 设备上的 Python | `/data/service/hnp/python.org/python_3.12/bin/python3.12`（musl，与引擎兼容）|
| IDA | x570 `ssh hu60@192.168.31.96`，`~/re/`（`libhiai_llm_engine.so` + `ida_decomp.py`）|

## 已解决的两大块

### ① `Prompt_SetText` 的 22 字节缺陷 —— 根因已确证

`Prompt_SetText(prompt, text)` 内部是 `std::string::assign()`，
其**堆分配路径在本机不可用**：实测边界**精确落在 22→23 字节**（libc++ 的 SSO 容量）。

```
'def add(a,b): return a'  (22 字节) → ✓ 正确代码
'def add(a,b): return a+' (23 字节) → ✗ in=1，输出恒为 'ampie...'
```

**修法**（已实现）：改用 `Prompt_SetTokenIds(prompt, int32*, count)`，
即**自己在外面分词**——这正是系统服务的做法。
分词器：`src/cann_llm/backends/hiai_tokenizer.py`（纯 Python，**已用往返验证**：
vocab 151643 / merges 151387 / ByteLevel / BPE / NFC / 22 个特殊 token）。

### ② API 差异（全部反编译确认）

| # | 差异 |
|---|---|
| 1 | Context/Executor 收 **JSON 内容**（`CreateFromContextJson` / `CreateFromJson`）|
| 2 | 符号名：内部是 `Executor_CreateFromJson`（NDK 是 `...CreateFromExecutorJson`）|
| 3 | **没有** `Prompt_SetTokenId`（NDK 有 `Prompt_SetTokenIds`，注意不同名）|
| 4 | `Executor_Generate` 第 3 参是 **`Prompt*` 对象**，不是文本 |
| 5 | **不要用 `Prompt_SetText`**（见 ①）|

## 当前卡点（唯一）

用 `CreateFromJson` 建的 Executor **初始化不全** → `Generate` 返回 1。
服务的正路是 `InitOption_*` + `Executor_Create` + `Init_Use_Option`：

```
InitOption_Create()                              → opt            ✓ 实测
SetModel(opt, modelType, modelInfo*)             → rc=0           ✓ 实测（但见下）
SetTokenizer(opt, tokType, path)                 → rc=0           ✓ 实测
SetInferType(opt, inferType)                     → rc=0           ✓ 实测
Executor_Create()                                → exec           ✓ 实测（无参）
Executor_Init_Use_Option(exec, opt)              → **rc=1**       ✗ 卡在这里
```

`Init_Use_Option` 的失败条件（反编译原文）：

```c
v2 = *a2;                          // ← opt+0（未知写入者）
*((_BYTE *)a2 + 89) = 0;
if ( v2 ) return sub_1129B4();      // ★ 若 opt+0 != 0 → 走另一条路（不检查 weightDir）
v3 = *((_QWORD *)a2 + 5);           // opt+40 → 当作 modelInfo 结构体
"initOptionImpl->modelInfo->weightDir.size() > 0"  "false, return FAIL."
```

**已排除**：
- `SetModel(opt, type, path)` 的 path **不是字符串，是 `modelInfo*`**（反编译：`*(_QWORD*)(opt+40) = a3`）
  → 传字符串（`.omc` / 目录）都是错的：`.omc` 得 rc=1、目录**段错误**
- `SetModelComponent(opt, modelInfo)` 是设**组件模型**的（modelType ∈ [3, 3+0x76)，如 5=lmhead），**不涉及 weightDir**
- `SetInferType` 的取值（0/1/2/5）**都改变不了** `opt+0`（0 还会段错误）

**已尝试**：写了 `hiai_shim.c`（用真 C++ 组装 `modelInfo{int model_type; char reserved[36]; std::string weight_dir;}` 并代调）→ **编译成功**（92,608 字节）但 `Init_Use_Option` **仍 rc=1** ✗
→ 说明该布局猜测**不对**。

## 下一步（按优先级）

1. **反编译 `SetTokenizer`(0xfc118) 与 `InitOption_Create`(0xfa468)** —— 找**谁写 `opt+0`**
   （后台任务 `bash-220` 正在跑，结果在 x570 `~/re/tokopt.txt`）
2. 若仍无线索 → 反编译 **`sub_1129B4`**（`opt+0 != 0` 时走的那条路）—— 看它需要什么，
   就知道了 `opt+0` 的语义，也能反推该由谁设
3. `modelInfo` 的**真实布局**：从 `Init_Use_Option` 里 `v3+40` 的用法反推
   （注意它读 `*(u8*)(v3+40)` 与 `*(u64*)(v3+48)` → 像是 `std::string` 的 size/ptr）
4. 通了之后：`Context_Create()` + `Context_Set*` + 四个回调 + `Prompt_SetTokenIds` +
   `Executor_GenerateAsync` + `Context_TerminateOnce`（签名与地址见 `hiai.py` 末尾注释块）

## 关键文件

| 文件 | 内容 |
|---|---|
| `cann-llm/src/cann_llm/backends/hiai.py` | 后端；**文件末尾注释块有完整施工图 + 全部地址** |
| `cann-llm/src/cann_llm/backends/hiai_tokenizer.py` | 纯 Python Qwen 分词器（已验证）|
| `cann-llm/src/cann_llm/backends/hiai_shim.c` | C++ 辅助库（编译通过，布局待修正）|
| `probe2.py` / `test_infer.py` / `test_shim.py` | 单变体探针（一变体一进程）|
| x570 `~/re/*.txt` | 各函数反编译结果 |

## 引擎符号地址备忘

```
InitOption_Create 0xfa468    SetInferType 0xfc0c4    SetTokenizer 0xfc118
SetModel 0xfc1a8             SetModelComponent 0xfca78
Executor_Create 0xfc2b8      Init_Use_Option 0xfc750
Prompt_Create 0xfa82c        Prompt_SetText 0xfaa00  Prompt_SetTokenIds 0xfbedc
Executor_Generate 0xfd884    GenerateAsync 0xfdbec    Context_Create 0x1083c8
Context_CreateFromContextJson 0x10d148               SetMaxGenTokens 0x108f30
```

---

## ★ 更正（Round 33）：`weightDir` 是**不可达代码**，卡点在 `sub_1129B4`

`InitOption_Create`(0xfa468) 的反编译：

```c
v0 = operator new(0x60);          // 96 字节
*v0 = 0xFF00000005LL;             // ★ opt+0 = 0xFF000000（构造时就非零）; opt+4 = 5
sub_E7A1C(v0 + 8, &unk_4BF74);    // opt+8  = std::string（tokenizerPath）
*((int *)v1 + 8) = 121;           // opt+32 = 121 = HIAI_LLMENGINE_UNDEFINED_MODEL
v1[5] = 0;                        // opt+40 = 0（modelInfo 初始为空）
```

因此 `Init_Use_Option`(0xfc750) 里：

```c
v2 = *a2;                        // = 0xFF000000 → 永远非零
if ( v2 ) return sub_1129B4();    // ★ 永远走这里
... weightDir 检查 ...            // ✗ 不可达！
```

**结论：`weightDir` / `modelInfo` 布局【都不是】卡点** ✗ —— 之前几轮在这上面花的力气全是弯路 ✓。
真正的初始化逻辑与失败原因都在 **`sub_1129B4`(0x1129B4)**。

### 另外两点更正

| 之前以为 | 实际 |
|---|---|
| `modelType = 0` 正确（抄 `api_config.json`）| ✗ 默认 **121 = UNDEFINED_MODEL**，合法范围 **[3, 121)**；传 0 非法 |
| `SetModel` 的第 3 参须是 `modelInfo*`（含 `weightDir` @+40）| ✓ 这一点**仍然成立**（`*(_QWORD*)(opt+40) = a3` ✓），只是它**不影响**当前失败 |

### 当前唯一卡点

`Executor_Init_Use_Option` → `sub_1129B4` 返回失败 ✗。
**下一步 = 读 `sub_1129B4` 的反编译**（x570 `~/re/sub1129.txt`，脚本 `~/re/run6.sh`），
看它校验哪些字段的什么组合 ✓。

### 已实测的 `InitOption` 字段写入位置（供参考）

| 偏移 | 内容 | 写入者 |
|---|---|---|
| +0 | `0xFF000000`（构造时定） | `InitOption_Create` |
| +4 | tokenizerType | `SetTokenizer` |
| +8 | `std::string` tokenizerPath（引擎自己 assign ✓ 安全）| `SetTokenizer` |
| +32 | modelType（默认 121）| `SetModel` / `SetModelComponent` |
| +40 | `modelInfo*`（原样存指针 ✗）| `SetModel` |
| +48 | 组件模型类型（如 5=lmhead）| `SetModelComponent` |

---

## ★★★ 关键突破（Round 35）：`Init_Use_Option` 需要 JSON 先载入 —— 两条路要**一起走**

`sub_1129B4`（`Init_Use_Option` 实际转去的函数）的断言原文：

```
"!isInit_.load()"
"InitComponentModel(info) == HIAI_LLMEngine_SUCCESS"
"initOptionPacker.SetInitOptionByJson(initOptionImpl, llmConfig_, specLlmConfig_) == hiai::SUCCESS"   ★★
"initOptionPacker.SetInitOption(initOptionImpl) == hiai::SUCCESS"
"initOptionPacker.ParseTokenizerConf(tokenizerInitParams) == hiai::SUCCESS"
"initOptionPacker.ParsePipeLineExecutorConfALL(pipeLineExecutorInitParams) == hiai::SUCCESS"
"pipelineExecutor_" "null, return FAIL."
"tokenizer" "null, return FAIL."
```

**第二行是钥匙**：`Init_Use_Option` 内部要 `SetInitOptionByJson(..., llmConfig_, specLlmConfig_)`，
而这两个成员**只有 `Executor_CreateFromJson` → `SetJsonParam` 才会填入**。

### 因此正确的建 Executor 序列是**两步**（这正是之前一直在缺的一环）

```c
exec = HIAI_LLMEngine_Executor_CreateFromJson(executor_json);   // ① 载 JSON → llmConfig_
opt  = HIAI_LLMEngine_InitOption_Create();
HIAI_LLMEngine_InitOption_SetModel(opt, modelType, modelInfo*);
HIAI_LLMEngine_InitOption_SetTokenizer(opt, tokType, path);
HIAI_LLMEngine_InitOption_SetInferType(opt, inferType);
HIAI_LLMEngine_Executor_Init_Use_Option(exec, opt);             // ② 现在 llmConfig_ 有了 → 才可能成功
```

### 这解释了两条路各自失败的原因

| 我曾单独走 | 结果 | 原因 |
|---|---|---|
| 只走 ① `CreateFromJson` | `Generate` 返回 1 ✗ | `pipelineExecutor_` / `tokenizer` 等仍未初始化 |
| 只走 ② `Create`+`Init_Use_Option` | `Init_Use_Option` rc=1 ✗ | `llmConfig_` 为空 → `SetInitOptionByJson` 失败 |

### 下一步（明确）

把 ① 和 ② **串起来**跑一次 → 若 `Init_Use_Option` 返回 0 ✓，则 Executor 完整 ✓ →
接 `Context_Create` + `Context_Set*` + 四回调 + `Prompt_SetTokenIds` + `GenerateAsync` + `TerminateOnce`。

注意 `modelType` 的合法范围是 **[3, 121)**（默认 121 = UNDEFINED_MODEL，传 0 非法）。

---

## ★★★ 重大进展（Round 36–37）：Executor 建起来了，且确认要用 GenerateAsync

### 1. Executor 初始化的正确序列（**卡点已解决** ✓）

```c
exec = HIAI_LLMEngine_Executor_CreateFromJson(executor_json);   // ① 载 JSON → llmConfig_
opt  = HIAI_LLMEngine_InitOption_Create();
HIAI_LLMEngine_InitOption_SetModel(opt, modelType /*∈[3,121)*/, modelInfo*);
HIAI_LLMEngine_InitOption_SetTokenizer(opt, tokType, path);
HIAI_LLMEngine_InitOption_SetInferType(opt, inferType);
rc = HIAI_LLMEngine_Executor_Init_Use_Option(exec, opt);        // ② → **实测 rc=0** ✓✓
```

实测（`test_both.py`）：`① CreateFromJson -> exec ✓  ② InitOption ✓  ③ Init_Use_Option rc=0 ✓✓`
**两条路必须串起来** —— 只走任一条都会失败（见上一节对照表）。

`modelInfo` 用 `hiai_shim.c` 构造（真 C++ `std::string`），
设备上编译：`clang -x c++ -std=c++17 -shared -fPIC -o libhiai_shim.so hiai_shim.c -L/system/lib64 -lhiai_llm_engine`
（**注意**：文件名是 .c 时必须加 `-x c++`，否则报 `invalid argument '-std=c++17' not allowed with 'C'`）

### 2. 同步 `Generate` 不可用，**必须用 `GenerateAsync`**

| 调用 | 结果 |
|---|---|
| `Executor_Generate(exec, ctx, prompt)` | **返回 1**（拒绝）✗ |
| **`Executor_GenerateAsync(exec, ctx, prompt)`** | **返回 0**（接受）✓✓ |

### 3. 但 async 目前无产出 → 需要**注册回调**

实测（`test_async.py`，已先 `Init_Use_Option rc=0`）：
```
[ 23B] rc=0 in=0 status=-1 len=0 60.6s   ← 调用成功，但 60 秒内零 token
```
`in=0` 且无输出 → 引擎靠**回调**驱动/回传，必须注册：
```
Context_SetOnPrefillGenerateDoneFunc
Context_SetOnSomeTokenGenerateDoneFunc     ← 流式的来源
Context_SetOnAllTokensGenerateDoneFunc
Context_SetOnGenerateAsyncFailed
```
**下一步 = 反编译这四个 `Set*Func` 的签名**（都在引擎导出表里，用 `nm -D` 取地址后 `ida_decomp.py`），
注册回调后再 `GenerateAsync`，并在 `OnAllTokensGenerateDone` 里读 `GetAllGeneration`。
收尾用 `Context_TerminateOnce(ctx)`。

---

## ★ 最后一块：四个回调注册签名（Round 39）

全部是 **两参** `(ctx, 函数指针)` —— 已反编译确认：

```c
__int64 HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(void *ctx, void *cb);   // 0x109864
__int64 HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(void *ctx, void *cb);   // 0x1099d4
__int64 HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(void *ctx, void *cb);         // 0x109b44
__int64 HIAI_LLMEngine_Context_SetOnPrefillGenerateDoneFunc(void *ctx, void *cb);     // 0x10c86c
// 另有 SetOnPrefillSomeTokenGenerateDoneFunc 0x10c9dc、SetOnGenerateDraftTokensCallback 0x10d3a0
```

函数体把它们包进 `std::function` 之类的对象（`v4[4]`/`v5` BYREF）—— 所以 **ctypes 的 CFUNCTYPE 指针可用**。

**回调自身的参数形态尚未反编译**（引擎内部调用点才看得出来）。可用最小假设试探：
`OnSomeTokenGenerateDone(void *userData, const char *text, int len)`。

### 完整的收尾流程（拿到回调形态后照此接线）

```
① exec = Executor_CreateFromJson(executor_json)            ← 载 JSON
② opt  = shim.hiai_make_opt(omc, 3, 4, "tokenizer.json", 模型目录, 0)
③ Init_Use_Option(exec, opt)  → 实测 rc=0 ✓
④ ctx  = Context_CreateFromContextJson(context_super.json)
⑤ Context_SetMaxGenTokens / SetTemperature / SetTopP / SetTopK / SetSeed / SetStopSeq
⑥ 注册四个回调：SetOnPrefill / SetOnSomeToken / SetOnAllTokens / SetOnGenerateAsyncFailed
⑦ prompt = Prompt_Create(); Prompt_SetTokenIds(prompt, ids, n)   ← ids 由 hiai_tokenizer 生成
⑧ Executor_GenerateAsync(exec, ctx, prompt)   → 实测 rc=0 ✓（未注册回调时 60s 零 token）
⑨ 在 OnAllTokensDone 回调里读 Context_GetAllGeneration
⑩ Context_TerminateOnce(ctx)
```

**已完成**：①②③ ✓（rc=0 实测）、分词器 ✓、⑥ 的签名 ✓。
**仅剩**：⑥ 回调自身的参数形态 → ⑧⑨ 取 token → 接进 `hiai.py` → CLI/HTTP → 测试 → 提交。
