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
| IDA | 开发机 `ssh <user>@<host>`，`~/re/`（`libhiai_llm_engine.so` + `ida_decomp.py`）|

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
分词器（**该模块后已删除**，此处保留历史）：`src/cann_llm/backends/hiai_tokenizer.py`（纯 Python，**已用往返验证**：
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
> 📌 该文件**已删除**（结论证明它无用：最终走 `Executor_CreateFromJson`）。此处仅保留探索记录。
→ 说明该布局猜测**不对**。

## 下一步（按优先级）

1. **反编译 `SetTokenizer`(0xfc118) 与 `InitOption_Create`(0xfa468)** —— 找**谁写 `opt+0`**
   （后台任务 `bash-220` 正在跑，结果在 开发机 `~/re/tokopt.txt`）
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
| ~~`cann-llm/src/cann_llm/backends/hiai_tokenizer.py`~~ | 纯 Python Qwen 分词器 —— **已删除**（全导出 API 重构后不需要自己分词）|
| ~~`cann-llm/src/cann_llm/backends/hiai_shim.c`~~ | C++ 辅助库 —— **已删除**（证明无用）|
| `probe2.py` / `test_infer.py` / `test_shim.py` | 单变体探针（一变体一进程）|
| 开发机 `~/re/*.txt` | 各函数反编译结果 |

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
**下一步 = 读 `sub_1129B4` 的反编译**（开发机 `~/re/sub1129.txt`，脚本 `~/re/run6.sh`），
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

`modelInfo` 用 `hiai_shim.c` 构造（真 C++ `std::string`；该文件**已删除**），
设备上编译（**文件已删除，仅存记录**）：`clang -x c++ -std=c++17 -shared -fPIC -o libhiai_shim.so hiai_shim.c -L/system/lib64 -lhiai_llm_engine`
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
② opt  = shim.hiai_make_opt(omc, 3, 4, "tokenizer.json", 模型目录, 0)   # shim 已删除，此路已废弃
③ Init_Use_Option(exec, opt)  → 实测 rc=0 ✓
④ ctx  = Context_CreateFromContextJson(context_super.json)
⑤ Context_SetMaxGenTokens / SetTemperature / SetTopP / SetTopK / SetSeed / SetStopSeq
⑥ 注册四个回调：SetOnPrefill / SetOnSomeToken / SetOnAllTokens / SetOnGenerateAsyncFailed
⑦ prompt = Prompt_Create(); Prompt_SetTokenIds(prompt, ids, n)   ← ✗ 这条路径已被证伪，见文末更正节
⑧ Executor_GenerateAsync(exec, ctx, prompt)   → 实测 rc=0 ✓（未注册回调时 60s 零 token）
⑨ 在 OnAllTokensDone 回调里读 Context_GetAllGeneration
⑩ Context_TerminateOnce(ctx)
```

**已完成**：①②③ ✓（rc=0 实测）、分词器 ✓、⑥ 的签名 ✓。
**仅剩**：⑥ 回调自身的参数形态 → ⑧⑨ 取 token → 接进 `hiai.py` → CLI/HTTP → 测试 → 提交。

---

## ★★★ Round 44–47：失败点的完整下钻（EngineExecutorImpl 内部调用链）

### 调用链（已逐层反编译）

```
Executor_GenerateAsync(exec, ctx, prompt)          // 0xfdbec：仅做参数校验 + 组装，然后转发
  └─ sub_118704(exec, ctx, params)                 // 0x118704：真正的异步入口
       lock(exec+8)
       if (!(exec+48 & 1))     → "isInit_.load()" → return 1        [测试 769]
       if (!*(void**)(exec+200)) → "pipelineExecutor_" null → 1     [测试 771]
       if (!ctx)               → "ctx" null → 1                     [测试 773]
       (*vtable+240)(pipelineExecutor_, exec+176, exec+188)
       if (exec[992]&1) sub_150794(pipelineExecutor_, ...)
       v17 = sub_153B98(*(void**)(exec+200), ctx, params)   ← ★ 真正干活，失败在这里
       if (v17) (*(vtable+232))(pipelineExecutor_)          ← ★ 触发 OnGenerateAsyncFailed 的路径
       return sub_139400(v17)
```

### 已实测排除的原因（别再试）

| 假设 | 实测结果 |
|---|---|
| `isInit_`(+48) 未置位 | ✗ **已置**（`+48=1`，`CreateFromJson` 之后就是 1）—— 直接读内存验证 |
| 手动清 `+48` 再 `Init_Use_Option` | ✗ 更糟：`GenerateAsync` 立刻返回 1（首检失败）|
| `Init_Use_Option` 初始化不全 | ✗ `CreateFromJson` **本身就完整初始化**（含 `+48`）；`Init_Use_Option` 走"已初始化"早退（返回 0 但无操作）|
| `pipelineExecutor_` 为空 | ✗ 否则不触发回调；我们**触发了** |
| Context 来源问题 | ✗ `CreateFromContextJson` 与 `Context_Create()`+全 setter **表现完全相同** |
| 缺 `SetInitTokenLen`/`SetCallbackFreq`/`SetDoSampleFlag` | ✗ 都设了、都返回 0，仍失败 |
| 回调未注册 | ✗ 四个注册均返回 0，且 fail 回调稳定触发 |

### 关键内存布局（EngineExecutorImpl）

| 偏移 | 含义 |
|---|---|
| `+8` | mutex |
| `+48` | **`isInit_`**（`CreateFromJson` 置 1，`sub_1129B4` 第 393 行也置 1）|
| `+50` | 另一个标志 —— **只在 `sub_1129B4` 第 360 行（真初始化路径）置 1**；实测始终为 0 |
| `+51` | `SetJsonParam` 里置 1（`atomic_store`）|
| `+200` | **`pipelineExecutor_`**（非空 ✓）|
| `+992` | 某个可选分支开关 |

### 回调

- 注册签名：`(ctx, 函数指针)` —— 四个都一样，全部返回 0 ✓
- **回调自身签名：`(void* ctx)`** —— 单参（实测第 2/3 参是垃圾 ✗）
- **不带错误码** ✗

### 下一步（唯一）

反编译 **`sub_153B98`**(0x153B98) —— 它是 prefill/decode 的实现，**失败断言就在里面**。
（开发机 `~/re/sub153.txt`，脚本 `~/re/run10.sh`）

---

## ★★★ Round 49–52：失败点定位到 `InputExecutorCheck(ctx)`，并发现引擎内有「Prompt KV 缓存」阶段

### 失败调用链（全部反编译确认）

```
Executor_GenerateAsync                                  0xfdbec
  └─ sub_118704                                         0x118704
       ├─ isInit_(exec+48) / pipelineExecutor_(exec+200) / ctx   全过 ✓
       └─ sub_153B98                                    0x153b98
            └─ (vtable+464)(pipelineExecutor, ctx)  ← ★ InputExecutorCheck(ctx) 失败
                 → AI_Log_Print("InputExecutorCheck(ctx) == hiai::SUCCESS" "false, return FAIL.")
                 → 上层调 (vtable+232) → 触发 OnGenerateAsyncFailed（我们观测到的 ✓）
```

### 由字符串挖出的流水线要求（同一函数族）

```
"GeneratePreproc(ctx) == hiai::SUCCESS"
"CalculateCachedTokenLen() == hiai::SUCCESS"
"GetPromptKVCacheSize(cacheSize, allSize) == hiai::SUCCESS"
"ApplyMemoryForPromptKVCache(promptKVCache.cacheAddr_, allSize) == hiai::SUCCESS"
"LoadCacheToTensor(promptKVCache, cacheSize) == hiai::SUCCESS"
"ClearCachedEmbed() == hiai::SUCCESS"
"FreeKVSparseResource() == hiai::SUCCESS"
"LoadPromptKVCache inputIds_ size: %ld"
"LoadPromptKVCache inputIds_ tokenizer encode time: %.5f ms"     ← ★★ 引擎自己会做 tokenizer encode
"ExecuteDraftPrefill(ctx) == hiai::SUCCESS"
"GenerateAsyncByEmbedding" / "GenerateByEmbedding"
```

### ★ 修正一个我此前的错误判断

我曾判断官方 `api_config.json` 里的 `pmtCacheOperation` / `pfxInitTokenLen` / `initTokenLen`
是"服务级字段、引擎不读"，因此在转换时丢弃了它们 ✗。

**上述字符串表明引擎内部确有 Prompt KV Cache 阶段**（`GetPromptKVCacheSize` /
`ApplyMemoryForPromptKVCache` / `LoadCacheToTensor` / `LoadPromptKVCache`），
而我当时排除它们的依据只是"`CreateFromContextJson` 的 JSON schema 只读 4 个键"——
**那只说明 JSON 入口的 schema，不等于引擎内部不需要这些状态** ✗。

### `InputExecutorCheck` 的候选实现（来自字符串 xref）

字符串 `InputExecutorCheck` @0x40238 被 4 个函数引用：
`sub_176AC8`(0x176ac8)、`sub_26F948`、`sub_27D188`、`sub_154DE0`(0x154de0)；
调用点断言字符串 @0x7ef5b 被 `sub_151888` / `sub_15326C` / `sub_153B98` 引用。
完整反编译在 开发机 `~/re/inexec.txt`（2494 行）。

**下一步**：用 vtable 定位真正的 `InputExecutorCheck`（它是 `pipelineExecutor` 类的虚方法，
槽位 `vtable+464`），或在上述候选中按"是否读取 ctx 的输入字段"逐个排除；
读出它的断言即可知道 Context 缺什么。

---

## ★★★★ Round 78–80：方向性修正 —— 失败**不在** `Generate` 内部

### 决定性证据（读内存字段，前后对比）

```
[GenerateAsync 前]  promptType(ctx+920) = 3     ← Context 的默认值（"未设定"）
[GenerateAsync 后]  promptType(ctx+920) = 3
[GenerateAsync 前]  inferring(pe+3697) = 0
[GenerateAsync 后]  inferring(pe+3697) = 0
batchSize(pe+2568)=1   resourceFreed(pe+3698)=0   modelExecutor_(pe+2944)非空
```

- `CheckPromptType`（`pipeline_executor_base.cpp:372`）应把类型写入 `ctx+920`（text=0 / tokenids=1 …）
  → **实测始终是默认值 3** ✗ ⇒ **该步从未执行**
- `SetInferringStatus`（同文件:373，`vtable+16`）会置 `pe+3697 = 1`
  → **实测始终 0** ✗ ⇒ **该步也从未执行**

**两条独立信号 ⇒ `PipelineExecutorBase::Generate` 根本没走到 372/373 行。**
⇒ **失败发生在 `Generate` 被调用【之前】**，即在 `sub_153B98` 内。

### 因此在 `Generate` 内部做的一切排查都是无效功（教训）

曾被推断为失败点、后被**实测排除**的共 **9 项**：

| 被排除的假设 | 依据 |
|---|---|
| `isInit_`(exec+48) 未置位 | 实测已置 1 |
| `modelExecutor_` 为空 | 实测非空 |
| `Init_Use_Option` 初始化不全 | 反而**有害**（会清零 `pipelineExecutor_`/`modelExecutor_`）|
| `GeneratePreproc` 失败 | 8 字节空函数，必返回 0 |
| `InputExecutorCheck` 失败 | 只查 `modelExecutor_ != NULL`，实测非空 |
| `CheckPromptType` 失败 | 非空 prompts 即返回 0（类型只存进 ctx+920）|
| `SetInferringStatus`（槽 16）| 实测返回 0 |
| `SetBatchInfo`（槽 680）| **永远返回 0** |
| `batchSize_` 为 0 / 资源已释放 | 实测 1 / 0 |

### 方法论教训（我反复踩的）

1. **只看函数的局部就断定返回值会失败** ✗ —— 必须看它的**直接封装者**（`CheckPromptType` 就是典型）
2. **按字符串猜函数** ✗ —— 会被**同名不同类**误导（spec-eagle/tree 版 `GeneratePreproc` 就是典型），应改用 **vtable 槽位 + 内存读取 + 基址换算**（本方法已在 `InputExecutorCheck`(槽 464→0x176ac8)、`GeneratePreproc`(槽 368→0x179458)、`SetInferringStatus`(槽 16→0x16d888)、`SetBatchInfo`(槽 680→0x178838) 上四次验证成功 ✓）

### 下一步（新方向）

反编译 **`sub_153B98`** 的**线程函数**及其调用：`sub_15933C` / `sub_158D00` / `sub_158458` / `sub_156C64`
（`sub_153B98` 本体反编译已在 开发机 `~/re/sub153.txt`）—— 失败在 `Generate` **之前**。

---

# ★★★★★ 可行方案（Round 87–88 实测打通）

## 结论：直接调引擎内部函数，绕过两个缺陷

```python
exec   = HIAI_LLMEngine_Executor_CreateFromJson(executor_json)   # ① 只调它！
ctx    = HIAI_LLMEngine_Context_Create()                          # ② 每请求【新建】
#        HIAI_LLMEngine_Context_SetMaxGenTokens(ctx, n)
prompt = HIAI_LLMEngine_Prompt_Create()
HIAI_LLMEngine_Prompt_SetTokenIds(prompt, ids, n)                 # ③ 自研分词器编码的 ids

vec = (c_uint64 * 3)(0, 0, 0)                                     # 空 vector<Prompt336>
sub_FDA28(byref(vec), prompt)          # 0xfda28 —— 原样搬 336 字节（含 prompt[39..41] 的 tokenids）
sub_118704(exec, ctx, byref(vec))      # 0x118704 —— 真正的执行入口（GenerateAsync 的内部调用）

ids  = HIAI_LLMEngine_Context_GetOneTokenGeneration(ctx, buf, len)  # ④ 取 token id 数组
text = tokenizer.decode(ids)                                        # ⑤ ★ 自己解码
```

## 实测结果（官方 Qwen2.5-Coder-7B）

| 指标 | 实测 |
|---|---|
| **成功回调 `OnAllTokensGenerateDone`** | ★ **触发**（此前每次都是 `OnGenerateAsyncFailed`）|
| `GetInputTokenCount` | **9 / 11 / 8** —— 与自研分词器编码长度**逐一对上** ✓✓ |
| `GetDecodeNum` / `GetOutputTokenCount` | **59 / 60** —— 真的生成了 |
| `GetOneTokenGeneration` | 返回 **token id 数组**（不是文本）⇒ 必须自己 `decode` |

## 为什么必须这样（两个缺陷）

1. **`Prompt_SetText` 的 `std::string` 堆路径坏掉**：实测边界**精确在 22 字节**（libc++ SSO 容量）✗；
   **已用 C 程序最终确认缺陷在引擎内部**（C 与 ctypes 表现完全相同，连 `ampie...` 胡话都一样）✗
   ⇒ **文本路径对长 prompt 不可用** ⇒ 必须走 tokenids
2. **`GenerateAsync` 内部的 `Prompt→params` 转换丢弃 tokenids** ✗：
   证据 = 流水线里 `CheckPromptType` 把类型写进 `ctx+920`，实测值恒为 **3**（"text 与 tokenids 都为空"）；
   而 `sub_FDA28` 只做 336 字节 memcpy，不做筛选 ⇒ **转换发生在它之前的那一步** ✗
   ⇒ 直接调 `sub_FDA28` 把 Prompt 【原样】塞进参数向量，**跳过转换**

## 关键地址

```
Executor_CreateFromJson  0xff4d0    Context_Create        0x1083c8
Prompt_Create            0xfa82c    Prompt_SetTokenIds    0xfbedc
Prompt_SetText           0xfaa00    SetMaxGenTokens       0x108f30
sub_FDA28 (push_back)    0xfda28    sub_118704 (执行入口)  0x118704
GetOneTokenGeneration    —          GetOutputTokenCount   —
```
运行时地址 = 模块基址 + 上述偏移（基址从 `/proc/self/maps` 读 `libhiai_llm_engine.so`）。

## 剩余（判据 #1–#4）

1. 用 `tokenizer.decode(ids)` 把输出变成文本
2. Context 过滤掉终止 token（`<|im_end|>` / `<|endoftext|>`）
3. 接进 `hiai.py` 的 `load()`/`generate()`，`supports_streaming` 依据 `SetOnSomeTokenGenerateDoneFunc`
4. CLI + HTTP 端到端验证（长 prompt / 中文 / 连续多次）
5. 测试与 git 提交

---

# ✅ 最终验证结果（Round 89，判据 #1 达成）

三个 prompt 全部通过上面那条路径跑通，**输出用自研分词器 `decode()` 得到连贯文本**：

| prompt | 字节 | 输入 token | 输出 token | 解码后的文本（截断）|
|---|---|---|---|---|
| `def add(a,b): return a+` | 23 | **9** ✓ | 60 | ``' b\n#def sub(a,b): return a-b\n#def mul(a,b): return a*b\n#def div(a,b): return a/b\n...'`` |
| `def add(a, b): return a + b` | 26 | **11** ✓ | 60 | ``'\n#def subtract(a, b): return a - b\n#def multiply(a, b): return a * b\n...'`` |
| `你好，请写一个加法函数` | 33 | **8** ✓ | 60 | ``'，输入两个数，返回它们的和。\n\n\ndef add(a, b):\n    return a + b\n\n# 示例使用\nresult = add(3, 5)\nprint("3 + 5 =", result)  # 输出: 3 + 5 = 8'`` |

- **长 prompt**（此前必崩）✓ 正确输出
- **中文 prompt**（此前必崩）✓ 正确输出中文 + 代码
- **`in=1` / `ampie...` 胡话彻底消失** ✓✓
- **`GetInputTokenCount` 与自研分词器编码长度逐一对上**（9/11/8）✓✓

取输出必须用 **`GetAllTokenGeneration(ctx, int32*, n)`**（int32 数组）——
`GetOneTokenGeneration` 只给一个 token，`GetAllGeneration` 返回 1（格式不符）。

## 剩余工作（判据 #2–#4）

1. 把上面 8 步搬进 `src/cann_llm/backends/hiai.py` 的 `load()` / `generate()`
2. 过滤终止 token（`<|im_end|>` / `<|endoftext|>`）
3. `supports_streaming`：用 `SetOnSomeTokenGenerateDoneFunc` + 轮询 `GetDecodeNum` 实现
4. CLI（`-b hiai`）与 HTTP 服务端到端验证
5. 测试 + 提交（判据 #4/#5 收尾）

---

# ✅ 判据 #3 / #4 / #5 收尾（Round 91–93）

## 判据 #3：CLI 与 HTTP 端到端 —— 达成 ✓

```bash
# CLI
cann-llm chat -b hiai -d <模型目录> -p "def add(a, b): return a + b" --maxtok 60
# → ```python\n#   def add(a, b):\n#       return a + b\n#   ``` ✓

# HTTP（OpenAI 兼容）
POST /v1/chat/completions
{"model":"hiai","messages":[{"role":"user","content":"def add(a, b): return a + b"}],"max_tokens":60}
# → {"object":"chat.completion","choices":[{"message":{"content":"The function `add(a, b)` takes two
#    arguments...\n```python\nresult = add(3, 5)\nprint(result)  # Output: 8\n```"},
#    "finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":60,"total_tokens":79}} ✓
```

## 判据 #4：测试 —— 达成 ✓

```
# 设备【不】自带 Python。两种 Python（应用市场「Python安装器」版 / harmonybrew 版）
# 都用这条命令装 —— 直接写 pip 有可能会落到不是你想用的那个解释器上。
python3 -m pip install pytest
PYTHONPATH=src pytest -q      # → 184 passed in 8.41s ✓
```

## ★ 第三个关键修复：**绝不能在生成期间轮询 Context**

症状：调用成功（内部执行入口返回 0），但随后 `libc++abi: Pure virtual function called!` → abort。

原因：推理是**异步**的，在引擎工作线程运行期间从外部读 Context（哪怕只是
`GetAllTokenGenerationLen`）会与工作线程**竞态**，导致虚表被破坏。

修法：**完全由回调驱动** ——

```python
import threading
ev_done, ev_fail = threading.Event(), threading.Event()
CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
cb_done = CB(lambda _p: ev_done.set())        # 必须长期持有引用，勿被 GC
cb_fail = CB(lambda _p: ev_fail.set())
Context_SetOnAllTokensGenerateDoneFunc(ctx, cast(cb_done, c_void_p))
Context_SetOnGenerateAsyncFailed(ctx, cast(cb_fail, c_void_p))

run(exec, ctx, byref(vec))                     # 返回 0 后
ev_done.wait(timeout=300)                      # ★ 等回调，【不要】轮询
if ev_fail.is_set():
    raise GenerationError("引擎报告生成失败")
```

## 另一处坑：`load()` 里不要预建 Context

预建（`Context_CreateFromContextJson`）再建 Executor → **SIGTRAP**。
正确顺序：**只调 `Executor_CreateFromJson`**；Context 每请求用 **`Context_Create()`（无参）** 新建。

## 判据状态

| # | 判据 | 状态 |
|---|---|---|
| 1 | 长 prompt / 中文 / 连续多次输出正确 | ✅ 达成 |
| 2 | 流式 `supports_streaming = True` | ✗ **唯一剩余**（`SetOnSomeTokenGenerateDoneFunc` 已定位）|
| 3 | CLI + HTTP 端到端 | ✅ 达成 |
| 4 | 测试通过不回归（184 passed）| ✅ 达成 |
| 5 | 提交到 git | ✅ 达成 |

---

## ⚠ 判据 #2（流式）—— 尝试过，导致 segfault，已回退

**当前状态**：`supports_streaming` 保持 `False`。其余判据 #1/#3/#4/#5 全部达成。

**尝试过的做法**（在 `generate()` 内）：

1. 注册 `HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(ctx, cb_some)`
2. `cb_some` **在回调内部**读 Context（`GetAllTokenGenerationLen` + `GetAllTokenGeneration`）
   —— 依据是"回调跑在引擎工作线程上，同线程读 Context 应无竞态"
3. 把 `decode(全量)[已发出长度:]` 作为增量推入 `queue.Queue`，生成器边等 `Event` 边 `yield`

**结果**：`supports_streaming = True` 生效、分词器就绪，随后 **Segmentation fault (core dumped)**。

**推测原因（供新会话排查）**：

- 该回调是从**引擎自己的工作线程**进入 Python 的；`ctypes.CFUNCTYPE` 回调在**外来线程**上
  重入 Python/GIL 是已知的脆弱点（`OnAllTokensDone` 回调只做 `Event.set()` 尚可，
  但 `_on_some` 里做了较多 Python 工作：ctypes 调用、列表切片、分词器 `decode`）
- 也可能是 `SetOnSomeTokenGenerateDoneFunc` 的**回调签名/触发时机**与假设不符
  （该 setter 的绑定在 `_HiaiBindings.SIGS` 里的签名 `(c_int, [c_void_p, c_void_p])` 需复核）

**建议的安全做法**（未验证）：

- 回调里**只**做最小动作（例如设一个 `ctypes.c_int` 标志或调用一次 `Event.set()`），
  **不在回调里解码/读 Context**；增量由**另一个线程**在生成期间读取 —— 但注意
  从**外部**轮询 Context 已验证会导致 `libc++abi Pure virtual function call` abort（见上文），
  所以这条路需要重新设计（例如让回调把 token 数写进一块 `ctypes` 预分配内存，
  由生成器只读那块内存，不碰 Context）

---

# 🎉 全部判据达成（Round 94–96）：流式已打通

## 判据 #2（流式）—— 达成 ✓

| 入口 | 实测 |
|---|---|
| 后端 `supports_streaming` | **True** ✓ |
| 后端 `generate()` | **30 token → 30 个分片**，约 60 ms/片，零崩溃 ✓ |
| CLI（默认即流式）| 短 prompt / 中文 prompt 均退出码 0，逐字输出 ✓ |
| HTTP `stream: true`（SSE）| **56 行 `data: {...chat.completion.chunk...}` + `data: [DONE]`** ✓ |

## ★★ 之前多次 segfault 的【真正根因】——两个，缺一不可

### 根因 1：`_HiaiBindings` 缺少这两个函数的 `argtypes`

```python
"HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc": (c_int, [c_void_p, c_void_p]),
"HIAI_LLMEngine_Context_GetOneTokenGeneration":         (c_int, [c_void_p, c_char_p, c_int]),
```

只补上这两行，**问题就从"必崩"变成"完全不崩"** —— 这是本次定位的关键
（定位手段：用"后端 `load()` + 独立脚本自设 argtypes"做二分，一次命中）。

### 根因 2：回调里**不能触碰 `self`**

引擎工作线程上访问 Python 对象的属性会 segfault。回调必须**只闭包局部名**：

```python
_acc: list = []
_one_fn = self._bind.lib.HIAI_LLMEngine_Context_GetOneTokenGeneration   # 先取到局部
_byref, _cast, _i32, _cp = ctypes.byref, ctypes.cast, ctypes.c_int32, ctypes.c_char_p

def _on_some(p: object) -> None:          # 只用 _one_fn / _acc / _byref …，不碰 self
    try:
        _v = _i32(0)
        if _one_fn(p, _cast(_byref(_v), _cp), 4) == 0:
            _acc.append(int(_v.value))
    except Exception:
        pass
```

## 流式的完整正确形态（已验证）

```
回调（引擎工作线程）          主线程（生成器）
─────────────────────        ──────────────────────────────
GetOneTokenGeneration(p)  →  _acc.append(id)
                             轮询 len(_acc) 变大
                             → decode(全量 id) → 与已发出文本 diff
                             → yield GenerationChunk(text=增量)
                             （OnAllTokensDone 后补最后一次 + finish_reason="stop"）
```

### 三条实测禁忌（都会 segfault 或 abort）

1. 回调里用 **`GetAllTokenGeneration`**（全量拷贝 → 重入引擎）✗
2. 回调里**解码** / 用 **`queue.put`** / 触碰 **`self`** ✗
3. 生成期间**从外部读 Context**（与工作线程竞态 → `libc++abi Pure virtual function called!`）✗

## 判据总表（全部达成）

| # | 判据 | 状态 |
|---|---|---|
| 1 | 长 prompt / 中文 / 连续多次输出正确，`in=1` 与 `ampie` 消失 | ✅ |
| 2 | 流式 `supports_streaming = True`（后端 + CLI + HTTP/SSE）| ✅ |
| 3 | CLI + HTTP 端到端（含 OpenAI 兼容流式响应）| ✅ |
| 4 | 测试通过不回归（**184 passed**）| ✅ |
| 5 | 提交到 git | ✅ |

---

# ★★★★★ 重大更正（重构完成）：**完全不需要内部函数**

本文档前述章节记录的「绕道内部函数（`0xFDA28` / `0x118704`）」是**基于错误前提**的做法，
现已**全部删除**。正确序列来自系统服务自己的实现。

## 证据来源：系统服务怎么调引擎

`/system/lib64/libhm_model_engine_service.z.so`（11434 服务）里
`AIMM::HIAI::HiaiSession`（源文件名 `hiai_session.cpp`）**只按名字直接导入 49 个
`HIAI_LLMEngine_*` 符号**（`.dynsym` UND），**全部是导出符号**，一个内部函数都没碰：

```
Prompt_*  : 只有 Prompt_Create / Prompt_Destroy      ← 没有 SetText、没有 SetTokenIds
Context_* : 41 个（含 SetPrefixPrompt / GetOneGeneration / GetGenerateStatus …）
Executor_*: Create / Deinit / Destroy / GenerateAsync / Init_Use_Option / SetInferencePerfMode
InitOption_*: Create / Destroy / SetInferType / SetModel / SetTokenizer
```

**反编译实锤的调用序列**（`svc_call.txt` / `svc_gen.txt`，均在 开发机 `~/re/`）：

```c
// HiaiSessionRun（hiai_session.cpp:1419-1434）
ApplyChatTemplate(req, …) → promptStr
Context_SetPrefixPrompt(ctx, promptStr.c_str())            // :1429  ★ 输入＝文本
Context_SetInitTokenLen(ctx, param.initTokenLen)           // :1431  ★ 两个参数（调用点实锤）
HiaiSession::LLMEngineRun(this, param)                     // :1434

// LLMEngineGenerateAsync（:786-811）
Context_SetOnAllTokensGenerateDoneFunc(ctx, …)             // :786
Context_SetOnSomeTokenGenerateDoneFunc(ctx, …)             // :793
Context_SetOnGenerateAsyncFailed(ctx, …)                   // :800
Executor_GenerateAsync(exec, ctx, str.c_str())             // :811  ★ 第 3 参＝文本，不是 Prompt*

// OnSomeTokensGenerated（回调，0x16c034）—— 流式的读就在回调里
Context_GetOneGenerationLen(ctx, &len)
Context_GetOneGeneration(ctx, buf, len)
```

上下文/执行器句柄在 `HiaiSession` 里是 `*(this + 376)`（context）与 `*(this + 384)`（executor）。
`Context_SetPrefixPrompt` 内部只是 `std::string::operator=(ctx + 576)`（`sub_15B0B0`）——
**它不分词**，分词由引擎自己做（tokenizer 在模型配置里，`InitOption_SetTokenizer` 告诉它）。

## 我此前错在哪

| 我原来的判断 | 真相 |
|---|---|
| `GenerateAsync` 第 3 参必须是 `Prompt*` | 是 `std::string::c_str()` **文本** |
| 输入只能靠 `Prompt_SetTokenIds` | 是 `Context_SetPrefixPrompt(ctx, 文本)` |
| 22 字节 SSO 缺陷逼我绕道 | 那条通道本就不该用；正确通道不涉及它 |
| 所以要直调内部 `0xFDA28` / `0x118704` | **完全不需要** |

**教训**：反编译单个函数得到的原型常常不完整（`SetInitTokenLen` 就骗过我一次）；
**调用点才是硬证据** —— 而「系统服务怎么调」是现成的、权威的答案。

## 三个实现坑（都踩过，已修）

1. **ctypes 的 `argtypes` 必须完整**：`GenerateAsync` 第 3 参要声明成 `c_char_p`
   （`c_void_p` + Python `bytes` 不会补 NUL → 引擎读越界 → segfault）；
   漏了声明（默认 `c_int`）会把指针截成 32 位 → 同样 segfault。
   自查办法：列出代码里 `self._bind.lib.X` 用到的每个符号，逐个核对 SIGS 里有声明。
2. **`Get*Generation` 家族的返回值不是成败标志**：实测 `GetAllGeneration` 返回 **1**
   而文本完全正确（`GetOneGeneration` 同样）。**只按长度与缓冲区内容判断**，别用 `!= 0` 当失败
   —— 这个坑让输出一直是空字符串。
3. **流式的读必须在回调里**（同线程，安全）；从外部主线程读会与引擎工作线程竞态，
   轻则读到空、重则 `libc++abi: Pure virtual function called!` abort。

## 现在的实现

```
load()      : Executor_CreateFromJson(executor_json)              // 保留
generate()  : ctx = Context_Create()
              Context_SetPrefixPrompt(ctx, prompt 文本)
              Context_SetInitTokenLen(ctx, init_token_len)
              Context_SetMaxGenTokens(ctx, maxgen)
              三个回调（AllTokens / SomeToken / Failed）
              Executor_GenerateAsync(exec, ctx, prompt 文本)
              OnSomeToken 回调内读 GetAllGeneration → 增量化 → Queue → 主线程 yield
              Context_Destroy
```

**实测**：27B/33B(中文)/23B prompt 全部输出正确；逐 token 流式 30 分片；
CLI、HTTP（含 SSE）、184 测试全过。

---

# 补充：第 4 个坑 —— **必须自己设停止序列**

`Context_SetStopSeq` 的签名（反编译实锤，`0x109224`）：

```c
HIAI_LLMEngine_Context_SetStopSeq(ctx, const char** stopSeq, unsigned int stopSeqLen)
// stopSeqLen 有效范围 1..9（否则报 STOP_SEQ_MAX_LEN 断言）；每个元素必须非空，否则 FAIL
```

**不设它的后果**：引擎生成完答案后**不会停**，会继续编出

```
"def add(a,b): return a+b<|im_end|>\n<|endoftext|><|endoftext|>Human: Can you …"
```

这种假对话（`out_tokens` 直接跑满 `max_tokens`）。设了之后同一 prompt 的
`out_tokens` 从 40 变成 **9** —— 在 `<|im_end|>` 处**真停** ✓。

停止序列**不用自己编**：官方模型的 `api_config.json` 里就有

```json
"stopSeq": ["<|im_end|>", "<|endoftext|>"]
```

实现里优先读它，读不到才用同款默认；读侧另做一次兜底截断（含 `<|im_start|>`），双保险。

## 最终实现（全导出 API）

```
load()      : Executor_CreateFromJson(executor_json)；读 kv_cache_max_len / initTokenLen / stopSeq
generate()  : ctx = Context_Create()
              [try]  Context_SetPrefixPrompt(ctx, prompt 文本)
                     Context_SetInitTokenLen(ctx, init_token_len)
                     Context_SetStopSeq(ctx, stopSeq[], n)
                     Context_SetMaxGenTokens(ctx, maxgen)
                     三个回调（AllTokens / SomeToken / Failed）
                     Executor_GenerateAsync(exec, ctx, prompt 文本)
                     OnSomeToken 回调内读 GetAllGeneration → 增量 → Queue → yield
              [finally] Context_Destroy
```

**用到的 15 个符号全部在 `nm -D` 导出表内**（可用下面的自查法复核）。

## 一条可复用的自查法（本次踩坑后总结）

```python
# 1) 列出代码里调用的每个引擎符号
called = set(re.findall(r'self\._bind\.lib\.(HIAI_LLMEngine_\w+)', src))
# 2) 列出 SIGS 里声明过的
declared = set(re.findall(r'"(HIAI_LLMEngine_\w+)":', src))
# 3) 列出 .so 导出的
exported = {nm -D ... 里 type 为 T/W 的符号}
# 要求：called == declared ⊆ exported
```

三处任一不齐都会以 **segfault** 的形式表现出来（第 3 条踩过两次：
漏声明 → 指针被当 int 截断；声明成 `c_void_p` → 传 bytes 不补 NUL）。

---

# 附：满屏 `Unknown class perfgenius_interface` 的来源（结论：系统 SELinux 策略，非本项目问题）

## 现象

stderr 反复出现：

```
Unknown class perfgenius_interface
```

（常与 `avc:  could not determine enforcing mode: Permission denied` 一起出现。）

## 完整链条（逐层实证）

```
我们的进程
 └─ libhiai_llm_engine.so
      └─ 运行时 dlopen("libperfgenius_client.z.so")      ← 目的：给 NPU 设置温控事件
           └─ NEEDED → libselinux.z.so
                └─ 解析/检查 SELinux 策略 → 警告写到 stderr
```

**证据**：

1. 该字符串出现在 `libai_text_analyzer_innerapi.z.so` / `…_image_…`（它们静态链接了
   libselinux 的策略解析代码），同库还有
   `SELinux: Class %s not defined in policy.` / `Unknown permission %s for class %s` /
   `%s/class/%s/perms` —— 都是 SELinux 策略解析器的措辞。
2. `libhiai_llm_engine.so` 的 NEEDED **只有** `libz / libhilog_ndk / libc++_shared / libc`，
   但内部含字符串 `while loading perfGenius, dlopen %s failed` 与
   `Load PerfGeniusFuncs timeout, might failed to set thermal control event`，
   且导入了 `dlopen/dlsym/dlclose` ⇒ **运行时 dlopen**，用途是温控。
3. 进程 `/proc/self/maps` 快照里确实有：
   `chipset-sdk/libperfgenius_client.z.so`、`chipset-sdk/libperfgenius_proxy_1.0.z.so`、
   `chipset-sdk-sp/libselinux.z.so`。
4. `perfgenius_interface` **在策略里是声明了的**
   （`/system/etc/selinux/system_common.cil:214` 的
   `(class perfgenius_interface (perfCmdHandle … perfSetMode …))`），
   同时 `system.cil`、`compatible/40.cil` 等也有相关 neverallow / typeattribute。
   ⇒ 警告的成因是**兼容层策略与主策略的解析可见性不一致**，不是"没声明"。

## 为什么不"修"

- **它不能靠注册 class 解决**：SELinux 的 object class 由策略声明、在加载策略时由解析器处理，
  **没有运行时注册接口**。
- **也不能改策略**：`/system/etc/selinux/*.cil` 是系统只读文件。
- **唯一能从源头消除的办法**是让引擎不去 dlopen perfgenius（例如用同名空 stub 抢在
  `LD_LIBRARY_PATH` 前面），但那会**连温控一起屏蔽** —— 对长时间 NPU 推理，
  失去温控的风险大于一行 stderr。

## 结论

**接受现状**。用户侧若确实嫌吵，可在命令末尾加 `2>/dev/null`，
但要知道它会屏蔽**所有** stderr（含真正的报错）。

---

## ★ 采样参数：每个 Context 都能设，而且**默认是关的**

### 症状

同一提示每次都得到**完全相同**的输出。

### 两个原因（都实测）

1. **采样默认关着**。新建 Context 的实测默认值是：

   ```
   GetDoSampleFlag = 0     ← ★ 采样关 = 等价于贪心
   GetSampleGreedy = 0
   GetSeed         = 99    ← 引擎写死的种子
   GetTopK         = 100
   ```

   所以必须**每个请求显式**下发 `SetDoSampleFlag(1)` + `SetSeed(...)` +
   `SetTemperature/TopK/TopP` —— 光靠 executor 的 JSON 字段不够，
   因为每个请求新建的 Context 不会带上后来改的值。

2. 原来的实现把 sampler 写进 executor JSON 就完事了，没有按请求下发。

### ★ 附带发现：引擎根本不读我们 JSON 里的 sampler 字段

我们合成的 executor/context JSON 里写了 `sampler`（topK / topP / temperature /
repetition_penalty / seed，值取自模型的 `api_config.json`），但实测新建 Context 的
默认值**不是那些**：

```
我们 JSON 里写了 topK=20（来自 api_config.json）
实测 GetTopK = 100        ← 引擎自己的默认，不是 20
实测 GetDoSampleFlag = 0  ← api_config.json 写的是 True
实测 GetSeed = 99
```

⇒ **那份 JSON 里的 sampler 键名引擎不认**，采样参数只能靠 `Context_Set*` 显式下发。

### 因此：采样默认值以「模型自带的 api_config.json」为准

`config.py` 里曾写死 `top_p=0.95`，而官方包写的是 **0.8** —— 每请求下发 setter 时
把模型真值盖掉了，且**从输出上完全看不出来**（topK / temperature /
repetitionPenalty 恰好与官方一致，只有 topP 露了馅）。

现在 `ModelConfig` 的采样字段默认是 `None`（= 未指定），由
`modelcfg.read_sampler()` 读模型目录里的 `api_config.json` 作为权威来源，
按 `用户显式给的 > 模型自带 > 内置兜底` 的顺序合并（见 `config.resolve_sampler`）。
`seed` **不**跟随模型：模型里那 99 是引擎默认，沿用会让每次输出完全相同。

### 签名（**不是猜的**：用「设进去再读回来」逐条验证）

```
int Context_SetTemperature(ctx, float)      // ★ float32，不是 double
int Context_SetTopP(ctx, float)             // ★ float32
int Context_SetTopK(ctx, int)
int Context_SetSeed(ctx, int)
int Context_SetDoSampleFlag(ctx, uint8)
int Context_SetSampleGreedy(ctx, uint8)
```

对应的 `Get*` 都是 `(ctx, T*)` 出参形式，返回 0 表示成功。

### ★ 坑一：浮点参数必须用 `c_float`

`SetTemperature(ctx, c_double(0.5))` 之后读回的是 **0.0**。

原因：aarch64 上 `c_double` 走 `d0`、`c_float` 走 `s0`（即 `d0` 的低半），
而 0.5 的 IEEE754 double 位模式是 `0x3FE0000000000000`，**低 32 位正好是 0**
→ 引擎按 float 读 `s0` 就得到 `0.0f`。

改成 `c_float` 后 0.5 / 0.25 / 0.7 / 1.0 全部往返成功。

### ★ 坑二：getter 写回的是 **1 字节**

`GetDoSampleFlag(ctx, byref(c_int(-1)))` 得到 **-255**（`0xFFFFFF01`）——
它只写了 1 个字节，高 3 字节留的是哨兵值。用 `c_ubyte` / `c_bool`，
或者把缓冲区先清零。

### ★★ 重要限制：`seed` 的作用范围是**进程级**

`SetSeed` 对该进程的**第一次**采样生效；之后随机数流继续往下走，
再设同一个 seed **不会**重置：

| 场景 | 实测结果 |
|---|---|
| 新进程（CLI）同一 seed 跑两次 | ✅ 输出完全相同 |
| 新进程（CLI）不同 seed | ✅ 输出不同 |
| 长驻服务（同一进程）同一 seed 连打 4 次 | ❌ **4 次各不相同** |

引擎没有导出任何随机数重置接口（`nm -D` 里找不到 `random` / `rng` / `reset`），
所以长驻服务里想"固定回答"只能把 `temperature` 设 0（贪心解码，
实测与服务是否长驻无关）。

---

## ★★★ 结论性发现：官方不用 `Executor_CreateFromJson`，用的是 `InitOption` 那条路

### 怎么发现的

Qwen3-8B（8B，官方包，从系统模型管理器的详情页跳转下载）在本项目里**加载即崩**：

```
libc++abi: terminating … nlohmann::json … type_error.302:
           type must be string, but is array
  hiai.py:361  ← Executor_CreateFromJson
```

用 `lldb_test/` 那套（设备侧 `huawei-debug-lldb-server` + lldb，见
`~/work/llm/lldb_test/`）抓到调用栈，抛异常处引擎正在**按空格拆字符串**：

```
unnamed_symbol4777 + 3340:
    mov w1, #0x20                              # ' '
    bl  std::string::find(char, unsigned long)
```

模块内偏移 **0x2335d4**（加载基址实测 0x5556a40000 → 0x5556c735d4）。
附近 .rodata 是 LoRA 那一摊：`dynamic_lora_rank`(0x39880)、`dynamicLoraRank`、
`LORA_RANK_SUPPORT…`、`loraConf.loraDat…`。

### 排除过的（都实测）

| 尝试 | 结果 |
|---|---|
| `architectures` 改字符串 / 改成 Qwen2 / 删掉 | ✗ 同一个错 |
| `model_type` 改成 qwen2 | ✗ 同一个错 |
| 把**我们合成的 executor 里所有数组**都改成字符串 | ✗ 同一个错 |
| 移走 `omc.omc.loraconf`/`loradata` + 清空 `loraCfgPath` + 删 `lora_rank` | ✗ 正确参数顺序下仍崩 |
| 换成 20251024 那份包（旧格式，无 LoRA） | ✗ 仍崩 |

⇒ **那个数组不在我们传进去的 JSON 里**；引擎是从别处按自己的规则读的。

### 真正的答案：看官方服务怎么调引擎

```
readelf --dyn-syms /system/lib64/libhm_model_engine_service.z.so | grep LLMEngine
```

它导入的创建相关符号是：

```
HIAI_LLMEngine_InitOption_Create
HIAI_LLMEngine_InitOption_SetInferType
HIAI_LLMEngine_InitOption_SetModel
HIAI_LLMEngine_InitOption_SetTokenizer
HIAI_LLMEngine_Executor_Create
HIAI_LLMEngine_Executor_Init_Use_Option      ← ★★★
```

**没有 `Executor_CreateFromJson`** ✗ —— 官方走的是

```
InitOption_Create()
  → InitOption_SetInferType / SetModel / SetTokenizer
  → Executor_Create()
  → Executor_Init_Use_Option(exec, option)
```

**⇒ 我们用的是次要入口（整份 JSON）**，JSON 里任何一处形状不符合它的期望，
就在解析期抛 `type_error` ✗。这也解释了：

* 那些 `executor.json` / `executor_super.json` **派生文件官方根本不用**
* 7B 能跑只是**我们的 JSON 恰好对了**，不是这条路本身可靠
* 官方这套 API 在 `libhiai_llm_engine.so` 里**全部导出**（已 `nm -D` 确认）

### 这与早先的记录吻合

本文档 Round 35 那条「`Init_Use_Option` 需要 JSON 先载入 —— 两条路要一起走」
就是这条路。当时走到了岔口，最后选了 `CreateFromJson`。

### 下一步（待做）

1. 逆向这几个符号的签名与 `InitOption_SetModel` 需要的
   `HIAI_LMEngine_ModelInfo` 结构布局（IDA：`~/re/ida_decomp.py`）
2. 后端改成：Create → SetInferType/SetModel/SetTokenizer → Executor_Create
   → Init_Use_Option
3. 回归：7B 仍要能跑；Qwen3-8B 应能加载

### 官方那条路的完整形态（IDA 反编译 `libai_large_model_enginesvr.z.so` 得到）

函数：`OHOS::AI::LargeModelEngineBase::SetInferTypeAndTokenizer` @ 0x1df2b8
      `OHOS::AI::LargeModelEngineBase::LoadEngine` @ 0x1deb04

```c
// SetInferTypeAndTokenizer(option, std::string const& tokenPath)
HIAI_LLMEngine_InitOption_SetInferType(option, 0);          // 非 0 = 失败
// tokenPath 的 data/len 直接传下去（libc++ string 的两个字）
HIAI_LLMEngine_InitOption_SetTokenizer(option, ptr, len);   // ★ 三个参数

// LoadEngine(ModelDataInfo&, Executor*, InitOption*, bool)
v10 = HIAI_LMEngine_ModelInfo_Create();
HIAI_LLMEngine_InitOption_SetModel(option, 0, v10);         // ★ 三个参数
HIAI_LMEngine_ModelInfo_SetModelType(v15, 3);               // 官方用 3
HIAI_LLMEngine_InitOption_SetModelComponent(option, v17);   // 可选（组件模型）
```

引擎导出的配套 API（`nm -D` 实查）：

```
HIAI_LLMEngine_InitOption_Create / Destroy / SetInferType / SetModel / SetTokenizer
                                  / SetModelComponent / SetKVCacheSwitchableFlag
HIAI_LMEngine_ModelInfo_Create / Destroy / SetModelPath / SetWeightDir / SetModelType
                              / SetModelCacheStrategy / SetModelBuffer
                              / SetPreprocessorConfigPath / SetUserData
HIAI_LLMEngine_Executor_Create / Init_Use_Option / InitGraph_Use_Option
                              / InitWeight_Use_Option / Deinit / Destroy
```

### 实现前还需确认的

> ✅ **这四条已全部查清、后端已实现并实测通过 —— 见本文档最后一节
> 「官方那条路已打通（Round 100）」。** 下面是提问当时的原文，保留以便对照。

1. `InitOption_SetTokenizer` 第 2/3 参是 `(const char*, size_t)` 还是别的组合
   （从 libc++ `std::string` 取 data/len 的次序要与反编译逐条对上）
2. `SetModelPath` / `SetWeightDir` / `SetUserData` 的签名
3. 原来 `executor.json` 里 `llm_config` 的那些数值（`num_hidden_layers` / `hidden_size` /
   `kv_cache_max_len` / embedding 权重文件名 …）在这条路上**从哪个入口进**
   —— 候选：`SetUserData`（可能是 JSON 串）、或 `InitGraph_Use_Option` /
   `InitWeight_Use_Option`。**这是实现的关键未知点。**
4. `Init_Use_Option` 的返回值语义与失败时该看哪条日志

---

# ★★★★★ 官方那条路已打通（Round 100）：7B 与 Qwen3-8B 都能跑

上面那四个「实现前还需确认的」**全部查清了**（两个途径：引擎侧反编译 + 官方服务侧
反编译；再在设备上逐模型单进程实测）。后端已改为这条路，旧入口
`Executor_CreateFromJson` 不再被调用，并且**已从 `SIGS` 里摘掉** ——
不声明就调不出去，顺手让本项目那条自查法（`called == declared ⊆ exported`）保持成立。

## 一、四个待确认项 —— 逐条实证

### 1. `InitOption_SetTokenizer` 的签名：`(opt, int tokenizerType, const char* path)`

引擎侧反编译 `libhiai_llm_engine.so` **@0xfc118**：

```c
__int64 __fastcall HIAI_LLMEngine_InitOption_SetTokenizer(__int64 a1, int a2, __int64 a3)
{
  if ( a1 ) {
    if ( a3 ) {
      *(_DWORD *)(a1 + 4) = a2;
      std::__n1::basic_string<...>::assign(a1 + 8, a3);   // ← a3 一路进 assign
      return 0;
    }
    ... "tokenizerPath" "null, return FAIL."
```

`a3` 直接喂给 `std::string::assign` ⇒ 是 **`const char*`**，不是
`(ptr, len)` 那一对。★ **这意味着那个 C++ shim 完全不需要了** —— 旧记录里
`hiai_shim.c` 存在的理由是"要自己拼一个真正的 `std::string` 塞进 ModelInfo"，
而官方 API 本来就直接收 C 字符串（引擎自己 `assign` 出一份）。

调用点佐证：官方服务 `LargeModelEngineBase::SetInferTypeAndTokenizer`（@0x1df2b8）里
`HIAI_LLMEngine_InitOption_SetTokenizer(a2, v12, v11)`，`v11` 就是
`std::string::c_str()`。

### 2. `SetModelPath` / `SetWeightDir` / `SetUserData` 的签名

| 导出符号 | 地址 | 反编译出的行为 | 结论 |
|---|---|---|---|
| `LMEngine_ModelInfo_SetModelPath` | 0x2cba9c | `std::string::assign(mi+16, a2)` | `(mi, const char*)` |
| `LMEngine_ModelInfo_SetWeightDir` | 0x2cbb20 | `std::string::assign(mi+40, a2)` | `(mi, const char*)` |
| `LMEngine_ModelInfo_SetModelType` | 0x2cbc28 | `*mi = a2` | `(mi, int)` |
| `LMEngine_ModelInfo_SetUserData` | 0x2cbba4 | `*(mi+88)=a2; *(mi+96)=a3` | `(mi, void*, size_t)` |
| `ModelInfo_Create` | 0x2cb8d8 | `new(0x70)`，+16/+40/+64 三个 `std::string` 置空 | 结构见下 |

```c
struct HIAI_LMEngine_ModelInfo {        // 0x70 = 112 字节
    int64_t     model_type;             // +0
    std::string model_path;             // +16
    std::string weight_dir;             // +40
    std::string preprocessor_cfg_path;  // +64   （由 SetPreprocessorConfigPath 填）
    void       *user_data;              // +88   ┐ SetUserData(ptr, len)
    size_t      user_data_len;          // +96   ┘
    int32_t     model_cache_strategy;   // +104
};
```

★ `SetUserData` 收的是 **`(void*, size_t)` 裸缓冲区**，不是字符串 ——
所以它**不是** llm_config 的入口（候选 C 排除）。而且
`libai_large_model_enginesvr.z.so` / `libhm_model_engine_service.z.so` 两个服务
**都没有导入它**。两个服务实际导入的 ModelInfo 系列只有
`Create / Destroy / SetModelPath / SetWeightDir`（前者多一个
`SetModelType / SetModelBuffer / SetPreprocessorConfigPath`）。

### 3. ★★ `llm_config` 那些数值从哪个入口进 —— 答案是「引擎自己去读文件」

既不是 `SetUserData`，也不是 `InitGraph_Use_Option` / `InitWeight_Use_Option`。
真正的机制在 `InitOptionPacker` 里，两条路**都会**先做同一件事：

```c
// InitOptionPacker::SetInitOption(option)                        @0x12f684
// （= 没走过 CreateFromJson 时走的那条；compare: SetInitOptionByJson @0x130288）
GetConfigFilePath(modelInfo->modelPath, configFile_);   // ← 推出配置文件路径
FileUtil::LoadToBuffer(&buf, configFile_);              // ← 从磁盘读它
nlohmann::json j = parse(buf);                          // ← 这就是 llm_config
```

而 `GetConfigFilePath`（**@0x130088，反编译**）只做一件事：

```c
i = model_path.rfind('.', -1);
if (i == -1) { log("model config path error"); return FAIL; }
out = model_path.substr(0, i) + ".json";
```

⇒ `qwen3_8b_ceval_g256.omc` → **`qwen3_8b_ceval_g256.json`**，
`qwen7b.omc` → `qwen7b.json`。

**这就是"官方包里模型配置与 `.omc` 同名"这条命名约定的由来** ——
引擎按约定自己去读，用户不必（也无法）通过 API 把那些数值传进去。
所以：

* `executor.json` / `executor_super.json` 这类派生文件官方确实不用 ✓（旧结论对了）
* 我们**合成**的 JSON 在官方这条路上根本不会被读 —— JSON 入口（`CreateFromJson`）
  才是那个"额外"的功能，它把 llm_config 直接塞进 `llmConfig_` 成员
* 哪条路走哪个 packer 由 `EngineExecutorImpl` 的一个字节（`+51`）决定：
  `SetJsonParam` 会把它置 1 → `Init` 里走 `SetInitOptionByJson`；
  没置 1（= `Executor_Create` + `Init_Use_Option`）→ 走 `SetInitOption`。
  （`sub_1129B4` = `Init`，反编译 @0x1129b4 第 108–111 行就是这两个分支）

### 4. `Init_Use_Option` 的返回值语义与失败时看哪条日志

* **0 = 成功**，非 0 = 失败（`HIAI_LLMEngine_SUCCESS` / `FAILURE`）。
* 它在自己的入口处有两条断言，都是**纯参数检查**，返回值直接来自这里：
  * `executor` / `initOption` 为空 → 日志 `HIAI_LLMEngine_Executor_Init_Use_Option(450/451)`
  * `inferType == 0` 时还要求 **`modelInfo->weightDir` 非空**（`@0xfc750`，
    测 `.size()`），为空 → 报
    `"initOptionImpl->modelInfo->weightDir.size() > 0" "false, return ..."`
    （`llm_engine_executor.cpp:455`）
* 再往里（`Init` @0x1129b4）失败会打 `AI_INFRA` 级别的
  `engine_executor_impl.cpp / init_option_packer.cpp` 断言原文，例如
  `"readConfigBuffer" "null, return FAIL."`（＝ 那个 `<omc 同名>.json` 没读到）。
* 我们的取法：出错时 `src/cann_llm/enginelog.py` 抓 `hilog -x` **按 pid 过滤**
  后原样附在异常里（`_engine_log_suffix()`）。本轮实测这条路上没触发过失败日志。

## 二、新实现（`src/cann_llm/backends/hiai.py`）

```python
opt = InitOption_Create()
InitOption_SetInferType(opt, inferType)                    # api_config.inferType
InitOption_SetTokenizer(opt, tokenizerType, tokenizerPath) # api_config 的同名字段
mi  = LMEngine_ModelInfo_Create()
LMEngine_ModelInfo_SetWeightDir(mi, api_config.weightDir)  # ★ 不能为空（见上）
LMEngine_ModelInfo_SetModelPath(mi, api_config.modelPath)
LMEngine_ModelInfo_SetModelType(mi, 0)                     # = Create 的默认值，见下
InitOption_SetModel(opt, 0, mi)                            # ★ 第 2 参官方传 0
exec = Executor_Create()
Executor_Init_Use_Option(exec, opt)                        # 0 = 成功
```

参数取值的依据：`InitOption_SetModel(opt, 0, mi)` 的 **0** 和 `modelType` 的取值
都是从官方服务的调用点直接读出来的
（`LargeModelEngineBase::LoadEngine` @0x1deb04）：

```c
v10 = HIAI_LMEngine_ModelInfo_Create();
ILargeModelEngine::SetEngineModelBuffer(v10, omcVector, <weightDir>, <omcPath>);
HIAI_LLMEngine_InitOption_SetModel(a4, 0, v10);            // ← 0
...
v15 = HIAI_LMEngine_ModelInfo_Create();
HIAI_LMEngine_ModelInfo_SetModelType(v15, 3);              // ← 3，但这是给【组件】的
HIAI_LLMEngine_InitOption_SetModelComponent(a4, v15);
```

而 `SetEngineModelBuffer`（`ILargeModelEngine::SetEngineModelBuffer` @0x1dee08，
反编译，参数名来自它自己的日志串）证明**基座**只设三样、且**没有** `SetModelType`：

```c
// SetEngineModelBuffer(ModelInfo* mi, HIAI_LM_Buffer& omc, const string& weightDir, const string& omcPath)
HIAI_LMEngine_ModelInfo_SetModelBuffer(mi, omc);
HIAI_LMEngine_ModelInfo_SetWeightDir(mi, weightDir.c_str());   // ← 先 weightDir
HIAI_LMEngine_ModelInfo_SetModelPath(mi, omcPath.c_str());     // ← 后 modelPath
```

⇒ 基座 `ModelInfo->modelType` 就是 `ModelInfo_Create` 的清零值 **0**
（日志也印证：`"...SetEngineModelBuffer weightDir is %s"` / `"omcPath is %s"`，
两个 `std::string` 的次序就是 `(weightDir, omcPath)`）。
实测基座给 0、给 3 都 rc=0；实现按官方取证取 **0**（并显式写出来，让
「调用的符号」与 `SIGS` 声明一一对应 —— 项目那条 `called == declared` 自查法）。

两个生命周期注意点（都写进代码注释了）：

* option / modelInfo 的指针被 executor 长期持有（`Init` 把 option 原样交给了流水线），
  **不能提前 Destroy**，也不能让它们被 GC 回收 —— 挂在后端实例上。
* `Context` 仍然每请求新建（`Context_Create()`），输入/输出仍走
  `SetPrefixPrompt` / `GenerateAsync` / `GetAllGeneration`，采样与停止序列
  仍按请求 `Context_Set*` 下发 —— **只有建 Executor 这一处换了入口**。

## 三、实测（设备上单进程逐个跑，probe 见 `scripts/probe_official.py`（本地临时脚本，未随仓库分发））

| 模型 | `Init_Use_Option` | 端到端生成 |
|---|---|---|
| `qwen25_coder_7b_omc1024` | **rc=0** | ✓ 中文回答正常 |
| `Qwen3-8B` | **rc=0** | ✓ 中文回答正常（旧路径在此 abort） |
| `Qwen3-8B-20251024` | **rc=0** | ✓ 中文回答正常 |

`Executor_CreateFromJson` 对 Qwen3-8B 是**加载即 abort**
（`nlohmann::json type_error.302`），换到这条路之后不再出现。
本轮**没有**再复现/复核那次 abort 的精确栈帧（任务前提里已确证），
所以"为什么 JSON 入口会崩"仍以旧记录为准 —— 本节的结论只管"新入口可用"。
