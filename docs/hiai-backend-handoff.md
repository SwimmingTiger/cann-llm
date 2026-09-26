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
（x570 `~/re/sub153.txt`，脚本 `~/re/run10.sh`）

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
完整反编译在 x570 `~/re/inexec.txt`（2494 行）。

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
（`sub_153B98` 本体反编译已在 x570 `~/re/sub153.txt`）—— 失败在 `Generate` **之前**。

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
