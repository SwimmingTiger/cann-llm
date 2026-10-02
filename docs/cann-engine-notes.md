# CANN LLM Engine 使用笔记（逆向 + 实测）

本文记录在鸿蒙设备上把 CANN LLM Engine 跑通的过程中，通过 IDA 反编译与
lldb 调试确认的行为。**这些结论不是从文档抄的**，改动 `backends/cann.py` 前请先读。

目标库：`/system/lib64/ndk/libcann_llm_engine.so`（NDK 层，无需 HAP / root）
以及 `/system/lib64/libhiai_llm_engine.so`（HIAI 层，官方 demo 链接的那个）。

---

## 1. `Generate` 的第 3 个参数是**文本**，不是 `Prompt*`

官方 demo 写的是：

```cpp
LLMEngine_Prompt* p = LLMEngine_Prompt_Create();
LLMEngine_Prompt_SetText(p, text);
LLMEngine_Executor_LLM_Generate(exec, ctx, p);      // 传 Prompt*
```

但反编译 `HMS_LLMEngineExecutor_Generate` 可见它**自己**构造 Prompt：

```c
memset(v7, 0, 0x30);                  // 栈上的局部 Prompt
std::string::assign(v7, a3);          // a3 被当作 const char* 文本
EngineExecutorImpl::Generate(exec, ctx, v7);
Prompt::~Prompt(v7);
```

**实测（两个库都试了）**：

| 传参 | `tokens_in` | 结果 |
|---|---|---|
| `Prompt*` | **1** | `raftraftraft…`（恒定重复） |
| 文本 | **5** | 正确 |

传 `Prompt*` 时，包装函数把 Prompt 对象的**原始字节**当 C 字符串读：
首字节是 libc++ string 的标识位（长字符串时为 `cap<<1|1`，例如 24 字符时
是 `0x31` = `'1'`），紧接着 `0x00` 截断 —— prompt 塌缩成 `"1"`，
只切出 1 个 token，模型只看到 1 个 token，于是输出恒定垃圾。

## 2. `HMS_LLMEngine_Context_Destroy` **会崩溃**，一律不调用

实测：每轮「新建 Context → 用完销毁」在**第二轮** core dump。
只新建不销毁、或复用同一个 Context，都正常。

策略：启动时建一个 Context 之后一直复用；仅在采样参数变化时新建，
旧的一律不释放（Context 很小，用 `CONTEXT_CACHE_MAX` 限制常驻数量）。

## 3. `GetAllTokenGeneration` / `...Len` **会引起堆破坏**

二分定位结果：

| 单独调用 | 结果 |
|---|---|
| `GetInputTokenCount` + `GetOutputTokenCount` | 两轮正常 |
| `GetAllGenerationLen` | 两轮正常 |
| `GetAllTokenGeneration` | 两轮正常 |
| **组合使用** | **core dump**（非确定性） |

既然对话/服务都不需要原始 token id，脚本里完全不用它们。
**要取 token id 的场合请改用别的路子。**

注意 `GetAllTokenGenerationLen` 返回的是**字节数**，
`GetAllTokenGeneration(ctx, buf, len)` 的 `len` 也是字节数：
`len = n * 4`，缓冲区要 `n * 4` 字节。

## 4. 逐字输出的回调签名

```
HMS_LLMEngineContext_SetOnOneTokenGenerateDoneFunc(ctx, func)
  → void (*)(const HIAI_LLMEngine_Context*)      存于 ctx + 144
HMS_LLMEngineContext_SetOnAllTokensGenerateDoneFunc(ctx, func)
  → 同类型                                        存于 ctx + 192
HMS_LLMEngineContext_SetOnGenerateAsyncFailed(ctx, func)
  → 同类型                                        存于 ctx + 240
```

类型信息来自反编译里的 `std::__h::__value_func<void ()(HIAI_LLMEngine_Context const*)>`。

* **只收一个 `ctx` 指针，没有 userdata** —— 回调里用同一个 ctx 调
  `GetAllGenerationLen` / `GetAllGeneration` 就能拿到"已生成的部分文本"
* 配 `context.json` 的 `generate_options.callback_freq = 1`，每 token 回调一次
* 回调运行在**引擎的工作线程**里；异常绝不能穿回 C 栈（会崩），
  实现里必须全部吞掉
* 必须持有回调对象的强引用，被 GC 后会崩溃

实测到达时间线（`List three colors…`）：

```
0.336s 'Blue'   0.514s ','   0.665s ' Green'   0.816s ','   0.967s ' Red'
```

## 5. 引擎不保存对话历史

每轮必须把完整 transcript 重新发过去。但引擎会**复用公共前缀的 KV 缓存**：
实测多轮时第二轮 1.9s vs 第一轮 7.1s。因此多轮不要新开 Context。

## 6. 两层库的 `CreateFrom*Json` 语义不同

| 层 | 参数 |
|---|---|
| HIAI（`libhiai_llm_engine.so`） | **JSON 内容字符串**（传路径会抛 `nlohmann::json parse_error`） |
| NDK（`libcann_llm_engine.so`） | **文件路径**（内部自己 fopen） |

NDK 层用路径更方便，本项目用的是 NDK 层。

## 7. 输入张量规格（Qwen2.5-1.5B 的实测值）

```
input_embed      int8  [1, 64, 1536]        ← int8 嵌入行
attention_mask   fp32  [1, 1, 64, 2048]     ← 加性掩码：0.0 有效 / -3.4e38 屏蔽
position_ids     int32 [1, 64]
new_kv_cache_pos int32 [64]
embed_scales     fp32  [1, 64, 1]           ← 反量化 scale，图中第一个 Mul 用它
past_key_in{i}   fp32  [2048, 2, 1, 128]
past_value_in{i} fp32  [2048, 2, 1, 128]
输出 lm_logits    fp32  [1, 64, 151936]
```

> 其中 `past_key_in{i}` / `past_value_in{i}` 的**第 0 维就是 KV 缓存上限**。
> 上面这两个 2048 属于 **Qwen2.5-1.5B** 这个模型 —— 它由转换/量化时的
> `kv_cache_max_len` 决定并固化进形状，**不是引擎常量**（详见第 9 节）。

引擎读 logits 的位置是 `lmLogits + (vaildLenLast_ - 1) * vocabAlignSize_`
（lldb 实测：`vaildLenLast_ = 5` 时指针 = 基址 + `4 × 151936 × 4` 字节 ✓）。

## 8. 模型产物：务必检查量化后的权重符号

> 这是本项目踩过的最大的坑，值得单独记住。**根因已查明：配置项写错，不是工具 bug。**

### 8.1 症状

用 `dopt` 做 W4 量化（`quant_strategy: "Quant_act_weight_eco"`）导出的
`fake_quant_weight.pth`，**权重张量里 99% 完全没有负值** ——
量化的负半轴被钳成了 0，权重的一半信息被抹掉。

症状极具误导性：

* 引擎返回 0，NPU kernel 确实在跑，改权重文件会影响输出 → 看起来"模型在工作"
* 但输出是恒定垃圾，与 prompt 内容/长度**无关**
* CPU（llama.cpp + 正确量化的 GGUF）却完全正常

### 8.2 根因：`quant_param_2` 写错了

`config.yaml` 里的 `quant_param_2` 必须**按目标平台**设置：

| 平台 | 取值 |
|---|---|
| kirinx90 | `False` |
| kirin9020 | `True` |

本项目早期把它写成了 `True`（kirin9020 的值），于是量化时负半轴被钳掉。

**2×2 实测**（同一台机器、同一份 HF 检查点、同一个 `dopt_config.json`）：

| DDK 版本 | `quant_param_2` | 权重负值占比 | 全非负的权重张量 |
|---|---|---|---|
| 5.1.1.1 | True | 13.17% | 196 / 198 (99%) ✗ |
| 5.1.1.1 | False | 43.98% | 0 / 198 (0%) ✓ |
| 6.1.1.0 | True | 13.17% | 196 / 198 (99%) ✗ |
| 6.1.1.0 | False | 43.98% | 0 / 198 (0%) ✓ |

两两数字逐位相同 —— **决定因素是那个配置项，与 DDK 版本无关**。
（未量化的 HF 原权重负值占比 43.24%，所以 43.98% 才是正常值。）

### 8.3 判定方法

最简单：加载 `fake_quant_weight.pth`，统计 `*.weight` 张量的负值占比。
正常应 ≈ **43%~44%**、全非负张量 **0 个**；被钳位则是 ≈13%、几乎全部无负值。

要追到 ONNX 层（逐个权重比对，注意 MatMul 存 `[in, out]`、HF 存 `[out, in]`）：

```
corr(导出权重, HF 权重)        ≈ 0.79 ~ 0.84      ← 被钳位时
corr(导出权重, ReLU(HF 权重))  ≈ 0.98             ← 这个特征最直接
ONNX 中负值占比 0.00%  vs  HF 中 50%

（正确量化后 corr(导出权重, HF 权重) 应该就有 0.98 ~ 0.99）
```

### 8.4 当初的排查路径（可复用于"模型输出垃圾"这类问题）

1. **ONNXRuntime 跑导出图** —— 需要先给 `ScatterND` 的 indices 插
   `Cast(int32→int64)`（ORT 要求 int64，CANN 导出的是 int32），
   并让节点名唯一，否则 ORT 拒绝加载。
   若 ORT 与 NPU 给出**同样的**垃圾 token，说明是图本身错，不是引擎。
2. **用 HF 检查点重建权重**（`onnx::MatMul_xxxx` 名字丢了，
   但**节点名**保留了，可按节点名映射回 HF 参数名），再跑 ORT。
3. ORT 给出正确结果（`12095` = `' Paris'`）后，再走 OMG 转换并部署。

> **修法**：把 `quant_param_2` 改成 `False` **重跑量化**即可，不需要绕行。
> 历史上用过的两条绕行（重建权重、切分 down_proj）保留在
> [`scripts/model-conversion/`](../scripts/model-conversion/) 与
> [model-conversion 附录 B](model-conversion.md)。

另外两条与模型产物相关的记录：

* OMG **不支持** 官方导出的 `model.layers.N.mlp.down_proj` MatMul
  （K=8960 过大），必须按 K 切块相加；但切块的节点改名后
  `--compress_conf` 的量化参数就**对不上**了（报
  `Node:…down_proj has quant params, but not in the graph`），
  所以只能改用 `--weight_data_type FP16` 放弃量化。
* 切 `down_proj` 时脚本必须**就地插入**节点：早期版本用 `g.node.append()`
  把子图追加到图末尾（layer 0 的节点排到 2005/2117），破坏拓扑序；
  另外当 K ≤ chunk 只切出 1 块时，原输出名永远不会被产生（悬空引用）。
  OMG 内部会重排节点，所以顺序本身不影响结果，但悬空引用会让 IR 生成失败。
* OMG 的 `LD_LIBRARY_PATH` 要包含 `tools_omg/master/lib64`，
  否则 `RmsNorm` 算子拿不到 infershape 函数。

## 9. 上下文上限：引擎的行为，不是错误

KV 缓存的形状是 `past_key_in{i} = [<上限>, 2, 1, 128]`，**第 0 维就是 token 容量**
（输入 + 输出之和）。

**这个上限取决于模型，不是引擎常量**：它由转换/量化时的 `kv_cache_max_len` 决定，
并固化进张量形状，所以改上限 = 重新转换模型。本节的实测用的是 **2048** 的那个
示例模型；官方 Qwen2.5-Coder-7B OMC 包则是 **4096**。实测三种表现：

| 输入规模（引擎报告的 in_tokens） | 引擎行为 |
|---|---|
| ≤ ~1708 | 输出正常 |
| ~2086 – 3360 | 引擎返回 0，输出退化成 `- - - - - - -` 这类重复文本 |
| ~6800（31KB 英文） | 引擎返回码 1 |
| `prompt=""` | 引擎返回码 1 |

**这不是缺陷，是模型的真实行为。** 输入超出训练/缓存范围时，模型本来就
会产出退化文本；引擎把它的输出如实交出来，这是正确的。

因此本框架**不做任何长度限制、不做历史裁剪、不把退化输出当错误**：

* 我们是推理框架，职责是把模型的行为如实传给调用方，而不是替调用方
  判断输出"该不该"。模型产出了什么，就交给调用方什么。
* 也不设"启发式上限"：引擎没有只分词不生成的接口，按字节估算必然估歪
  （实测「字节 // 2」对英文偏高约 2.3 倍）。给一个会误伤的上限，比不给更糟。
* 长度控制是调用方的决定：CLI 用 `/reset`，HTTP 由客户端管理 `messages`。

引擎返回非零时，我们**如实报告返回码、不替它断言原因** —— 从返回码区分不出
是输入超长、含无法分词的字符还是引擎内部错误。

需要 token 数时用 `GenerationStats.prompt_tokens`（来自引擎的
`GetInputTokenCount`，准确）；`count_prompt_tokens()` 只是给日志看的粗估
（系数按实测标定为「字节 // 4」），**不作为任何限制依据**。

## 9.1 引擎不告诉你「因何而停」

`HMS_LLMEngineContext_*` 里**没有**任何 stop/finish reason 的接口。逐个试过：

```
GetStopReason  GetFinishReason  GetEndReason  GetGenerateState
GetIsFinished  IsFinished       GetStatus     GetLastError  GetErrorCode
```

全部不存在。而且引擎会把命中的 `stop_sequence` 从输出里**剥掉** —— 实测
`max_tokens=64` 让模型自然说完时，`GetAllGeneration` 拿到的是 `'谢谢！'`，
不含 `<|im_end|>`；被截断时同样不含。所以没法从文本反推。

唯一可用的证据是 `GetOutputTokenCount()`。由此推断：

| 条件 | 判断 | 确定性 |
|---|---|---|
| `out_tokens < max_tokens` | `stop` | **确定**（引擎只会在「够到上限」或「命中停止符」时停） |
| `out_tokens == max_tokens` | `length` | 不确定 —— 也可能是模型恰好在第 N 个 token 自然结束 |

边界歧义无法消除。选 `length` 是因为「该继续却被截断」远比「刚好说完」常见，
且把截断误报成 `stop` 会让调用方把半句话当成完整回答（后果更严重）。

实测到的歧义例子：`max_tokens=3` 时模型恰好说 3 个 token「谢谢！」，
会被判成 `length` 返回。

## 10. 其它环境事实

* `hilog` 抓 CANN 域日志经常拿到 0 行，需要时用终端面板而不是管道
* 设备上 `/tmp` 只读，临时文件写工作区
* `embedding_input_type: "int8"` 时，embedding 文件是
  `[vocab, hidden]` 的 int8 行 + 每 token 一个 fp32 scale
