# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed — 用第三方 Python（如 harmonybrew）启动会 segfault

现象：`./scripts/start_chat.sh -d …` 在 brew 的 Python 下**直接 core dump**，
换成 `export PATH=/data/service/hnp/bin:$PATH` 就正常。

**根因（已查明）**：本机系统 libc 是 **musl**，`libcann_llm_engine.so` 也是按 musl
编的；而 harmonybrew 提供的 Python 是 **glibc 构建**，靠 `libmusl_compat.so` 垫片在
musl 系统上跑。把 musl 版的引擎加载进这种进程，调
`HMS_LLMEngineExecutor_CreateFromExecutorJson` 时**直接段错误** ——
在 Python 层**没有任何可捕获的异常**。

实测对照（`faulthandler` 定位到 `backends/cann.py` 的 `executor_create` 那一行）：

| 解释器 | `/proc/self/maps` 里有 libmusl_compat | 结果 |
|---|---|---|
| harmonybrew python 3.14.7 | 有 | **segfault，退出码 139** |
| `/data/service/hnp/bin/python3` 3.12.8 | 无 | 正常 |

**修法**：

- `backends/cann.py` 新增 `interpreter_libc_conflict()`：判据是进程的
  `/proc/self/maps` 里是否出现 `libmusl_compat`（实测可靠；非 Linux 读不到时不拦）。
  `load()` 在碰 NDK 之前先检查，冲突就抛 `BackendUnavailableError` 并给出
  可操作的建议（含 `PYTHON=/data/service/hnp/bin/python3 …` 的示例），
  **把 core dump 变成一句人话**。
- `start_chat.sh` / `start_server.sh` 新增 `resolve_python()`：没显式给 `PYTHON`
  时，在候选里挑第一个与引擎 libc 兼容的（`python3` -> `/data/service/hnp/bin/python3`），
  于是**不用再手工 export PATH**；显式指定的仍优先，不兼容时由后端报清楚。

验证：
- 默认 PATH（python3 = brew 3.14）下 `start_chat.sh` 自动改用 hnp，正常进入对话
- `PYTHON=<brew python>` 显式指定时给出清晰报错，退出码 1（原来 139）
- `make test` 184 个全过（新增 3 个用例）

### Added — 上下文长度/输出窗口现在都可配（各归其位）

用户问"上下文只有 2K 吗，能提升吗"。结论：**能，但性质不同**——

| | 上下文长度（KV 缓存） | 输出窗口（max_tokens） |
|---|---|---|
| 性质 | **编译期**属性，写进 .omc 的静态形状 | 运行时参数 |
| 配置方式 | 转换时用 `--kv-len` 指定，需重跑导出+OMG | CLI / 配置 / API 请求体 |

**上下文长度**：新增 `scripts/model-conversion/build_model.py`，把整条链路收敛成
一个 `--kv-len` 参数。它从 HF config 自动推断架构/层数/hidden/KV 头数，生成导出
yaml，跑导出，跑 OMG（`omg_convert.py --kv-len`），再装配模型目录 —— 保证
**三处 KV 长度始终一致**（yaml / OMG input_shape / executor.json），这是之前手工
改三处最容易漏的地方。支持 `--dry-run` / `--only-yaml` / `--skip-export` /
`--skip-omg` 分步执行，并会在 dry-run 里报出内存开销与每 token 的 KV 大小。

内存不是瓶颈（4B 每 token 144 KB：8K=1.1 GB、32K=4.5 GB，设备有 14~18 GB 可用）；
真正的代价是**速度** —— 图里 KV 是静态形状，每个 token 都要读完整个 KV 张量，
与实际用了多少上下文无关，所以解码时间随 KV 长度线性上升。

**输出窗口**：本来就可配（配置文件 `max_tokens` / CLI `--maxtok` / 交互 `/maxtok` /
API 请求体的 `max_tokens`），这次给 **server 补上了 `--max-tokens`** 命令行开关
（chat 早就有，server 一直缺）。

**顺带修正**：`CannNdkBackend.context_length` 原本写死默认 2048，与实际模型无关，
会让 `/v1/models` 的元信息和"输入超出 KV 缓存（本模型 2048 token）"这类提示误导人。
改为**默认从模型目录的 `executor.json` 读 `kv_cache_max_len`**（显式传参仍优先），
读不到才回退 `DEFAULT_CONTEXT_LEN`。

新增 `tests/test_config.py` 里 3 个用例覆盖自动读取（正常/缺失或损坏/显式优先）。
测试 178 -> 181，全过。

### Fixed — 服务启动信息重复打印，且在绑定成功前就宣称"监听"

前台启动时，端点信息打印了两遍：`start_server.sh` 在 `exec` 服务之前先
`print_endpoints` 打了一遍（此刻**还没 bind**），服务起来后 `serve()` 又打了
自己的横幅。前者既造成重复，也违反"监听地址应当在绑定成功后再报"。

- 前台分支不再自己打印端点：交给 `serve()` 那一次（它本就是先
  `LlmHttpServer(...)` 绑定、再打印），只保留一句"正在加载模型…"
- 后台分支保持不变：探活成功后才打印（本来就正确）
- 顺带给绑定失败加了友好提示（原先直接抛 traceback）：
  `EADDRINUSE` / `EACCES` / `EADDRNOTAVAIL` 分别给出可操作的建议，
  并明确返回 1、**不**出现任何"监听"字样

验证：
- 前台：横幅只出现一次，且在加载+绑定之后
- 端口被占用（脚本预检 & 绕过预检的竞态两条路径）：0 次"监听"字样
- 后台 -B：就绪后才打印；健康检查返回 `model=qwen3_4b`

### Fixed — 换模型后名字仍是 `qwen2.5-1.5b`

`ModelConfig.model_id` 默认值被写死成字符串 `"qwen2.5-1.5b"`，而 `-d/--model-dir`
只覆盖 `model_dir`、从不设置 `model_id`（chat CLI 甚至没有这个参数）。
于是不管加载哪个模型，横幅、`/v1/models`、响应里的 `model` 字段一律显示
`qwen2.5-1.5b`。

改为**默认取模型目录名**：

- `ModelConfig.model_id` 默认值改为 `None`（表示"未显式指定"）
- 新增 `ModelConfig.resolved_id` 属性：有显式 `model_id` 就用它，否则取
  `model_dir` 的目录名（`/srv/models/qwen3_4b` -> `qwen3_4b`；目录也没有则 `cann-llm`）
- 全部用户可见处（chat 横幅、`/v1/models`、`chat/completions` 与 `completions`
  响应、导出命令）改用 `resolved_id`（共 14 处）
- chat CLI 补上 `--model-id`（server 早就有），需要固定名字时可显式指定

实测：`models/qwen611` -> `qwen611`、`models/qwen3_4b` -> `qwen3_4b`、
`models/rebuilt` -> `rebuilt`；`-d` 换目录后跟着变；显式指定优先。

新增 `tests/test_config.py`（8 个用例，含"默认值不应含 1.5b 字样"的回归断言）。
测试 170 -> 178，全过。

### Added

**核心**
- `types.py`：`GenerationRequest` / `GenerationChunk` / `GenerationResult` /
  `GenerationStats` / `ModelInfo` 等数据契约，以及 `aggregate()` 聚合
- `errors.py`：统一异常层次，携带到 HTTP 状态码与 OpenAI `error.type` 的映射
- `config.py`：TOML + 环境变量配置加载（`CANN_LLM_MODEL__XXX`），零第三方依赖

**后端**
- `backends/base.py`：`EngineBackend` 协议 + 注册表 + `SerializedBackend`
  （把只支持单路推理的引擎串行化，带排队上限与超时）
- `backends/cann.py`：华为 CANN LLM Engine NDK 后端（ctypes），
  支持逐 token 流式输出

**对话**
- `chat/template.py`：可插拔对话模板（`chatml` / `plain`）+ 注册表
- `chat/session.py`：多轮会话、历史裁剪、用量统计

**接口**
- `cli/chat.py`：交互式命令行，逐字流式输出，斜杠命令
- `api/openai.py`：OpenAI 协议映射（请求解析 / 响应构造 / 错误映射 / SSE 帧）
- `api/server.py`：OpenAI 兼容 HTTP 服务（stdlib `http.server`）
  - `GET /healthz`、`GET /v1/models`
  - `POST /v1/chat/completions`（流式 SSE 与非流式）
  - `POST /v1/completions`（流式与非流式）
  - `stream_options.include_usage`、鉴权、CORS、并发上限（503）

**工程**
- 72 个单元测试（stdlib `unittest`，配假后端，不依赖 NPU）
- `scripts/stream_check.py`：流式输出自检
- `Makefile`、`examples/config.example.toml`
- `docs/architecture.md`、`docs/cann-engine-notes.md`、`docs/openai-api.md`

### Changed — 移除工具结果截断

`max_result_chars`（默认 4000）会把工具返回值截断后再交给模型和调用方。
问题在于截断发生在结果返回给调用方**之前**，所以它同时做了两件事：构造 prompt
（可讨论）与**销毁调用方的数据**（没有理由）。

实测：工具返回 12000 字符时，模型与调用方都只拿到 4018 字符，完整数据不可取回。

现在完全不截断：工具返回多少，模型就看到多少，调用方也拿到多少。
需要控制长度请在工具实现里自己分块/摘要/返回引用。

### Changed — 保留每轮工具声明（KV 缓存）

用户指出：末轮撤掉工具声明会不会让缓存无法命中？实测确实会，而且代价很大。

工具声明注入在第一个 system 轮次内部，即 prompt **最开头**；撤掉它公共前缀
只剩 87 字符，之后全部重新 prefill。同一对话只改这一处的实测：

| | in_tokens | prefill | 每 token |
|---|---|---|---|
| 带声明 | 226 | 327 ms | 1.44 ms |
| 撤掉声明 | 94 | 693 ms | 7.37 ms |

token 少一半多，prefill 反而慢一倍以上（每 token 慢 5.1 倍）。

现在**每一轮都传完全相同的工具声明**；`max_steps` 只决定何时停止，不再改
模型看到的东西。用满步数时若模型仍停在"想调工具"的状态，如实交给调用方
（`Final.tool_calls`），CLI 会打印一行说明而不是假装有答案。

参照：DSH 全包搜不到 `maxSteps` / `stepLimit` / `maxIterations` 之类的概念。

### Changed — finish_reason 不再被改写

`AgentLoop` 里曾把「最后一轮被截断」改写成「模型想调工具」：

```python
if finish_reason == "length" and pending_calls:
    finish_reason = "tool_calls"
```

两个问题：`pending_calls` 累积的是整个运行过程的调用，而 `finish_reason`
描述的是最后一轮 —— 两件不同的事被混在一起；后果是**被截断的半截回答**
被报成 `tool_calls`，按 pi-ai 的映射（`tool_calls` → `stopReason: "toolUse"`）
等于告诉调用方「模型还想执行工具」。

现在如实透传最后一轮的引擎推断值；本次运行用过哪些工具看 `Final.tool_calls`。

另：`max_steps` 用尽时最后一轮不传工具声明这一行为**保留**（它是 agent 循环
自身的控制流，不是对模型输出的解释），但已在文档中写明 —— 用满步数时模型
看到的最后一轮 prompt 与前面不同。

### Changed — 工具失败只陈述事实

工具失败时不再追加「请修正参数后重新调用同一个工具；在拿到成功结果之前不要
凭猜测作答」这类指导语，只回**错误原文**（`Error: <原因>`）。

判定失败本身是 agent 循环的职责（DSH 的 `dsh-agent-loop` 同样设
`isError: true`），问题在于**判完之后多说了话** —— 那属于替调用方指挥模型。

查证 pi-ai 的 wire 转换（`dist/api/openai-completions.js`）：tool 消息只发
`content` / `tool_call_id`，**`isError` 不发给模型**（只是框架内部元数据）。
所以 content 里的文字是模型唯一能看到的失败信号，`Error:` 前缀是必须的，
而任何指导语都不是。

同时删除了从未被使用的死配置 `announce_tool_calls`。

### Changed — 彻底移除「裸 JSON 兜底」

`parse_tool_calls` 原先默认把**不带 `<tool_call>` 标签**的裸 JSON 也当成工具
调用。实测它会扭曲输出并且**抹掉正文**：

```
输入:  The tool takes {"name": "search", "arguments": {"q": "x"}} as input.
原先:  判成调用 + 从正文删掉该 JSON → 用户看到 "The tool takes  as input."
现在:  不判为调用，正文原样保留
```

最要命的场景是用户**主动要 JSON** 时：用户说「给我一个 JSON-RPC 请求体的
例子」，模型给出 ` ```json {"name": "getUserProfile", "arguments": {...}} ``` `，
原先会被判成调用并从正文删除 —— 用户拿到的是一个**被掏空的代码块**。

根因是它做了两件越权的事：在模型**没有表达调用意图**时替它认定意图，并把
模型写下的文本从输出里抹掉。`<tool_call>` 标签的存在本身就是"这是一个调用"
的协议信号，也是唯一可靠的判据；模型忘记标签属于它自身的格式偏离，推理框架
应当如实呈现，想让它稳定用标签该改的是 prompt 而不是解析器。

**彻底移除**（先改为默认关闭，随后连开关一并删掉），`extract_json_objects`
也一并删除（只被它使用）。

### Fixed — Qwen3-4B 端到端跑通：FP16 导出会被 RoPE 融合 pass 拒绝

上面那条「端到端尚未成功」已解决。根因与修法如下。

**根因**：`export_model_single_qwen3.py` 硬编码 FP32 导出，但在 31 GB 内存的机器上
FP32 装不下 4B（确定性 OOM）。当时把精度改成 FP16 绕开 —— 导出能过、OMG 也能出
`.omc`，但 OMG 日志里出现 **36 层 × 4 = 144 条**
`rope_llm_fusion_pass.cc CheckMul0: mul0 weight size invalid 0 != 1`：
FP16 引入的额外 Cast 破坏了 RoPE 的模式匹配，融合失败后图里留下 kirinx90 执行不了的
RoPE，于是引擎**能加载模型但 `Generate` 恒返回 1**。

**修法**：换一台大内存的机器跑 FP32 导出，而不是降精度。实测在 62 GB 的机器上：
  - FP32 导出通过（437 秒，`.pb` 16.36 GB）
  - OMG 的 RoPE 融合错误 **0 条**（对比 FP16 当时是 144 条）
  - 模型在 NPU 上**正常出词**

**Qwen3-4B 实测结果**（设备侧）：
  `The capital of France is` -> `Paris.`
  `What is the capital of Japan?` -> `The capital of Japan is Tokyo.`
  `1+1=` -> `1 + 1 = 2.`
  80 token 长文本通顺；真流式逐 token；2.5~4.1 tok/s
  （同设备 Qwen2.5-1.5B 为 13.1 tok/s）

顺带踩到并记录的第 6 个坑：`tools_omg/master/omg` 的 ELF 解释器被指到
`/tmp/ld-linux-x86-64-2.35.so.2`，换到干净机器做 OMG 时忘了建这个链接会以
`FileNotFoundError: .../master/omg` 的形式失败（文件其实在），极易误判。

**跨机协作分工**（本次实际用的）：
  - 量化要 8 GB 显存 -> RTX 3080 Ti（12 GB）那台
  - FP32 导出要 >31 GB 内存 -> 62 GB 那台
  - OMG 纯 CPU -> 随便哪台
  两台的目录保持同一个绝对路径，脚本里的绝对路径就不用改；
  唯一要补的是 venv 的 `bin/python` 符号链接和上面那个 ld.so 链接。

文档：`docs/model-conversion.md` 附录 D 的状态说明改为「已跑通」，坑 4 补上
FP32/FP16 的实测对照表，新增坑 6。

### Added — Qwen3 系列的转换坑（实测）+ 两个新工具

尝试把链路从 Qwen2.5-1.5B 扩展到 **Qwen3-4B-Instruct-2507**，一路踩到 5 个官方示例代码
层面的问题，逐个定位并给出了修法与工具。**端到端尚未成功**（见下面第 4 条），
但过程与结论都已写进 `docs/model-conversion.md` 附录 D。

新工具（都在 `scripts/model-conversion/`，并在真实数据上验证过判断）：

- **`normalize_tokenizer_merges.py`** ★ 最要命的一个
  新版 HF tokenizer.json 把 BPE merges 存成「数组的数组」，旧版是空格分隔字符串。
  引擎解析新版会抛 `nlohmann::json type_error.302: type must be string, but is array`
  然后 **core dumped**。本脚本转成旧格式并去掉 `model.ignore_merges`。
  实测 151387 条 merges 秒级转换，转换后模型从「加载即崩」变为「能正常加载」。
- **`set_quant_strategy.py`**
  dopt 首次运行只生成一份「全是 float（不量化）」的配置就退出，必须人工填策略。
  本脚本按规则自动填（Linear 量化、lm_head/embedding 保持 float）。
  回归验证：在 7B 的生成配置上运行，产出与实测跑通的 1.5B 配置**逐字段一致 198/198**。
- **`patch_qwen3_embedding.py`**
  qwen3 导出脚本把独立 embedding 导出的整段代码注释掉了，导致 ONNX 有 `input_embed`
  输入却没有 embedding 文件。本脚本按 qwen2 的写法恢复。
- **`patch_qwen3_export_mem.py`**
  qwen3 导出脚本把 `from dopt.do_opt import ...` 写成了不存在的路径（两个 DDK 版本
  都没有顶层 `dopt.do_opt`，真实定义在示例自带的 `do_opt.py` 里）——本脚本改为
  `from do_opt import`；同时让 `onnxsim.simplify` 可由 `CANN_SKIP_ONNX_SIMPLIFY=1` 跳过。

关键结论（写进附录 D）：

1. qwen3 导出脚本的 dopt import 路径是错的（qwen2 脚本用的是本地 `do_opt`，所以
   1.5B 那条路没碰上）；
2. 独立 embedding 导出被整段注释掉；
3. tokenizer merges 新旧格式不兼容会导致引擎 core dump；
4. **导出精度必须 FP32**：FP32 在 31 GB 内存的机器上装不下 4B（确定性 OOM，
   实测可用内存掉到 33 MB）；改成 FP16 虽然能导出、能出 OMC，但会让 OMG 的
   RoPE 融合 pass 匹配失败（`rope_llm_fusion_pass.cc CheckMul0: mul0 weight size
   invalid 0 != 1`，36 层 × 4 = 144 条），结果是引擎能加载模型但 `Generate`
   恒返回 1。**这一步需要 ≥ 64 GB 内存的转换机**；
5. Qwen3 的 `q_norm` / `k_norm` 是被 OMG 正常支持的，不是坑。

### Fixed — 模型转换：量化钳位的真正原因是配置项，不是工具版本

用 DDK 6.1.1.0 完整重跑了转换链路，并把之前那个「权重负半轴被钳成 0」的问题
**归因查清了**。

**先前结论是错的**：一度以为 6.1.1.0 修了这个 bug。做 2×2 隔离实验后确认
根因是 `config.yaml` 里的 **`quant_param_2` 取值**，与 DDK 版本无关：

| DDK 版本 | `quant_param_2` | 权重负值占比 | 全非负的权重张量 |
|---|---|---|---|
| 5.1.1.1 | True | 13.17% | 196 / 198 (99%) ✗ |
| 5.1.1.1 | False | 43.98% | 0 / 198 (0%) ✓ |
| 6.1.1.0 | True | 13.17% | 196 / 198 (99%) ✗ |
| 6.1.1.0 | False | 43.98% | 0 / 198 (0%) ✓ |

两两数字逐位相同。正确取值：**kirinx90 → False / kirin9020 → True**
（官方文档就是这么写的）。未量化的 HF 原权重负值占比 43.24%，所以 43.98%
才是正常值。

**修正后的完整链路**（6.1.1.0，全程官方工具，无需任何自定义脚本改图）：

    dopt 三阶段量化 → 导出 ONNX → OMG 转换 → 装配模型目录

实测结果：`The capital of France is` → `Paris`；`1+1=` → `2`；
80 token 长文本通顺连贯；流式逐 token 到达；6.9~13.7 tok/s。

文档改动：
- `docs/model-conversion.md` **整体重写** —— 主线是 6.1.1.0 的官方标准流程；
  旧版的问题与两条绕行手段（重建权重 / 切分 down_proj）降级为附录 B，
  并写明真正原因是配置项
- `docs/cann-engine-notes.md` 第 8 节重写：补上 2×2 实测表与正确取值，
  原「诊断路径」保留为 8.4（对"模型输出垃圾"这类问题仍然可复用）
- `README.md` 顶部入口的描述改为「全程官方标准流程」，并点明那个易写错的配置项
- 新增 `scripts/model-conversion/check_quant_clamp.py`：一条命令判断量化产物
  有没有被钳位（已在已知好/坏两份 3.2 GB 产物上验证判断正确）

### Docs — 两处「启发式」的依据已查实

用户要求核实：`_finish_reason` 的推断是什么、`StreamFilter` 删掉 `<tool_call>`
是否合理。结论如下（已写进代码注释与文档）：

- **`_finish_reason` 的依据**：引擎 `HMS_LLMEngineContext_*` **没有**任何
  stop/finish reason 接口（逐个试过 GetStopReason / GetFinishReason /
  GetEndReason / GetGenerateState / IsFinished 等，均不存在），且会把命中的
  `stop_sequence` 从输出里剥掉（实测原始文本不含 `<|im_end|>`）。
  唯一可用证据是 `GetOutputTokenCount()`：
  `out_tokens < max_tokens` → 确定是 stop；
  `out_tokens == max_tokens` → 说不准（可能是恰好在第 N 个 token 自然结束），
  判为 length。边界歧义无法消除，已如实记录（实测 max_tokens=3 时模型恰好
  说完「谢谢！」会被判成 length）。
- **`StreamFilter` 是协议重编码，不是丢信息**：查了 DSH 的源码 —— 它用的
  pi-ai（`dist/api/openai-completions.js`）在流式处理里**只认
  `choice.delta.tool_calls`**，整个仓库没有对 `<tool_call>` 文本标记的解析。
  不过滤的话调用方只会把那段当成普通文本，agent 循环直接断掉。
  且只在客户端声明了 tools 时才过滤，不带 tools 的请求原样透传 ——
  想要原始 Qwen 格式的调用方依然拿得到。

### Changed — 不扭曲模型行为

用户指出的原则：**推理框架的职责是如实传递模型的行为，不替调用方判断输出
"该不该"**。据此审计并修正了我自己的四处扭曲：

- **采样参数不再做「哨兵值」替换**。原先 `_merge_params` 把「值恰好等于
  dataclass 默认」当成「未指定」，于是后端默认 temperature=0.1 时，调用方
  显式要求 0.7（恰好等于默认值）会拿到 0.1。服务端的
  `req.params != GenerationParams()` 是同一类 bug。现在默认值只在解析阶段
  显式兜底，进入后端后参数原样使用。
- **不再截断模型的工具调用**。原先 `max_calls_per_step`（默认 4）会丢弃
  第 5 个之后的调用 —— 那是在丢模型输出。现在发几个执行几个。
- **`force_tool_use` 已删除**。它会往 prompt 里塞调用方没写的指令，属于改变
  模型行为。查证了 llama.cpp 的做法（`common/chat-auto-parser-generator.cpp`）：
  工具格式说明来自**模型自带的 chat template**，框架只负责应用；「强制调用」
  用 **grammar 做 token 级约束**，且只在调用方要求 `tool_choice: required` 时
  启用（`auto` 下是 lazy grammar，从不强迫模型）。llama.cpp 里搜不到任何框架
  自撰的「你必须调用工具」文字。想强化工具使用请写在调用方自己的 system prompt。
  同时删除了从未被使用的 `requires_tool_call` / `forbids_tool_call` 两个属性。
- **引擎非零返回不再断言原因**。原先会猜「输入超出上下文」并返回 400
  `context_length_exceeded`；我们其实区分不出是超长、含无法分词的字符还是
  引擎内部错误。现在如实报告返回码 + 列出可能性，状态码回到 500。
  相应地删除了已成死代码的 `ContextLengthExceededError`。

文档同步去掉了把这个现象称作「危险区 / 坑」的说法 —— 模型在输入超范围时
产出退化文本是它的真实行为，框架如实传出去就是正确的。

### Changed — 移除上下文长度限制

- **后端不再对 prompt 长度设限**，也不再做历史裁剪。原先的 `max_prompt_tokens`
  （默认 1800）会误拦请求：它按「字节 // 2」估算，实测对英文偏高约 2.3 倍
  （实际 1316 token 估成 3011）。既然没有 tokenize 接口、无法可靠预判，
  就不该给一个会误伤的上限 —— 改为让调用方观察模型输出自行控制。
- `ChatSession` 与 `AgentLoop` 的静默历史裁剪一并移除：丢历史会让模型在
  调用方不知情的情况下换掉上下文，比报错更难排查
- `count_prompt_tokens()` 系数按实测标定为「字节 // 4」，且明确它**仅供
  日志展示，不作为任何限制依据**；准确值用 `usage.prompt_tokens`
- 引擎非零返回若输入确实超长，改判为 400 `context_length_exceeded`
  （原来是 500，会让调用方误以为该重试）
- 文档补充引擎的分层行为：≤2048 正常 / ~2086–3360 **成功但输出垃圾** /
  很大则报错（见 `docs/cann-engine-notes.md` 第 9 节）

### Added — Agent（工具调用）

- **prompt-based function calling**：模型用 Qwen 官方格式
  `<tool_call>{...}</tool_call>` 表达调用，服务端解析、执行、回填
  `<tool_response>`，循环至给出最终回答
- `tools/`：零依赖的 JSON Schema 子集校验器、工具注册表（`dangerous` 门禁）、
  内置工具（`get_current_time` / `calculator` / `http_get` 带 SSRF 防护）
- `agent/parser.py`：容错的 tool_call 解析（12 种畸形输出）
- `agent/loop.py`：agent 循环 + `StreamFilter`（流式时隐藏工具协议标记）
- HTTP 层按 **OpenAI 标准**返回 `tool_calls` 由客户端执行；流式时过滤掉
  `<tool_call>` 协议标记并按标准发 `delta.tool_calls`
- CLI：`--tools all|<名字>`、`--list-tools`、`/tools` 命令、调用过程可视化
  （CLI 没有"客户端"可代劳，所以直接用 `AgentLoop` 执行工具）
- 新增 `[agent]` 配置段（`max_steps` / `max_calls_per_step` /
  `max_result_chars` / `force_tool_use`），只作用于 CLI
- 文档 `docs/agent.md`

### Changed

- **不支持的字段改为默认「接受但忽略」**（原为一律 400）。
  绝大多数现代客户端即使普通聊天也会带 `tools`，一律报错会让它们完全不可用。
  现在会忽略并在响应头 `X-Cann-Llm-Ignored-Fields` 回报、日志记一行；
  新增 `server.reject_unsupported = true` 可恢复严格模式。
  `logprobs` 与 `image_url` 仍然始终拒绝 —— 忽略它们会产出错误结果。
- 启动横幅改为显式打印 **base_url**（`http://host:port/v1`），
  端点单独列出，避免把端点路径误当成 base_url。
- 404/405 给出可操作提示（base_url 写法、允许的方法），不再只有一句"未知路径"。

### 说明

- **核心零第三方依赖**：目标环境（鸿蒙设备）上没有 FastAPI/uvicorn/pytest，
  HTTP 层用标准库实现。可选依赖组 `fastapi` / `dev` 已预留。
- 一次只跑一路推理（引擎限制），并发请求排队，超出上限返回 503。

[Unreleased]: https://example.invalid/cann-llm/compare/v0.0.0...HEAD
