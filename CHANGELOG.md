# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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
