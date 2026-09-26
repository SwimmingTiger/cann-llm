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
