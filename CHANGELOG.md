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

### 说明

- **核心零第三方依赖**：目标环境（鸿蒙设备）上没有 FastAPI/uvicorn/pytest，
  HTTP 层用标准库实现。可选依赖组 `fastapi` / `dev` 已预留。
- 一次只跑一路推理（引擎限制），并发请求排队，超出上限返回 503。

[Unreleased]: https://example.invalid/cann-llm/compare/v0.0.0...HEAD
