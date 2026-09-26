# cann-llm

华为 **CANN LLM Engine**（鸿蒙 NPU）上的对话与 **OpenAI 兼容推理服务**。

直接以 `ctypes` 调用系统自带的 NDK 库 `/system/lib64/ndk/libcann_llm_engine.so`，
**不需要 HAP、不需要 root、核心零第三方依赖**。

## 特性

- 🗣 **交互式对话 CLI** —— 逐字流式输出、多轮上下文、采样参数热更新
- 🔌 **OpenAI 兼容 HTTP 服务** —— `/v1/chat/completions`（含 SSE 流式）、`/v1/completions`、`/v1/models`
- 🧩 **可插拔后端** —— 统一的 `EngineBackend` 协议，CANN NDK 是首个实现，
  将来可接 llama.cpp / vLLM / 远端 OpenAI 服务而不改上层
- 📦 **零第三方依赖核心** —— HTTP 层基于 `http.server`，设备上开箱即用
- 🧪 **可测** —— 分块聚合、对话模板、协议映射都有单元测试；`scripts/stream_check.py` 自检流式

## 快速开始

```bash
# 1) 准备模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json 配置）
#    参考 docs/cann-engine-notes.md

# 2) 对话（逐字流式）
PYTHONPATH=src python3 -m cann_llm.cli.chat -d /path/to/model_dir

# 3) OpenAI 兼容服务
PYTHONPATH=src python3 -m cann_llm.api.server -d /path/to/model_dir --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

## 项目结构

```
src/cann_llm/
├── types.py        请求 / 分块 / 结果 / 统计（后端与上层之间的唯一契约）
├── errors.py       统一异常
├── config.py       配置（TOML / 环境变量 / 命令行）
├── backends/       推理后端
│   ├── base.py     EngineBackend 协议 + 注册表
│   └── cann.py     CANN LLM Engine NDK（ctypes）实现
├── chat/           对话模板（ChatML 等）与多轮会话
├── cli/chat.py     交互式命令行
└── api/            OpenAI 兼容 HTTP 服务（stdlib http.server）
```

## 文档

| 文档 | 内容 |
|---|---|
| `docs/architecture.md` | 分层、数据流、为什么这样设计 |
| `docs/cann-engine-notes.md` | **踩坑记录**：API 调用约定、回调签名、必须避开的崩溃点 |
| `docs/openai-api.md` | 兼容范围与差异 |

## 许可

MIT
