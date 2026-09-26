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

## 依赖与构建

**纯 Python 项目，没有任何需要编译的部分。**

| 项 | 情况 |
|---|---|
| 本项目源码 | 全部是 Python，无需 C/C++ 工具链、无需 `pip install` 即可运行 |
| 第三方依赖 | **零**（HTTP 层用标准库 `http.server`，测试用 `unittest`） |
| 原生库 | 运行期由 `ctypes` 加载**系统自带**的 `/system/lib64/ndk/libcann_llm_engine.so`（鸿蒙 NDK 提供，不是本项目编译的） |
| 模型产物 | `.omc` + `SubGraph_0.weight` 由华为的 **OMG 离线转换工具**生成，属于离线步骤，不在本仓库内（见 `docs/cann-engine-notes.md`） |
| Python 版本 | ≥ 3.9（用到 `tomllib`） |

`import` 本包**不会**加载任何 `.so`；原生库只在 `backend.load()` 时加载，
所以在非鸿蒙机器上仍可正常 `import`、跑测试、做协议层开发。

想装成命令也行（可选，仍然没有编译）：

```bash
pip install -e .            # 提供 cann-llm-chat / cann-llm-server 两个命令
```

## 快速开始

```bash
# 1) 准备模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json 配置）
#    参考 docs/cann-engine-notes.md

# 2) 交互式对话（逐字流式）—— 一键脚本
scripts/start_chat.sh -d /path/to/model_dir
scripts/start_chat.sh -d /path/to/model_dir -p "你好"          # 单轮

# 2.5) 工具调用（agent）
scripts/start_chat.sh -d /path/to/model_dir --tools all      # 启用内置工具
python3 -m cann_llm.cli.chat --list-tools                    # 看有哪些工具

# 3) OpenAI 兼容推理服务 —— 一键脚本
scripts/start_server.sh -d /path/to/model_dir                 # 前台
scripts/start_server.sh -d /path/to/model_dir -b              # 后台，等就绪后返回
scripts/start_server.sh --status                              # 看状态
scripts/start_server.sh --stop                                # 停止

# 不带参数也行：会自动在 models/<名字>/ 下找模型目录
# 也可以用环境变量：export CANN_LLM_MODEL_DIR=/path/to/model_dir

# 客户端 base_url 只到 /v1：
#     OpenAI(base_url="http://127.0.0.1:8000/v1")     ✓
#     OpenAI(base_url="http://127.0.0.1:8000")        ✗ 少了 /v1
#     OpenAI(base_url="http://127.0.0.1:8000/v1/chat/completions")  ✗ 多了端点路径
# 填错时服务端会返回带提示的 404（见 docs/openai-api.md「排错」）。
# 浏览器打开 http://127.0.0.1:8000/ 可以直接看到全部端点。

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

不用脚本、直接跑模块也可以：

```bash
PYTHONPATH=src python3 -m cann_llm.cli.chat -d /path/to/model_dir
PYTHONPATH=src python3 -m cann_llm.api.server -d /path/to/model_dir --port 8000
make chat MODEL=/path/to/model_dir
make server MODEL=/path/to/model_dir PORT=8000
```

`start_server.sh` 启动前会做预检：Python 版本、NDK 库、模型目录完整性、
端口占用；后台模式还会轮询 `/healthz` 等到就绪才返回。

## 工具调用（agent）

本服务支持 **prompt-based function calling**：模型用 Qwen 官方格式
`<tool_call>{...}</tool_call>` 表达调用意图，服务端解析、执行、把结果回填，
再让模型作答。详见 **[docs/agent.md](docs/agent.md)**。

```bash
# CLI
scripts/start_chat.sh -d /path/to/model_dir --tools all
you> 帮我算一下 (23*7+11)/4
  → 调用 calculator({'expression': '(23*7+11)/4'})
    ✓ 0ms  {"expression": "(23*7+11)/4", "result": 43.0}
bot> 计算结果是 43.0。

# HTTP：按 OpenAI 标准，声明工具后返回 tool_calls 由客户端执行
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "2 的 10 次方是多少？"}],
  "tools": [{"type": "function", "function": {"name": "calculator",
             "description": "计算算术表达式",
             "parameters": {"type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"]}}}]}'
# → {"finish_reason": "tool_calls", "message": {"tool_calls": [...]}}
```

**HTTP 层只实现 OpenAI 标准语义**（服务端不执行工具）—— 你的应用能访问的
东西（数据库、内部 API）和本服务注册的工具不是一回事，把执行权交给客户端
才通用。**CLI 是例外**：它没有"客户端"可以代劳，所以直接用 agent 循环执行。

内置工具：`get_current_time`、`calculator`（默认启用）、`http_get`（带 SSRF
防护，标记 `dangerous` 需显式开启）。**框架刻意不内置 shell / 文件读写工具** ——
要加请自行注册并评估风险。加工具只需一个装饰器，见 docs/agent.md。

**HTTP 层不执行工具**，只按标准返回 `tool_calls`；执行循环由你的客户端负责
（用 `openai` SDK 的话就是十来行，见 docs/agent.md 的完整示例）。

## 上下文长度

**本框架不做任何长度限制，也不裁剪历史。** 推理框架的职责是把模型的行为如实
传给调用方 —— 模型产出了什么就交出去什么，不替调用方判断输出"该不该"。

长度控制由调用方决定：CLI 用 `/reset`，HTTP 由客户端自己管理 `messages`。
真实 token 数看响应里的 `usage.prompt_tokens`（直接来自引擎的 `GetInputTokenCount`，准确）；
`count_prompt_tokens()` 只是给日志看的粗估，不作为任何依据。

引擎的实测行为（KV 缓存 2048 token，含输出）：

| 输入规模 | 行为 |
|---|---|
| ≤ ~1708 | 输出正常 |
| ~2086 – 3360 | 模型产出退化文本（重复的 `- - - -` 之类）—— 这是模型的真实输出，如实返回 |
| 很大（如 31KB 英文） | 引擎返回码 1 → 服务端如实报 500（不替引擎断言原因） |

详见 [cann-engine-notes](docs/cann-engine-notes.md) 第 9 节。

## 项目结构

```
scripts/
├── start_server.sh   一键启动推理服务（预检 + 前台/后台 + status/stop）
├── start_chat.sh     一键启动交互式对话（支持 --tools）
└── stream_check.py   流式输出自检

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
