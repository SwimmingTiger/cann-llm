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

# 3) OpenAI 兼容推理服务 —— 一键脚本
scripts/start_server.sh -d /path/to/model_dir                 # 前台
scripts/start_server.sh -d /path/to/model_dir -b              # 后台，等就绪后返回
scripts/start_server.sh --status                              # 看状态
scripts/start_server.sh --stop                                # 停止

# 不带参数也行：会自动在 models/<名字>/ 下找模型目录
# 也可以用环境变量：export CANN_LLM_MODEL_DIR=/path/to/model_dir

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

## 项目结构

```
scripts/
├── start_server.sh   一键启动推理服务（预检 + 前台/后台 + status/stop）
├── start_chat.sh     一键启动交互式对话
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
