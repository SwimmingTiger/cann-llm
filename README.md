# cann-llm

华为 **CANN LLM Engine**（鸿蒙 NPU）上的对话与 **OpenAI 兼容推理服务**，代码和文档均由DeepSeek Harness中的DeepSeek V4.1 Flash等模型生成。

直接以 `ctypes` 调用系统自带的 NDK 库 `/system/lib64/ndk/libcann_llm_engine.so`，
**不需要 HAP、不需要 root；Python 包层面零依赖**（但鸿蒙 PC 需先装 Python 运行时，
见[依赖与构建](#依赖与构建)）。

> 📦 **模型还没准备好？两条路，任选一条：**
>
> - 📥 **[下载官方模型](docs/get-models.md)** —— Matrix 模型库的**网页点击步骤**、
>   选包提醒（认准 `OMC` 包）、SHA256 校验。**最快，推荐先用这条。**
>
> - 🔧 **[转换自己的模型](docs/model-conversion.md)** —— 从 HuggingFace 检查点
>   到能在 NPU 上跑的**一条我们实际走通过的路径**：
>   dopt 三阶段量化 → 导出 ONNX → OMG 转换 → 装配模型目录 → 验证。
>   ⚠️ **官方工具链对模型结构/版本的兼容性有限，不保证你的模型能走通** ——
>   这更像"可以尝试"而不是"照着做就能成"。遇到走不通的模型，欢迎开 issue 交流。
>
>   ⚠️ 有一个配置项容易写错：`quant_param_2` 必须按平台设 ——
>   **kirinx90 用 `False`**。写成 `True` 会让量化把权重负半轴钳成 0，
>   模型能跑但输出恒定垃圾（实测对照见[附录 B](docs/model-conversion.md)）。
>   配套脚本与诊断工具在 [`scripts/model-conversion/`](scripts/model-conversion/)。

## 特性

- 🗣 **交互式对话 CLI** —— 逐字流式输出、多轮上下文、采样参数热更新
- 🔌 **OpenAI 兼容 HTTP 服务** —— `/v1/chat/completions`（含 SSE 流式）、`/v1/completions`、`/v1/models`
- 🧩 **三个后端，同一套上层** —— 统一 `EngineBackend` 协议下目前实现了三个：
  **`cann`** 走华为官方 NDK 接口（`/system/lib64/ndk/libcann_llm_engine.so`），
    **`hiai`** 走系统内部引擎（`/system/lib64/libhiai_llm_engine.so`），
    **`nnrt`** 用 MindSpore Lite NDK 跑**第三方离线模型**（`.ms`）—— 它**不经过
    华为 LLM 引擎**，所以模型结构不受「逐层一对 K/V」那类约束，见
    [docs/offline-model-nnrt.md](docs/offline-model-nnrt.md)。
    用 `-b cann` / `-b hiai` / `-b nnrt` 切换，**CLI、HTTP 服务、对话模板都不用改**
- 📦 **Python 包层面零依赖** —— HTTP 层基于标准库 `http.server`。
  不过鸿蒙 PC **不自带 Python**，需先从应用市场装「Python安装器」，
  见[依赖与构建](#依赖与构建)
- 🧪 **可测** —— 分块聚合、对话模板、协议映射都有单元测试；`scripts/stream_check.py` 自检流式

## 依赖与构建

**纯 Python 项目，没有任何需要编译的部分。**

| 项 | 情况 |
|---|---|
| 本项目源码 | 全部是 Python，无需 C/C++ 工具链、无需 `pip install` 即可运行 |
| **Python 运行时** | ★ **这是唯一的第三方依赖，且鸿蒙 PC 不预装** —— 见下方说明 |
| Python 包依赖 | **零**（HTTP 层用标准库 `http.server`；测试本体是标准库 `unittest`，pytest 只是可选的跑法） |
| 原生库 | 运行期由 `ctypes` 加载**系统自带**的 `/system/lib64/ndk/libcann_llm_engine.so`（鸿蒙 NDK 提供，不是本项目编译的） |
| 模型产物 | `.omc` + `SubGraph_0.weight` 由华为的 **OMG 离线转换工具**生成，属于离线步骤，不在本仓库内（见 `docs/cann-engine-notes.md`） |
| Python 版本 | ≥ 3.9（用到 `tomllib`） |

### 先装 Python 运行时（鸿蒙 PC 必做）

**鸿蒙 PC 不自带 Python。** 必须先从应用市场安装「**Python安装器**」：

> **Python安装器**（`com.develop.opensource.ohdpc.python.launcherforpython312`）
> <https://appgallery.huawei.com/app/detail?id=com.develop.opensource.ohdpc.python.launcherforpython312>

装好后它会落在 `/data/service/hnp/bin/python3`。**`scripts/start_chat.sh` /
`scripts/start_server.sh` 会自动把它追加到 `PATH` 末尾并选中它**，无需手工配置。

> ⚠️ **不要用 glibc 构建的 Python**（例如 pyenv / harmonybrew 装的那些）。
> 本机系统 libc 是 musl，引擎按 musl 编译；glibc 的 Python 靠 `libmusl_compat`
> 垫片运行，把 musl 版引擎加载进来会**直接段错误**。脚本在发现这类解释器时会
> 逐个跳过并说明原因。

## 终端权限问题

以下应用的内置终端没有打开华为推理框架的权限，与`cann-llm`不兼容：

* MKCode
* BitFun
* WorkBuddy
* CodeArts Agent (注意不是`CodeArts IDE`)

在这些应用内运行`cann-llm`，**日志里**（hilog，不是进程的 stdout/stderr）会出现这句：

```
Error loading header libneural_network_runtime.so: failed to map header
```

以下应用的内置终端有权限，可以正常使用`cann-llm`进行推理：

* HiShell (系统自带终端)
* DevEco Studio
* CodeArts IDE (注意不是`CodeArts Agent`)

目测大部分第三方应用都无权打开华为推理框架，所以建议在系统自带`HiShell`终端内运行`cann-llm`推理。

## 快速开始

```bash
# 1) 准备模型目录（含 omc / SubGraph_0.weight / embedding / tokenizer / json 配置）
#    参考 docs/get-models.md 或 docs/model-conversion.md

# 2) 交互式对话（逐字流式）—— 一键脚本
scripts/start_chat.sh -d /path/to/model_dir
scripts/start_chat.sh -d /path/to/model_dir -p "你好"          # 单轮

#    采样默认是【开着】的，而且各项**跟随模型自带的 api_config.json**
#    （官方包是 temperature=0.7 / topK=20 / topP=0.8 / repetitionPenalty=1.1），
#    所以同一提示每次回答都不一样。想固定下来：
#      --temp 0          贪心解码（最可靠，与服务是否长驻无关）
#      --seed 42         固定种子（CLI 每次是新进程，可复现。server 要复现 seed 需重启服务）
#    对话里也可以临时改：/temp 0.9   /seed random   /params

# 2.5) 工具调用（agent）
scripts/start_chat.sh -d /path/to/model_dir --tools all      # 启用内置工具
scripts/start_chat.sh --list-tools                           # 看有哪些工具

# 3) OpenAI 兼容推理服务 —— 一键脚本
scripts/start_server.sh -d /path/to/model_dir                 # 前台
scripts/start_server.sh -d /path/to/model_dir -B              # 后台，等就绪后返回
scripts/start_server.sh --status                              # 看状态
scripts/start_server.sh --stop                                # 停止

# 5) 出问题时的诊断模式
#    记录 HTTP 请求/响应（含客户端到底带没带 max_tokens）与引擎的
#    原始输入/输出，写到 log/<日期>-<时间>-<pid>.log
scripts/start_server.sh -d /path/to/model_dir --debug

# 4) 切换后端（默认 hiai）
#    hiai = 系统内部引擎。更快、输出干净、支持停止序列 —— 官方 OMC 包推荐用它
#    cann = 官方 NDK 后端
scripts/start_chat.sh   -d /path/to/model_dir -b cann        # 改用 cann
scripts/start_server.sh -d /path/to/model_dir -b cann
#    也可以固定住：export CANN_LLM_BACKEND=hiai
#
# 5) stderr 里满屏 Unknown class perfgenius_interface 时
#    这是引擎为给 NPU 设温控而 dlopen 华为 perfgenius 客户端、连带加载 libselinux
#    解析系统策略时产生的警告 —— 不是本项目的错误，也不影响功能。
#    嫌吵可以把它屏蔽掉：
scripts/start_chat.sh -d /path/to/model_dir 2>/dev/null
#    ⚠️ 这会屏蔽【所有】stderr（含真正的报错），只在确认其它一切正常时用。
#    完整调查见 docs/hiai-backend-handoff.md。

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

> 上手只需要上面两个脚本：`scripts/start_chat.sh` 与 `scripts/start_server.sh`。
> 直接用模块、`make`、`pip install` 的方式属于**维护者/开发**用法，见文末「维护者」一节。

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

**KV 缓存上限不是引擎的固定值，而是模型自己的属性** —— 它在转换/量化时由
`kv_cache_max_len` 决定，并**固化进模型的张量形状**（`past_key_in{i}` 的第 0 维）。
所以不同模型不一样：本项目用于这组实测的 1.5B 示例模型是 **2048**，
官方 Qwen2.5-Coder-7B OMC 包则是 **4096**。

下面这组实测数据来自 **2048** 那个模型（口径是 **输入 + 输出之和**，不是输入上限）：

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
├── import_model.sh   导入官方 OMC 模型包（解压 + 转换；预检 .zip 后缀）
└── stream_check.py   流式输出自检

src/cann_llm/
├── types.py        请求 / 分块 / 结果 / 统计（后端与上层之间的唯一契约）
├── errors.py       统一异常
├── config.py       配置（TOML / 环境变量 / 命令行）
├── backends/       推理后端
│   ├── base.py     EngineBackend 协议 + 注册表
│   └── cann.py     CANN LLM Engine NDK（ctypes）实现
├── chat/           对话模板（ChatML 等）与多轮会话
├── agent/          工具调用循环与解析
├── tools/          工具注册表与内置工具
├── cli/chat.py     交互式命令行
└── api/            OpenAI 兼容 HTTP 服务（stdlib http.server）
```

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/deployment.md](docs/deployment.md)** | **部署与运行**（装依赖、拿模型、启动、验证、常见问题）—— 上手第一篇 |
| **[docs/get-models.md](docs/get-models.md)** | **下载官方模型**（Matrix 模型库网页点击步骤、选包提醒、SHA256） |
| **[docs/model-conversion.md](docs/model-conversion.md)** | **模型转换全流程**（从 HF 检查点到能上 NPU，含最容易踩的量化坑） |
| **[docs/offline-model-nnrt.md](docs/offline-model-nnrt.md)** | **离线模型 + NNRt**：把厂商离线模型包成 `.ms`，**不经过华为 LLM 引擎**、**不受**「逐层一对 K/V」那类结构约束的部署路（含自己编转换器） |
| `docs/architecture.md` | 分层、数据流、为什么这样设计 |
| `docs/cann-engine-notes.md` | **引擎笔记**：API 调用约定、回调签名、必须避开的崩溃点、上下文上限 |
| `docs/agent.md` | 工具调用（function calling）的原理、用法与可靠性 |
| `docs/openai-api.md` | 兼容范围、与官方的差异、排错 |

## 维护者

`import` 本包**不会**加载任何 `.so`；原生库只在 `backend.load()` 时加载，
所以在非鸿蒙机器上仍可正常 `import`、跑测试、做协议层开发。

> 以下是**开发/排错**用法。普通用户只需要 `scripts/start_chat.sh` 与 `scripts/start_server.sh`。

底层示例在 [`examples/`](examples/)：

* [`examples/npu-probe/`](examples/npu-probe/) —— 一百多行的 C 程序，**不用 python**、
  直接 `dlopen` 系统引擎在 NPU 上跑一轮推理。用来确认"原生 ELF 有没有权限打开 NPU"、
  或者绕开 python 层做最小复现。README 里记了三个坑（引擎是
  `/system/lib64/libhiai_llm_engine.so`；`modelPath` 只写文件名且要先 chdir 到模型目录；
  `GenerateAsync` 第 3 参是 prompt 文本、引擎自己分词）。

* [`examples/mslite-nnrt/`](examples/mslite-nnrt/) —— 单文件 C 程序，在设备上用 **MindSpore Lite NDK + NNRt 后端**加载 `.ms` 做一次推理。配套的转换步骤见 **[docs/offline-model-nnrt.md](docs/offline-model-nnrt.md)**；这条路也接成了后端：`cann-llm -b nnrt -d <放 .ms 的目录>`。

* [`examples/nnrt-probe/`](examples/nnrt-probe/) —— 走 **Neural Network Runtime**
  （`libneural_network_runtime.so`）这条路：自己在线构图、在这块 NPU 上编译并执行，
  **完全不经过 LLM 引擎**。LLM 引擎要求模型必须是「逐层一对 K/V + 每层 `past_key_value`
  进出」的结构，新架构（比如 Gemma 4 的每层不同配置、K=V、K/V 跨层共享）塞不进去；
  而 NNRt 的输入/输出张量由模型自己声明，能容纳这类结构。README 里记了实测的算子支持矩阵、
  NNRt 的建图规格（几条容易一直踩的坑），以及**怎么用 hilog 问出建图失败的真实原因**。

### 安装成 pip 包（可选，仍然没有编译）

```bash
python3 -m pip install -e .  # 提供 cann-llm-chat / cann-llm-server 两个命令
```

### 直接用模块运行（不经过脚本）

```bash
PYTHONPATH=src python3 -m cann_llm.cli.chat -d /path/to/model_dir
PYTHONPATH=src python3 -m cann_llm.api.server -d /path/to/model_dir --port 8000
```

### Makefile 快捷方式

```bash
make chat   MODEL=/path/to/model_dir                 # 交互式对话
make server MODEL=/path/to/model_dir PORT=8000       # 启动服务
make test                                            # 单元测试（pytest；自动挑可用的解释器）
make test-v                                          # 单元测试（详细输出）
make test-unittest                                   # 同上，但不需要 pytest（stdlib unittest）
```

`make` 的 `chat` / `server` 支持 `BACKEND=cann|hiai`（默认 `cann`）。

## 许可

MIT
