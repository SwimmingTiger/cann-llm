# 架构

## 分层

```
        ┌───────────────────────┐   ┌────────────────────────┐
        │  cli/chat.py          │   │  api/server.py         │   ← 交互层
        │  交互式对话（逐字输出）│   │  OpenAI 兼容 HTTP 服务 │
        └───────────┬───────────┘   └───────────┬────────────┘
                    │                           │
        ┌───────────▼───────────────────────────▼────────────┐
        │  chat/session.py   多轮会话（历史、裁剪、统计）      │   ← 会话层
        │  chat/template.py  对话模板（ChatML / plain）        │
        └───────────┬────────────────────────────────────────┘
                    │
        ┌───────────▼────────────────────────────────────────┐
        │  types.py    GenerationRequest / Chunk / Result      │   ← 契约
        │  errors.py   统一异常 + HTTP 映射                    │
        └───────────┬────────────────────────────────────────┘
                    │  EngineBackend 协议
        ┌───────────▼────────────────────────────────────────┐
        │  backends/base.py    协议 + 注册表 + 串行化包装      │   ← 后端层
        │  backends/cann.py    CANN LLM Engine NDK（ctypes）   │
        └────────────────────────────────────────────────────┘
```

**依赖方向严格自上而下**。`backends/` 不 import `chat/`、`cli/`、`api/`；
`api/openai.py` 只做 dict ↔ 项目类型的映射，不碰 HTTP。因此：

* 换 HTTP 框架（FastAPI/ASGI）不影响 `api/openai.py`
* 加新后端不影响上层
* 加新对话模板不影响后端

## 三个关键设计选择

### 1. 流式优先

后端只实现 `generate() -> Iterator[GenerationChunk]`，**逐块产出增量文本**；
"非流式"就是 `types.aggregate()` 把块收干。于是：

* CLI 的逐字输出 = 直接写 stdout
* OpenAI 的 `stream=true` = 把块包成 SSE
* OpenAI 的 `stream=false` = 收干后包成一个响应

三条路径共用同一段生成逻辑，行为天然一致（连 usage 都来自同一个最终分块）。

### 2. 阻塞 C 调用 + 回调 → 生成器

CANN 的 `Generate` 是阻塞调用，而它每产一个 token 会回调我们的函数；
Python 生成器**不能**从回调里 `yield`。所以 `backends/cann.py` 里：

```
工作线程：调 Generate（阻塞）
回调（引擎线程）：把增量 push 进 queue.Queue
生成器（调用方线程）：从 queue 取出来 yield
```

上层因此只面对一个普通的生成器，不感知线程。

### 3. 串行化放在后端层

CANN 引擎一个 Executor 同时只能跑一路推理。这个约束被封装成
`SerializedBackend` 包装器，而不是散落在服务层的加锁逻辑里。好处：

* 服务层代码保持线性、无锁
* 排队上限 / 超时 / `BusyError`（→ HTTP 503）只有一处实现
* 将来接上支持并发的后端（vLLM 等）时，直接不套这层即可

## 核心零依赖

`pyproject.toml` 里 `dependencies = []`。HTTP 层用标准库 `http.server`，
测试用 `unittest`。原因很实际：**目标运行环境（鸿蒙设备）上装不了
FastAPI/uvicorn/pytest**，而本服务的瓶颈在 NPU 推理（~7 tok/s），
HTTP 层性能完全不是问题。

`[project.optional-dependencies]` 里保留了 `fastapi` / `dev` 两组，
需要时再装。

## 数据流（一次 HTTP 流式请求）

```
POST /v1/chat/completions {"stream": true}
  → api/openai.parse_chat_request()   校验 + 转成 ChatCompletionRequest
  → chat/template.chatml.render()     消息列表 → ChatML 文本
  → SerializedBackend.generate()      排队 + 加锁
  → CannNdkBackend.generate()         工作线程跑 Generate，回调 push 队列
  → （逐块）api/server._stream_chat() 每块发一帧 SSE
  → data: [DONE]
```

## 扩展点

| 想做什么 | 改哪里 |
|---|---|
| 加后端（vLLM / llama.cpp / 远端） | 新建 `backends/xxx.py` + `@register_backend("xxx")`，在 `backends/__init__.py` import 一次 |
| 加对话模板（GLM / HunYuan） | `chat/template.py` 里实现 `ChatTemplate` + `register_template` |
| 换 HTTP 框架 | 复用 `api/openai.py`，重写 `api/server.py` |
| 加 tokenize / logprobs 接口 | `GenerationChunk.token_id` 已预留；后端补实现即可 |
| 多进程 / 多卡 | 后端层已隔离，`SerializedBackend` 换成进程池即可 |

## 已知限制

* **一次只跑一路推理**（引擎限制）；并发请求排队，超出上限返回 503
* `generate()` 是生成器：若调用方中途放弃且不 `close()`，锁要等 GC 才释放
  （HTTP 层总会收干，CLI 的 `break` 会触发 `GeneratorExit`→`finally`）
* 没有独立的 tokenize 接口，输入长度只能用字符数保守预检
* `os.chdir(模型目录)`：引擎按相对路径找模型文件，所以进程 CWD 会被改掉
