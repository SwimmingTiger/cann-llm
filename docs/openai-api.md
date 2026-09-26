# OpenAI 兼容接口

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz`（或 `/health`） | 健康检查，**不需要鉴权** |
| GET | `/v1/models` | 列出模型 |
| POST | `/v1/chat/completions` | 对话补全（流式 / 非流式） |
| POST | `/v1/completions` | 文本补全（流式 / 非流式） |

启动：

```bash
./scripts/start_server.sh -d /path/to/model_dir --host 127.0.0.1 --port 8000 --api-key sk-xxx
```

## 请求示例

```bash
# 非流式
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is the capital of France?"}
    ],
    "max_tokens": 64, "temperature": 0.2
  }'

# 流式（SSE）
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from one to five."}],"stream":true}'
```

官方 Python SDK 直接可用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-xxx")

resp = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)

for chunk in client.chat.completions.create(
        model="qwen2.5-1.5b",
        messages=[{"role": "user", "content": "写一首短诗"}],
        stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 兼容性

### 支持

| 字段 | 说明 |
|---|---|
| `model` | 忽略具体值（服务只有一个模型），空则用配置的 `model_id` |
| `messages` | `role` ∈ system/user/assistant/tool；`content` 支持字符串或内容块数组（仅 `type: text`） |
| `prompt` | `/v1/completions`，只支持单个字符串 |
| `stream` | SSE；`data: [DONE]` 结束 |
| `stream_options.include_usage` | 末尾额外发一帧 `choices: []` 且带 `usage` 的块 |
| `max_tokens` / `max_completion_tokens` | 后者优先 |
| `temperature` / `top_p` / `stop` / `seed` | 直接映射 |
| `n` | 只支持 `1` |

### 扩展字段（OpenAI 没有，本服务支持）

| 字段 | 说明 |
|---|---|
| `top_k` | top-k 采样。OpenAI 无此字段，但本引擎支持且很有用 |
| `repetition_penalty` | 重复惩罚，默认 1.1 |

### 接受但忽略（默认）

`tools`、`functions`、`tool_choice`、`function_call`、`parallel_tool_calls`、
`frequency_penalty`、`presence_penalty`、`response_format`、`logit_bias`、`user`

引擎没有对应能力。**这些字段不会导致报错**，服务端会：

1. 在响应头回报被忽略的字段名：
   ```
   X-Cann-Llm-Ignored-Fields: tools, tool_choice, frequency_penalty
   ```
2. 在服务端日志里记一行 `忽略字段: tools, tool_choice`

**为什么默认是忽略而不是报错**：绝大多数现代客户端（Continue / Cline /
Roo Code / LangChain / 各类 agent 框架）**即使只是普通聊天也会带上
`tools`**。一律 400 会让这些客户端完全无法使用本服务。

代价是：模型不会真的调用函数，它只会用普通文本回答。客户端通常能接受
（当作一次最终回答），但**不要指望本服务完成 function calling 闭环**。

### 明确拒绝（400，任何模式）

| 字段 | 为什么必须拒绝 |
|---|---|
| `logprobs` / `top_logprobs` | 响应里不会出现 `logprobs` 字段，客户端解析会出错 |
| `content` 里的 `image_url` | 模型看不见图片，照常回答等于骗人 |
| `n > 1` | 引擎一次只产一路，返回的 `choices` 数量会对不上 |
| 批量 `prompt` | 同上 |

### 严格模式

如果你更希望「宁可报错，也别给我"看起来正常"的假象」：

```toml
[server]
reject_unsupported = true
```

或 `CANN_LLM_SERVER__REJECT_UNSUPPORTED=true`。开启后「接受但忽略」那一组
也会返回 400。上表里的字段**无论哪种模式都拒绝**。

## 响应示例

非流式：

```json
{
  "id": "chatcmpl-2d6792a884e249f7b7fc0840",
  "object": "chat.completion",
  "created": 1790384891,
  "model": "qwen2.5-1.5b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital city of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 26, "completion_tokens": 9, "total_tokens": 35}
}
```

流式（每行一个 SSE 帧）：

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"One"},"finish_reason":null}]}

…

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

错误：

```json
{"error": {"message": "本服务只支持 n=1（引擎一次只产一路）",
           "type": "invalid_request_error", "param": null, "code": null}}
```

## 错误码

| HTTP | `error.type` | 触发条件 |
|---|---|---|
| 400 | `invalid_request_error` | 参数非法、不支持的字段、`n>1`、prompt 为空 |

| 400 | `invalid_request_error` | JSON 解析失败 |
| 401 | `invalid_api_key` | 配置了 `api_key` 但请求未带或错误 |
| 404 | `not_found` | 未知路径 |
| 413 | `invalid_request_error` | 请求体超过 4 MB |
| 500 | `generation_failed` / `model_load_failed` / `internal_error` | 服务端错误 |
| 503 | `server_busy` | 并发超上限或排队超时（引擎一次只跑一路） |
| 503 | `backend_unavailable` | 动态库缺失 / 非鸿蒙环境 |

## 与 OpenAI 的差异

1. **一次只跑一路推理**。并发请求会在服务端排队（`server.max_queue`，
   默认 8），超出返回 503 `server_busy`。这是 CANN 引擎的限制，
   不是实现偷懒。
2. **不做 token 计费意义的分词**。`usage` 里的 token 数来自引擎自身的
   计数（`GetInputTokenCount`/`GetOutputTokenCount`）；服务端的输入长度
   预检只能按字符数保守估算。
3. **`finish_reason` 是近似值**。引擎没有直接暴露"因何而停"，
   实现上用"输出 token 数是否顶到 `max_tokens`"判断 `length`，否则 `stop`。
4. **不支持函数调用、多模态、logprobs**（引擎无对应能力）。
5. **没有 `/v1/embeddings`**。模型未导出 embedding 接口。
   将来要加时，在 `EngineBackend` 上扩一个方法即可。


## 排错

### `base_url` 该填什么？

**只到 `/v1`**，不要带端点路径：

```python
# ✓ 正确
OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="...")

# ✗ 错误：SDK 会在后面拼 /models、/chat/completions
OpenAI(base_url="http://127.0.0.1:8000/v1/chat/completions")
OpenAI(base_url="http://127.0.0.1:8000/chat/completions")        # 少了 /v1
OpenAI(base_url="http://127.0.0.1:8000")                        # 少了 /v1
```

填错时服务端会返回带提示的 404，例如：

```
GET /v1/chat/completions/models  → 404
未知路径 /v1/chat/completions/models。
路径 /v1/chat/completions/models 看起来是把端点路径拼进了 base_url。
OpenAI 客户端的 base_url 只应到 /v1，例如 http://127.0.0.1:8000/v1
—— 不要带上 /chat/completions 之类的端点路径。
```

### 直接用浏览器打开根路径

`GET /` 会返回一个小索引，列出全部端点并重复 base_url 的说明：

```json
{
  "service": "cann-llm",
  "endpoints": {
    "GET /v1/models": "列出模型",
    "POST /v1/chat/completions": "对话补全（支持 stream）",
    "POST /v1/completions": "文本补全（支持 stream）",
    "GET /healthz": "健康检查"
  },
  "note": "OpenAI 客户端的 base_url 只应到 /v1，例如 http://127.0.0.1:8000/v1 …"
}
```

### 常见状态码速查

| 看到 | 多半是 |
|---|---|
| `404 未知路径 …/v1/chat/completions/models` | base_url 带上了端点路径 |
| `404 未知路径 /chat/completions` | base_url 少了 `/v1` |
| `405 GET 不被 /v1/chat/completions 支持` | 用错了 HTTP 方法（端点是 POST）；响应头有 `Allow` |
| `401 invalid_api_key` | 服务端开了 `api_key`，但请求没带 `Authorization: Bearer` |
| `503 server_busy` | 并发超过 `server.max_queue`（引擎一次只跑一路），稍后重试 |
| `400 invalid_request_error` | 参数问题，`error.message` 里有具体原因 |
| `503 backend_unavailable` | 不在鸿蒙环境 / 找不到 NDK 库 |
| `500 generation_failed` | 引擎返回非零。服务端只如实报告返回码，不替它断言原因（可能：输入超出**该模型**的 KV 缓存上限 / 含无法分词的字符 / 引擎内部错误） |

服务端日志里也会带上同样的提示（404/405 会打印原因），方便对着日志排查。
