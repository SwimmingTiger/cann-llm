# Agent（工具调用）

## 它是什么

本服务实现的是 **prompt-based function calling**：把工具声明按 Qwen 官方
格式写进 prompt，模型用 `<tool_call>{...}</tool_call>` 表达调用意图，服务端
解析后执行，把结果以 `<tool_response>` 回填，再让模型作答。如此往复。

为什么不用原生 API：CANN LLM Engine 只接受**文本**输入、只返回**文本**，
没有结构化的 tools/tool_calls 接口。而 Qwen2.5 本身是用这套 XML 格式训练过的
（设备 tokenizer 里 `<tool_call>`=151657 / `</tool_call>`=151658 是独立特殊
token），所以走 prompt 路线是可行且正确的。

**实测确实可用**：

```
用户: 北京现在天气怎么样？
模型: <tool_call>{"name": "get_current_weather", "arguments": {"city": "北京"}}</tool_call>
执行: {"city":"北京","temperature_c":18,"condition":"多云有小雨"}
喂回: <|im_start|>user\n<tool_response>…</tool_response><|im_end|>
模型: 北京现在的天气是多云有小雨，温度大约是18摄氏度。建议您携带雨具并注意保暖。
```

## 一次迭代做了什么

```
渲染 prompt（messages + 工具声明）
  → 生成（流式）
  → 边流边过滤 <tool_call> 标记（用户看不到协议文本）
  → 解析出工具调用
      ├─ 没有调用 → 结束，产出最终回答
      └─ 有调用   → 逐个执行 → assistant(tool_calls) 与 tool 结果入历史 → 下一轮
```

到 `max_steps` 就**不再传工具声明**，逼模型用自然语言收尾，避免无限调用。

## 工具

### 内置

| 工具 | 说明 | 默认 |
|---|---|---|
| `get_current_time` | 当前日期与时间（可指定时区偏移、格式） | ✅ 启用 |
| `calculator` | 算术表达式，走 AST 白名单（不是 `eval`） | ✅ 启用 |
| `http_get` | 抓取 http/https URL | ⚠️ `dangerous`，需显式启用 |

**框架刻意不内置** shell / 读写文件 / 任意网络请求这类工具。那些能力一旦
默认打开，一个会写 prompt 的攻击面就能升级成 RCE。要加就在应用侧注册，
并自行评估风险。

### `http_get` 的安全措施（SSRF 防护）

只允许 http/https；解析后的地址不能是私有 / 回环 / 链路本地 / 保留 / 组播 /
未指定网段（含 IPv6 `::1`）；限制响应体大小与超时；不跟随到非 http(s) 的重定向。
确需访问内网时把参数 `allow_private_network` 设为 `true`（会写明风险）。

### 自己加一个工具

```python
from cann_llm.tools import default_registry

@default_registry().register(
    name="query_db",
    description="查询订单状态。",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
    dangerous=False,          # True 则默认不启用，需显式授权
    timeout_s=5.0,
)
def query_db(order_id: str) -> str:
    return json.dumps({"order_id": order_id, "status": "已发货"})
```

要点：

* `parameters` 用 JSON Schema 描述。**描述写清楚很重要** —— 见下面"可靠性"。
* handler 抛异常不会中断对话：异常会被包成 `ok=False` 的工具结果回给模型，
  让它自己修正。
* 返回值建议返回 JSON 字符串（模型对结构化结果理解更好）。

## 可靠性（如实说明）

工具调用是 **prompt 驱动的**，可靠性取决于模型本身。在 **Qwen2.5-1.5B** 上
实测：

| 场景 | 表现 |
|---|---|
| 提问里明确提到工具/参数 | 稳定调用 |
| 一次给多个工具、问复合问题 | 能选对工具并分别调用 |
| 工具失败（语法错等） | **会**把错误回给模型；加上显式重试指令后能自我修正 |
| 提问很短、工具参数全可选 | **可能不调用**，转而向用户索要信息 |

最后一条是最常见的失败模式。缓解办法（已内建）：

* `AgentConfig.force_tool_use = True`（默认）会在工具说明后追加：
  *"If the user's request can be answered by any of the tools above, you MUST
  emit the tool call immediately. Never ask the user for information that a tool
  can provide…"* —— 同一问题实测从 **0/4 提升到 4/4**。
* 把工具的 `description` 写具体（说明"无需参数，可直接调用"之类）。

**务必知道**：更大/更强的模型会显著更可靠。换模型只需改配置里的
`model_dir`，agent 这一层不用动。

## 用法

### 命令行

```bash
# 启用全部安全工具
scripts/start_chat.sh -d /path/to/model_dir --tools all

# 只启用指定工具
scripts/start_chat.sh -d /path/to/model_dir --tools get_current_time,calculator

# 单轮
scripts/start_chat.sh -d /path/to/model_dir --tools all -p "现在几点？"

# 列出可用工具
python3 -m cann_llm.cli.chat --list-tools
```

交互中：`/tools` 看列表与启用情况，`/tools calculator` 切换。

### HTTP（OpenAI 兼容）

HTTP 层**只实现 OpenAI 标准语义**：服务端不执行工具，只把模型表达的调用
返回给客户端，由客户端执行后再把结果发回来。

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "2 的 10 次方是多少？"}],
  "tools": [{"type": "function", "function": {
      "name": "calculator", "description": "计算算术表达式",
      "parameters": {"type": "object",
                     "properties": {"expression": {"type": "string"}},
                     "required": ["expression"]}}}]}'
```

声明的工具会渲染进 prompt；模型如果发起调用，响应就是：

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant", "content": null,
      "tool_calls": [{"id": "call_3f2a1b9c", "type": "function",
                      "function": {"name": "calculator",
                                   "arguments": "{\"expression\": \"2**10\"}"}}]
    }
  }]
}
```

客户端执行后把结果发回来，服务端继续：

```json
{"messages": [
  {"role": "user", "content": "2 的 10 次方是多少？"},
  {"role": "assistant", "content": null, "tool_calls": [{...}]},
  {"role": "tool", "tool_call_id": "call_3f2a1b9c", "content": "{\"result\": 1024}"}
], "tools": [{...}], "model": "qwen2.5-1.5b"}
```

流式时同样按标准发 `delta.tool_calls`，并且 **`<tool_call>` 协议标记会被
过滤掉**，不会出现在 `delta.content` 里。

> 为什么服务端不替你执行工具：本服务注册的工具（`get_current_time` 等）
> 和你的应用能访问的东西（数据库、内部 API）不是一回事。让客户端执行是
> OpenAI 的既定语义，也更通用。CLI 是例外 —— 它没有"客户端"可以代劳，
> 所以直接走 `AgentLoop`。

### Python 客户端示例

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")

TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "计算算术表达式",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]

def calculator(expression: str) -> str:
    return json.dumps({"result": eval(expression)})     # 示例，别在生产里用 eval

messages = [{"role": "user", "content": "2 的 10 次方是多少？"}]
while True:
    resp = client.chat.completions.create(
        model="qwen2.5-1.5b", messages=messages, tools=TOOLS)
    msg = resp.choices[0].message
    if not msg.tool_calls:
        print(msg.content)
        break
    messages.append(msg)
    for call in msg.tool_calls:
        out = calculator(**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
```

## 性能

工具调用是**多次生成**：一次发起调用、一次消化结果给答案。1.5B 模型在
本设备上约 7–9 tok/s：

| 场景 | 耗时 |
|---|---|
| 普通回答（几十 token） | 2–4 s |
| 一次工具调用 + 收尾 | 6–17 s |

HTTP 层不跑循环，所以没有"轮数"概念；CLI 的 `agent.max_steps`
（默认 4）限制最多几轮，防止长尾。

## 代码结构

| 文件 | 职责 |
|---|---|
| `tools/schema.py` | JSON Schema 子集校验（零依赖） |
| `tools/registry.py` | 工具注册表、`dangerous` 门禁 |
| `tools/builtin.py` | 内置工具 + SSRF 防护 |
| `agent/parser.py` | 从模型输出解析 tool_call（容错） |
| `agent/loop.py` | agent 循环（CLI 用）、`StreamFilter` |
| `chat/template.py` | 把工具声明/调用/结果渲染成 Qwen 官方格式 |
| `api/openai.py` | OpenAI 协议映射（tools 解析、tool_calls 响应） |
| `api/server.py` | HTTP 入口（按 OpenAI 标准返回 `tool_calls`） |

## 扩展点

| 想做什么 | 改哪里 |
|---|---|
| 加工具 | `ToolRegistry.register`（见上） |
| 换工具调用格式（GLM / Llama 等） | `chat/template.py` 里新增 `ChatTemplate` 子类，覆盖 `render` 与 `tool_call_tags` |
| 改 agent 策略（重试、并行、反思） | `agent/loop.py` 的 `run()` |
| 换更强的模型 | 只改配置 `model_dir` |
