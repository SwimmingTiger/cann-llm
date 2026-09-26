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

### 工具结果不做截断

工具返回多少字符就原样给模型多少，也原样交给调用方。**框架不设上限**。

原先是截到 4000 字符，问题在于截断发生在结果交给调用方**之前** —— 于是它同时
干了两件事：

* 截断进入 prompt 的内容（这是 agent 循环构造 prompt 的工作，尚有讨论空间）；
* 截断调用方拿到的 `ToolResult.content`（这是**销毁数据** —— 工具是调用方
  注册的，返回值是调用方的东西，框架没理由砍掉还让它看不出来）。

实测：工具返回 12000 字符时，模型和调用方都只能拿到 4018 字符，完整数据无法取回。

现在一条也不截。工具结果过大时，请在工具实现里自己分块、摘要，或返回引用
（如文件路径/ID）而不是全文 —— 这属于调用方的设计决定。

### 解析范围：只认标准标签

解析器只把 ``<tool_call>…</tool_call>`` 里的内容当作工具调用。**模型不带标签
直接吐裸 JSON 时，不认。**

曾经做过「裸 JSON 容错」——把形如 ``{"name": ..., "arguments": ...}`` 的
JSON 也当成调用。实测它会把用户**明确要求生成**的 JSON 代码删掉：

```
用户:   给我一个 JSON-RPC 请求体的例子
模型:   当然，这是一个 JSON-RPC 请求体：
         ```json
         {"name": "getUserProfile", "arguments": {"userId": 42}}
         ```
         把它 POST 到 /rpc 即可。

原先:   判成工具调用 + 从正文删除该 JSON
         → 用户拿到的是被掏空的代码块： ```json\n\n```
现在:   不判为调用，回答原样保留
```

也会把模型举例说明用的 JSON 从正文里抹掉（``The tool takes {"name": ...}
as input.`` → ``The tool takes  as input.``）。

根因是它替模型认定了并不存在的调用意图，并把模型写下的文本从输出里删掉。

**模型忘记标签属于它自身的格式偏离，推理框架应当如实呈现。** 想让它稳定用标签，
该改的是 prompt 里的格式说明，而不是在解析器里猜。``<tool_call>`` 标签的存在
本身就是"这是一个调用"的协议信号 —— 这也是唯一可靠的判据。

### 为什么要把 `<tool_call>` 从流里过滤掉

`<tool_call>` 是 **Qwen 的文本格式**，不是 OpenAI 协议。实测 DSH 所用的
pi-ai（`node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js`）
在流式处理里只认 `choice.delta.tool_calls`：

```js
if (choice?.delta?.tool_calls) {
    for (const toolCall of choice.delta.tool_calls) {
        const block = ensureToolCallBlock(toolCall);
        ...
```

整个 DSH 仓库**搜不到任何对 `<tool_call>` 文本标记的解析**。也就是说：
不过滤的话，调用方只会把这段当成普通的 assistant 文本，**agent 循环直接断掉**。

所以这**不是"删掉模型输出"，而是按 OpenAI 协议重新编码** —— 调用内容完整地
出现在 `delta.tool_calls` 里，信息没有丢失。

另外两条边界，保证信息不会不可逆地丢失：

* **只在客户端声明了 `tools` 时才过滤**。不带 tools 的请求原样透传，
  想直接拿原始 Qwen 文本格式的调用方依然拿得到。
* 未闭合的工具段（被 max_tokens 截断）会连同后续文本一起抑制，但服务端
  仍会尝试解析它并把结果放进 `tool_calls`；解析不了就带 `ok=false` 回到
  agent 循环，不会静默吞掉。

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
| 工具失败（语法错等） | 把错误原文回给模型，由它自己决定怎么办 —— 框架不加任何指导 |
| 提问很短、工具参数全可选 | **可能不调用**，转而向用户索要信息 |

最后一条是最常见的失败模式。**框架不提供任何「强制调用」开关** —— 缓解只能
从两处入手，都在调用方自己手里：

* **把工具的 `description` 写具体**（说明「无需参数，可直接调用」之类）。
  实测这对模型的影响很大。
* **在调用方自己的 system prompt 里写清期望**。框架不会替你加这类文字。

### `finish_reason` 如实透传

`Final.finish_reason` 用的是**最后一轮**引擎推断出的值，不做任何改写。

曾经有过这样一段：

```python
if finish_reason == "length" and pending_calls:
    finish_reason = "tool_calls"
```

它是错的，两个原因：

1. `pending_calls` 累积的是**整个运行过程**中发起过的调用，而 `finish_reason`
   描述的是**最后一轮**怎么结束的 —— 两件不同的事被混在一起；
2. 后果是误导：最后一轮明明被 `max_tokens` 截断（半句话），却被报成
   `tool_calls`。按 pi-ai 的映射 `tool_calls` → `stopReason: "toolUse"`，
   等于告诉调用方「模型还想执行工具」，而实际是输出被截断了。

想知道本次运行用过哪些工具，看 `Final.tool_calls`（那个字段保留）。

### 最后一轮不再传工具声明

`max_steps` 用尽时，最后一轮渲染的 prompt 里**不含工具声明** —— 目的是逼模型
用自然语言收尾，避免无限调用下去。

这一条**保留**，因为它和上面几处性质不同：它是 agent 循环**自身的控制流**
（决定循环何时必须结束、怎么结束），不是在解释或改写模型的行为。但它确实
改变了模型看到的 prompt，所以在这里写明：**用满 `max_steps` 时，模型看到的
最后一轮 prompt 与前面几轮不同**（没有 `<tools>` 段）。

### 工具失败：只陈述事实

工具抛异常、参数不符 schema、工具名不存在、调用格式无法解析 —— 这四种情况
由 agent 循环判定为失败（`ToolResult.ok = False`）并**把错误原文回给模型**，
**不追加任何指导**。

判定失败确实是 agent 循环的职责，这一点有参照实现佐证。DSH 的
`dsh-agent-loop` 在工具失败时构造：

```js
{ content: [{ type: "text", text: "Error: tool call aborted before dispatch" }],
  isError: true,
  error: { message: "...", info: { name: "AbortError", code: ... } } }
```

即「事实 + 结构化元数据」，**没有一句「请重试」**。

这一点很要紧：pi-ai 的 wire 转换（`dist/api/openai-completions.js`）里，

```js
const toolResultMsg = { role: "tool", content: sanitizeSurrogates(toolResultText),
                        tool_call_id: toolMsg.toolCallId };
```

—— **`isError` 根本不发给模型**，它只是框架内部给 UI/会话记录用的元数据。
所以模型判断「这是失败」的唯一依据就是 content 里的文字，那个 `Error:` 前缀
（而非任何指导语）才是必须的部分。

曾经在这里追加过「请修正参数后重新调用同一个工具；在拿到成功结果之前不要凭
猜测作答」。实测那样确实能让小模型少编答案，但那属于**替调用方指挥模型** ——
与 `force_tool_use` 同类，已移除。想让模型失败后重试，请写在调用方自己的
system prompt 里。

### 为什么框架不写「你必须调用工具」

曾经加过一个 `force_tool_use`（往工具说明后面追加 *"you MUST emit the tool
call immediately…"*），实测对小模型很有效（同一问题 **0/4 → 4/4**）。但它被
移除了，因为它同时改了调用方的 prompt 和模型的行为。

对照 llama.cpp 的源码（`common/chat-auto-parser-generator.cpp`、`common/chat.h`）：

| | llama.cpp | 原先的 `force_tool_use` |
|---|---|---|
| 机制 | **grammar（约束解码，token 级硬约束）** | 往 prompt 里塞文字 |
| 谁发起 | **调用方**要求 `tool_choice: required` | **框架**默认开启 |
| `tool_choice: auto` 时 | **完全不强制** —— 用 lazy grammar，只在模型自己开始输出触发标记后才约束格式 | 塞了「你必须调用」 |

```cpp
bool include_grammar = has_response_format || (has_tools &&
        ((tool_choice == AUTO && !trigger_marker.empty()) ||
          tool_choice == REQUIRED));
data.grammar_lazy = !has_response_format && tool_choice == AUTO;
```

而且 llama.cpp 里**搜不到任何框架自撰的「你必须调用工具」文字** —— 工具格式
说明那段文字来自**模型自带的 chat template**（如
`models/templates/Qwen-Qwen2.5-7B-Instruct.jinja`），框架只负责应用。
本项目的 `render_tools_block()` 就是照抄那份模板，逐字一致。

结论：**「强制调用」是调用方的诉求，且正统做法是约束解码而非提示词。**
CANN 引擎没有 grammar 能力，所以我们做不到真正的强制；能做的只是如实告知
`tool_choice: required` 不被支持（见响应头 `X-Cann-Llm-Ignored-Fields`），
而不是用一段模型可以无视的文字假装做到了。

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
