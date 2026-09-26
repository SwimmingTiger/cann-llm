"""Agent 循环：让模型调用工具、把结果喂回去、直到给出最终回答。

一次迭代（step）：
    渲染 prompt（含工具声明） → 生成 → 解析 tool_call
      ├─ 没有调用   → 结束，产出最终回答
      └─ 有调用     → 逐个执行 → 把 assistant(tool_calls) 与 tool 结果
                      追加进消息列表 → 下一轮

细节与取舍
----------
* **流式输出要隐藏工具协议**：模型吐 ``<tool_call>…</tool_call>`` 时这段
  文本不能给用户看。用 :class:`StreamFilter` 做增量过滤（含"半个开标签"的
  延迟判定），所以调用标记不会漏出去，也不会让正文卡住。
* **工具失败不中断对话**：执行异常 / 参数不符 schema / 工具不存在，都变成
  ``ok=False`` 的 tool 结果回给模型，让它自己纠正 —— 这是 agent 能自我修复的
  关键。只有"生成"本身失败才向上抛。
* **步数用尽就强制收尾**：最后一轮不再传工具，逼模型用自然语言作答，
  而不是无限调用下去。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..backends.base import EngineBackend
from ..chat.template import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ChatTemplate,
    Message,
)
from ..tools.registry import ToolError, ToolRegistry
from ..types import (
    GenerationChunk,
    GenerationParams,
    GenerationRequest,
    GenerationStats,
    ToolCall,
    ToolResult,
)
from .parser import parse_tool_calls

__all__ = [
    "AgentConfig", "AgentLoop", "AgentEvent",
    "TextDelta", "ToolCallReady", "ToolCallDone", "StepStarted", "Final",
    "StreamFilter",
]


# ------------------------------------------------------------------ 事件

@dataclass
class TextDelta:
    """给用户看的正文增量（已剔除工具协议标记）。"""

    text: str


@dataclass
class ToolCallReady:
    """即将执行一次工具调用。"""

    call: ToolCall
    step: int


@dataclass
class ToolCallDone:
    """一次工具调用执行完毕。"""

    result: ToolResult
    duration_ms: float
    step: int


@dataclass
class StepStarted:
    """开始新一轮迭代。"""

    step: int
    max_steps: int


@dataclass
class Final:
    """循环结束。"""

    text: str
    finish_reason: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    steps: int = 0
    stats: Optional[GenerationStats] = None
    messages: List[Message] = field(default_factory=list)


AgentEvent = Any   # Union[TextDelta, ToolCallReady, ToolCallDone, StepStarted, Final]


# ------------------------------------------------------------------ 流式过滤

def _prefix_hold(text: str, tag: str) -> int:
    """返回 text 末尾「是 tag 真前缀」的最长长度（需要继续等后续字符）。"""
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class StreamFilter:
    """把模型输出里的 ``<tool_call>…</tool_call>`` 段从流中剔除。

    为什么需要它：``<tool_call>`` 是 Qwen 的**文本格式**，不是 OpenAI 协议。
    实测 DSH 所用的 pi-ai（`dist/api/openai-completions.js`）在流式处理里
    只认 ``choice.delta.tool_calls``，整个仓库搜不到任何对 ``<tool_call>``
    文本标记的解析。也就是说：**不过滤的话调用方根本不会把这段当工具调用**，
    它只会看到一段普通的 assistant 文本，agent 循环直接断掉。

    所以这不是"删掉模型输出"，而是**按 OpenAI 协议做重新编码** ——
    调用内容会完整地出现在 ``delta.tool_calls`` 里，信息没有丢失。
    另外：只在客户端**声明了 tools** 时才过滤；不带 tools 的请求原样透传，
    想拿原始 Qwen 格式的调用方依然拿得到。

    流式场景下只能边收边判断，所以需要缓冲"半个开标签"：最多延迟
    ``len('<tool_call>') - 1`` 个字符，实际约一个 token。
    """

    def __init__(self, open_tag: str = TOOL_CALL_OPEN,
                 close_tag: str = TOOL_CALL_CLOSE) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._tail = ""
        self._in_tool = False

    @property
    def in_tool_call(self) -> bool:
        return self._in_tool

    def feed(self, delta: str) -> str:
        """喂入一段增量，返回应当发给用户的可见文本。"""
        self._tail += delta
        out: List[str] = []
        while self._tail:
            if self._in_tool:
                i = self._tail.find(self.close_tag)
                if i < 0:
                    # 只保留可能是"半个闭标签"的尾巴，其余丢弃
                    keep = _prefix_hold(self._tail, self.close_tag)
                    self._tail = self._tail[len(self._tail) - keep:] if keep else ""
                    break
                self._tail = self._tail[i + len(self.close_tag):]
                self._in_tool = False
                continue
            i = self._tail.find(self.open_tag)
            if i < 0:
                hold = _prefix_hold(self._tail, self.open_tag)
                if hold:
                    out.append(self._tail[:len(self._tail) - hold])
                    self._tail = self._tail[len(self._tail) - hold:]
                else:
                    out.append(self._tail)
                    self._tail = ""
                break
            out.append(self._tail[:i])
            self._tail = self._tail[i + len(self.open_tag):]
            self._in_tool = True
        return "".join(out)

    def flush(self) -> str:
        """流结束时吐出剩余可见文本（未闭合的工具段一律不吐）。"""
        text = "" if self._in_tool else self._tail
        self._tail = ""
        return text


# ------------------------------------------------------------------ 配置

@dataclass
class AgentConfig:
    #: 最多迭代几轮（含最后一轮强制收尾）
    max_steps: int = 4


# ------------------------------------------------------------------ 主循环

class AgentLoop:
    """把「生成 + 工具调用」串成循环。"""

    def __init__(self, backend: EngineBackend, template: ChatTemplate,
                 tools: ToolRegistry, *, config: Optional[AgentConfig] = None,
                 system_prompt: Optional[str] = None,
                 params: Optional[GenerationParams] = None):
        self.backend = backend
        self.template = template
        self.tools = tools
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.params = params or GenerationParams()

    # ---- 工具执行 ----
    def _execute(self, call: ToolCall) -> tuple[ToolResult, float]:
        t0 = time.time()
        tool = self.tools.get(call.name) if call.name else None

        if not call.name:
            raw = call.arguments.get("__unparsable__", call.raw)
            return (ToolResult(call.id, "", f"Error: could not parse tool call: {raw!r}",
                               ok=False), 0.0)
        if tool is None:
            return (ToolResult(call.id, call.name,
                               f"Error: no tool named {call.name!r}. Available: "
                               f"{', '.join(self.tools.names())}", ok=False), 0.0)

        try:
            if tool.timeout_s:
                value = self._call_with_timeout(tool, call.arguments, tool.timeout_s)
            else:
                value = tool.call(call.arguments)
            content = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=str)
            ok = True
        except (ToolError, PermissionError, KeyError) as e:
            content, ok = f"Error: {e}", False
        except Exception as e:                    # noqa: BLE001 - 工具是外部代码
            content, ok = f"Error: {type(e).__name__}: {e}", False

        # 失败时只陈述事实，**不追加任何指导**。
        #
        # 依据 DSH 的实现（dsh-agent-loop）：工具失败时它构造
        #   { content: [{type:"text", text:"Error: tool call aborted before dispatch"}],
        #     isError: true, error: {message, info:{name, code}} }
        # 也就是「事实 + 结构化元数据」，没有一句「请重试」。
        #
        # 这一点很要紧：查过 pi-ai 的 wire 转换（dist/api/openai-completions.js），
        # tool 消息只发 content / tool_call_id —— **isError 根本不发给模型**。
        # 所以模型判断"这是失败"的唯一依据就是 content 里的文字，
        # 那个 "Error:" 前缀（而非任何指导语）才是必须的部分。
        #
        # 曾经在这里追加过「请修正参数后重新调用同一个工具；在拿到成功结果之前
        # 不要凭猜测作答」—— 实测那样确实能让小模型少编答案，但那属于替调用方
        # 指挥模型，与 force_tool_use 同类，已移除。
        #
        # 工具返回多少就用多少，也不截断：截断既让调用方拿不回自己工具返回的
        # 数据（数据销毁，不是构造 prompt），又替调用方决定了模型能看到多少。
        return ToolResult(call.id, call.name, content, ok=ok), (time.time() - t0) * 1000

    @staticmethod
    def _call_with_timeout(tool, arguments: Dict[str, Any], timeout_s: float) -> Any:
        box: Dict[str, Any] = {}

        def run() -> None:
            try:
                box["value"] = tool.call(arguments)
            except BaseException as e:            # noqa: BLE001
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            raise ToolError(f"执行超过 {timeout_s:g}s 超时")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # ---- 主流程 ----
    def run(self, messages: Sequence[Message], *,
            stream: bool = True) -> Iterator[AgentEvent]:
        """跑完整个循环，逐个 yield 事件。

        调用方可以从事件里拿正文增量、工具调用过程，以及最终的 :class:`Final`。
        """
        convo: List[Message] = list(messages)
        if self.system_prompt and not any(m.role == "system" for m in convo):
            convo.insert(0, Message("system", self.system_prompt))

        steps = 0
        last_stats: Optional[GenerationStats] = None
        final_text = ""
        finish_reason = "stop"
        #: 本轮运行里累计发起过的工具调用（供调用方查看，**不影响 finish_reason**）
        pending_calls: List[ToolCall] = []

        for step in range(1, self.config.max_steps + 1):
            steps = step
            last_step = step >= self.config.max_steps
            yield StepStarted(step=step, max_steps=self.config.max_steps)

            # **每一轮都传完全相同的工具声明**，包括最后一轮。
            #
            # 曾经在最后一轮把工具声明撤掉，想逼模型用自然语言收尾。但那会
            # 破坏 KV 缓存：工具声明注入在第一个 system 轮次内部，也就是
            # prompt 的最开头，撤掉它等于从第 87 个字符起就与前面几轮不同。
            # 实测（同一对话，只有这一处不同）：
            #     带工具声明  in=226  prefill=346ms   → 1.5 ms/token
            #     撤掉声明    in= 94  prefill=771ms   → 8.2 ms/token（慢 5.5 倍）
            # token 少一半多，prefill 反而更慢 —— 缓存完全没命中。
            #
            # 步数上限仍然保留（循环必须能终止），但只用来决定**何时停止**，
            # 不再去改模型看到的东西。用满步数时模型可能停在"还想调工具"的
            # 状态，那就如实交给调用方（Final.tool_calls 里能看出来）。
            #
            # 参照实现佐证：DSH 全包（dsh-agent-loop 等）搜不到 maxSteps /
            # stepLimit / maxIterations 之类的概念 —— 它不做这种事。
            prompt = self._render(convo, tools=self.tools.openai_schemas())

            flt = StreamFilter()
            raw_parts: List[str] = []
            visible_parts: List[str] = []
            step_stats: Optional[GenerationStats] = None
            step_finish = "stop"

            for chunk in self.backend.generate(
                    GenerationRequest(prompt=prompt, params=self.params)):
                if chunk.text:
                    raw_parts.append(chunk.text)
                    piece = flt.feed(chunk.text)
                    if piece:
                        visible_parts.append(piece)
                        if stream:
                            yield TextDelta(piece)
                if chunk.stats is not None:
                    step_stats = chunk.stats
                if chunk.finish_reason:
                    step_finish = chunk.finish_reason

            tail = flt.flush()
            if tail:
                visible_parts.append(tail)
                if stream:
                    yield TextDelta(tail)

            raw = "".join(raw_parts)
            parsed = parse_tool_calls(raw)
            last_stats = step_stats or last_stats
            finish_reason = step_finish

            if last_step or not parsed.tool_calls:
                final_text = "".join(visible_parts).strip()
                # 写回历史，供上层做多轮对话
                convo.append(Message("assistant", final_text))
                break

            # 模型发了几个就执行几个 —— 截断等于丢弃模型输出
            calls = parsed.tool_calls
            pending_calls.extend(calls)
            convo.append(Message("assistant", parsed.text, tool_calls=tuple(calls)))

            for call in calls:
                yield ToolCallReady(call=call, step=step)
                result, ms = self._execute(call)
                convo.append(Message("tool", result.content,
                                     tool_call_id=result.tool_call_id, name=result.name))
                yield ToolCallDone(result=result, duration_ms=ms, step=step)

        # finish_reason 如实使用**最后一轮**的引擎推断值，不做任何改写。
        #
        # 曾经有过 `if finish_reason == "length" and pending_calls:
        # finish_reason = "tool_calls"` —— 那是错的，两个原因：
        #   1. pending_calls 累积的是整个运行过程的调用，而 finish_reason
        #      描述的是最后一轮怎么结束的 —— 两件不同的事被混在一起；
        #   2. 后果是误导：最后一轮明明是被 max_tokens 截断（半句话），
        #      却被报成 tool_calls。按 pi-ai 的映射 tool_calls → stopReason
        #      "toolUse"，等于告诉调用方「模型还想执行工具」。
        #
        # 调用方若需要知道本次运行用过哪些工具，看 Final.tool_calls 即可。
        yield Final(text=final_text, finish_reason=finish_reason,
                    tool_calls=pending_calls, steps=steps, stats=last_stats,
                    messages=convo)

    def _render(self, messages: Sequence[Message],
                tools: Optional[List[Dict[str, Any]]]) -> str:
        """渲染完整消息 —— **不裁剪历史，也不追加任何指令**。

        两件事都不做，都是有意的：

        * **不裁剪历史**：静默丢弃会让模型在调用方不知情的情况下换掉上下文。
        * **不追加「你必须调用工具」之类的文字**：那等于往 prompt 里塞调用方
          没写的指令。查过 llama.cpp 的做法（源码见文档）：工具格式说明来自
          **模型自带的 chat template**，框架只负责应用；而「强制调用」用
          grammar 做 token 级约束，且只在调用方明确要求 `tool_choice: required`
          时启用（`auto` 下是 lazy grammar，从不强迫模型）。框架自己在 prompt
          里塞「你必须」既没有先例，又只是建议而非保证。

        想让模型更爱用工具，请写在调用方自己的 system prompt 里 —— 那是
        调用方的地方。
        """
        return self.template.render(messages, tools=tools or None)
