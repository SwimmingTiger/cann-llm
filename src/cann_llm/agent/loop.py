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

    为什么需要它：工具调用是**内部协议**，用户不该看到。而流式场景下
    我们是边收边发，只能在收到足够字符后才能判断"这是不是工具标记的开头"。
    这里最多延迟 ``len('<tool_call>') - 1`` 个字符，实际约一个 token。
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
    #: 单步内最多执行几次工具调用（防止模型一次吐一堆）
    max_calls_per_step: int = 4
    #: 工具结果回填时截断长度，避免撑爆上下文
    max_result_chars: int = 4000
    #: 是否把工具调用过程也流给用户（作为提示，不含协议标记）
    announce_tool_calls: bool = True
    #: 在工具说明后追加"该用就用、别向用户索要工具能给的信息"的强指令。
    #: 实测对小模型是决定性的（同一问题 0/4 → 4/4），故默认开启。
    force_tool_use: bool = True


# ------------------------------------------------------------------ 主循环

class AgentLoop:
    """把「生成 + 工具调用」串成循环。"""

    def __init__(self, backend: EngineBackend, template: ChatTemplate,
                 tools: ToolRegistry, *, config: Optional[AgentConfig] = None,
                 system_prompt: Optional[str] = None,
                 params: Optional[GenerationParams] = None,
                 max_prompt_tokens: int = 1800):
        self.backend = backend
        self.template = template
        self.tools = tools
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.params = params or GenerationParams()
        self.max_prompt_tokens = max_prompt_tokens

    # ---- 工具执行 ----
    def _execute(self, call: ToolCall) -> tuple[ToolResult, float]:
        t0 = time.time()
        tool = self.tools.get(call.name) if call.name else None

        if not call.name:
            raw = call.arguments.get("__unparsable__", call.raw)
            return (ToolResult(call.id, "", f"你上一条工具调用格式无法解析：{raw!r}。"
                                            "请严格按 <tool_call>{\"name\": ..., "
                                            "\"arguments\": {...}}</tool_call> 重新输出。",
                               ok=False), 0.0)
        if tool is None:
            return (ToolResult(call.id, call.name,
                               f"没有名为 {call.name!r} 的工具。可用工具："
                               f"{', '.join(self.tools.names())}。"
                               "请改用其中某个工具重新调用。", ok=False), 0.0)

        try:
            if tool.timeout_s:
                value = self._call_with_timeout(tool, call.arguments, tool.timeout_s)
            else:
                value = tool.call(call.arguments)
            content = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=str)
            ok = True
        except (ToolError, PermissionError, KeyError) as e:
            content, ok = f"工具调用失败：{e}", False
        except Exception as e:                    # noqa: BLE001 - 工具是外部代码
            content, ok = f"工具执行异常：{type(e).__name__}: {e}", False

        if not ok:
            # 实测：小模型拿到"失败"后倾向于直接编一个答案，而不是重试。
            # 显式要求它修正参数后重新调用，能显著改善。
            content += ("\n请修正参数后重新调用同一个工具；"
                        "在拿到成功结果之前不要凭猜测作答。")

        limit = self.config.max_result_chars
        if len(content) > limit:
            content = content[:limit] + f"…（已截断，原长 {len(content)} 字符）"
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
        pending_calls: List[ToolCall] = []

        for step in range(1, self.config.max_steps + 1):
            steps = step
            last_step = step >= self.config.max_steps
            yield StepStarted(step=step, max_steps=self.config.max_steps)

            # 最后一轮不再给工具，逼模型用自然语言收尾
            active_tools = None if last_step else self.tools.openai_schemas()
            prompt = self._render(convo, tools=active_tools)

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
                break

            calls = parsed.tool_calls[:self.config.max_calls_per_step]
            pending_calls.extend(calls)
            convo.append(Message("assistant", parsed.text, tool_calls=tuple(calls)))

            for call in calls:
                yield ToolCallReady(call=call, step=step)
                result, ms = self._execute(call)
                convo.append(Message("tool", result.content,
                                     tool_call_id=result.tool_call_id, name=result.name))
                yield ToolCallDone(result=result, duration_ms=ms, step=step)

        if finish_reason == "length" and pending_calls:
            finish_reason = "tool_calls"
        yield Final(text=final_text, finish_reason=finish_reason,
                    tool_calls=pending_calls, steps=steps, stats=last_stats,
                    messages=convo)

    #: 追加在工具说明之后的强指令（针对小模型"向用户索要工具能提供的信息"的失败模式）
    TOOL_USE_DIRECTIVE = (
        "\n\nIMPORTANT: If the user's request can be answered by any of the tools above, "
        "you MUST emit the tool call immediately. Never ask the user for information that "
        "a tool can provide, and never answer from memory when a tool applies."
    )

    def _render(self, messages: Sequence[Message],
                tools: Optional[List[Dict[str, Any]]]) -> str:
        if tools and self.config.force_tool_use:
            tools = [dict(t) for t in tools]
            # 把指令挂在最后一个工具的 description 上（模板会原样渲染）
            fn = tools[-1].setdefault("function", {})
            fn["description"] = (fn.get("description", "") or "") + self.TOOL_USE_DIRECTIVE
        rendered = self.template.render(messages, tools=tools or None)
        est = self.backend.count_prompt_tokens(rendered)
        if self.max_prompt_tokens > 0 and est > self.max_prompt_tokens:
            # 从最旧的 user 轮开始丢，保留 system 与最近的工具往返
            trimmed = list(messages)
            while len(trimmed) > 2:
                idx = 1 if trimmed[0].role == "system" else 0
                del trimmed[idx]
                rendered = self.template.render(trimmed, tools=tools or None)
                if self.backend.count_prompt_tokens(rendered) <= self.max_prompt_tokens:
                    break
        return rendered
