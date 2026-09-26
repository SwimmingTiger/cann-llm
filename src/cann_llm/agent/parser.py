"""从模型输出里解析工具调用。

小模型的输出一定会畸形，所以这里刻意做得宽容：

* ``<tool_call>…</tool_call>`` 正常配对
* 只有开标签、没有闭标签（被 max_tokens 截断）
* JSON 被包在 ```` ```json ```` 代码块里
* 直接吐裸 JSON 而不带标签（模型常见"忘记格式"）
* ``arguments`` 是 JSON 字符串而不是对象（双重编码）
* 单引号 / 尾逗号 / 前后夹带解释文字

**解析失败不会丢信息**：无法解析的片段会以 ``ok=False`` 的
:class:`~cann_llm.types.ToolCall` 形式带回去（``raw`` 里是原文），
由 agent 循环决定怎么向模型反馈，而不是静默吞掉。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import ToolCall

__all__ = ["ParsedOutput", "parse_tool_calls", "repair_json"]

#: 匹配 <tool_call>…</tool_call>；闭标签可选（可能被截断）
_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
#: ```json … ``` / ``` … ```
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```?$", re.DOTALL)

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ParsedOutput:
    """解析结果。

    :param text: 去掉工具调用片段之后的文本（即"模型对用户说的话"）
    :param tool_calls: 解析出的工具调用
    :param had_invalid: 是否存在解析失败的调用片段
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    had_invalid: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def finish_reason_hint(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


def repair_json(raw: str) -> Optional[Any]:
    """尽力修复常见的 JSON 畸形，修不好返回 ``None``。"""
    raw = raw.strip()
    if not raw:
        return None

    # 去掉代码围栏
    m = _FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()

    # 直接试
    for candidate in (raw,
                      _TRAILING_COMMA_RE.sub(r"\1", raw),
                      _TRAILING_COMMA_RE.sub(r"\1", raw).replace("'", '"')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 截断的 JSON：尝试补齐括号
    for opener, closer in (("{", "}"), ("[", "]")):
        if raw.count(opener) > raw.count(closer):
            patched = raw + closer * (raw.count(opener) - raw.count(closer))
            try:
                return json.loads(_TRAILING_COMMA_RE.sub(r"\1", patched))
            except json.JSONDecodeError:
                pass

    # 从第一个 { 或 [ 开始截取到最后一个 } 或 ]
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        i, j = raw.find(start_ch), raw.rfind(end_ch)
        if 0 <= i < j:
            try:
                return json.loads(_TRAILING_COMMA_RE.sub(r"\1", raw[i:j + 1]))
            except json.JSONDecodeError:
                pass
    return None


def _normalize_arguments(args: Any) -> Dict[str, Any]:
    """把 arguments 归一到 dict。"""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        # 双重编码：arguments 是个 JSON 字符串
        parsed = repair_json(args)
        if isinstance(parsed, dict):
            return parsed
        return {"__raw__": args}
    if isinstance(args, list):
        return {"__raw__": args}
    return {"__raw__": args}


def _to_tool_call(obj: Any, raw: str) -> Optional[ToolCall]:
    """把解析出来的 JSON 对象转成 ToolCall。"""
    if not isinstance(obj, dict):
        return None

    # 常见几种键名
    name = obj.get("name") or obj.get("function") or obj.get("tool") or obj.get("tool_name")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))

    # OpenAI 风格嵌套: {"function": {"name": ..., "arguments": ...}}
    fn = obj.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or name
        args = fn.get("arguments", args)

    if isinstance(name, dict):          # arguments 里套了 name 的畸形情况
        args = name.get("arguments", args)
        name = name.get("name")

    if not isinstance(name, str) or not name.strip():
        return None

    call_id = obj.get("id") if isinstance(obj.get("id"), str) else ""
    return ToolCall(name=name.strip(), arguments=_normalize_arguments(args),
                    id=call_id or "", raw=raw.strip())


def parse_tool_calls(text: str) -> ParsedOutput:
    """解析模型输出，只认 ``<tool_call>…</tool_call>`` 标签里的内容。

    **不做「裸 JSON 容错」**。曾经尝试过：模型没写标签时，把形如
    ``{"name": ..., "arguments": ...}`` 的 JSON 也当成工具调用。实测它会：

    * 把用户**明确要求生成**的 JSON 代码当成调用，并从回答里删掉 ——
      实测用户说"给我一个 JSON-RPC 请求体的例子"、模型给出
      ` ```json {"name": "getUserProfile", "arguments": {...}} ``` `，
      结果代码块被掏空成 ` ```json\n\n``` `；
    * 把模型举例说明用的 JSON 从正文里抹掉，句子被挖空
      （``The tool takes {"name": ...} as input.`` → ``The tool takes  as input.``）。

    根因是它替模型认定了并不存在的调用意图，并把模型写下的文本从输出里删掉。
    **模型忘记标签属于它自身的格式偏离**：推理框架应当如实呈现，由调用方决定
    怎么办 —— 想让它稳定用标签，该改的是 prompt 里的格式说明，而不是在这里
    猜。``<tool_call>`` 标签的存在本身就是"这是一个调用"的协议信号。
    """
    if not text:
        return ParsedOutput()

    calls: List[ToolCall] = []
    consumed: List[Tuple[int, int]] = []
    had_invalid = False

    for m in _BLOCK_RE.finditer(text):
        inner = m.group(1).strip()
        consumed.append((m.start(), m.end()))
        if not inner:
            had_invalid = True
            continue

        payload = repair_json(inner)
        candidates: Sequence[Any]
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            candidates = [payload]
        else:
            candidates = []

        got = False
        for cand in candidates:
            tc = _to_tool_call(cand, inner)
            if tc is not None:
                calls.append(tc)
                got = True
        if not got:
            # 解析失败也要把原文带回去，让 agent 能告诉模型哪里错了
            had_invalid = True
            calls.append(ToolCall(name="", arguments={"__unparsable__": inner},
                                  raw=inner))

    # 去掉消费掉的片段，剩下的就是给用户看的话
    remaining = text
    for start, end in sorted(consumed, key=lambda x: -x[0]):
        remaining = remaining[:start] + remaining[end:]
    remaining = remaining.strip()

    # 统一补 id
    fixed: List[ToolCall] = []
    for i, tc in enumerate(calls):
        if tc.id:
            fixed.append(tc)
        else:
            fixed.append(ToolCall(name=tc.name, arguments=tc.arguments,
                                  id=f"call_{uuid.uuid4().hex[:8]}", raw=tc.raw))
    return ParsedOutput(text=remaining, tool_calls=fixed, had_invalid=had_invalid)
