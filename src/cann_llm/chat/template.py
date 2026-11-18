"""对话模板。

模板负责把「消息列表」渲染成模型能吃的纯文本。CANN LLM Engine 只接受文本，
不接受结构化消息，所以这一步必须在客户端完成。

内置：
* ``chatml`` —— Qwen2.5 / Qwen3 系列（``<|im_start|>role\\n…<|im_end|>``），当前默认
* ``plain``  —— 不做任何包装，直接把消息内容拼起来（调试 / 续写用）
* ``gemma``  —— Gemma 系列（``<start_of_turn>user…<end_of_turn>``，**没有 system 角色**）

要加新模板（GLM、HunYuan 等）：实现 :class:`ChatTemplate` 并用
:func:`register_template` 注册即可，上层零改动。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..errors import InvalidRequestError
from ..types import ToolCall

#: 允许的角色（与 OpenAI 对齐；tool 先放行，等接入工具调用再细化）
ROLES = ("system", "user", "assistant", "tool")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class Message:
    """一条对话消息。

    ``tool_calls`` 只对 assistant 有意义（模型请求调用的工具）；
    ``tool_call_id`` / ``name`` 只对 role=tool 有意义（工具结果回填）。
    """

    role: str
    content: str
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise InvalidRequestError(f"不支持的角色 {self.role!r}；可用：{', '.join(ROLES)}")


class ChatTemplate(ABC):
    """对话模板协议。"""

    name: str = "base"

    @abstractmethod
    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        """渲染成提示文本。

        :param add_generation_prompt: 是否在末尾追加「轮到 assistant 说话」的引导段。
        :param tools: OpenAI 格式的工具声明；非空时会把「可用工具 + 调用格式」
            注入 system 段（与 Qwen 官方 chat template 的行为一致）。
        """

    def stop_strings(self) -> Sequence[str]:
        """该模板对应的停止串（会额外传给引擎的 stop_sequence）。"""
        return ()

    @property
    def default_system(self) -> str:
        return "You are a helpful assistant."


#: Qwen 官方 chat template 里注入的工具说明（逐字对齐，改动需谨慎）
TOOLS_PREAMBLE = (
    "\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
    "\n<tools>"
)
TOOLS_EPILOGUE = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def render_tools_block(tools: Sequence[Dict[str, Any]]) -> str:
    """把工具声明渲染成官方模板要求的 ``<tools>…</tools>`` 段。"""
    lines = [TOOLS_PREAMBLE]
    for t in tools:
        lines.append("\n" + json.dumps(t, ensure_ascii=False))
    lines.append(TOOLS_EPILOGUE)
    return "".join(lines)


def render_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    """assistant 消息里的工具调用段。"""
    out = []
    for tc in tool_calls:
        payload = {"name": tc.name, "arguments": tc.arguments}
        out.append(f"\n{TOOL_CALL_OPEN}\n"
                   f"{json.dumps(payload, ensure_ascii=False)}"
                   f"\n{TOOL_CALL_CLOSE}")
    return "".join(out)


class ChatMLTemplate(ChatTemplate):
    """Qwen2.5 / Qwen3 的 ChatML 模板（含 function calling 支持）。"""

    name = "chatml"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        parts: List[str] = []
        tools_block = render_tools_block(tools) if tools else ""
        tools_emitted = False
        n = len(messages)
        i = 0
        while i < n:
            m = messages[i]

            if m.role == "tool":
                # 连续多条 tool 结果合并进同一个 user 轮次（对齐官方模板）
                parts.append(f"{IM_START}user")
                while i < n and messages[i].role == "tool":
                    parts.append(f"\n{TOOL_RESPONSE_OPEN}\n"
                                 f"{messages[i].content}\n{TOOL_RESPONSE_CLOSE}")
                    i += 1
                parts.append(f"{IM_END}\n")
                continue

            content = m.content
            is_first_system = m.role == "system" and not tools_emitted
            if is_first_system and tools_block:
                content = content + tools_block
                tools_emitted = True
            if m.role == "assistant" and m.tool_calls:
                content = content + render_tool_calls(m.tool_calls)
            parts.append(f"{IM_START}{m.role}\n{content}{IM_END}\n")
            i += 1

        # 没有 system 消息但带了工具 → 补一个 system 轮次承载工具说明
        if tools_block and not tools_emitted:
            parts.insert(0, f"{IM_START}system\n{tools_block.lstrip(chr(10))}{IM_END}\n")

        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)

    def stop_strings(self) -> Sequence[str]:
        return (IM_END,)

    @property
    def default_system(self) -> str:
        # 旧行为：chatml 的默认系统提示就是 Qwen 那句（config 里不再写死 ✓）
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    @property
    def tool_call_tags(self) -> Sequence[str]:
        return (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)


#: gemma4 结构化数据里的类型名（大写）
_GEMMA4_TYPE_NAMES = {
    "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def gemma4_quote(text: str) -> str:
    """gemma4 结构化数据里的字符串定界：``<|"|>…<|"|>``（官方规定）。"""
    return f'<|"|>{text}<|"|>'


def gemma4_value(value: Any) -> str:
    """把 Python 值渲染成 gemma4 结构化字面量（字符串用 ``<|"|>`` 包）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return gemma4_quote(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(gemma4_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{gemma4_value(v)}" for k, v in value.items()) + "}"
    return gemma4_quote(str(value))


def gemma4_tool_declaration(tool: Dict[str, Any]) -> str:
    """OpenAI 工具声明 → gemma4 ``<|tool>declaration:…<tool|>``。"""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = str(fn.get("name") or "")
    desc = str(fn.get("description") or "")
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    inner: List[str] = []
    for pname, psch in props.items():
        psch = psch or {}
        ptype = _GEMMA4_TYPE_NAMES.get(str(psch.get("type", "string")).lower(), "STRING")
        pdesc = str(psch.get("description") or "")
        inner.append(f"{pname}:{{description:{gemma4_quote(pdesc)},type:{gemma4_quote(ptype)}}}")
    if required:
        inner.append("required:" + gemma4_value([str(r) for r in required]))
    inner.append(f"type:{gemma4_quote('OBJECT')}")
    return (f"<|tool>declaration:{name}{{description:{gemma4_quote(desc)},"
            f"parameters:{{{','.join(inner)}}}}}<tool|>")


def gemma4_tool_call_block(call: ToolCall) -> str:
    """模型请求调用的渲染：``<|tool_call>call:name{…}<tool_call|>``。"""
    return f"<|tool_call>call:{call.name or ''}{gemma4_value(dict(call.arguments or {}))}<tool_call|>"


def gemma4_tool_response_block(name: str, result: Any) -> str:
    """工具执行结果回填：``<|tool_response>response:name{…}<tool_response|>``。"""
    return f"<|tool_response>response:{name or ''}{gemma4_value(result)}<tool_response|>"


class GemmaTemplate(ChatTemplate):
    """Gemma 系列的对话模板（**gemma4 词表**：``<|turn>`` / ``<turn|>``）。

    ⚠ gemma4 的 tokenizer 里**没有** ``<start_of_turn>`` / ``<end_of_turn>`` 这两个
    字符串（那是 gemma3 的标记）—— 写成 gemma3 形式会被 BPE 拆成十几段普通文本
    （实测 31 token → 14 token），模型行为随之变差 ✗。
    gemma4 的规范格式（tokenizer.json 里 105/106，且 generation_config 的
    ``eos_token_id=[1,106,50]`` 把 ``<turn|>`` 也算 EOS ✓）：

        <|turn>user\n{用户内容}<turn|>\n<|turn>model\n

    Gemma **没有 system 角色**：system 内容前置到第一条 user 前面 ✓（不丢信息，
    也不引入模型不认的标签）。结尾 ``<|turn>model\n`` 是「轮到模型说话」的引导段。
    """

    name = "gemma"

    START = "<|turn>"
    END = "<turn|>"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        """渲染 gemma4 对话；``tools`` 非空时注入官方 function calling 协议。

        协议（tokenizer.json 里 46/47/48/49/50/51 + 52 作字符串定界）：
        工具声明进 system 回合：``<|tool>declaration:name{…}<tool|>``；
        模型调用：``<|tool_call>call:name{…}<tool_call|>``；
        结果回填：``<|tool_response>response:name{…}<tool_response|>``。
        工具握手时 model 回合**保持打开**（不再补 ``<|turn>model``）✓。
        """
        system_parts: List[str] = []
        decls = ""
        if tools:
            system_parts = [m.content.strip() for m in messages
                            if m.role == "system" and (m.content or "").strip()]
            decls = "".join(gemma4_tool_declaration(t) for t in tools)
        parts: List[str] = []
        if system_parts or decls:
            parts.append(f"{self.START}system\n{chr(10).join(system_parts)}{decls}{self.END}\n")
        emitted_system = bool(system_parts or decls)

        open_model = False          # model 回合是否还没闭合（工具握手期间为 True）
        pending_names: List[str] = []
        system_prefix = ""
        for m in messages:
            if m.role == "system":
                if emitted_system:
                    continue        # 已并进上面的 system 回合
                system_prefix = system_prefix or (m.content or "").strip()
                continue
            if m.role == "user":
                if open_model:
                    parts.append(f"{self.END}\n")
                    open_model = False
                content = m.content or ""
                if system_prefix:
                    content = f"{system_prefix}\n\n{content}"
                    system_prefix = ""
                parts.append(f"{self.START}user\n{content}{self.END}\n")
                continue
            if m.role == "assistant":
                if not open_model:
                    parts.append(f"{self.START}model\n")
                    open_model = True
                parts.append(m.content or "")
                calls = list(m.tool_calls or ())
                for tc in calls:
                    parts.append(gemma4_tool_call_block(tc))
                if calls:
                    pending_names = [tc.name or "" for tc in calls]
                else:
                    parts.append(f"{self.END}\n")
                    open_model = False
                continue
            if m.role == "tool":
                if not open_model:
                    parts.append(f"{self.START}model\n")
                    open_model = True
                name = m.name or (pending_names.pop(0) if pending_names else "")
                raw = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
                try:
                    payload = json.loads(raw)
                except Exception:      # 非 JSON 就按字符串回填 ✓
                    payload = raw
                parts.append(gemma4_tool_response_block(name, payload))
                continue

        if system_prefix and not emitted_system:
            parts.insert(0, f"{self.START}user\n{system_prefix}{self.END}\n")
        if add_generation_prompt and not open_model:
            parts.append(f"{self.START}model\n")
        return "".join(parts)

    def stop_strings(self) -> Sequence[str]:
        return (self.END,)

    @property
    def default_system(self) -> str:
        return ""


class PlainTemplate(ChatTemplate):
    """不包装：把各条消息内容直接拼接。"""

    name = "plain"

    def render(self, messages: Sequence[Message], *,
               add_generation_prompt: bool = True,
               tools: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        return "".join(m.content for m in messages)


_REGISTRY: Dict[str, Callable[[], ChatTemplate]] = {}


def register_template(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


# 注册内置模板
register_template("chatml")(ChatMLTemplate)
register_template("plain")(PlainTemplate)
register_template("gemma")(GemmaTemplate)


def pick_template_name(configured: str, backend_name: str = "") -> str:
    """选模板名：**配置显式改过就尊重**，仍是默认 ``chatml`` 时跟随后端自报。

    后端的 ``ModelInfo.chat_template`` 是按模型给的（cllm/gemma 模型 → ``gemma``）。
    不跟随的话，``-b cllm`` 跑 gemma 会套 ChatML ⇒ 模型看到它不认识的标签 ✗
    （实测输出会跑偏）。
    """
    if configured and configured != "chatml":
        return configured
    return backend_name or configured or "chatml"


def get_template(name: str) -> ChatTemplate:
    if name not in _REGISTRY:
        raise InvalidRequestError(
            f"未知对话模板 {name!r}；可用：{', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]()


def available_templates() -> List[str]:
    return sorted(_REGISTRY)
