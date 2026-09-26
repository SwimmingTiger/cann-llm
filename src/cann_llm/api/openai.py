"""OpenAI 兼容协议的请求解析与响应构造。

**本模块不涉及 HTTP**，只做 dict <-> 项目类型的双向映射，因此可以单独单测，
也方便将来换成 FastAPI 时复用。

兼容范围与差异见 ``docs/openai-api.md``。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import CannLlmError, InvalidRequestError
from ..types import GenerationParams
from ..chat.template import Message

# ------------------------------------------------------------------ 请求


@dataclass
class ChatCompletionRequest:
    """``POST /v1/chat/completions`` 的请求体。"""

    model: str
    messages: List[Message]
    stream: bool = False
    params: GenerationParams = field(default_factory=GenerationParams)
    #: OpenAI 格式的工具声明（原样透传给模板）
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[str] = None
    # 记录被接受但未生效的字段，便于在响应头/日志里说明
    ignored: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    #: 引擎一次只能产一路，n>1 直接拒绝
    n: int = 1

    @property
    def tool_names(self) -> List[str]:
        out = []
        for t in self.tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                out.append(str(fn["name"]))
        return out

    @property
    def requires_tool_call(self) -> bool:
        return self.tool_choice == "required"

    @property
    def forbids_tool_call(self) -> bool:
        return self.tool_choice == "none"


#: 永远拒绝：这些字段一旦被忽略，调用方会拿到"看起来正常但答案是错的"结果。
#:   * image_url —— 模型看不见图片，照常回答等于骗人
#:   * logprobs  —— 响应里不会出现该字段，客户端解析会出错
_ALWAYS_REJECT = {
    "logprobs": "本服务不返回 logprobs（响应里不会有该字段）",
    "top_logprobs": "本服务不返回 logprobs",
}

#: 默认「接受但忽略」，并在响应头 X-Cann-Llm-Ignored-Fields 里回报、服务端日志记一行。
#: 之所以不直接报错：绝大多数现代客户端（Continue / Cline / LangChain 等）
#: 即使只是普通聊天也会带上 tools，一律 400 会让这些客户端完全用不了。
#: 需要「宁可报错也别给我假象」的场景，把 server.reject_unsupported 设成 true。
_IGNORABLE = {
    # 注：tools 不在忽略之列了 —— 它是 agent 能力的入口，见 parse_chat_request。
    # tool_choice 目前只支持 "auto" 的语义（模型自行决定），其余取值忽略。
    "tool_choice": "仅支持 auto 语义（由模型自行决定是否调用）",
    "function_call": "请改用 tools",
    "frequency_penalty": "引擎无对应能力",
    "presence_penalty": "引擎无对应能力",
    "response_format": "本服务不强制输出格式",
    "logit_bias": "引擎无对应能力",
    "user": "无实际作用",
}


def _as_text(content: Any) -> str:
    """OpenAI 允许 content 是字符串或「内容块数组」，这里统一成字符串。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in (None, "text") and "text" in part:
                    out.append(str(part["text"]))
                elif part.get("type") == "image_url":
                    raise InvalidRequestError("本服务为纯文本模型，不支持 image_url")
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    raise InvalidRequestError(f"content 类型不支持: {type(content).__name__}")


def parse_chat_request(body: Dict[str, Any], *, default_model: str = "",
                       strict: bool = False) -> ChatCompletionRequest:
    """把 OpenAI 风格的请求体解析成本项目的类型。

    :param strict: ``True`` 时，所有本服务无法实现的字段都直接报错；
        默认 ``False`` 则把 :data:`_IGNORABLE` 里的字段记录到 ``ignored``
        并继续（否则带 tools 的客户端会完全用不了）。
        :data:`_ALWAYS_REJECT` 里的字段无论哪种模式都报错。
    """
    if not isinstance(body, dict):
        raise InvalidRequestError("请求体必须是 JSON 对象")

    for key, why in _ALWAYS_REJECT.items():
        if body.get(key) not in (None, False, [], {}):
            raise InvalidRequestError(f"字段 {key} 不支持：{why}")

    raw_msgs = body.get("messages")
    if not isinstance(raw_msgs, list) or not raw_msgs:
        raise InvalidRequestError("messages 必须是非空数组")

    messages: List[Message] = []
    for i, m in enumerate(raw_msgs):
        if not isinstance(m, dict) or "role" not in m:
            raise InvalidRequestError(f"messages[{i}] 缺少 role")
        messages.append(_parse_message(m, i))
    if not any(m.role != "system" for m in messages):
        raise InvalidRequestError("messages 至少需要一条非 system 消息")

    n = body.get("n", 1)
    if not isinstance(n, int) or n != 1:
        raise InvalidRequestError("本服务只支持 n=1（引擎一次只产一路）")

    tools = _parse_tools(body.get("tools"))
    tool_choice_raw = body.get("tool_choice")
    tool_choice: Optional[str] = None
    if isinstance(tool_choice_raw, str):
        if tool_choice_raw not in ("auto", "none", "required"):
            raise InvalidRequestError(
                f"tool_choice 只支持 auto/none/required，收到 {tool_choice_raw!r}")
        tool_choice = tool_choice_raw
    elif isinstance(tool_choice_raw, dict):
        # 指定具体函数：本服务不支持强制某一个，退化成 required
        tool_choice = "required"

    if body.get("stream") and body.get("stream_options"):
        # 只支持 include_usage，其它忽略
        so = body["stream_options"]
        if isinstance(so, dict) and set(so) - {"include_usage"}:
            pass

    ignored: List[str] = []
    for key, why in _IGNORABLE.items():
        if body.get(key) in (None, [], {}):
            continue
        # tool_choice="auto" 就是本服务的行为，不算被忽略
        if key == "tool_choice" and body.get(key) == "auto":
            continue
        if strict:
            raise InvalidRequestError(f"字段 {key} 不支持：{why}")
        ignored.append(key)

    params = _params_from_body(body)

    return ChatCompletionRequest(
        model=str(body.get("model") or default_model),
        messages=messages,
        stream=bool(body.get("stream", False)),
        params=params,
        tools=tools,
        tool_choice=tool_choice,
        ignored=ignored,
        raw=body,
        n=1,
    )


def _parse_tools(raw: Any) -> List[Dict[str, Any]]:
    """校验并归一化 OpenAI ``tools`` 数组。"""
    if raw in (None, [], {}):
        return []
    if not isinstance(raw, list):
        raise InvalidRequestError("tools 必须是数组")
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise InvalidRequestError(f"tools[{i}] 必须是对象")
        fn = t.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            raise InvalidRequestError(f"tools[{i}] 缺少 function.name")
        if t.get("type", "function") != "function":
            raise InvalidRequestError(f"tools[{i}].type 只支持 function")
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise InvalidRequestError(f"tools[{i}].function.parameters 必须是对象")
        out.append({
            "type": "function",
            "function": {
                "name": str(fn["name"]),
                "description": str(fn.get("description") or ""),
                "parameters": params or {"type": "object", "properties": {}},
            },
        })
    return out


def _parse_message(m: Dict[str, Any], index: int) -> Message:
    """把一条 OpenAI 消息转成 :class:`Message`（含工具相关字段）。"""
    from ..types import ToolCall

    role = str(m["role"])
    content = _as_text(m.get("content"))

    calls: Tuple[ToolCall, ...] = ()
    raw_calls = m.get("tool_calls")
    if raw_calls:
        if not isinstance(raw_calls, list):
            raise InvalidRequestError(f"messages[{index}].tool_calls 必须是数组")
        parsed = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                # OpenAI 规范里 arguments 是 JSON 字符串
                import json as _json
                try:
                    args = _json.loads(args) if args.strip() else {}
                except _json.JSONDecodeError:
                    args = {"__raw__": args}
            if not isinstance(args, dict):
                args = {"__raw__": args}
            parsed.append(ToolCall(name=str(fn.get("name") or ""), arguments=args,
                                   id=str(tc.get("id") or "")))
        calls = tuple(parsed)

    return Message(role=role, content=content, tool_calls=calls,
                   tool_call_id=(str(m["tool_call_id"]) if m.get("tool_call_id") else None),
                   name=(str(m["name"]) if m.get("name") else None))


def _stop_list(body: Dict[str, Any]) -> Tuple[str, ...]:
    stop = body.get("stop")
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    if isinstance(stop, list):
        if len(stop) > 4:
            raise InvalidRequestError("stop 最多 4 个")
        return tuple(str(s) for s in stop)
    raise InvalidRequestError("stop 必须是字符串或字符串数组")


def _params_from_body(body: Dict[str, Any], *, base: Optional[GenerationParams] = None) -> GenerationParams:
    """从请求体构造采样参数；未给出的字段沿用 base（后端默认）。"""
    p = base or GenerationParams()
    kw: Dict[str, Any] = {
        # OpenAI 新字段 max_completion_tokens 优先
        "max_tokens": body.get("max_completion_tokens", body.get("max_tokens", p.max_tokens)),
        "temperature": body.get("temperature", p.temperature),
        "top_p": body.get("top_p", p.top_p),
        "repetition_penalty": body.get("repetition_penalty", p.repetition_penalty),
        "seed": body.get("seed", p.seed),
        "stop": _stop_list(body) or p.stop,
    }
    # OpenAI 没有 top_k，但它在本引擎里很有用，作为扩展字段支持
    if "top_k" in body:
        kw["top_k"] = body["top_k"]
    else:
        kw["top_k"] = p.top_k
    try:
        return GenerationParams(**kw)
    except (ValueError, TypeError) as e:
        raise InvalidRequestError(f"采样参数非法: {e}") from e


@dataclass
class CompletionRequest:
    """``POST /v1/completions`` 的请求体。"""

    model: str
    prompt: str
    stream: bool = False
    params: GenerationParams = field(default_factory=GenerationParams)
    ignored: List[str] = field(default_factory=list)


def parse_completion_request(body: Dict[str, Any], *,
                             default_model: str = "",
                             strict: bool = False) -> CompletionRequest:
    if not isinstance(body, dict):
        raise InvalidRequestError("请求体必须是 JSON 对象")
    for key, why in _ALWAYS_REJECT.items():
        if body.get(key) not in (None, False, [], {}):
            raise InvalidRequestError(f"字段 {key} 不支持：{why}")
    ignored_fields = []
    for key, why in _IGNORABLE.items():
        if body.get(key) in (None, [], {}):
            continue
        if strict:
            raise InvalidRequestError(f"字段 {key} 不支持：{why}")
        ignored_fields.append(key)
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise InvalidRequestError("本服务只支持单个 prompt（不支持 batch）")
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt:
        raise InvalidRequestError("prompt 必须是非空字符串")
    n = body.get("n", 1)
    if not isinstance(n, int) or n != 1:
        raise InvalidRequestError("本服务只支持 n=1")
    return CompletionRequest(
        model=str(body.get("model") or default_model),
        prompt=prompt,
        stream=bool(body.get("stream", False)),
        params=_params_from_body(body),
        ignored=ignored_fields,
    )


# ------------------------------------------------------------------ 响应


def describe_ignored(fields: Iterable[str]) -> str:
    """把被忽略的字段拼成一句人话，用于日志。"""
    msgs = [f"{f}（{_IGNORABLE.get(f, '')}）" for f in fields]
    return "已忽略：" + "、".join(msgs) if msgs else ""


def new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())


def models_response(model_ids: Iterable[str], *, created: Optional[int] = None) -> Dict[str, Any]:
    """``GET /v1/models``。"""
    ts = created if created is not None else now()
    data = [{"id": mid, "object": "model", "created": ts, "owned_by": "cann-llm"}
            for mid in model_ids]
    return {"object": "list", "data": data}


def chat_completion_response(*, req_id: str, model: str, text: str,
                             finish_reason: str, usage: Optional[Dict[str, int]] = None,
                             created: Optional[int] = None,
                             tool_calls: Optional[Sequence[Any]] = None,
                             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """非流式 ``chat.completion``。

    :param tool_calls: 非空时按 OpenAI 规范放进 message（客户端需自行执行），
        并把 ``content`` 置为 null、``finish_reason`` 置为 ``tool_calls``。
    :param extra: 非标准扩展字段（以 ``x_`` 开头），客户端一般会忽略。
    """
    message: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = [tc.to_openai() for tc in tool_calls]
        message["content"] = text or None
        finish_reason = "tool_calls"
    payload = {
        "id": req_id,
        "object": "chat.completion",
        "created": created if created is not None else now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    if extra:
        payload.update(extra)
    return payload


def chat_completion_chunk(*, req_id: str, model: str, created: int,
                          delta: Dict[str, Any],
                          finish_reason: Optional[str] = None) -> Dict[str, Any]:
    """流式 ``chat.completion.chunk``。"""
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def tool_call_delta(index: int, *, call_id: Optional[str] = None,
                    name: Optional[str] = None,
                    arguments: Optional[str] = None) -> Dict[str, Any]:
    """构造 ``delta.tool_calls`` 条目（OpenAI 流式工具调用的形状）。"""
    entry: Dict[str, Any] = {"index": index}
    if call_id is not None:
        entry["id"] = call_id
        entry["type"] = "function"
    fn: Dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    if fn:
        entry["function"] = fn
    return entry


def completion_response(*, req_id: str, model: str, text: str, finish_reason: str,
                        usage: Optional[Dict[str, int]] = None,
                        created: Optional[int] = None) -> Dict[str, Any]:
    """非流式 ``text_completion``。"""
    return {
        "id": req_id,
        "object": "text_completion",
        "created": created if created is not None else now(),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None,
                     "finish_reason": finish_reason}],
        **({"usage": usage} if usage else {}),
    }


def completion_chunk(*, req_id: str, model: str, created: int, text: str,
                     finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None,
                     "finish_reason": finish_reason}],
    }


def error_payload(message: str, *, err_type: str = "invalid_request_error",
                  param: Optional[str] = None, code: Optional[str] = None) -> Dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}


def error_from_exception(exc: BaseException) -> Tuple[int, Dict[str, Any]]:
    """把项目异常映射成 (状态码, OpenAI 风格错误体)。"""
    if isinstance(exc, CannLlmError):
        return exc.http_status, error_payload(str(exc), err_type=exc.error_type)
    return 500, error_payload(f"{type(exc).__name__}: {exc}", err_type="internal_error")


def usage_payload(stats) -> Dict[str, int]:
    return stats.as_openai_usage() if stats is not None else {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def sse_data(payload: Any) -> bytes:
    """SSE 帧：``data: <json>\\n\\n``。"""
    import json

    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"
