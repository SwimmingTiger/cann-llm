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
    # 记录被接受但未生效的字段，便于在响应头/日志里说明
    ignored: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    #: 引擎一次只能产一路，n>1 直接拒绝
    n: int = 1


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
    "tools": "暂不支持 function calling",
    "functions": "暂不支持 function calling",
    "tool_choice": "暂不支持 tools",
    "function_call": "暂不支持 function calling",
    "parallel_tool_calls": "暂不支持 tools",
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
        messages.append(Message(role=str(m["role"]), content=_as_text(m.get("content"))))
    if not any(m.role != "system" for m in messages):
        raise InvalidRequestError("messages 至少需要一条非 system 消息")

    n = body.get("n", 1)
    if not isinstance(n, int) or n != 1:
        raise InvalidRequestError("本服务只支持 n=1（引擎一次只产一路）")

    if body.get("stream") and body.get("stream_options"):
        # 只支持 include_usage，其它忽略
        so = body["stream_options"]
        if isinstance(so, dict) and set(so) - {"include_usage"}:
            pass

    ignored: List[str] = []
    for key, why in _IGNORABLE.items():
        if body.get(key) in (None, [], {}):
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
        ignored=ignored,
        raw=body,
        n=1,
    )


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
                             created: Optional[int] = None) -> Dict[str, Any]:
    """非流式 ``chat.completion``。"""
    payload = {
        "id": req_id,
        "object": "chat.completion",
        "created": created if created is not None else now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage
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
