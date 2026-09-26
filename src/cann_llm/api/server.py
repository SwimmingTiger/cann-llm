#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI 兼容的 HTTP 推理服务。

刻意只用标准库 ``http.server``：鸿蒙设备上没装 FastAPI/uvicorn，而本服务的
瓶颈在 NPU 推理（~7 tok/s），HTTP 层的性能完全不是问题。想要 ASGI 时，
``api/openai.py`` 那层映射逻辑可以直接复用。

    cann-llm-server -d /path/to/model_dir --port 8000
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen2.5-1.5b","messages":[{"role":"user","content":"hi"}]}'
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .. import version
from ..backends import available_backends, create_backend
from ..backends.base import EngineBackend, SerializedBackend
from ..chat.template import get_template
from ..config import AppConfig, load_config
from ..agent.loop import StreamFilter
from ..agent.parser import parse_tool_calls
from ..errors import CannLlmError, InvalidRequestError
from ..types import GenerationParams
from . import openai as oa

#: 请求体上限（防止误发大文件把内存打满）
MAX_BODY_BYTES = 4 * 1024 * 1024

#: 路由表：路径 -> 允许的方法
ROUTES: Dict[str, Tuple[str, ...]] = {
    "/": ("GET",),
    "/healthz": ("GET",),
    "/health": ("GET",),
    "/v1/models": ("GET",),
    "/v1/chat/completions": ("POST",),
    "/v1/completions": ("POST",),
}

#: 这些子路径常被误当成 base_url，404 时给出针对性提示
_ENDPOINT_LIKE = ("/v1/chat/completions", "/v1/completions", "/v1/models")

BASE_URL_HINT = (
    "OpenAI 客户端的 base_url 只应到 /v1，例如 "
    "http://127.0.0.1:{port}/v1 —— 不要带上 /chat/completions 之类的端点路径。"
)


@dataclass
class AppState:
    """服务运行期共享状态。"""

    cfg: AppConfig
    backend: EngineBackend
    template: Any
    started_at: float = field(default_factory=time.time)

    def base_params(self) -> GenerationParams:
        mc = self.cfg.model
        return GenerationParams(
            max_tokens=mc.max_tokens, temperature=mc.temperature, top_k=mc.top_k,
            top_p=mc.top_p, repetition_penalty=mc.repetition_penalty,
            stop=tuple(self.template.stop_strings()))


def build_state(cfg: AppConfig) -> AppState:
    mc = cfg.model
    if not mc.model_dir:
        raise SystemExit("未指定模型目录（-d / CANN_LLM_MODEL__MODEL_DIR / 配置文件）")
    kw: Dict[str, Any] = {
        "model_dir": mc.model_dir,
        "model_id": mc.resolved_id,
        "context_length": mc.context_length,
        "default_params": GenerationParams(
            max_tokens=mc.max_tokens, temperature=mc.temperature, top_k=mc.top_k,
            top_p=mc.top_p, repetition_penalty=mc.repetition_penalty),
    }
    if mc.backend == "cann":
        kw["lib_path"] = os.environ.get("CANN_LLM_LIB", version.CANN_NDK_LIB)
    inner = create_backend(mc.backend, **kw)
    inner.load()
    # 引擎一次只能跑一路，统一在后端层串行化（见 backends/base.SerializedBackend）
    backend = SerializedBackend(inner, max_queue=cfg.server.max_queue,
                                timeout_s=cfg.server.queue_timeout_s)
    return AppState(cfg=cfg, backend=backend, template=get_template(mc.chat_template))


# ------------------------------------------------------------------ 处理器


class Handler(BaseHTTPRequestHandler):
    server_version = f"cann-llm/{version.__version__}"
    protocol_version = "HTTP/1.1"

    #: 本轮请求里被忽略的字段（由 parse_* 填充），会在响应头回报
    ignored_fields: Tuple[str, ...] = ()
    #: 额外的响应头。注意：必须在 send_response 之后、end_headers 之前发，
    #: 所以统一存到这里由 _emit_headers 发出，不要在调用方直接 send_header
    #: （那样会把头插到状态行之前，响应就废了）。
    extra_headers: Tuple[Tuple[str, str], ...] = ()

    def _emit_headers(self) -> None:
        for k, v in self.extra_headers:
            self.send_header(k, v)
        if self.ignored_fields:
            self.send_header("X-Cann-Llm-Ignored-Fields", ", ".join(self.ignored_fields))

    # ---- 工具 ----
    @property
    def state(self) -> AppState:
        return self.server.state          # type: ignore[attr-defined]

    def _path(self) -> str:
        """取归一化后的路径（去掉查询串与结尾斜杠）。"""
        return urlparse(self.path).path.rstrip("/") or "/"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: Any,
                   extra_headers: Optional[Sequence[Tuple[str, str]]] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        if extra_headers:
            self.extra_headers = tuple(extra_headers)
        self._emit_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str,
                         err_type: str = "invalid_request_error") -> None:
        self._send_json(status, oa.error_payload(message, err_type=err_type))

    # ---- SSE ----
    def _sse_start(self) -> None:
        """开始一个 text/event-stream 响应（chunked 传输，保持连接）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # 让 nginx 之类的反代不要缓冲
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self._emit_headers()
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_raw(self, data: bytes) -> None:
        self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _sse_write(self, data: bytes) -> None:
        self._sse_raw(data)

    def _sse_end(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _sse_abort(self, exc: BaseException) -> None:
        """流已经开始、无法再改状态码时的错误上报。"""
        _status, payload = oa.error_from_exception(exc)
        try:
            self._sse_write(oa.sse_data(payload))
            self._sse_write(oa.SSE_DONE)
            self._sse_end()
        except OSError:
            pass

    # ---- 路由错误提示 ----
    def _hint_for(self, path: str) -> Optional[str]:
        """对常见误用给出针对性提示（而不是干巴巴一句 404）。"""
        base = BASE_URL_HINT.format(port=self.state.cfg.server.port)
        # 把端点路径拼进了 base_url，例如 /v1/chat/completions/models
        for ep in _ENDPOINT_LIKE:
            if path != ep and ep in path:
                return f"路径 {path} 看起来是把端点路径拼进了 base_url。" + base
        # 少了或多了一层 /v1
        if path in ("/v1", "/chat/completions", "/completions", "/models"):
            return "base_url 的路径部分应为 /v1。" + base
        if path.startswith("/v1/"):
            return ("本服务提供的路径：GET /v1/models、POST /v1/chat/completions、"
                    "POST /v1/completions、GET /healthz。")
        return None

    def _not_found(self, path: str) -> None:
        hint = self._hint_for(path)
        self.log_message("404 %s %s%s", self.command, path,
                         f"  —— {hint}" if hint else "")
        self._send_error_json(404, f"未知路径 {path}" + (f"。{hint}" if hint else ""),
                              "not_found")

    def _method_not_allowed(self, path: str, allowed: Tuple[str, ...]) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", ", ".join(allowed))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        body = json.dumps(oa.error_payload(
            f"{self.command} 不被 {path} 支持；允许的方法：{', '.join(allowed)}",
            err_type="method_not_allowed"), ensure_ascii=False).encode()
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self._emit_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        origin = self.state.cfg.server.cors_allow_origin
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _authorized(self) -> bool:
        want = self.state.cfg.server.api_key
        if not want:
            return True
        got = self.headers.get("Authorization", "")
        if got.startswith("Bearer "):
            got = got[7:]
        return got == want

    def _read_body(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_error_json(400, "Content-Length 非法")
            return None
        if length <= 0:
            self._send_error_json(400, "请求体为空")
            return None
        if length > MAX_BODY_BYTES:
            self._send_error_json(413, f"请求体超过 {MAX_BODY_BYTES} 字节")
            return None
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send_error_json(400, f"JSON 解析失败: {e}")
            return None
        if not isinstance(body, dict):
            self._send_error_json(400, "请求体必须是 JSON 对象")
            return None
        return body

    # ---- 路由 ----
    def do_OPTIONS(self) -> None:            # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:                # noqa: N802
        path = self._path()
        if path in ("/healthz", "/health"):
            st = self.state
            self._send_json(200, {
                "status": "ok",
                "model": st.cfg.model.resolved_id,
                "backend": st.cfg.model.backend,
                "uptime_s": round(time.time() - st.started_at, 1),
                "version": version.__version__,
            })
            return
        if path == "/":
            self._send_json(200, {
                "service": "cann-llm",
                "version": version.__version__,
                "model": self.state.cfg.model.resolved_id,
                "endpoints": {
                    "GET /v1/models": "列出模型",
                    "POST /v1/chat/completions": "对话补全（支持 stream）",
                    "POST /v1/completions": "文本补全（支持 stream）",
                    "GET /healthz": "健康检查",
                },
                "note": BASE_URL_HINT.format(port=self.state.cfg.server.port),
            })
            return
        if path == "/v1/models":
            if not self._authorized():
                self._send_error_json(401, "鉴权失败", "invalid_api_key")
                return
            self._send_json(200, oa.models_response([self.state.cfg.model.resolved_id]))
            return
        if path in ROUTES:
            self._method_not_allowed(path, ROUTES[path])
            return
        self._not_found(path)

    def do_POST(self) -> None:               # noqa: N802
        path = self._path()
        if path not in ("/v1/chat/completions", "/v1/completions"):
            if path in ROUTES:
                self._method_not_allowed(path, ROUTES[path])
            else:
                self._not_found(path)
            return
        if not self._authorized():
            self._send_error_json(401, "鉴权失败", "invalid_api_key")
            return
        body = self._read_body()
        if body is None:
            return
        try:
            if path == "/v1/chat/completions":
                self._handle_chat(body)
            else:
                self._handle_completion(body)
        except CannLlmError as e:
            status, payload = oa.error_from_exception(e)
            self._send_json(status, payload)
        except Exception as e:               # noqa: BLE001
            self.log_message("内部错误: %r", e)
            status, payload = oa.error_from_exception(e)
            self._send_json(status, payload)

    # ---- 工具模式 ----
    def _note_ignored(self, fields) -> None:
        """记录本轮被忽略的字段：写响应头 + 日志一行（便于排查客户端行为）。"""
        self.ignored_fields = tuple(fields or ())
        if self.ignored_fields:
            self.log_message("忽略字段: %s", ", ".join(self.ignored_fields))

    def _handle_chat(self, body: Dict[str, Any]) -> None:
        st = self.state
        req = oa.parse_chat_request(body, default_model=st.cfg.model.resolved_id,
                                    strict=st.cfg.server.reject_unsupported,
                                    base_params=st.base_params())
        self._note_ignored(req.ignored)
        # req.params 在解析阶段已用配置默认值补齐，直接用，不做任何比较 ——
        # 曾经的 `req.params != GenerationParams()` 会在调用方显式给出的值
        # 恰好等于 dataclass 默认值时把它替换掉，属于吞掉调用方意图。
        params = req.params
        prompt = st.template.render(req.messages, tools=req.tools or None)

        if req.stream:
            self._stream_chat(req, prompt, params)
            return

        from ..types import GenerationRequest, aggregate

        result = aggregate(st.backend.generate(GenerationRequest(prompt=prompt, params=params)))

        tool_calls = None
        if req.tools:
            # 按 OpenAI 标准：把 tool_calls 交给客户端自行执行
            parsed = parse_tool_calls(result.text)
            tool_calls = parsed.tool_calls or None
            if tool_calls:
                result = replace(result, text=parsed.text, finish_reason="tool_calls")

        self._send_json(200, oa.chat_completion_response(
            req_id=oa.new_id("chatcmpl"), model=req.model or st.cfg.model.resolved_id,
            text=result.text, finish_reason=result.finish_reason,
            usage=oa.usage_payload(result.stats), tool_calls=tool_calls))

    def _stream_chat(self, req: oa.ChatCompletionRequest, prompt: str,
                     params: GenerationParams) -> None:
        """SSE 流式 chat.completion.chunk。

        声明了 tools 时，会把工具协议标记从流里过滤掉 —— 客户端不该看到
        ``<tool_call>`` 这种内部格式；解析出的调用放到末尾的 delta.tool_calls 里，
        按 OpenAI 标准交给客户端执行。
        """
        from ..types import GenerationRequest

        st = self.state
        req_id = oa.new_id("chatcmpl")
        created = oa.now()
        model = req.model or st.cfg.model.resolved_id
        include_usage = bool((req.raw.get("stream_options") or {}).get("include_usage"))
        use_tools = bool(req.tools)

        self._sse_start()
        try:
            # 首个分块先声明角色（与 OpenAI 行为一致）
            self._sse_write(oa.sse_data(oa.chat_completion_chunk(
                req_id=req_id, model=model, created=created,
                delta={"role": "assistant", "content": ""})))

            finish_reason = "stop"
            stats = None
            raw_parts: List[str] = []
            flt = StreamFilter() if use_tools else None
            for chunk in st.backend.generate(GenerationRequest(prompt=prompt, params=params)):
                if chunk.text:
                    if flt is not None:
                        raw_parts.append(chunk.text)
                        piece = flt.feed(chunk.text)
                    else:
                        piece = chunk.text
                    if piece:
                        self._sse_write(oa.sse_data(oa.chat_completion_chunk(
                            req_id=req_id, model=model, created=created,
                            delta={"content": piece})))
                if chunk.stats is not None:
                    stats = chunk.stats
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason

            if flt is not None:
                parsed = parse_tool_calls("".join(raw_parts))
                if parsed.tool_calls:
                    self._sse_write(oa.sse_data(oa.chat_completion_chunk(
                        req_id=req_id, model=model, created=created,
                        delta={"tool_calls": [
                            oa.tool_call_delta(i, call_id=tc.id, name=tc.name,
                                               arguments=json.dumps(tc.arguments,
                                                                    ensure_ascii=False))
                            for i, tc in enumerate(parsed.tool_calls)]})))
                    finish_reason = "tool_calls"

            self._sse_write(oa.sse_data(oa.chat_completion_chunk(
                req_id=req_id, model=model, created=created,
                delta={}, finish_reason=finish_reason)))
            if include_usage:
                usage_chunk = oa.chat_completion_chunk(
                    req_id=req_id, model=model, created=created, delta={})
                usage_chunk["choices"] = []
                usage_chunk["usage"] = oa.usage_payload(stats)
                self._sse_write(oa.sse_data(usage_chunk))
            self._sse_write(oa.SSE_DONE)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            # 客户端提前断开（例如 Ctrl-C），不算错误
            self.close_connection = True
        except Exception as e:               # noqa: BLE001
            self.log_message("流式生成失败: %r", e)
            self._sse_abort(e)


    def _handle_completion(self, body: Dict[str, Any]) -> None:
        st = self.state
        req = oa.parse_completion_request(body, default_model=st.cfg.model.resolved_id,
                                          strict=st.cfg.server.reject_unsupported,
                                          base_params=st.base_params())
        self._note_ignored(req.ignored)
        params = req.params

        if req.stream:
            self._stream_completion(req, params)
            return

        from ..types import GenerationRequest, aggregate

        result = aggregate(st.backend.generate(GenerationRequest(prompt=req.prompt,
                                                                params=params)))
        self._send_json(200, oa.completion_response(
            req_id=oa.new_id("cmpl"), model=req.model or st.cfg.model.resolved_id,
            text=result.text, finish_reason=result.finish_reason,
            usage=oa.usage_payload(result.stats)))

    def _stream_completion(self, req: oa.CompletionRequest,
                           params: GenerationParams) -> None:
        """SSE 流式 text_completion。"""
        from ..types import GenerationRequest

        st = self.state
        req_id = oa.new_id("cmpl")
        created = oa.now()
        model = req.model or st.cfg.model.resolved_id

        self._sse_start()
        try:
            finish_reason = "stop"
            for chunk in st.backend.generate(GenerationRequest(prompt=req.prompt,
                                                               params=params)):
                if chunk.text:
                    self._sse_write(oa.sse_data(oa.completion_chunk(
                        req_id=req_id, model=model, created=created, text=chunk.text)))
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
            self._sse_write(oa.sse_data(oa.completion_chunk(
                req_id=req_id, model=model, created=created, text="",
                finish_reason=finish_reason)))
            self._sse_write(oa.SSE_DONE)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:               # noqa: BLE001
            self.log_message("流式生成失败: %r", e)
            self._sse_abort(e)


class LlmHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: Tuple[str, int], state: AppState):
        self.state = state
        super().__init__(addr, Handler)


# ------------------------------------------------------------------ 入口


def serve(cfg: AppConfig) -> int:
    state = build_state(cfg)
    host, port = cfg.server.host, cfg.server.port
    httpd = LlmHttpServer((host, port), state)
    shown = "127.0.0.1" if host in ("", "0.0.0.0") else host

    base = f"http://{shown}:{port}/v1"
    print(f"cann-llm {version.__version__}  ·  {cfg.model.resolved_id}  "
          f"({cfg.model.backend}/{cfg.model.chat_template})")
    print(f"  监听      : http://{shown}:{port}")
    print(f"  base_url  : {base}      ← OpenAI 客户端的 base_url 填这个")
    print(f"  端点      : GET {base}/models · POST {base}/chat/completions"
          f" · POST {base}/completions")
    print(f"  健康检查  : http://{shown}:{port}/healthz（无需鉴权）"
          f"    鉴权: {'开启' if cfg.server.api_key else '关闭'}")
    if host in ("", "0.0.0.0"):
        print(f"  注意      : 绑定在 0.0.0.0，局域网其它机器请把 127.0.0.1 换成本机 IP")

    stop = threading.Event()

    def _bye(signum, frame):                 # noqa: ARG001
        stop.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        state.backend.close()
        print("\n已停止。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cann-llm-server",
        description="华为 CANN LLM Engine 的 OpenAI 兼容推理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--model-dir", help="模型目录")
    ap.add_argument("-c", "--config", help="TOML 配置文件")
    ap.add_argument(
        "-b", "--backend", default="hiai",
        help=f"后端，可用：{', '.join(available_backends())}（默认 hiai）")
    ap.add_argument("--host", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, help="监听端口（默认 8000）")
    ap.add_argument("--api-key", help="非空则要求 Authorization: Bearer <key>")
    ap.add_argument("--model-id", help="对外暴露的模型 id")
    ap.add_argument("--max-tokens", type=int, dest="max_tokens",
                    help="默认输出窗口（单轮最多生成多少 token；请求里的 max_tokens 优先）")
    ap.add_argument("--workers", type=int, help="忽略，保留参数位（兼容习惯）")
    ap.add_argument("-V", "--version", action="version", version=f"cann-llm {version.__version__}")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    over = {}
    for key, val in (("model_dir", args.model_dir), ("backend", args.backend),
                     ("model_id", args.model_id), ("max_tokens", args.max_tokens)):
        if val is not None:
            over[key] = val
    if over:
        cfg = cfg.merged(model=over)
    sover = {}
    for key, val in (("host", args.host), ("port", args.port), ("api_key", args.api_key)):
        if val is not None:
            sover[key] = val
    if sover:
        cfg = cfg.merged(server=sover)

    try:
        return serve(cfg)
    except OSError as e:
        # 绑定失败要说清楚"还没开始监听" —— 别让人以为服务起来了。
        host, port = cfg.server.host, cfg.server.port
        if e.errno == errno.EADDRINUSE:
            msg = (f"端口 {port} 已被占用（{host}）。"
                   f"换一个端口（--port），或先停掉占用它的进程。")
        elif e.errno == errno.EACCES:
            msg = f"没有权限绑定 {host}:{port}（1024 以下的端口通常需要特权）。"
        elif e.errno == errno.EADDRNOTAVAIL:
            msg = f"地址 {host} 不是本机可用的地址（--host 指定错了？）。"
        else:
            msg = f"绑定 {host}:{port} 失败：{e}"
        print(f"启动失败：{msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
