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
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .. import version
from ..backends import available_backends, create_backend
from ..backends.base import EngineBackend, SerializedBackend
from ..chat.template import get_template
from ..config import AppConfig, load_config
from ..errors import CannLlmError, InvalidRequestError
from ..types import GenerationParams
from . import openai as oa

#: 请求体上限（防止误发大文件把内存打满）
MAX_BODY_BYTES = 4 * 1024 * 1024


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
        "model_id": mc.model_id,
        "context_length": mc.context_length,
        "max_prompt_tokens": mc.max_prompt_tokens,
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

    # ---- 工具 ----
    @property
    def state(self) -> AppState:
        return self.server.state          # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str,
                         err_type: str = "invalid_request_error") -> None:
        self._send_json(status, oa.error_payload(message, err_type=err_type))

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
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/healthz", "/health"):
            st = self.state
            self._send_json(200, {
                "status": "ok",
                "model": st.cfg.model.model_id,
                "backend": st.cfg.model.backend,
                "uptime_s": round(time.time() - st.started_at, 1),
                "version": version.__version__,
            })
            return
        if path == "/v1/models":
            if not self._authorized():
                self._send_error_json(401, "鉴权失败", "invalid_api_key")
                return
            self._send_json(200, oa.models_response([self.state.cfg.model.model_id]))
            return
        self._send_error_json(404, f"未知路径 {path}", "not_found")

    def do_POST(self) -> None:               # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_error_json(404, f"未知路径 {path}", "not_found")
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

    # ---- 业务 ----
    def _handle_chat(self, body: Dict[str, Any]) -> None:
        st = self.state
        req = oa.parse_chat_request(body, default_model=st.cfg.model.model_id)
        prompt = st.template.render(req.messages)
        params = req.params if req.params != GenerationParams() else st.base_params()

        if req.stream:
            self._send_error_json(
                400, "本服务尚未实现流式输出（stream=true），请用 stream=false",
                "invalid_request_error")
            return

        from ..types import GenerationRequest, aggregate

        result = aggregate(st.backend.generate(GenerationRequest(prompt=prompt, params=params)))
        self._send_json(200, oa.chat_completion_response(
            req_id=oa.new_id("chatcmpl"), model=req.model or st.cfg.model.model_id,
            text=result.text, finish_reason=result.finish_reason,
            usage=oa.usage_payload(result.stats)))

    def _handle_completion(self, body: Dict[str, Any]) -> None:
        st = self.state
        req = oa.parse_completion_request(body, default_model=st.cfg.model.model_id)
        params = req.params if req.params != GenerationParams() else st.base_params()

        if req.stream:
            self._send_error_json(
                400, "本服务尚未实现流式输出（stream=true），请用 stream=false",
                "invalid_request_error")
            return

        from ..types import GenerationRequest, aggregate

        result = aggregate(st.backend.generate(GenerationRequest(prompt=req.prompt,
                                                                params=params)))
        self._send_json(200, oa.completion_response(
            req_id=oa.new_id("cmpl"), model=req.model or st.cfg.model.model_id,
            text=result.text, finish_reason=result.finish_reason,
            usage=oa.usage_payload(result.stats)))


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

    print(f"cann-llm {version.__version__}  ·  {cfg.model.model_id}  "
          f"({cfg.model.backend}/{cfg.model.chat_template})")
    print(f"  监听 http://{shown}:{port}")
    print(f"  OpenAI 兼容端点: /v1/models  /v1/chat/completions  /v1/completions")
    print(f"  健康检查: /healthz    鉴权: {'开启' if cfg.server.api_key else '关闭'}")

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
    ap.add_argument("-b", "--backend", help=f"后端，可用：{', '.join(available_backends())}")
    ap.add_argument("--host", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, help="监听端口（默认 8000）")
    ap.add_argument("--api-key", help="非空则要求 Authorization: Bearer <key>")
    ap.add_argument("--model-id", help="对外暴露的模型 id")
    ap.add_argument("--workers", type=int, help="忽略，保留参数位（兼容习惯）")
    ap.add_argument("-V", "--version", action="version", version=f"cann-llm {version.__version__}")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    over = {}
    for key, val in (("model_dir", args.model_dir), ("backend", args.backend),
                     ("model_id", args.model_id)):
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

    return serve(cfg)


if __name__ == "__main__":
    sys.exit(main())
