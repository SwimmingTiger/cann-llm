"""HTTP 层测试：用假后端起一个真实的 http.server，走 urllib 请求。"""

import json
import threading
import unittest
import urllib.error
import urllib.request

from cann_llm.api.server import AppState, LlmHttpServer
from cann_llm.backends.base import SerializedBackend
from cann_llm.chat.template import get_template
from cann_llm.config import AppConfig, ModelConfig, ServerConfig
from cann_llm.types import GenerationParams, GenerationStats

from .fakes import FakeBackend


def _start(reply="hello world", *, api_key=None, max_queue=4):
    inner = FakeBackend(reply, stats=GenerationStats(prompt_tokens=4, completion_tokens=2,
                                                     prefill_ms=1.0, decode_ms=2.0))
    inner.load()
    cfg = AppConfig(
        model=ModelConfig(model_id="fake-model", backend="fake"),
        server=ServerConfig(host="127.0.0.1", port=0, api_key=api_key, max_queue=max_queue),
    )
    state = AppState(cfg=cfg, backend=SerializedBackend(inner), template=get_template("chatml"))
    httpd = LlmHttpServer(("127.0.0.1", 0), state)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


def _post(url, body, headers=None):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.base = _start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_healthz(self):
        status, body = _get(self.base + "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_models(self):
        status, body = _get(self.base + "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "fake-model")

    def test_chat_non_stream(self):
        status, body = _post(self.base + "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        d = json.loads(body)
        self.assertEqual(d["object"], "chat.completion")
        self.assertEqual(d["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        self.assertEqual(d["usage"]["prompt_tokens"], 4)

    def test_chat_stream_sse(self):
        status, body = _post(self.base + "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(status, 200)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))
        frames = [json.loads(l[6:]) for l in body.splitlines()
                  if l.startswith("data: ") and l[6:] != "[DONE]"]
        # 首帧声明 role
        self.assertEqual(frames[0]["choices"][0]["delta"].get("role"), "assistant")
        text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
        self.assertEqual(text, "hello world")
        # 末帧带 finish_reason 且 delta 为空
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(frames[-1]["choices"][0]["delta"], {})

    def test_completion_non_stream_and_stream(self):
        status, body = _post(self.base + "/v1/completions", {"prompt": "x"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["object"], "text_completion")

        status, body = _post(self.base + "/v1/completions", {"prompt": "x", "stream": True})
        self.assertEqual(status, 200)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    def test_include_usage(self):
        _status, body = _post(self.base + "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "x"}],
                               "stream": True, "stream_options": {"include_usage": True}})
        frames = [json.loads(l[6:]) for l in body.splitlines()
                  if l.startswith("data: ") and l[6:] != "[DONE]"]
        usage_frames = [f for f in frames if "usage" in f]
        self.assertEqual(len(usage_frames), 1)
        self.assertEqual(usage_frames[0]["choices"], [])

    def test_errors(self):
        status, body = _post(self.base + "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")

        # tools 默认被忽略而不是报错（否则现代客户端用不了），仍返回 200
        status, body = _post(self.base + "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "x"}], "tools": [{}]})
        self.assertEqual(status, 200)

        # logprobs 这种「忽略就会给出错误结果」的字段任何时候都拒绝
        status, _ = _post(self.base + "/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "x"}], "logprobs": True})
        self.assertEqual(status, 400)

        status, body = _get(self.base + "/v1/nope")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["type"], "not_found")

        status, _ = _post(self.base + "/v1/chat/completions", {"not": "valid"})
        self.assertEqual(status, 400)

    def test_bad_json(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(cm.exception.code, 400)

    def test_options_preflight(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 204)


class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.base = _start(api_key="sk-test")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_missing_key_401(self):
        status, body = _get(self.base + "/v1/models")
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_api_key")

    def test_wrong_key_401(self):
        status, _ = _get(self.base + "/v1/models", {"Authorization": "Bearer nope"})
        self.assertEqual(status, 401)

    def test_correct_key_ok(self):
        status, _ = _get(self.base + "/v1/models", {"Authorization": "Bearer sk-test"})
        self.assertEqual(status, 200)

    def test_healthz_needs_no_auth(self):
        status, _ = _get(self.base + "/healthz")
        self.assertEqual(status, 200)


class TestBusy(unittest.TestCase):
    def test_503_when_queue_full(self):
        """并发上限 1 时，第二个并发请求应拿到 503。"""
        class Blocking(FakeBackend):
            def __init__(self):
                super().__init__("x")
                self.gate = threading.Event()

            def generate(self, request):
                self.gate.wait(5)
                yield from super().generate(request)

        inner = Blocking()
        inner.load()
        cfg = AppConfig(model=ModelConfig(model_id="m", backend="fake"),
                        server=ServerConfig(host="127.0.0.1", port=0, max_queue=1))
        state = AppState(cfg=cfg, backend=SerializedBackend(inner, max_queue=1, timeout_s=5),
                         template=get_template("chatml"))
        httpd = LlmHttpServer(("127.0.0.1", 0), state)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"

        results = []
        ready = threading.Event()

        def slow():
            ready.set()
            results.append(_post(base + "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "a"}]}))

        t = threading.Thread(target=slow, daemon=True)
        t.start()
        ready.wait(2)
        threading.Event().wait(0.2)          # 等第一个请求真正占住
        status, body = _post(base + "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "b"}]})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["type"], "server_busy")

        inner.gate.set()
        t.join(5)
        httpd.shutdown()
        httpd.server_close()




class TestIgnoredFields(unittest.TestCase):
    """被忽略的字段要在响应头里回报，不能让调用方蒙在鼓里。"""

    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.base = _start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post_with_headers(self, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()

    def test_header_lists_ignored_fields(self):
        status, headers, _ = self._post_with_headers(
            {"messages": [{"role": "user", "content": "hi"}],
             "tools": [{}], "tool_choice": "auto"})
        self.assertEqual(status, 200)
        got = headers.get("X-Cann-Llm-Ignored-Fields", "")
        self.assertIn("tools", got)
        self.assertIn("tool_choice", got)

    def test_header_absent_when_nothing_ignored(self):
        _status, headers, _ = self._post_with_headers(
            {"messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("X-Cann-Llm-Ignored-Fields", headers)

    def test_streaming_carries_header(self):
        _status, headers, _ = self._post_with_headers(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True, "tools": [{}]})
        self.assertIn("tools", headers.get("X-Cann-Llm-Ignored-Fields", ""))


class TestStrictMode(unittest.TestCase):
    """server.reject_unsupported = True 时恢复「宁可报错」的行为。"""

    @classmethod
    def setUpClass(cls):
        inner = FakeBackend("ok")
        inner.load()
        cfg = AppConfig(model=ModelConfig(model_id="m", backend="fake"),
                        server=ServerConfig(host="127.0.0.1", port=0,
                                            reject_unsupported=True))
        state = AppState(cfg=cfg, backend=SerializedBackend(inner),
                         template=get_template("chatml"))
        cls.httpd = LlmHttpServer(("127.0.0.1", 0), state)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_tools_rejected(self):
        status, body = _post(self.base + "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "x"}],
                              "tools": [{}]})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")

    def test_plain_request_still_ok(self):
        status, _ = _post(self.base + "/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 200)


class TestRoutingHelpfulness(unittest.TestCase):
    """路由误用要给出可操作的提示，而不是干巴巴 404。"""

    @classmethod
    def setUpClass(cls):
        cls.httpd, cls.base = _start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _err_message(self, method, path):
        req = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, ""
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())["error"]["message"]

    def test_base_url_with_endpoint_appended(self):
        """用户实际遇到的：base_url 写成了 .../v1/chat/completions。"""
        status, msg = self._err_message("GET", "/v1/chat/completions/models")
        self.assertEqual(status, 404)
        self.assertIn("base_url", msg)
        self.assertIn("/v1", msg)

    def test_missing_v1_prefix(self):
        status, msg = self._err_message("GET", "/chat/completions")
        self.assertEqual(status, 404)
        self.assertIn("base_url", msg)

    def test_unknown_v1_subpath_lists_endpoints(self):
        status, msg = self._err_message("GET", "/v1/bogus")
        self.assertEqual(status, 404)
        self.assertIn("/v1/chat/completions", msg)

    def test_get_on_post_endpoint_is_405_with_allow(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", method="GET")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 405)
        self.assertIn("POST", cm.exception.headers.get("Allow", ""))
        body = json.loads(cm.exception.read().decode())
        self.assertEqual(body["error"]["type"], "method_not_allowed")

    def test_post_on_get_endpoint_is_405(self):
        req = urllib.request.Request(self.base + "/v1/models", method="POST", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 405)
        self.assertIn("GET", cm.exception.headers.get("Allow", ""))

    def test_index_lists_endpoints(self):
        status, body = _get(self.base + "/")
        self.assertEqual(status, 200)
        d = json.loads(body)
        # endpoints 的 key 形如 "POST /v1/chat/completions"
        self.assertTrue(any("/v1/chat/completions" in k for k in d["endpoints"]))
        self.assertTrue(any("/v1/models" in k for k in d["endpoints"]))
        self.assertIn("base_url", d["note"])

    def test_trailing_slash_is_tolerated(self):
        status, _ = _get(self.base + "/v1/models/")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
