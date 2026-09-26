import json
import unittest

from cann_llm.api import openai as oa
from cann_llm.errors import (
    BusyError,
    CannLlmError,
    ContextLengthExceededError,
    InvalidRequestError,
)
from cann_llm.types import GenerationStats


class TestParseChat(unittest.TestCase):
    def test_minimal(self):
        r = oa.parse_chat_request({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(len(r.messages), 1)
        self.assertEqual(r.messages[0].content, "hi")
        self.assertFalse(r.stream)

    def test_full(self):
        r = oa.parse_chat_request({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "stream": True, "temperature": 0.1, "max_tokens": 9, "top_k": 3,
            "top_p": 0.5, "repetition_penalty": 1.2, "seed": 7, "stop": "E",
            "frequency_penalty": 1.0,
        })
        self.assertEqual(r.model, "m")
        self.assertTrue(r.stream)
        self.assertEqual(r.params.temperature, 0.1)
        self.assertEqual(r.params.max_tokens, 9)
        self.assertEqual(r.params.top_k, 3)
        self.assertEqual(r.params.seed, 7)
        self.assertEqual(r.params.stop, ("E",))
        self.assertIn("frequency_penalty", r.ignored)

    def test_max_completion_tokens_wins(self):
        r = oa.parse_chat_request({"messages": [{"role": "user", "content": "x"}],
                                   "max_tokens": 1, "max_completion_tokens": 42})
        self.assertEqual(r.params.max_tokens, 42)

    def test_content_blocks(self):
        r = oa.parse_chat_request({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
        self.assertEqual(r.messages[0].content, "ab")

    def test_content_null_becomes_empty(self):
        r = oa.parse_chat_request({"messages": [{"role": "assistant", "content": None},
                                                {"role": "user", "content": "x"}]})
        self.assertEqual(r.messages[0].content, "")

    def test_rejects(self):
        bad = [
            ({"messages": []}, "空数组"),
            ({"messages": "x"}, "非数组"),
            ({"messages": [{"content": "x"}]}, "缺 role"),
            ({"messages": [{"role": "user", "content": "x"}], "n": 3}, "n=3"),
            ({"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
            ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "image"),
            ({"messages": [{"role": "user", "content": "x"}], "temperature": -5}, "温度"),
            ({"messages": [{"role": "user", "content": "x"}], "stop": ["a"] * 5}, "stop 过多"),
            ({"messages": [{"role": "system", "content": "only"}]}, "只有 system"),
        ]
        for body, why in bad:
            with self.assertRaises(InvalidRequestError, msg=why):
                oa.parse_chat_request(body)

    def test_default_model_used_when_absent(self):
        r = oa.parse_chat_request({"messages": [{"role": "user", "content": "x"}]},
                                  default_model="fallback")
        self.assertEqual(r.model, "fallback")


class TestUnsupportedFieldTiers(unittest.TestCase):
    """默认宽容（带 tools 的客户端要能用），但绝不静默给出错误结果。"""

    def test_tools_accepted_and_reported_by_default(self):
        r = oa.parse_chat_request({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto", "parallel_tool_calls": True,
        })
        self.assertIn("tools", r.ignored)
        self.assertIn("tool_choice", r.ignored)
        self.assertIn("parallel_tool_calls", r.ignored)

    def test_tools_rejected_in_strict_mode(self):
        with self.assertRaises(InvalidRequestError):
            oa.parse_chat_request(
                {"messages": [{"role": "user", "content": "x"}], "tools": [{}]},
                strict=True)

    def test_logprobs_always_rejected(self):
        for strict in (False, True):
            with self.assertRaises(InvalidRequestError, msg=f"strict={strict}"):
                oa.parse_chat_request(
                    {"messages": [{"role": "user", "content": "x"}], "logprobs": True},
                    strict=strict)

    def test_image_url_always_rejected(self):
        for strict in (False, True):
            with self.assertRaises(InvalidRequestError, msg=f"strict={strict}"):
                oa.parse_chat_request({"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "x"}}]}]}, strict=strict)

    def test_ignored_empty_when_nothing_unsupported(self):
        r = oa.parse_chat_request({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.ignored, [])

    def test_completion_request_tiers(self):
        r = oa.parse_completion_request({"prompt": "x", "frequency_penalty": 1.0})
        self.assertIn("frequency_penalty", r.ignored)
        with self.assertRaises(InvalidRequestError):
            oa.parse_completion_request({"prompt": "x", "logprobs": 1}, strict=True)

    def test_describe_ignored(self):
        self.assertEqual(oa.describe_ignored([]), "")
        self.assertIn("tools", oa.describe_ignored(["tools"]))


class TestParseCompletion(unittest.TestCase):
    def test_basic(self):
        r = oa.parse_completion_request({"prompt": "hi"})
        self.assertEqual(r.prompt, "hi")

    def test_single_element_list_ok(self):
        self.assertEqual(oa.parse_completion_request({"prompt": ["x"]}).prompt, "x")

    def test_rejects_batch_and_empty(self):
        for body in ({"prompt": ["a", "b"]}, {"prompt": ""}, {"prompt": 3}, {}):
            with self.assertRaises(InvalidRequestError):
                oa.parse_completion_request(body)


class TestResponses(unittest.TestCase):
    def test_models(self):
        p = oa.models_response(["m1"])
        self.assertEqual(p["object"], "list")
        self.assertEqual(p["data"][0]["id"], "m1")

    def test_chat_completion(self):
        p = oa.chat_completion_response(req_id="i", model="m", text="t",
                                        finish_reason="stop", usage={"a": 1})
        self.assertEqual(p["object"], "chat.completion")
        self.assertEqual(p["choices"][0]["message"]["content"], "t")
        self.assertEqual(p["usage"], {"a": 1})

    def test_chunk_shapes(self):
        c = oa.chat_completion_chunk(req_id="i", model="m", created=1, delta={"content": "x"})
        self.assertEqual(c["object"], "chat.completion.chunk")
        self.assertIsNone(c["choices"][0]["finish_reason"])
        self.assertEqual(oa.completion_chunk(req_id="i", model="m", created=1, text="x")["object"],
                         "text_completion")

    def test_usage_payload_none(self):
        self.assertEqual(oa.usage_payload(None)["total_tokens"], 0)
        st = GenerationStats(prompt_tokens=2, completion_tokens=3)
        self.assertEqual(oa.usage_payload(st)["total_tokens"], 5)

    def test_sse_frames(self):
        self.assertTrue(oa.sse_data({"a": 1}).startswith(b"data: "))
        self.assertTrue(oa.sse_data({"a": 1}).endswith(b"\n\n"))
        self.assertEqual(oa.SSE_DONE, b"data: [DONE]\n\n")

    def test_sse_keeps_non_ascii(self):
        self.assertIn("中文".encode(), oa.sse_data({"t": "中文"}))


class TestErrorMapping(unittest.TestCase):
    def test_http_status_per_exception(self):
        cases = [
            (InvalidRequestError("x"), 400, "invalid_request_error"),
            (ContextLengthExceededError("x"), 400, "context_length_exceeded"),
            (BusyError("x"), 503, "server_busy"),
        ]
        for exc, status, etype in cases:
            got_status, payload = oa.error_from_exception(exc)
            self.assertEqual(got_status, status)
            self.assertEqual(payload["error"]["type"], etype)
            self.assertEqual(payload["error"]["message"], "x")

    def test_unknown_exception_is_500(self):
        status, payload = oa.error_from_exception(ValueError("boom"))
        self.assertEqual(status, 500)
        self.assertIn("boom", payload["error"]["message"])

    def test_all_errors_carry_status_and_type(self):
        for cls in (CannLlmError, InvalidRequestError, BusyError):
            self.assertTrue(hasattr(cls, "http_status"))
            self.assertTrue(hasattr(cls, "error_type"))


class TestJsonRoundTrip(unittest.TestCase):
    def test_payload_is_json_serializable(self):
        p = oa.chat_completion_response(req_id="i", model="m", text="中文",
                                        finish_reason="stop", usage={"a": 1})
        self.assertIn("中文", json.dumps(p, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
