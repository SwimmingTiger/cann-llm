import json
import time
import unittest

from cann_llm.agent.loop import (
    AgentConfig,
    AgentLoop,
    Final,
    StreamFilter,
    TextDelta,
    ToolCallDone,
    ToolCallReady,
)
from cann_llm.agent.parser import parse_tool_calls, repair_json
from cann_llm.chat.template import Message, get_template
from cann_llm.tools import ToolError, ToolRegistry
from cann_llm.types import GenerationParams, GenerationStats, ToolCall

from .fakes import ScriptedBackend

CALL = '<tool_call>\n{"name": "echo", "arguments": {"s": "hi"}}\n</tool_call>'


def make_registry(calls):
    reg = ToolRegistry()

    @reg.register("echo", "回显", {"type": "object",
                                   "properties": {"s": {"type": "string"}},
                                   "required": ["s"]})
    def _echo(s):
        calls.append({"s": s})
        return json.dumps({"echo": s})

    @reg.register("broken", "总是抛错", {"type": "object"})
    def _broken():
        raise ToolError("故意的")

    @reg.register("slow", "很慢", {"type": "object"})
    def _slow():
        time.sleep(2)
        return "done"

    return reg


def run(replies, registry, **cfg):
    backend = ScriptedBackend(replies)
    backend.load()
    loop = AgentLoop(backend, get_template("chatml"), registry,
                     config=AgentConfig(**cfg) if cfg else AgentConfig(),
                     system_prompt="SYS",
                     params=GenerationParams(max_tokens=32))
    events = list(loop.run([Message("user", "问题")]))
    return backend, events


def final_of(events):
    return next(e for e in events if isinstance(e, Final))


class TestParser(unittest.TestCase):
    def test_standard(self):
        r = parse_tool_calls(CALL)
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual(r.tool_calls[0].name, "echo")
        self.assertEqual(r.tool_calls[0].arguments, {"s": "hi"})
        self.assertEqual(r.text, "")

    def test_truncated_block(self):
        r = parse_tool_calls('<tool_call>\n{"name": "echo", "arguments": {"s": "hi"}}')
        self.assertEqual(len(r.tool_calls), 1)

    def test_surrounding_text_kept(self):
        r = parse_tool_calls(f"我先查一下。{CALL}稍等。")
        self.assertEqual(r.tool_calls[0].name, "echo")
        self.assertIn("我先查一下", r.text)
        self.assertIn("稍等", r.text)

    def test_code_fence_and_quotes_and_trailing_comma(self):
        for raw in ['<tool_call>\n```json\n{"name": "echo", "arguments": {"s": "hi"}}\n```\n</tool_call>',
                    "<tool_call>{'name': 'echo', 'arguments': {'s': 'hi'}}</tool_call>",
                    '<tool_call>{"name": "echo", "arguments": {"s": "hi",},}</tool_call>']:
            r = parse_tool_calls(raw)
            self.assertEqual(len(r.tool_calls), 1, raw)
            self.assertEqual(r.tool_calls[0].name, "echo", raw)

    def test_double_encoded_arguments(self):
        r = parse_tool_calls('<tool_call>{"name": "echo", "arguments": "{\\"s\\": \\"hi\\"}"}</tool_call>')
        self.assertEqual(r.tool_calls[0].arguments, {"s": "hi"})

    def test_openai_nested_shape(self):
        r = parse_tool_calls('<tool_call>{"function": {"name": "echo", "arguments": {"s": "hi"}}}</tool_call>')
        self.assertEqual(r.tool_calls[0].name, "echo")

    def test_multiple_and_array(self):
        two = ('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
               '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
        self.assertEqual([c.name for c in parse_tool_calls(two).tool_calls], ["a", "b"])
        arr = '<tool_call>[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{}}]</tool_call>'
        self.assertEqual([c.name for c in parse_tool_calls(arr).tool_calls], ["a", "b"])

    def test_bare_json(self):
        r = parse_tool_calls('{"name": "echo", "arguments": {"s": "hi"}}')
        self.assertEqual(len(r.tool_calls), 1)

    def test_plain_text_and_plain_json_not_calls(self):
        self.assertFalse(parse_tool_calls("巴黎是法国的首都。").tool_calls)
        self.assertFalse(parse_tool_calls('{"temperature": 18}').tool_calls)

    def test_unparsable_keeps_raw(self):
        r = parse_tool_calls("<tool_call>我想查天气</tool_call>")
        self.assertTrue(r.had_invalid)
        self.assertEqual(r.tool_calls[0].name, "")
        self.assertIn("我想查天气", r.tool_calls[0].arguments["__unparsable__"])

    def test_ids_assigned_and_unique(self):
        two = ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
               '<tool_call>{"name":"b","arguments":{}}</tool_call>')
        ids = [c.id for c in parse_tool_calls(two).tool_calls]
        self.assertTrue(all(ids))
        self.assertEqual(len(set(ids)), 2)

    def test_repair_json_variants(self):
        self.assertEqual(repair_json('{"a": 1,}'), {"a": 1})
        self.assertEqual(repair_json("{'a': 1}"), {"a": 1})
        self.assertEqual(repair_json('{"a": 1'), {"a": 1})
        self.assertIsNone(repair_json("不是 json"))


class TestStreamFilter(unittest.TestCase):
    def test_each_char(self):
        f = StreamFilter()
        raw = '好的<tool_call>{"name":"w"}</tool_call>完成'
        out = "".join(f.feed(c) for c in raw) + f.flush()
        self.assertEqual(out, "好的完成")

    def test_chunked(self):
        f = StreamFilter()
        out = "".join(f.feed(p) for p in ['好的<tool_', 'call>{"na', 'me":"w"}</tool_',
                                          'call>完成']) + f.flush()
        self.assertEqual(out, "好的完成")

    def test_plain_text_untouched(self):
        f = StreamFilter()
        text = "这是一段普通回答。"
        self.assertEqual("".join(f.feed(c) for c in text) + f.flush(), text)

    def test_less_than_sign_not_swallowed(self):
        f = StreamFilter()
        text = "a < b 且 x <tool 不是标记"
        self.assertEqual("".join(f.feed(c) for c in text) + f.flush(), text)

    def test_unterminated_suppressed(self):
        f = StreamFilter()
        out = "".join(f.feed(c) for c in '开始<tool_call>{"name": "w"') + f.flush()
        self.assertEqual(out, "开始")


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer_no_tools(self):
        backend, events = run(["你好呀"], make_registry([]))
        self.assertEqual(backend.call_count, 1)
        f = final_of(events)
        self.assertEqual(f.text, "你好呀")
        self.assertEqual(f.steps, 1)
        self.assertFalse(f.tool_calls)

    def test_tool_call_then_answer(self):
        calls = []
        backend, events = run([CALL, "结果是我"], make_registry(calls))
        self.assertEqual(calls, [{"s": "hi"}])
        self.assertEqual(backend.call_count, 2)
        f = final_of(events)
        self.assertEqual(f.text, "结果是我")
        self.assertEqual(f.steps, 2)
        self.assertEqual([c.name for c in f.tool_calls], ["echo"])
        # 第二轮 prompt 里应包含工具声明与 tool_response
        self.assertIn("</tool_response>", backend.prompts[1])
        self.assertIn("<tools>", backend.prompts[1])

    def test_events_order_and_text_deltas_exclude_protocol(self):
        _, events = run([CALL, "答案"], make_registry([]))
        kinds = [type(e).__name__ for e in events]
        self.assertEqual(kinds[0], "StepStarted")
        self.assertIn("ToolCallReady", kinds)
        self.assertIn("ToolCallDone", kinds)
        self.assertEqual(kinds[-1], "Final")
        streamed = "".join(e.text for e in events if isinstance(e, TextDelta))
        self.assertNotIn("<tool_call>", streamed)
        self.assertIn("答案", streamed)

    def test_unknown_tool_result_is_error_and_model_continues(self):
        _, events = run(['<tool_call>{"name": "nope", "arguments": {}}</tool_call>', "好的"],
                        make_registry([]))
        done = [e for e in events if isinstance(e, ToolCallDone)]
        self.assertEqual(len(done), 1)
        self.assertFalse(done[0].result.ok)
        self.assertIn("nope", done[0].result.content)
        self.assertEqual(final_of(events).text, "好的")

    def test_schema_violation_reported(self):
        _, events = run(['<tool_call>{"name": "echo", "arguments": {}}</tool_call>', "ok"],
                        make_registry([]))
        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        self.assertFalse(done.result.ok)
        self.assertIn("s", done.result.content)

    def test_tool_exception_becomes_error_result(self):
        _, events = run(['<tool_call>{"name": "broken", "arguments": {}}</tool_call>', "ok"],
                        make_registry([]))
        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        self.assertFalse(done.result.ok)
        self.assertIn("故意的", done.result.content)

    def test_unparsable_call_reported(self):
        _, events = run(["<tool_call>胡说八道</tool_call>", "ok"], make_registry([]))
        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        self.assertFalse(done.result.ok)
        self.assertIn("无法解析", done.result.content)

    def test_max_steps_forces_final_without_tools(self):
        calls = []
        # 模型每轮都想调工具，永不收尾
        backend, events = run([CALL] * 6, make_registry(calls), max_steps=3)
        f = final_of(events)
        self.assertEqual(f.steps, 3)
        self.assertEqual(backend.call_count, 3)
        # 最后一轮的 prompt 不应再包含工具声明
        self.assertNotIn("<tools>", backend.prompts[-1])

    def test_max_calls_per_step_truncated(self):
        many = "".join(f'<tool_call>{{"name":"echo","arguments":{{"s":"{i}"}}}}</tool_call>'
                       for i in range(6))
        calls = []
        _, events = run([many, "完"], make_registry(calls), max_calls_per_step=2)
        self.assertEqual(len(calls), 2)

    def test_result_truncated(self):
        reg = ToolRegistry()
        reg.register("big", "大结果", {"type": "object"})(lambda: "x" * 10000)
        _, events = run(['<tool_call>{"name":"big","arguments":{}}</tool_call>', "ok"],
                        reg, max_result_chars=100)
        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        self.assertLess(len(done.result.content), 200)
        self.assertIn("已截断", done.result.content)

    def test_tool_timeout(self):
        reg = make_registry([])
        reg.get("slow").timeout_s = 0.1
        _, events = run(['<tool_call>{"name":"slow","arguments":{}}</tool_call>', "ok"], reg)
        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        self.assertFalse(done.result.ok)
        self.assertIn("超时", done.result.content)

    def test_history_contains_tool_messages(self):
        reg = make_registry([])
        backend, events = run([CALL, "最终答案"], reg)
        f = final_of(events)
        roles = [m.role for m in f.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(f.messages[3].tool_call_id, f.messages[2].tool_calls[0].id)
        self.assertEqual(f.messages[-1].content, "最终答案")

    def test_force_tool_use_directive_in_prompt(self):
        backend, _ = run([CALL, "ok"], make_registry([]))
        self.assertIn("MUST emit the tool call", backend.prompts[0])

    def test_force_tool_use_can_be_disabled(self):
        backend, _ = run([CALL, "ok"], make_registry([]), force_tool_use=False)
        self.assertNotIn("MUST emit the tool call", backend.prompts[0])


if __name__ == "__main__":
    unittest.main()
