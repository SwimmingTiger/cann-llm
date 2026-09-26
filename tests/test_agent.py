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

    def test_json_the_user_asked_for_is_untouched(self):
        """用户明确要生成 JSON 代码时，绝不能把它当成工具调用删掉。"""
        text = "\n".join([
            "当然，这是一个 JSON-RPC 请求体：", "",
            "```json",
            '{"name": "getUserProfile", "arguments": {"userId": 42}}',
            "```", "", "把它 POST 到 /rpc 即可。",
        ])
        r = parse_tool_calls(text)
        self.assertEqual(r.tool_calls, [])
        self.assertEqual(r.text, text)                    # 一字不动
        self.assertIn("getUserProfile", r.text)           # 代码块内容还在

    def test_bare_json_is_never_a_call(self):
        """模型没写标签就是没表达调用意图 —— 不替它认定。"""
        for text in ('{"name": "echo", "arguments": {"s": "hi"}}',
                     'The tool takes {"name": "search", "arguments": {"q": "x"}} as input.'):
            r = parse_tool_calls(text)
            self.assertEqual(r.tool_calls, [], text)
            self.assertEqual(r.text, text, text)

    def test_there_is_no_bare_json_switch(self):
        """该「容错」已被彻底移除，不留开关。"""
        import inspect
        from cann_llm.agent import parser as pmod
        src = inspect.getsource(pmod)
        self.assertNotIn("allow_bare_json", src)
        self.assertNotIn("extract_json_objects", src)

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
        self.assertIn("Error:", done.result.content)
        self.assertIn("could not parse", done.result.content)

    def test_failure_content_is_factual_only(self):
        """失败内容只陈述事实，不追加任何指导 —— 依据 DSH 的做法。

        DSH 的 dsh-agent-loop 在工具失败时构造
          { content: "Error: tool call aborted before dispatch",
            isError: true, error: {message, info:{name, code}} }
        只有事实 + 结构化元数据，没有「请重试」之类的指导。

        而且 pi-ai 的 wire 转换只发 content / tool_call_id —— isError 不发给
        模型，所以 content 里的文字是模型唯一能看到的失败信号。
        """
        cases = [
            ('<tool_call>{"name": "nope", "arguments": {}}</tool_call>', "no tool named"),
            ('<tool_call>{"name": "echo", "arguments": {}}</tool_call>', "Error:"),
            ('<tool_call>{"name": "broken", "arguments": {}}</tool_call>', "故意的"),
            ("<tool_call>胡说八道</tool_call>", "could not parse"),
        ]
        for raw, expect in cases:
            _, events = run([raw, "ok"], make_registry([]))
            done = [e for e in events if isinstance(e, ToolCallDone)][0]
            self.assertFalse(done.result.ok, raw)
            self.assertIn(expect, done.result.content, raw)
            # 不许出现任何指挥模型的话
            for banned in ("请", "重新调用", "不要凭猜测", "must ", "should "):
                self.assertNotIn(banned, done.result.content, f"{raw} → {banned}")

    def test_max_steps_forces_final_without_tools(self):
        calls = []
        # 模型每轮都想调工具，永不收尾
        backend, events = run([CALL] * 6, make_registry(calls), max_steps=3)
        f = final_of(events)
        self.assertEqual(f.steps, 3)
        self.assertEqual(backend.call_count, 3)
        # 最后一轮的 prompt 不应再包含工具声明
        self.assertNotIn("<tools>", backend.prompts[-1])

    def test_all_tool_calls_are_executed(self):
        """模型发了几个就执行几个 —— 截断等于丢弃模型输出。"""
        many = "".join(f'<tool_call>{{"name":"echo","arguments":{{"s":"{i}"}}}}</tool_call>'
                       for i in range(6))
        calls = []
        _, events = run([many, "完"], make_registry(calls))
        self.assertEqual(len(calls), 6)

    def test_tool_result_is_never_truncated(self):
        """工具返回多少就给模型多少 —— 截断会销毁调用方的数据。"""
        big = "x" * 10000
        reg = ToolRegistry()
        reg.register("big", "大结果", {"type": "object"})(lambda: big)
        backend, events = run(['<tool_call>{"name":"big","arguments":{}}</tool_call>', "ok"], reg)

        done = [e for e in events if isinstance(e, ToolCallDone)][0]
        # 调用方拿到的必须是完整的
        self.assertEqual(done.result.content, big)
        # 进 prompt 的也必须完整
        prompt = backend.prompts[1]
        self.assertIn(big, prompt)
        self.assertNotIn("已截断", prompt)

    def test_no_result_limit_knob(self):
        self.assertFalse(hasattr(AgentConfig(), "max_result_chars"))
        from cann_llm.config import AgentSettings
        self.assertFalse(hasattr(AgentSettings(), "max_result_chars"))

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

    def test_prompt_is_never_edited(self):
        """框架不往 prompt 里塞任何指令 —— 那是调用方 system prompt 的地方。

        llama.cpp 可作对照：工具格式说明来自**模型自带的 chat template**
        （框架只负责应用），「强制调用」走 grammar 且只在调用方要求
        tool_choice=required 时启用。没有主流框架会自己写「你必须调用工具」。
        """
        backend, _ = run([CALL, "ok"], make_registry([]))
        prompt = backend.prompts[0]
        self.assertNotIn("MUST emit the tool call", prompt)
        self.assertNotIn("IMPORTANT:", prompt)
        # 工具声明本身仍在（它来自模型自带的 chat template，不是我们编的）
        self.assertIn("# Tools", prompt)


if __name__ == "__main__":
    unittest.main()
