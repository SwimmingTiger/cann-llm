import unittest

from cann_llm.chat.session import ChatSession
from cann_llm.chat.template import (
    IM_END,
    IM_START,
    ChatMLTemplate,
    Message,
    available_templates,
    get_template,
)
from cann_llm.errors import InvalidRequestError
from cann_llm.types import GenerationParams, GenerationStats

from .fakes import FakeBackend


class TestTemplate(unittest.TestCase):
    def test_chatml_render(self):
        t = get_template("chatml")
        out = t.render([Message("system", "S"), Message("user", "U")])
        self.assertEqual(
            out,
            f"{IM_START}system\nS{IM_END}\n{IM_START}user\nU{IM_END}\n{IM_START}assistant\n")

    def test_chatml_without_generation_prompt(self):
        t = get_template("chatml")
        out = t.render([Message("user", "U")], add_generation_prompt=False)
        self.assertFalse(out.endswith("assistant\n"))

    def test_stop_strings(self):
        self.assertEqual(tuple(get_template("chatml").stop_strings()), (IM_END,))

    def test_plain(self):
        self.assertEqual(get_template("plain").render([Message("user", "abc")]), "abc")

    def test_unknown_role_rejected(self):
        with self.assertRaises(InvalidRequestError):
            Message("bogus", "x")

    def test_unknown_template_rejected(self):
        with self.assertRaises(InvalidRequestError):
            get_template("nope")

    def test_registry_lists_builtins(self):
        self.assertIn("chatml", available_templates())
        self.assertIn("plain", available_templates())

    def test_custom_template_registration(self):
        from cann_llm.chat.template import register_template

        @register_template("dummy-for-test")
        class Dummy(ChatMLTemplate):
            def render(self, messages, *, add_generation_prompt=True):
                return "dummy"

        self.assertEqual(get_template("dummy-for-test").render([Message("user", "x")]),
                         "dummy")


class TestSession(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend("hi there")
        self.session = ChatSession(backend=self.backend, template=get_template("chatml"),
                                   system_prompt="SYS", params=GenerationParams(max_tokens=8))

    def test_stream_and_history(self):
        chunks = list(self.session.ask("hello"))
        self.assertEqual("".join(c.text for c in chunks), "hi there")
        self.assertEqual([m.role for m in self.session.messages],
                         ["user", "assistant"])
        self.assertEqual(self.session.messages[1].content, "hi there")
        self.assertEqual(self.session.turn_count, 1)

    def test_stats_extracted_from_final_chunk(self):
        list(self.session.ask("hello"))
        self.assertIsNotNone(self.session.last_stats)
        self.assertEqual(self.session.last_stats.completion_tokens, len("hi there"))
        self.assertEqual(self.session.last_finish_reason, "stop")

    def test_ask_sync(self):
        self.assertEqual(self.session.ask_sync("q"), "hi there")

    def test_render_includes_system_and_history(self):
        list(self.session.ask("first"))
        prompt = self.session.render()
        self.assertIn("SYS", prompt)
        self.assertIn("first", prompt)
        self.assertIn("hi there", prompt)
        self.assertTrue(prompt.endswith("assistant\n"))

    def test_early_break_still_records_partial(self):
        gen = self.session.ask("hello")
        next(gen)                      # 只取第一个分块就放弃
        gen.close()
        self.assertEqual(self.session.messages[-1].role, "assistant")
        self.assertTrue(self.session.messages[-1].content)

    def test_reset_and_set_system(self):
        list(self.session.ask("x"))
        self.session.reset()
        self.assertEqual(self.session.messages, [])
        self.session.set_system("NEW")
        self.assertEqual(self.session.system_prompt, "NEW")
        self.assertEqual(self.session.messages, [])

    def test_history_is_never_trimmed(self):
        """历史一律完整保留 —— 静默丢弃会让模型换掉上下文，比报错更难排查。"""
        s = ChatSession(backend=self.backend, template=get_template("chatml"),
                        system_prompt="SYS")
        for i in range(30):
            s.messages.append(Message("user", f"question number {i} with several words"))
            s.messages.append(Message("assistant", f"answer number {i} with several words"))
        rendered = s.render()
        self.assertIn("SYS", rendered)
        # 最旧的与最新的都在
        self.assertIn("question number 0", rendered)
        self.assertIn("question number 29", rendered)


if __name__ == "__main__":
    unittest.main()
