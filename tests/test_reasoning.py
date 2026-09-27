"""`<think>` 拆分：标记必须【独立成行】才算标记。

为什么这条重要：用户消息里可能内联写着 "<think> </think>"（讨论这个标签），
模型思考时也会引用它。若按"任意位置出现"判定，内联的 `</think>` 会被当成结束标记，
思考段从中间截断、后半段错当成正文露出去 —— 实测踩过。
"""
import unittest

from cann_llm.api.reasoning import split_reasoning


class TestSplitReasoning(unittest.TestCase):
    def check(self, src, want_reasoning, want_content):
        got = split_reasoning(src)
        self.assertEqual(got, (want_reasoning, want_content), msg=repr(src))

    def test_normal(self):
        self.check("<think>\n推理\n</think>\n\n正文", "推理\n", "正文")

    def test_inline_close_is_not_a_marker(self):
        self.check("<think>\n讨论 </think> 标签\n</think>\n\n正文", "讨论 </think> 标签\n", "正文")

    def test_inline_open_is_not_a_marker(self):
        self.check("看到 <think> 就该开始吗\n正文", "", "看到 <think> 就该开始吗\n正文")

    def test_truncated_inside_thinking(self):
        self.check("<think>\n还没写完", "还没写完", "")

    def test_variable_newline_count(self):
        # 换行个数不能假设。规律：标记【后】的换行是分隔符、吃光；
        # 思考段【内部】（含紧贴 </think> 之前）的换行原样保留。
        self.check("<think>\n推理\n</think>\n正文", "推理\n", "正文")
        self.check("<think>\n\n推理\n\n</think>\n\n正文", "推理\n\n", "正文")
        self.check("<think>\n\n\n推理\n\n\n</think>\n\n\n正文", "推理\n\n\n", "正文")
        self.check("<think>\n\n\n推理\n\n</think>\n正文", "推理\n\n", "正文")

    def test_no_marker(self):
        self.check("直接回答", "", "直接回答")

    def test_markers_mentioned_again_in_content(self):
        self.check("<think>\n想\n</think>\n\n正文说 <think> 与 </think>",
                   "想\n", "正文说 <think> 与 </think>")


if __name__ == "__main__":
    unittest.main()
