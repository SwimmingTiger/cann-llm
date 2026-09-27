"""debuglog 的单元测试：截断、密钥抹除、开关语义。

这三件事都关乎"日志能不能安全地给人看"：
截断防止日志被超长 prompt 淹掉；抹除防止 API key 落盘；默认关闭防止
不该开的时候产生输出。
"""
import io
import unittest

from cann_llm import debuglog


class TestClamp(unittest.TestCase):
    def test_str_passthrough(self):
        self.assertEqual(debuglog.clamp("abc", 10), "abc")

    def test_truncates_with_original_length(self):
        out = debuglog.clamp("x" * 100, 10)
        self.assertTrue(out.startswith("x" * 10))
        self.assertIn("100", out)          # 标出原长度，便于判断被截了多少

    def test_bytes_are_decoded_not_repr(self):
        # 引擎的 prompt 是 bytes；不能打成 b'...' 那种字面量（实测踩过）
        out = debuglog.clamp("你好".encode("utf-8"), 100)
        self.assertEqual(out, "你好")
        self.assertNotIn("b'", out)

    def test_bad_bytes_do_not_raise(self):
        out = debuglog.clamp(b"\xff\xfe", 100)
        self.assertIsInstance(out, str)


class TestRedact(unittest.TestCase):
    def test_secrets_are_replaced(self):
        h = {"Authorization": "Bearer sk-123", "Content-Type": "application/json"}
        out = debuglog.redact(h)
        self.assertNotIn("sk-123", out["Authorization"])
        self.assertEqual(out["Content-Type"], "application/json")

    def test_case_insensitive(self):
        self.assertNotIn("k", debuglog.redact({"AUTHORIZATION": "k"})["AUTHORIZATION"])

    def test_non_mapping_is_empty(self):
        self.assertEqual(debuglog.redact(None), {})


class TestSwitch(unittest.TestCase):
    def _reset(self):
        lg = debuglog._logger()
        for h in list(lg.handlers):
            lg.removeHandler(h)

    def test_off_by_default(self):
        # 不能因为"有人在用这个模块"就产生输出
        self._reset()
        self.assertFalse(debuglog.enabled())
        self.assertIsNone(debuglog.log_file())

    def test_enable_to_memory_stream(self):
        buf = io.StringIO()
        debuglog.enable(stream=buf)
        try:
            self.assertTrue(debuglog.enabled())
            debuglog.log("测试", "内容")
            self.assertIn("内容", buf.getvalue())
        finally:
            self._reset()


if __name__ == "__main__":
    unittest.main()
