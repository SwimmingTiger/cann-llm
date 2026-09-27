"""debuglog 的单元测试。

守两条规矩：

1. **debug 里不许有任何过滤、拦截、改写、限制** —— 诊断的价值就在完整原文，
   任何"帮你收一收"都是在藏东西。所以最长的文本要原样落盘、坏字节也不许丢。
2. **默认关闭**（不该开的时候不能产生输出），且请求头里的密钥要抹掉。

★ 断言的是【模块产出的记录】而不是"某个流收到了什么"：
  pytest 的 logging 插件会插手 handler，按流断言会让测试依赖环境
  （实测：同一用例单独跑通过、跟其它用例一起跑就失败）。
"""
import io
import logging
import unittest

from cann_llm import debuglog


class _Capture(logging.Handler):
    """把记录原样收集起来 —— 与 enable() 指向哪个流无关。"""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _Base(unittest.TestCase):
    def setUp(self):
        # ★ 顺序要紧：enable() 会清掉已有 handler，所以必须【先 enable 再挂自己的】。
        #   反过来写会让捕获 handler 被清掉，测试时好时坏（实测踩过）。
        debuglog.enable(stream=io.StringIO())      # 让 enabled() 为真
        self.cap = _Capture()
        debuglog._logger().addHandler(self.cap)

    def tearDown(self):
        lg = debuglog._logger()
        lg.removeHandler(self.cap)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    def output(self):
        return "\n".join(self.cap.messages)


class TestNoLimiting(_Base):
    """第 1 条规矩：不截断、不汇总、不丢字节。"""

    def test_no_limiting_helpers_exist(self):
        # clamp() / frame_cap() 曾经存在，已被删除 —— 别让它们悄悄回来
        self.assertFalse(hasattr(debuglog, "clamp"))
        self.assertFalse(hasattr(debuglog, "frame_cap"))

    def test_long_text_written_whole(self):
        long = "很长的一句话。" * 20000             # 约 14 万字符
        debuglog.log("原文", long)
        out = self.output()
        # 只用长度与首尾判断：断言失败时不会把十几万字符灌进终端
        self.assertGreater(len(out), len(long))
        self.assertTrue(out.endswith(long))
        self.assertNotIn("已截断", out)

    def test_bytes_keep_every_byte(self):
        debuglog.log("原始字节", b"good\xffend")     # 坏字节也不能被抹掉
        out = self.output()
        self.assertIn("good", out)
        self.assertIn("end", out)
        self.assertIn("xff", out)                  # 以 \xff 形式保留下来

    def test_utf8_bytes_decode_to_text(self):
        debuglog.log("中文", "你好".encode("utf-8"))
        out = self.output()
        self.assertIn("你好", out)                  # 正常 UTF-8 应可读
        self.assertNotIn("b'", out)                # 而不是 b'...' 字面量


class TestNothingIsRewritten(_Base):
    """debug 不许改写任何内容 —— 连请求头里的密钥也不抹（用户明确要求）。"""

    def test_no_redact_helper_exists(self):
        # redact() 曾经存在，会把 Authorization 抹成 ***已隐去*** —— 已删除
        self.assertFalse(hasattr(debuglog, "redact"))

    def test_auth_header_is_kept_verbatim(self):
        debuglog.kv("HTTP 请求", headers={"Authorization": "Bearer sk-abc"})
        out = self.output()
        self.assertIn("sk-abc", out)          # 原样保留，不做任何处理
        self.assertNotIn("已隐去", out)


class TestSwitch(unittest.TestCase):
    def _clear(self):
        lg = debuglog._logger()
        for h in list(lg.handlers):
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    def test_off_by_default(self):
        # 不能因为"有人在用这个模块"就产生输出
        self._clear()
        self.assertFalse(debuglog.enabled())
        self.assertIsNone(debuglog.log_file())

    def test_kv_and_log_content(self):
        lg = debuglog._logger()
        debuglog.enable(stream=io.StringIO())      # 先 enable（它会清 handler）
        cap = _Capture()
        lg.addHandler(cap)
        try:
            debuglog.log("测试", "内容")
            debuglog.kv("键值", a=1, b="二")
            out = "\n".join(cap.messages)
            self.assertIn("内容", out)
            self.assertIn("a = 1", out)
            self.assertIn("b = 二", out)
        finally:
            lg.removeHandler(cap)
            self._clear()


if __name__ == "__main__":
    unittest.main()
