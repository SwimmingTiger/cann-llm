"""调试启动参数构造（server 与 chat 共用那份逻辑）。"""
import os
import unittest
from unittest import mock

from cann_llm import lldb_launch


class TestBuildDebugArgv(unittest.TestCase):
    def test_uses_gdbserver_when_present(self):
        # 本机：有 huawei-debug-lldb-server 时必须走 gdbserver 中转
        # （直接 lldb -- <python> 会 "error: 'A' packet returned an error: 8"）
        with mock.patch.object(os.path, "exists", lambda p: p == lldb_launch.DEFAULT_GDBSERVER):
            argv, hints, err = lldb_launch.build_debug_argv("/usr/bin/python3", ["-m", "x"])
        self.assertEqual(err, "")
        self.assertEqual(argv[0], lldb_launch.DEFAULT_GDBSERVER)
        self.assertIn("gdbserver", argv)
        self.assertIn("--native-regs", argv)
        self.assertEqual(argv[argv.index("--") + 1:], ["/usr/bin/python3", "-X", "faulthandler", "-m", "x"])
        self.assertTrue(any("gdb-remote" in h for h in hints))

    def test_port_is_configurable(self):
        with mock.patch.object(os.path, "exists", lambda p: p == lldb_launch.DEFAULT_GDBSERVER), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB_PORT": "6000"}):
            argv, hints, _ = lldb_launch.build_debug_argv("py", ["-m", "x"])
        self.assertIn("127.0.0.1:6000", argv)
        self.assertTrue(any("6000" in h for h in hints))

    def test_falls_back_to_plain_lldb(self):
        with mock.patch.object(os.path, "exists", lambda p: p == "/some/lldb"), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/some/lldb"}):
            argv, hints, err = lldb_launch.build_debug_argv("py", ["-m", "x"])
        self.assertEqual(err, "")
        self.assertEqual(argv, ["/some/lldb", "--", "py", "-X", "faulthandler", "-m", "x"])

    def test_reports_when_no_debugger(self):
        with mock.patch.object(os.path, "exists", lambda p: False), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/nope"}):
            argv, hints, err = lldb_launch.build_debug_argv("py", ["-m", "x"])
        self.assertIsNone(argv)
        self.assertIn("找不到调试器", err)


class TestStripFlag(unittest.TestCase):
    def test_removes_flag(self):
        rest, found = lldb_launch.strip_flag(["-d", "m", "--lldb", "--temp", "0"])
        self.assertTrue(found)
        self.assertEqual(rest, ["-d", "m", "--temp", "0"])

    def test_absent(self):
        rest, found = lldb_launch.strip_flag(["-d", "m"])
        self.assertFalse(found)
        self.assertEqual(rest, ["-d", "m"])


if __name__ == "__main__":
    unittest.main()
