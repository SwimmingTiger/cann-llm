"""调试启动参数构造（server 与 chat 共用那份逻辑）。

两件必须成立的事：

1. 本机没有可用的 `lldb -- <binary>`（会报 `'A' packet returned an error: 8`），
   所以有 huawei-debug-lldb-server 时必须走 gdbserver 中转；
2. 传给调试器的必须是**真二进制**（ELF）—— gdbserver 用 execve 直接拉起进程、
   不解析 shebang，给 `#!/bin/sh` 包装器会得到 `execve failed: Operation not
   permitted`。这种情况要求直接报错，不做启发式替换。
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

from cann_llm import lldb_launch


def _script_wrapper():
    """造一个「脚本包装器」临时文件，返回路径（调用方负责删除）。"""
    fh = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    fh.write("#!/bin/sh\nexec python3.12 \"$@\"\n")
    fh.close()
    return fh.name


class TestBuildDebugArgv(unittest.TestCase):
    def test_uses_gdbserver_when_present(self):
        # 本机：有 huawei-debug-lldb-server 时必须走 gdbserver 中转
        # （直接 lldb -- <python> 会 "error: 'A' packet returned an error: 8"）
        with mock.patch.object(os.path, "exists", lambda p: p == lldb_launch.DEFAULT_GDBSERVER):
            argv, hints, err = lldb_launch.build_debug_argv(sys.executable, ["-m", "x"])
        self.assertEqual(err, "")
        self.assertEqual(argv[0], lldb_launch.DEFAULT_GDBSERVER)
        self.assertIn("gdbserver", argv)
        self.assertIn("--native-regs", argv)
        self.assertEqual(argv[argv.index("--") + 1:],
                         [sys.executable, "-X", "faulthandler", "-m", "x"])
        self.assertTrue(any("gdb-remote" in h for h in hints))

    def test_port_is_configurable(self):
        with mock.patch.object(os.path, "exists", lambda p: p == lldb_launch.DEFAULT_GDBSERVER), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB_PORT": "6000"}):
            argv, hints, _ = lldb_launch.build_debug_argv(sys.executable, ["-m", "x"])
        self.assertIn("127.0.0.1:6000", argv)
        self.assertTrue(any("6000" in h for h in hints))

    def test_falls_back_to_plain_lldb(self):
        with mock.patch.object(os.path, "exists", lambda p: p == "/some/lldb"), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/some/lldb"}):
            argv, hints, err = lldb_launch.build_debug_argv(sys.executable, ["-m", "x"])
        self.assertEqual(err, "")
        self.assertEqual(argv, ["/some/lldb", "--", sys.executable,
                                "-X", "faulthandler", "-m", "x"])

    def test_reports_when_no_debugger(self):
        with mock.patch.object(os.path, "exists", lambda p: False), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/nope"}):
            argv, hints, err = lldb_launch.build_debug_argv(sys.executable, ["-m", "x"])
        self.assertIsNone(argv)
        self.assertIn("找不到调试器", err)


class TestRequiresRealExecutable(unittest.TestCase):
    """--lldb 只接受真二进制（ELF），脚本包装器直接报错。"""

    def test_detects_script_wrapper(self):
        path = _script_wrapper()
        try:
            self.assertFalse(lldb_launch.is_real_executable(path))
        finally:
            os.unlink(path)

    def test_real_binary_is_accepted(self):
        self.assertTrue(lldb_launch.is_real_executable(sys.executable))

    def test_missing_path_is_not_executable(self):
        self.assertFalse(lldb_launch.is_real_executable("/nonexistent/python3"))

    def test_script_wrapper_is_rejected_with_message(self):
        path = _script_wrapper()
        try:
            with mock.patch.object(os.path, "exists",
                                   lambda p: p == lldb_launch.DEFAULT_GDBSERVER):
                argv, hints, err = lldb_launch.build_debug_argv(path, ["-m", "x"])
            self.assertIsNone(argv)
            self.assertIn("不是 ELF", err)
            self.assertIn("shebang", err)          # 说明为什么要报错
        finally:
            os.unlink(path)


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
