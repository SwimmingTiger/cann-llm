"""`--large-mem`：补丁表 + 启动参数构造。

三件必须成立的事：

1. **补丁表是单一来源**（`cann_llm.large_mem`），lldb 侧脚本从它导入 —— 两处各写
   一份迟早会漂移 ✗；而且每条都带"期望原字节"，补丁前先比对，版本一变就明确跳过
   （别把别处改坏）。
2. **`memmove_s` 那条极性相反**：它是 `b.hs` 跳向【正常】路径，`nop` 反而会掉进
   `ERANGE` ⇒ 必须改成无条件跳转（`06 00 00 14` = `b 0x5150`）。这是实测踩过的坑，
   单测把它钉住，防止以后"顺手改成 nop"。
3. argv 指向的**编排脚本与 lldb 脚本必须真的在仓库里**（发布包里也要在）。
"""
import ast
import os
import sys
import tempfile
import unittest
from unittest import mock

from cann_llm import large_mem, lldb_launch

NOP = b"\x1f\x20\x03\xd5"


def _script_wrapper():
    fh = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    fh.write("#!/bin/sh\nexec python3.12 \"$@\"\n")
    fh.close()
    return fh.name


class TestPatchTable(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(len(large_mem.PATCHES), 4)
        for p in large_mem.PATCHES:
            self.assertEqual(len(p.expect), 4, p.what)
            self.assertEqual(len(p.patch), 4, p.what)
            self.assertNotEqual(p.expect, p.patch, p.what)
            self.assertTrue(p.module.endswith(".so"), p.module)

    def test_modules_are_the_two_we_know(self):
        mods = {p.module for p in large_mem.PATCHES}
        self.assertEqual(mods, {"libhiai_ir.so", "libsec_shared.z.so"})

    def test_offsets_unique_per_module(self):
        seen = set()
        for p in large_mem.PATCHES:
            key = (p.module, p.offset)
            self.assertNotIn(key, seen, "偏移重复：%s" % (key,))
            seen.add(key)

    def test_nop_patches(self):
        """三处「跳向错误路径」的 cbnz ⇒ nop 即可。"""
        nops = [p for p in large_mem.PATCHES if p.patch == NOP]
        self.assertEqual(len(nops), 3)

    def test_memmove_patch_is_branch_not_nop(self):
        """★memmove_s 极性相反：必须是无条件跳转，不能是 nop★"""
        memmove = [p for p in large_mem.PATCHES if "memmove" in p.what]
        self.assertEqual(len(memmove), 1)
        p = memmove[0]
        self.assertEqual(p.patch, b"\x06\x00\x00\x14")   # b 0x5150
        self.assertNotEqual(p.patch, NOP)
        self.assertIn("极性", p.note + p.what + large_mem.describe_patches()[0])

    def test_cbnz_encodings_differ_between_sites(self):
        """★两条 cbnz x9 的立即数不同 —— 不能互相照抄★

        实测踩过：`memset_s` 那处记成了 memcpy_s 的 `09 03 00 b5`，
        实际是 `89 01 00 b5`（同为 `cbnz x9`，但跳转距离不同）。
        这正是"补丁前必须校验原字节"的原因 —— 照抄就会在运行期被拦下。
        """
        by_what = {p.what: p for p in large_mem.PATCHES}
        memcpy = [p for p in large_mem.PATCHES if "memcpy_s（导出）" in p.what][0]
        memset = by_what["securec memset_s"]
        self.assertEqual(memcpy.expect, b"\x09\x03\x00\xb5")
        self.assertEqual(memset.expect, b"\x89\x01\x00\xb5")
        self.assertNotEqual(memcpy.expect, memset.expect)

    def test_describe_lists_all(self):
        lines = large_mem.describe_patches()
        self.assertEqual(len(lines), len(large_mem.PATCHES))
        joined = "\n".join(lines)
        self.assertIn("libhiai_ir.so", joined)
        self.assertIn("libsec_shared.z.so", joined)


class TestStripFlag(unittest.TestCase):
    def test_removes_flag(self):
        rest, found = large_mem.strip_large_mem(["-d", "m", "--large-mem", "--temp", "0"])
        self.assertTrue(found)
        self.assertEqual(rest, ["-d", "m", "--temp", "0"])

    def test_removes_equals_form(self):
        rest, found = large_mem.strip_large_mem(["--large-mem=1"])
        self.assertTrue(found)
        self.assertEqual(rest, [])

    def test_absent(self):
        rest, found = large_mem.strip_large_mem(["-d", "m"])
        self.assertFalse(found)
        self.assertEqual(rest, ["-d", "m"])


class TestRendezvous(unittest.TestCase):
    """★第一次 build 之前的"报到—等放行"★（--large-mem 专用）。

    三件必须成立的事：
    1. 没设 `CANN_LLM_LARGE_MEM_RENDEZVOUS` 时**立刻返回**（普通运行零开销 ✓）；
    2. 放行文件已存在时**不等**（调试器先到的情况）⇒ 不浪费那几十秒 ✓；
    3. 放行文件不出现时**必须超时走人** —— 绝不能把程序永久卡住 ✗
    """

    def test_noop_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(large_mem.RENDEZVOUS_ENV, None)
            self.assertFalse(large_mem.rendezvous(log=lambda _m: None))

    def test_returns_immediately_when_release_exists(self):
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.join(d, "go")
            open(rel, "w").write("go\n")
            with mock.patch.dict(os.environ,
                                 {large_mem.RENDEZVOUS_ENV: rel}, clear=False):
                with mock.patch.object(large_mem, "preload_targets",
                                       return_value=[]) as pre:
                    # 幂等标志：这里要单独跑，先复位
                    large_mem._rendezvous_done = False
                    msgs = []
                    self.assertTrue(large_mem.rendezvous(log=msgs.append))
                    self.assertTrue(any("调试器已就位" in m for m in msgs))
                    pre.assert_called_once()          # 还是要预加载 ✓

    def test_times_out_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as d:
            rel = os.path.join(d, "never")
            with mock.patch.dict(os.environ,
                                 {large_mem.RENDEZVOUS_ENV: rel,
                                  large_mem.WAIT_ENV: "0.2"}, clear=False):
                with mock.patch.object(large_mem, "preload_targets", return_value=[]):
                    large_mem._rendezvous_done = False
                    msgs = []
                    self.assertTrue(large_mem.rendezvous(log=msgs.append))
                    self.assertTrue(any("超时" in m for m in msgs))


class TestBuildCallsRendezvous(unittest.TestCase):
    """app 侧：两条 build 路径都必须在 build **之前**调用 rendezvous ✓"""

    def test_nnrt_backend_calls_it_before_build(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "src", "cann_llm",
                                "backends", "nnrt.py"), encoding="utf-8").read()
        i = src.index("_large_mem_rendezvous()")
        j = src.index("OH_AI_ModelBuildFromFile(")
        self.assertLess(i, j, "rendezvous 必须在 build 之前调用 ✗")

    def test_gemma4_runner_calls_it_before_build(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "src", "cann_llm",
                                "backends", "gemma4_runner.py"), encoding="utf-8").read()
        i = src.index("_large_mem_rendezvous()")
        j = src.index("L.OH_AI_ModelBuildFromFile(")
        self.assertLess(i, j, "rendezvous 必须在 build 之前调用 ✗")


class TestShippedFiles(unittest.TestCase):
    """argv 会把这些路径交给 sh / lldb，所以它们必须真的在仓库里。"""

    def test_driver_exists(self):
        self.assertTrue(os.path.exists(large_mem.large_mem_driver_path()),
                        large_mem.large_mem_driver_path())

    def test_lldb_script_exists(self):
        self.assertTrue(os.path.exists(large_mem.large_mem_script_path()),
                        large_mem.large_mem_script_path())

    def test_lldb_script_registers_commands(self):
        with open(large_mem.large_mem_script_path(), encoding="utf-8") as fh:
            text = fh.read()
        # lldb 通过这两个命令名调用它
        self.assertIn("command script add", text)
        self.assertIn("large_mem_install", text)
        self.assertIn("large_mem_report_exit", text)


class TestBuildLargeMemArgv(unittest.TestCase):
    def _exists_for(self, paths):
        return lambda p: p in paths

    def test_uses_driver(self):
        driver = large_mem.large_mem_driver_path()
        script = large_mem.large_mem_script_path()
        with mock.patch.object(os.path, "exists", self._exists_for({driver, script, "/some/lldb"})), \
             mock.patch.object(lldb_launch, "find_gdbserver", return_value="/some/gdbserver"), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/some/lldb"}):
            argv, hints, err = large_mem.build_large_mem_argv(
                sys.executable, "cann_llm.cli.chat", ["-d", "m"])
        self.assertEqual(err, "")
        self.assertEqual(argv[0], "/bin/sh")
        self.assertEqual(argv[1], driver)
        # 编排脚本拿到的是 (python, 模块名, 模块参数…)：它用 runpy 在同进程跑模块
        self.assertEqual(argv[2:], [sys.executable, "cann_llm.cli.chat", "-d", "m"])
        text = "\n".join(hints)
        self.assertIn(large_mem.LIMIT_STOCK, text)
        self.assertIn(large_mem.LIMIT_PATCHED, text)
        self.assertIn("large_mem", " ".join(argv) + text)   # 提示里点明脚本名

    def test_port_is_configurable(self):
        driver = large_mem.large_mem_driver_path()
        script = large_mem.large_mem_script_path()
        with mock.patch.object(os.path, "exists", self._exists_for({driver, script, "/some/lldb"})), \
             mock.patch.object(lldb_launch, "find_gdbserver", return_value="/some/gdbserver"), \
             mock.patch.dict(os.environ, {"CANN_LLM_LLDB": "/some/lldb",
                                          "CANN_LLM_LLDB_PORT": "6001"}):
            argv, hints, _ = large_mem.build_large_mem_argv(
                sys.executable, "cann_llm.cli.chat", [])
        self.assertTrue(any("6001" in h for h in hints))

    def test_rejects_script_wrapper(self):
        path = _script_wrapper()
        try:
            with mock.patch.object(os.path, "exists", lambda p: True):
                argv, hints, err = large_mem.build_large_mem_argv(
                    path, "cann_llm.cli.chat", [])
            self.assertIsNone(argv)
            self.assertIn("不是 ELF", err)
        finally:
            os.unlink(path)

    def test_reports_missing_gdbserver(self):
        driver = large_mem.large_mem_driver_path()
        script = large_mem.large_mem_script_path()
        with mock.patch.object(os.path, "exists", self._exists_for({driver, script})), \
             mock.patch.object(lldb_launch, "find_gdbserver", return_value=""):
            argv, hints, err = large_mem.build_large_mem_argv(
                sys.executable, "cann_llm.cli.chat", [])
        self.assertIsNone(argv)
        self.assertIn("huawei-debug-lldb-server", err)

    def test_reports_missing_lldb(self):
        driver = large_mem.large_mem_driver_path()
        script = large_mem.large_mem_script_path()
        with mock.patch.object(os.path, "exists", self._exists_for({driver, script})), \
             mock.patch.object(lldb_launch, "find_gdbserver", return_value="/some/gdbserver"), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CANN_LLM_LLDB", None)
            os.environ.pop("LLDB", None)
            argv, hints, err = large_mem.build_large_mem_argv(
                sys.executable, "cann_llm.cli.chat", [])
        self.assertIsNone(argv)
        self.assertIn("lldb", err)


class TestLldbScriptApiUsage(unittest.TestCase):
    """lldb 的 SWIG 接口对参数类型很挑 —— 静态查几处已知的坑，防回归。

    实测踩过：`SBListener.WaitForEvent(1.0, event)` 里第二个参数是 **uint32_t**，
    传 float 直接

        TypeError: in method 'SBListener_WaitForEvent', argument 2 of type 'uint32_t'

    ★而它只在"进程活得够久"时才暴露★：短命进程还没轮到那行就退出了 ⇒
    单测里跑个假 lldb 也照不出来，所以这里做静态检查 ✓
    """

    def _calls(self, attr):
        with open(large_mem.large_mem_script_path(), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr]

    def test_waitforevent_gets_int_seconds(self):
        calls = self._calls("WaitForEvent")
        self.assertTrue(calls, "脚本里应当有 WaitForEvent 调用")
        for call in calls:
            self.assertTrue(call.args, "WaitForEvent 必须给超时参数")
            arg = call.args[0]
            self.assertIsInstance(arg, ast.Constant, "超时应当是字面量")
            self.assertIsInstance(arg.value, int, "★超时必须是 int（uint32_t），不能是 float★")
            self.assertNotIsInstance(arg.value, bool)


if __name__ == "__main__":
    unittest.main()
