"""``launcher_server.run_server`` 的回归测试（§166 ✓）。

只测「参数解析 → 后端判定 → 模型目录解析」这一段 ✓ —— **不真的起 uvicorn** ✗
（用不存在的模型目录让它走到 ``die()`` 为止 ✓）。

§166 的 bug：``run_server`` 里曾有一句函数内的
``from .launcher import detect_backend, find_model_dir`` ✗ —— 因为出现在函数体内，
``find_model_dir`` 在整个函数里都被当成**局部名** ✓；用户一旦显式给 ``-b``（那段被跳过 ✗），
后面 ``d = find_model_dir(root, …)`` 就 ``UnboundLocalError`` 崩掉 ✗
（实测：``scripts/start_server.sh -d ../models/Qwen3-8B -b hiai`` 必崩 ✓）。
"""

import os
import sys
import tempfile
import unittest

from cann_llm import launcher_server


class TestRunServerBackendAndDir(unittest.TestCase):
    """两条路径都必须能走到模型目录解析（而不是崩在局部名未绑定上 ✓）。"""

    def setUp(self):
        self._real_pick = launcher_server.pick_engine_lib
        # 引擎库换成一定存在的文件 ⇒ 免得在无设备的机器上提前 die ✗
        launcher_server.pick_engine_lib = lambda backend: (
            "CANN_LLM_ENGINE_LIB", sys.executable, "假引擎")
        self._env = os.environ.get("LD_LIBRARY_PATH")

    def tearDown(self):
        launcher_server.pick_engine_lib = self._real_pick
        if self._env is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = self._env

    def test_explicit_backend_hits_dir_resolution(self):
        """★给了 -b 也不能崩✓★（原 bug 的复现路径 ✓）。"""
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit) as cm:
                launcher_server.run_server(
                    root, ["-d", "/definitely/missing/model", "-b", "hiai"])
            self.assertNotEqual(cm.exception.code, 0)

    def test_auto_backend_hits_dir_resolution(self):
        """不传 -b ⇒ 走 detect_backend 分支 ✓，同样要能走到目录解析 ✓。"""
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit) as cm:
                launcher_server.run_server(root, [])
            self.assertNotEqual(cm.exception.code, 0)

    # 注：不在这里测"分段布局能通过目录解析" ✗ —— 那会一路走到真正拉起 uvicorn ✗，
    # 不适合放进单元测试 ✓。后端判定本身由 tests/test_backend_detect.py 覆盖 ✓。


if __name__ == "__main__":                                   # pragma: no cover
    unittest.main()
