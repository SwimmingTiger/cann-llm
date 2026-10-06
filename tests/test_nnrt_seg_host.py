"""``nnrt_seg`` 的**主机侧词表表**加载语义（与设备无关 ✓）。

§164 立的规矩：
  · 没有 ``emb_manifest.json`` ⇒ 返回 None ✓（这个模型不需要主机侧表 ✓）
  · 清单在、后续任何一步失败 ⇒ ★抛 ModelLoadError 并带出真因✗★
    —— 尤其【缺 numpy】✗：原来 `except Exception: return None` 会把它吞掉，
       上层于是报成"没找到主机侧嵌入表"，完全指错方向（实测踩到）。

这里刻意不加载 ``libmindspore_lite_ndk.so`` ✓（那需要鸿蒙设备 ✓）。
"""

import os
import struct
import tempfile
import unittest

from cann_llm.backends import nnrt_seg
from cann_llm.backends.nnrt_seg import _HostTable, _load_host_table
from cann_llm.errors import ModelLoadError

try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:                                          # pragma: no cover
    _HAS_NUMPY = False


def _write_manifest(d: str, **info) -> str:
    import json
    path = os.path.join(d, "emb_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    return path


class TestNoManifest(unittest.TestCase):
    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_load_host_table(d))

    def test_dir_with_other_files_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tokenizer.json"), "w") as fh:
                fh.write("{}")
            self.assertIsNone(_load_host_table(d))


class TestBrokenManifestRaises(unittest.TestCase):
    """★清单在 ⇒ 不许再静默返回 None ✗★"""

    def test_missing_weight_file(self):
        with tempfile.TemporaryDirectory() as d:
            man = _write_manifest(d, shape=[2, 3], dtype="float16", file="emb_f16.bin")
            with self.assertRaises(ModelLoadError) as cm:
                _load_host_table(d)
            msg = str(cm.exception)
            self.assertIn("emb_f16.bin", msg)
            self.assertIn(man, msg)                 # 把清单路径也带出来 ✓
            self.assertIn("不在", msg)

    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "emb_manifest.json"), "w", encoding="utf-8") as fh:
                fh.write("{ not json")
            with self.assertRaises(ModelLoadError) as cm:
                _load_host_table(d)
            self.assertIn("JSON", str(cm.exception))

    def test_missing_shape(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, dtype="float16", file="emb_f16.bin")
            with self.assertRaises(ModelLoadError) as cm:
                _load_host_table(d)
            self.assertIn("shape", str(cm.exception))

    def test_numpy_missing_message_is_actionable(self):
        """★缺 numpy 要指名道姓✗★（而不是说"没找到嵌入表" ✗）。"""
        class _Boom:
            def __init__(self, *a, **kw):
                raise ImportError("No module named 'numpy'")

        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, shape=[2, 3], dtype="float16", file="emb_f16.bin")
            with open(os.path.join(d, "emb_f16.bin"), "wb") as fh:
                fh.write(b"\x00" * (2 * 3 * 2))
            real, nnrt_seg._HostTable = nnrt_seg._HostTable, _Boom
            try:
                with self.assertRaises(ModelLoadError) as cm:
                    _load_host_table(d)
            finally:
                nnrt_seg._HostTable = real
            msg = str(cm.exception)
            self.assertIn("numpy", msg)
            self.assertIn("install_numpy.sh", msg)  # 一指禅到脚本 ✓（鸿蒙上只跑 pip 不够 ✓）
            self.assertIn("PYTHON=", msg)           # 第二条出路也给出 ✓


@unittest.skipUnless(_HAS_NUMPY, "需要 numpy ✓")
class TestHostTableMath(unittest.TestCase):
    def test_row_and_matmul(self):
        with tempfile.TemporaryDirectory() as d:
            vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            with open(os.path.join(d, "emb_f16.bin"), "wb") as fh:
                for r in vals:
                    fh.write(struct.pack("<3e", *r))     # 'e' = fp16 ✓
            _write_manifest(d, shape=[2, 3], dtype="float16", file="emb_f16.bin")
            tab = _load_host_table(d)
            self.assertIsInstance(tab, _HostTable)
            self.assertEqual((tab.rows, tab.cols), (2, 3))
            self.assertEqual([round(float(x), 3) for x in tab.row(1)], [4.0, 5.0, 6.0])
            lg = tab.matmul([1.0, 1.0, 1.0])             # 行内积 ✓
            self.assertEqual([round(v, 3) for v in lg], [6.0, 15.0])
            # 常驻 fp32 副本一旦建好，结果必须一致 ✓（§163 的提速路径 ✓）
            again = tab.matmul([1.0, 1.0, 1.0])
            self.assertEqual([round(v, 3) for v in again], [6.0, 15.0])


if __name__ == "__main__":                                   # pragma: no cover
    unittest.main()
