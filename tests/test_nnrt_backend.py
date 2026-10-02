"""``nnrt`` 后端的单元测试 —— 只覆盖**与设备无关**的部分。

这里刻意不加载 ``libmindspore_lite_ndk.so``：那需要鸿蒙设备。设备相关的部分
（Build / Predict / 数值比对）用 ``examples/mslite-nnrt/`` 里的两个 C 程序验证，
见 ``docs/offline-model-nnrt.md``。
"""

import os
import tempfile
import unittest

from cann_llm.backends import available_backends, create_backend
from cann_llm.backends.nnrt import (
    _DTYPE_FLOAT16,
    _DTYPE_FLOAT32,
    _DTYPE_NAMES,
    _DTYPE_SIZES,
    NnrtBackend,
)
from cann_llm.errors import InvalidRequestError, ModelLoadError


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("nnrt", available_backends())

    def test_create_by_name(self):
        be = create_backend("nnrt", model_dir=".")
        self.assertIsInstance(be, NnrtBackend)
        self.assertEqual(be.name, "nnrt")

    def test_sampler_defaults_empty(self):
        """离线模型不带采样配置 ⇒ 空字典，让上层用自己的兜底值。"""
        self.assertEqual(create_backend("nnrt", model_dir=".").sampler_defaults(), {})


class TestDataTypeEnum(unittest.TestCase):
    """枚举值取自 native/sysroot/usr/include/mindspore/data_type.h。

    踩过的坑：这些是 MindSpore 的 TypeId，不是 0,1,2… 的紧凑编号 ——
    ``FLOAT32`` 是 43，当初按直觉写 1 导致"模型没有 float32 输入"。
    """

    def test_float32_is_43(self):
        self.assertEqual(_DTYPE_FLOAT32, 43)
        self.assertEqual(_DTYPE_FLOAT16, 42)
        self.assertEqual(_DTYPE_NAMES[43], "FLOAT32")
        self.assertEqual(_DTYPE_SIZES[43], 4)
        self.assertEqual(_DTYPE_SIZES[42], 2)

    def test_all_named_types_have_size(self):
        for code in _DTYPE_NAMES:
            self.assertIn(code, _DTYPE_SIZES, f"{code} 缺 size")


class TestModelDirResolution(unittest.TestCase):
    def test_missing_dir(self):
        be = NnrtBackend(model_dir="/nonexistent/nnrt-model")
        with self.assertRaises(ModelLoadError):
            be._find_ms()

    def test_no_ms_in_dir(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "readme.txt"), "w").close()
            with self.assertRaises(ModelLoadError) as cm:
                NnrtBackend(model_dir=d)._find_ms()
            self.assertIn(".ms", str(cm.exception))

    def test_single_ms_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.ms")
            open(p, "w").close()
            self.assertEqual(NnrtBackend(model_dir=d)._find_ms(), p)

    def test_multiple_ms_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("a.ms", "b.ms"):
                open(os.path.join(d, n), "w").close()
            with self.assertRaises(ModelLoadError) as cm:
                NnrtBackend(model_dir=d)._find_ms()
            self.assertIn("多个", str(cm.exception))


class TestPromptParsing(unittest.TestCase):
    def test_parse_csv(self):
        be = NnrtBackend(model_dir=".")
        self.assertEqual(be._parse_prompt("1, 2.5, -3"), [1.0, 2.5, -3.0])

    def test_parse_newlines_and_empty(self):
        be = NnrtBackend(model_dir=".")
        self.assertEqual(be._parse_prompt(""), [])
        self.assertEqual(be._parse_prompt("1\n2,,\n3"), [1.0, 2.0, 3.0])

    def test_non_number_raises(self):
        be = NnrtBackend(model_dir=".")
        with self.assertRaises(InvalidRequestError):
            be._parse_prompt("1,oops")


class TestLifecycle(unittest.TestCase):
    def test_generate_before_load_raises(self):
        from cann_llm.types import GenerationRequest

        be = NnrtBackend(model_dir=".")
        with self.assertRaises(Exception):
            list(be.generate(GenerationRequest(prompt="1")))

    def test_close_is_reentrant(self):
        """退出阶段会被调多次；而且我们**故意不销毁** Context（会 core dump）。"""
        be = NnrtBackend(model_dir=".")
        be.close()
        be.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
