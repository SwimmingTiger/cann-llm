"""cllm 后端（cann-local-lm 引擎）里**纯 Python** 那部分的单测。

为什么只测这些：真正执行推理要 ARM 设备上的 ``libcann_lm_engine.so`` + NPU，
在 CI 上加载不了 ✗。但**路径解析 / 库查找 / 状态码翻译**是纯逻辑，而且是用户
最容易配错的地方（§「文档里那 7 条」），所以逐条钉住 ✓。

对应文档：``docs/cann-local-lm.md``（转换 → 编译 → 部署 → 运行）。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from cann_llm.backends import available_backends, create_backend
from cann_llm.backends.cllm import (
    CLLM_LIB_ENV,
    CllmBackend,
    find_engine_lib,
    resolve_artifact,
    status_text,
)
from cann_llm.errors import BackendUnavailableError, ModelLoadError


def _touch(path: str, text: str = "{}") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _layout(root: str) -> None:
    """标准部署包布局（与 cann-local-lm 的产出对应）。"""
    _touch(os.path.join(root, "prefill", "prefill.omc"))
    _touch(os.path.join(root, "decode", "decode.omc"))
    os.makedirs(os.path.join(root, "prefill", "weight"), exist_ok=True)
    os.makedirs(os.path.join(root, "decode", "weight"), exist_ok=True)
    _touch(os.path.join(root, "model.json"),
           json.dumps({"model_config": {"kv_cache_max_len": 576, "vocab_size": 262144,
                                        "num_hidden_layers": 35,
                                        "model_type": "gemma4"}}))
    _touch(os.path.join(root, "tokenizer.json"),
           json.dumps({"model": {"type": "BPE"}}))


class TestResolveArtifact(unittest.TestCase):
    def test_default_layout(self):
        with tempfile.TemporaryDirectory() as d:
            _layout(d)
            art = resolve_artifact(d)
            self.assertEqual(art["prefill_omc"], os.path.join(d, "prefill", "prefill.omc"))
            self.assertEqual(art["decode_omc"], os.path.join(d, "decode", "decode.omc"))
            self.assertEqual(art["meta_json"], os.path.join(d, "model.json"))
            self.assertEqual(art["prefill_weight"], os.path.join(d, "prefill", "weight"))
            self.assertEqual(art["tokenizer_path"], os.path.join(d, "tokenizer.json"))

    def test_testcase_json_overrides(self):
        """``testcase.json`` 里的相对路径按模型目录展开、绝对路径原样用。"""
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            _layout(d)
            _touch(os.path.join(other, "p.omc"))
            _touch(os.path.join(d, "testcase.json"), json.dumps({
                "model_path_prefill": "custom/p.omc",
                "model_path_decode": os.path.join(other, "p.omc"),
                "model_meta_config": "model.json",
                "model_weight_dir_prefill": "prefill/weight",
                "model_weight_dir_decode": "decode/weight",
            }))
            _touch(os.path.join(d, "custom", "p.omc"))
            art = resolve_artifact(d)
            self.assertEqual(art["prefill_omc"], os.path.join(d, "custom", "p.omc"))
            self.assertEqual(art["decode_omc"], os.path.join(other, "p.omc"))

    def test_missing_lists_all_and_hints_doc(self):
        """缺文件一次列全（不要一个一个报），并指到文档 ✓。"""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ModelLoadError) as cm:
                resolve_artifact(d)
            msg = str(cm.exception)
            self.assertIn("prefill_omc", msg)
            self.assertIn("decode_weight", msg)
            self.assertIn("docs/cann-local-lm.md", msg)

    def test_bad_testcase_json_is_load_error(self):
        with tempfile.TemporaryDirectory() as d:
            _layout(d)
            _touch(os.path.join(d, "testcase.json"), "{not json")
            with self.assertRaises(ModelLoadError):
                resolve_artifact(d)


class TestFindEngineLib(unittest.TestCase):
    """注意：查找顺序里 ` <cann-llm>/lib ` 排很前 —— 本机装过引擎库时它会**先命中** ✓，
    所以这里把 repo_root 指到空目录，测的才是"模型目录/环境变量"这两级 ✓。"""

    def setUp(self):
        from cann_llm import engine_build as eb
        self._eb = eb
        self._real = eb.repo_root
        self._tmp = os.path.join(eb.repo_root(), ".tmp", "unit-cllm-backend")
        os.makedirs(self._tmp, exist_ok=True)
        eb.repo_root = lambda: self._tmp

    def tearDown(self):
        self._eb.repo_root = self._real

    def test_env_wins(self):
        with tempfile.TemporaryDirectory() as d:
            lib = os.path.join(d, "libcann_lm_engine.so")
            _touch(lib, "x")
            old = os.environ.get(CLLM_LIB_ENV)
            os.environ[CLLM_LIB_ENV] = lib
            try:
                self.assertEqual(find_engine_lib(d), lib)
            finally:
                if old is None:
                    os.environ.pop(CLLM_LIB_ENV, None)
                else:
                    os.environ[CLLM_LIB_ENV] = old

    def test_model_dir_lib_or_py_layout(self):
        """部署包两种放法都认：``lib/`` 与 ``py/cann_local_lm/lib/``。"""
        for rel in (("lib",), ("py", "cann_local_lm", "lib")):
            with tempfile.TemporaryDirectory() as d:
                lib = os.path.join(d, *rel, "libcann_lm_engine.so")
                _touch(lib, "x")
                old = os.environ.pop(CLLM_LIB_ENV, None)
                try:
                    self.assertEqual(find_engine_lib(d), lib)
                finally:
                    if old is not None:
                        os.environ[CLLM_LIB_ENV] = old

    def test_missing_raises_with_paths_and_hint(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.pop(CLLM_LIB_ENV, None)
            try:
                with self.assertRaises(BackendUnavailableError) as cm:
                    find_engine_lib(d)
            finally:
                if old is not None:
                    os.environ[CLLM_LIB_ENV] = old
            msg = str(cm.exception)
            self.assertIn(CLLM_LIB_ENV, msg)
            self.assertIn("docs/cann-local-lm.md", msg)


class TestStatusAndHelpers(unittest.TestCase):
    def test_status_text(self):
        self.assertEqual(status_text(0), "SUCCESS")
        self.assertEqual(status_text(2), "INVALID_ARGUMENT")
        self.assertEqual(status_text(3), "INVALID_STATE")
        self.assertEqual(status_text(99), "status 99")

    def test_default_tokenizer_type(self):
        with tempfile.TemporaryDirectory() as d:
            _layout(d)
            self.assertEqual(CllmBackend(model_dir=d)._info.backend, "cllm")
            from cann_llm.backends.cllm import _default_tokenizer_type
            self.assertEqual(_default_tokenizer_type(d), "BPE")
            _touch(os.path.join(d, "tokenizer.json"), json.dumps({"model": {"type": "Unigram"}}))
            self.assertEqual(_default_tokenizer_type(d), "SPM")

    def test_chat_template_is_gemma_for_gemma(self):
        from cann_llm.backends.cllm import _chat_template
        self.assertEqual(_chat_template({"model_type": "gemma4"}), "gemma")
        self.assertEqual(_chat_template({"model_type": "qwen3"}), "chatml")


class TestChatTemplateDetect(unittest.TestCase):
    """没有 model_type 字段时，按 tokenizer 标记认出 gemma ✓（否则会套 ChatML ✗）。"""

    def _write(self, tmp, added):
        import json, os
        with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as fh:
            json.dump({"added_tokens": [{"content": t, "id": i} for i, t in enumerate(added)],
                       "model": {"vocab": {}}}, fh)

    def test_gemma_markers(self):
        import tempfile
        from cann_llm.backends.cllm import _chat_template
        with tempfile.TemporaryDirectory() as d:
            self._write(d, ["<|turn>", "<turn|>"])
            self.assertEqual(_chat_template({}, d), "gemma")

    def test_plain_is_chatml(self):
        import tempfile
        from cann_llm.backends.cllm import _chat_template
        with tempfile.TemporaryDirectory() as d:
            self._write(d, ["<|im_start|>", "<|im_end|>"])
            self.assertEqual(_chat_template({}, d), "chatml")

    def test_model_type_wins(self):
        from cann_llm.backends.cllm import _chat_template
        self.assertEqual(_chat_template({"model_type": "gemma4"}, None), "gemma")


class TestSpecialStopIds(unittest.TestCase):
    """gemma4 的 <turn|>(106) 必须进 stop_token_ids（否则一直重复它 ✗）。"""

    def test_parse_eos_list_and_end_markers(self):
        """停止 id：模型自报 eos 列表 + 结束型 special token；**开始标记不算** ✓"""
        from cann_llm.backends.cllm import special_stop_ids
        meta = {"tokenizer_config": {"special_tokens": {"<turn|>": 106, "<|turn>": 105}}}
        self.assertEqual(special_stop_ids(meta, {"eos_token_id": [1, 106, 50]}), [1, 106, 50])
        self.assertEqual(special_stop_ids(meta), [106])          # <|turn> 不参与
        self.assertEqual(special_stop_ids({}, {"eos_token_id": 1}), [1])

    def test_missing_or_bad(self):
        from cann_llm.backends.cllm import special_stop_ids
        self.assertEqual(special_stop_ids({}), [])
        self.assertEqual(special_stop_ids({"tokenizer_config": {}}), [])
        self.assertEqual(special_stop_ids({"tokenizer_config": {"special_tokens": None}}), [])
        self.assertEqual(special_stop_ids(
            {"tokenizer_config": {"special_tokens": {"bad": "abc"}}}), [])


class TestBackendLoadErrors(unittest.TestCase):
    def test_incomplete_model_dir_raises_before_touching_lib(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ModelLoadError):
                CllmBackend(model_dir=d).load()

    def test_bad_lib_path_is_backend_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            _layout(d)
            bogus = os.path.join(d, "libcann_lm_engine.so")
            _touch(bogus, "not an elf")
            with self.assertRaises(BackendUnavailableError):
                CllmBackend(model_dir=d, lib_path=bogus).load()

    def test_registered_and_constructible(self):
        self.assertIn("cllm", available_backends())
        with tempfile.TemporaryDirectory() as d:
            backend = create_backend("cllm", model_dir=d, model_id="m", context_length=0)
            self.assertIsInstance(backend, CllmBackend)
            self.assertEqual(backend.name, "cllm")
            self.assertTrue(backend.supports_streaming)


class TestSplitStop(unittest.TestCase):
    """流式 stop 裁剪 —— 引擎会把 stop 串留在文本里 ✗，必须在后端裁掉 ✓。"""

    def test_hit_trims_and_flags(self):
        from cann_llm.backends.cllm import split_stop
        self.assertEqual(split_stop("你好<end_of_turn>", ["<end_of_turn>"]),
                         ("你好", "", True))

    def test_partial_prefix_is_held_back(self):
        from cann_llm.backends.cllm import split_stop
        self.assertEqual(split_stop("你好<end", ["<end_of_turn>"]),
                         ("你好", "<end", False))

    def test_no_stops_passthrough(self):
        from cann_llm.backends.cllm import split_stop
        self.assertEqual(split_stop("abc", []), ("abc", "", False))

    def test_earliest_stop_wins(self):
        from cann_llm.backends.cllm import split_stop
        out, hold, hit = split_stop("aa<e1>bb<e2>", ["<e2>", "<e1>"])
        self.assertEqual((out, hold, hit), ("aa", "", True))

    def test_across_chunks(self):
        """分两块到达也能正确裁掉（模拟流式两个 token）。"""
        from cann_llm.backends.cllm import split_stop
        out1, hold, hit1 = split_stop("你好<en", ["<end_of_turn>"])
        self.assertEqual((out1, hold, hit1), ("你好", "<en", False))
        out2, hold2, hit2 = split_stop(hold + "d_of_turn>", ["<end_of_turn>"])
        self.assertEqual((out2, hold2, hit2), ("", "", True))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
