"""配置层的测试：模型 id 的解析、配置合并优先级。"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cann_llm.config import (  # noqa: E402
    AppConfig, ModelConfig, ServerConfig, load_config,
)


class TestResolvedModelId(unittest.TestCase):
    """model_id 默认取模型目录名 —— 换模型时名字要跟着变。"""

    def test_dir_name_when_id_absent(self):
        mc = ModelConfig(model_dir="/srv/models/qwen3_4b")
        self.assertEqual(mc.resolved_id, "qwen3_4b")

    def test_trailing_slash(self):
        mc = ModelConfig(model_dir="/srv/models/qwen3_4b/")
        self.assertEqual(mc.resolved_id, "qwen3_4b")

    def test_explicit_id_wins(self):
        mc = ModelConfig(model_dir="/srv/models/qwen3_4b", model_id="my-model")
        self.assertEqual(mc.resolved_id, "my-model")

    def test_fallback_without_dir(self):
        self.assertEqual(ModelConfig().resolved_id, "cann-llm")
        # 只有 id 时用它
        self.assertEqual(ModelConfig(model_id="x").resolved_id, "x")

    def test_default_is_not_a_hardcoded_model_name(self):
        """回归：默认值曾是写死的 'qwen2.5-1.5b'，换模型后名字不变。"""
        mc = ModelConfig(model_dir="/srv/models/qwen3_4b")
        self.assertNotIn("1.5b", mc.resolved_id)
        self.assertIsNone(mc.model_id)


class TestMergedKeepsResolvedId(unittest.TestCase):
    """merged() 用 -d 覆盖 model_dir 后，resolved_id 要跟着新目录走。"""

    def test_merged_dir_updates_resolved_id(self):
        cfg = AppConfig(model=ModelConfig(model_dir="/srv/models/old"))
        self.assertEqual(cfg.model.resolved_id, "old")

        cfg2 = cfg.merged(model={"model_dir": "/srv/models/qwen3_4b"})
        self.assertEqual(cfg2.model.resolved_id, "qwen3_4b")

    def test_explicit_override_survives(self):
        cfg = AppConfig(model=ModelConfig(model_dir="/srv/models/old"))
        cfg2 = cfg.merged(model={"model_dir": "/srv/models/new", "model_id": "pinned"})
        self.assertEqual(cfg2.model.resolved_id, "pinned")


class TestLoadConfig(unittest.TestCase):

    def test_load_toml_without_model_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.toml")
            with open(p, "w") as f:
                f.write('[model]\nmodel_dir = "/srv/models/qwen3_4b"\n')
            cfg = load_config(p)
            self.assertIsNone(cfg.model.model_id)
            self.assertEqual(cfg.model.resolved_id, "qwen3_4b")


if __name__ == "__main__":
    unittest.main()


class TestContextLengthAutoDetect(unittest.TestCase):
    """context_length 应从模型的 executor.json 自动读，而不是写死 2048。"""

    def _fake_model(self, d, kv):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "executor.json"), "w") as f:
            import json as _j
            _j.dump({"llm_config": {"kv_cache_max_len": kv}}, f)
        return d

    def test_reads_from_executor_json(self):
        from cann_llm.backends.cann import CannNdkBackend
        with tempfile.TemporaryDirectory() as d:
            self._fake_model(d, 8192)
            self.assertEqual(CannNdkBackend._read_context_length(d), 8192)

    def test_none_when_missing_or_broken(self):
        from cann_llm.backends.cann import CannNdkBackend
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(CannNdkBackend._read_context_length(d))
            with open(os.path.join(d, "executor.json"), "w") as f:
                f.write("{ not json")
            self.assertIsNone(CannNdkBackend._read_context_length(d))
            with open(os.path.join(d, "executor.json"), "w") as f:
                f.write('{"llm_config": {}}')
            self.assertIsNone(CannNdkBackend._read_context_length(d))

    def test_explicit_wins(self):
        from cann_llm.backends.cann import CannNdkBackend
        with tempfile.TemporaryDirectory() as d:
            self._fake_model(d, 8192)
            be = CannNdkBackend(d, context_length=4096)
            self.assertEqual(be.context_length, 4096)


class TestInterpreterLibcConflict(unittest.TestCase):
    """解释器 libc 与引擎不匹配时必须能提前判出来（否则是段错误，无法捕获）。"""

    def test_returns_none_or_message_never_raises(self):
        from cann_llm.backends.cann import interpreter_libc_conflict
        out = interpreter_libc_conflict()          # 不该抛异常
        self.assertTrue(out is None or isinstance(out, str))

    def test_message_mentions_the_fix(self):
        """真有冲突时，提示里要给出可操作的解法。"""
        from cann_llm.backends.cann import interpreter_libc_conflict
        out = interpreter_libc_conflict()
        if out is None:
            self.skipTest("当前解释器本来就兼容（本机是 musl 原生 python）")
        self.assertIn("musl", out)
        self.assertIn("PYTHON=", out)

    def test_non_linux_is_not_blocked(self):
        """读不到 /proc/self/maps（非 Linux）时不应误拦。"""
        import cann_llm.backends.cann as m
        real_open = open
        def fake_open(path, *a, **k):
            if str(path) == "/proc/self/maps":
                raise OSError("no procfs")
            return real_open(path, *a, **k)
        m.open = fake_open
        try:
            self.assertIsNone(m.interpreter_libc_conflict())
        finally:
            m.open = real_open
