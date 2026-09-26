import threading
import time
import unittest

from cann_llm.backends.base import (
    EngineBackend,
    SerializedBackend,
    available_backends,
    create_backend,
    register_backend,
)
from cann_llm.errors import BusyError, InvalidRequestError
from cann_llm.types import GenerationChunk, GenerationRequest, aggregate

from .fakes import FakeBackend


class TestRegistry(unittest.TestCase):
    def test_fake_is_registered(self):
        self.assertIn("fake", available_backends())

    def test_create_by_name(self):
        be = create_backend("fake", reply="ok")
        self.assertIsInstance(be, FakeBackend)
        self.assertEqual("".join(c.text for c in be.generate(
            GenerationRequest(prompt="x"))), "ok")

    def test_unknown_backend(self):
        with self.assertRaises(InvalidRequestError):
            create_backend("nope")

    def test_register_rejects_non_class(self):
        with self.assertRaises(TypeError):
            register_backend("bad")(lambda: None)

    def test_register_sets_name(self):
        @register_backend("tmp-for-test")
        class Tmp(FakeBackend):
            pass
        self.assertEqual(Tmp.name, "tmp-for-test")
        self.assertIn("tmp-for-test", available_backends())


class TestSerializedBackend(unittest.TestCase):
    def test_passthrough(self):
        inner = FakeBackend("abc")
        inner.load()
        be = SerializedBackend(inner)
        self.assertEqual(be.name, inner.name)
        # FakeBackend 没声明流式能力 → False；包装层必须原样透传
        self.assertFalse(be.supports_streaming)

        class Streaming(FakeBackend):
            @property
            def supports_streaming(self):
                return True
        inner2 = Serializing = Streaming()      # noqa: F841
        inner2.load()
        self.assertTrue(SerializedBackend(inner2).supports_streaming)
        r = aggregate(be.generate(GenerationRequest(prompt="x")))
        self.assertEqual(r.text, "abc")
        self.assertIs(be.info, inner.info)
        be.close()
        self.assertTrue(inner.closed)

    def test_serializes_concurrent_calls(self):
        """两个线程同时生成时不能重叠。"""
        overlap = []

        class Slow(FakeBackend):
            def __init__(self):
                super().__init__(reply="")
                self._active = 0
                self._guard = threading.Lock()

            def generate(self, request):
                with self._guard:
                    self._active += 1
                    if self._active > 1:
                        overlap.append(self._active)
                time.sleep(0.05)
                yield GenerationChunk(text="x", finish_reason="stop")
                with self._guard:
                    self._active -= 1

        inner = Slow()
        inner.load()
        be = SerializedBackend(inner, max_queue=4, timeout_s=5)

        def run():
            list(be.generate(GenerationRequest(prompt="x")))

        ts = [threading.Thread(target=run) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(overlap, [], "检测到并发重叠")

    def test_in_flight_counter(self):
        inner = FakeBackend("x")
        inner.load()
        be = SerializedBackend(inner)
        self.assertEqual(be.in_flight, 0)
        list(be.generate(GenerationRequest(prompt="z")))
        self.assertEqual(be.in_flight, 0)

    def test_queue_limit_raises_busy(self):
        class Blocking(FakeBackend):
            def __init__(self):
                super().__init__(reply="")
                self.gate = threading.Event()

            def generate(self, request):
                self.gate.wait(2)
                yield GenerationChunk(finish_reason="stop")

        inner = Blocking()
        inner.load()
        be = SerializedBackend(inner, max_queue=1, timeout_s=5)

        t = threading.Thread(target=lambda: list(be.generate(GenerationRequest(prompt="x"))),
                             daemon=True)
        t.start()
        time.sleep(0.1)                 # 让第一个占住锁
        with self.assertRaises(BusyError):
            list(be.generate(GenerationRequest(prompt="y")))
        inner.gate.set()
        t.join(3)

    def test_timeout_raises_busy(self):
        class Blocking(FakeBackend):
            def __init__(self):
                super().__init__(reply="")
                self.gate = threading.Event()

            def generate(self, request):
                self.gate.wait(3)
                yield GenerationChunk(finish_reason="stop")

        inner = Blocking()
        inner.load()
        be = SerializedBackend(inner, max_queue=4, timeout_s=0.1)
        t = threading.Thread(target=lambda: list(be.generate(GenerationRequest(prompt="x"))),
                             daemon=True)
        t.start()
        time.sleep(0.05)
        with self.assertRaises(BusyError):
            list(be.generate(GenerationRequest(prompt="y")))
        inner.gate.set()
        t.join(3)


class TestBackendContract(unittest.TestCase):
    def test_error_propagates_out_of_generator(self):
        from .fakes import ExplodingBackend
        be = ExplodingBackend()
        be.load()
        with self.assertRaises(RuntimeError):
            aggregate(be.generate(GenerationRequest(prompt="x")))

    def test_engine_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            EngineBackend()             # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()


class TestNoContextLimit(unittest.TestCase):
    """后端不应对 prompt 长度设限 —— 由调用方自己观察模型行为。"""

    def test_long_prompt_is_not_rejected_by_cann_backend(self):
        import inspect

        from cann_llm.backends import cann
        src = inspect.getsource(cann.CannNdkBackend)
        self.assertNotIn("max_prompt_tokens", src)
        # 只保留「不能为空」这一类真正的输入错误
        self.assertIn("prompt 不能为空", src)

    def test_agent_loop_does_not_trim_history(self):
        import inspect

        from cann_llm.agent.loop import AgentLoop
        src = inspect.getsource(AgentLoop)
        self.assertNotIn("max_prompt_tokens", src)
        self.assertNotIn("count_prompt_tokens", src)

    def test_config_has_no_context_limit_knob(self):
        from cann_llm.config import ModelConfig
        self.assertFalse(hasattr(ModelConfig(), "max_prompt_tokens"))

    def _fake_cann(self, status, context_length=2048):
        """造一个只让 _run 走到「引擎返回非零」那一步的后端。"""
        from cann_llm.backends.cann import CannNdkBackend

        be = CannNdkBackend.__new__(CannNdkBackend)
        be.context_length = context_length
        be.model_dir = "."

        class Ndk:
            def generate(self, *args):
                return status

        be._ndk = Ndk()
        be._context_for = lambda p: 1
        be._executor = 1
        return be

    def test_engine_failure_reports_raw_status_without_guessing(self):
        """如实报告引擎返回码，不替它断言原因 —— 我们区分不出到底为什么失败。"""
        from cann_llm.errors import GenerationError
        from cann_llm.types import GenerationParams

        be = self._fake_cann(status=1)
        for prompt in ("x" * 40000, "short"):
            with self.assertRaises(GenerationError) as cm:
                be._run(prompt, GenerationParams(), None)
            msg = str(cm.exception)
            self.assertIn("返回 1", msg)
            self.assertIn("无法从返回码判断", msg)
            # 不应断言某个具体原因
            self.assertNotIn("错误类型", msg)
            self.assertEqual(cm.exception.http_status, 500)

    def test_params_are_used_verbatim(self):
        """采样参数原样使用：不因「值恰好等于默认」而被替换。"""
        from cann_llm.backends import cann
        import inspect
        src = inspect.getsource(cann.CannNdkBackend)
        self.assertNotIn("_merge_params", src)

    def test_token_estimate_is_calibrated(self):
        """按实测标定：原先「字节 // 2」对英文偏高约 2.3 倍，改为「字节 // 4」。"""
        from cann_llm.backends.cann import CannNdkBackend

        be = CannNdkBackend.__new__(CannNdkBackend)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        # 实测该量级约 100 token 上下，估算不应再偏出 2 倍以上
        self.assertLess(be.count_prompt_tokens(text), 180)
        self.assertGreater(be.count_prompt_tokens(text), 40)

    def test_empty_prompt_still_rejected(self):
        from cann_llm.backends.cann import CannNdkBackend
        from cann_llm.errors import InvalidRequestError
        from cann_llm.types import GenerationParams
        be = CannNdkBackend.__new__(CannNdkBackend)
        with self.assertRaises(InvalidRequestError):
            be._check_prompt("", GenerationParams())
