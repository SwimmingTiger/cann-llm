"""采样器的单元测试（不需要设备）。"""
import random
import unittest
from dataclasses import replace

from cann_llm.sampling import sample_token
from cann_llm.types import GenerationParams


class TestSampling(unittest.TestCase):
    def test_temperature_zero_is_greedy(self):
        """temperature=0 ⇒ 取 argmax（便于复现问题）。"""
        p = replace(GenerationParams(), temperature=0.0)
        for _ in range(20):
            self.assertEqual(sample_token([1.0, 9.0, 3.0], p), 1)

    def test_reproducible_with_seed(self):
        p = replace(GenerationParams(), temperature=1.0, top_k=0, top_p=1.0, seed=1234)
        a = [sample_token([1.0, 2.0, 3.0, 2.5], p, rng=random.Random(p.seed)) for _ in range(5)]
        b = [sample_token([1.0, 2.0, 3.0, 2.5], p, rng=random.Random(p.seed)) for _ in range(5)]
        self.assertEqual(a, b)

    def test_top_k_1_equals_greedy(self):
        p = replace(GenerationParams(), temperature=1.0, top_k=1, top_p=1.0)
        self.assertEqual(sample_token([1.0, 9.0, 3.0], p), 1)

    def test_repetition_penalty_pushes_away_seen(self):
        """把已出现过的 token 压下去后，argmax 应当换人。"""
        p = replace(GenerationParams(), temperature=0.0, repetition_penalty=100.0)
        self.assertEqual(sample_token([5.0, 4.0], p), 0)              # 无历史 ⇒ 取 0
        self.assertEqual(sample_token([5.0, 4.0], p, history=[0]), 1)  # 0 被压 ⇒ 取 1

    def test_top_p_keeps_only_nucleus(self):
        p = replace(GenerationParams(), temperature=1.0, top_k=0, top_p=0.5)
        # 第一项占了 0.99 的概率质量 ⇒ top-p=0.5 只留它
        for _ in range(10):
            self.assertEqual(sample_token([10.0, 0.0, 0.0], p), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
