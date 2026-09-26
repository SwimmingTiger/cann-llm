import unittest

from cann_llm.types import (
    FINISH_LENGTH,
    FINISH_STOP,
    GenerationChunk,
    GenerationParams,
    GenerationResult,
    GenerationStats,
    aggregate,
)


class TestGenerationParams(unittest.TestCase):
    def test_defaults_are_valid(self):
        p = GenerationParams()
        self.assertFalse(p.greedy)
        self.assertEqual(p.max_tokens, 256)

    def test_temperature_zero_is_greedy(self):
        self.assertTrue(GenerationParams(temperature=0).greedy)

    def test_rejects_bad_values(self):
        for kw in ({"max_tokens": 0}, {"temperature": -1}, {"top_k": -1},
                   {"top_p": 0}, {"top_p": 1.5}, {"repetition_penalty": 0}):
            with self.assertRaises(ValueError, msg=str(kw)):
                GenerationParams(**kw)

    def test_frozen(self):
        p = GenerationParams()
        with self.assertRaises(Exception):
            p.temperature = 0.1        # type: ignore[misc]


class TestStats(unittest.TestCase):
    def test_openai_usage(self):
        st = GenerationStats(prompt_tokens=5, completion_tokens=7)
        self.assertEqual(st.as_openai_usage(),
                         {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})

    def test_tokens_per_second(self):
        st = GenerationStats(completion_tokens=20, decode_ms=1000.0)
        self.assertAlmostEqual(st.tokens_per_second, 20.0)

    def test_zero_decode_does_not_divide_by_zero(self):
        self.assertEqual(GenerationStats(completion_tokens=5, decode_ms=0).tokens_per_second, 0.0)


class TestAggregate(unittest.TestCase):
    def test_collects_text_and_stats(self):
        stats = GenerationStats(prompt_tokens=1, completion_tokens=2)
        def gen():
            yield GenerationChunk(text="he")
            yield GenerationChunk(text="llo")
            yield GenerationChunk(finish_reason=FINISH_STOP, stats=stats)
        r = aggregate(gen())
        self.assertIsInstance(r, GenerationResult)
        self.assertEqual(r.text, "hello")
        self.assertEqual(r.finish_reason, FINISH_STOP)
        self.assertEqual(r.chunk_count, 3)
        self.assertIs(r.stats, stats)

    def test_defaults_when_no_stats(self):
        def gen():
            yield GenerationChunk(text="x", finish_reason=FINISH_LENGTH)
        r = aggregate(gen())
        self.assertEqual(r.stats.as_openai_usage(),
                         {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(r.finish_reason, FINISH_LENGTH)

    def test_empty_stream(self):
        r = aggregate(iter(()))
        self.assertEqual(r.text, "")
        self.assertEqual(r.chunk_count, 0)

    def test_final_chunk_property(self):
        self.assertTrue(GenerationChunk(finish_reason=FINISH_STOP).is_final)
        self.assertFalse(GenerationChunk(text="a").is_final)


if __name__ == "__main__":
    unittest.main()
