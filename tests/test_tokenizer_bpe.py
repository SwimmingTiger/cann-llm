"""``tokenizer_bpe`` 的单元测试。

参考值由 **transformers 官方分词器**（x570 上 `AutoTokenizer.from_pretrained`）生成，
是权威答案；这里把它们钉住，防止改动正则或 BPE 循环时悄悄回归。

需要一份真实的 ``tokenizer.json``：没有就整体跳过（设置 ``CANN_LLM_TEST_TOKENIZER``
指向它）。
"""

import json
import os
import unittest

from cann_llm.tokenizer_bpe import QwenTokenizer

#: 权威参考：transformers 的 Qwen2Tokenizer 输出
REF = {
    "你好，世界": [108386, 3837, 99489],
    "Hello, world!": [9707, 11, 1879, 0],
    "1+1=2": [16, 10, 16, 28, 17],
    "  leading spaces": [220, 6388, 12621],
    "Qwen2.5-0.5B": [48, 16948, 17, 13, 20, 12, 15, 13, 20, 33],
    "混合 mixed 文本123": [105063, 9519, 53040, 21894, 16, 17, 18],
    "a\nb\n\nc": [64, 198, 65, 271, 66],
}

#: 若某条参考值不确定就置 None，测试会跳过该条
_KNOWN = {k: v for k, v in REF.items() if v is not None}


def _find_tokenizer() -> str:
    p = os.environ.get("CANN_LLM_TEST_TOKENIZER")
    if p and os.path.isfile(p):
        return p
    for cand in ("/storage/Users/currentUser/work/llm/models/qwen15b_e2e/tokenizer.json",
                 "/storage/Users/currentUser/work/llm/models/qwen25_coder_7b_omc1024/tokenizer.json"):
        if os.path.isfile(cand):
            return cand
    return ""


_TOKENIZER = _find_tokenizer()


@unittest.skipUnless(_TOKENIZER, "没有可用的 tokenizer.json（设 CANN_LLM_TEST_TOKENIZER）")
class TestQwenTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = QwenTokenizer.from_file(_TOKENIZER)

    def test_vocab_loaded(self):
        self.assertGreater(len(self.tok.vocab), 100000)
        self.assertGreater(len(self.tok.ranks), 100000)

    def test_matches_transformers(self):
        """逐条与 transformers 的输出比对 —— 这是最硬的判据。"""
        for text, want in _KNOWN.items():
            with self.subTest(text=text):
                self.assertEqual(self.tok.encode(text), want)

    def test_roundtrip(self):
        for text in ("你好，世界", "Hello, world!", "混合 mixed 文本123"):
            with self.subTest(text=text):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_special_token_kept_whole(self):
        """``<|im_start|>`` 这类 added token 不能被拆开。"""
        ids = self.tok.encode("<|im_start|>user")
        self.assertEqual(ids[0], self.tok.special_ids["<|im_start|>"])

    def test_empty(self):
        self.assertEqual(self.tok.encode(""), [])

    def test_space_merge_regression(self):
        """回归：Qwen 的前缀模式允许【空格】，所以 " world" 要并成一个 token。

        我们曾经把它写成 ``[^\\r\\n\\W\\d_]?``（把空格也排除），
        结果 "Hello, world!" 被切成 [9707, 11, 220, 14615, 0]。
        """
        self.assertEqual(self.tok.encode("Hello, world!"), [9707, 11, 1879, 0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
