"""★后端自动选择★（``detect_backend``）的判据 —— 必须与各后端自己的接受条件一致。

§165 的教训：判据曾经只认"段图目录 + weights/ 或 graphP/" ✗，于是本仓库自己导出的
分段模型（``tokenizer.json`` + ``seg0…seg11/``、没有 weights/ ✗）被判成 None ⇒ 回落到 hiai ✗
⇒ 用户不写 ``-b nnrt`` 就撞"缺少 api_config.json" ✗。
而 ``NnrtBackend.load()`` 认的正是"``tokenizer.json`` + 至少一个 ``seg*`` 目录" ✓ ——
两边判据必须对齐，所以这里逐条钉住 ✓。
"""

import os
import tempfile
import unittest

from cann_llm.launcher import detect_backend


def _dirs(root, *names):
    for n in names:
        os.makedirs(os.path.join(root, n), exist_ok=True)


def _files(root, *names):
    for n in names:
        path = os.path.join(root, n)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{}")


class TestDetectBackend(unittest.TestCase):
    # ---- nnrt：本仓库自装配的两种布局 -------------------------------------

    def test_segmented_repo_layout_is_nnrt(self):
        """★本仓库导出的分段模型 ⇒ nnrt★（就是这次踩到的那个 ✓）"""
        with tempfile.TemporaryDirectory() as d:
            _files(d, "tokenizer.json", "emb_manifest.json")
            _files(d, os.path.join("seg0", "seg0.ms"))
            self.assertEqual(detect_backend(d), "nnrt")

    def test_segmented_needs_dir_not_just_prefix_file(self):
        """名字以 seg 开头但【不是目录】的普通文件不算 ✓（避免误判 ✓）"""
        with tempfile.TemporaryDirectory() as d:
            _files(d, "tokenizer.json", "segments.json")
            self.assertIsNone(detect_backend(d))

    def test_seg_dir_without_tokenizer_is_none(self):
        """只有段图目录、没有 tokenizer.json ⇒ 还认不出 ✓（nnrt 也需要 python 侧分词 ✓）"""
        with tempfile.TemporaryDirectory() as d:
            _files(d, os.path.join("seg0", "seg0.ms"))
            self.assertIsNone(detect_backend(d))

    def test_old_layout_weights(self):
        with tempfile.TemporaryDirectory() as d:
            _dirs(d, "seg0", "weights")
            self.assertEqual(detect_backend(d), "nnrt")

    def test_old_layout_graphp(self):
        with tempfile.TemporaryDirectory() as d:
            _dirs(d, "dec0", "pre0", "graphP")
            self.assertEqual(detect_backend(d), "nnrt")

    def test_gemma4_style_dirs(self):
        """gemma4 系（dec0/pre0 + weights ✓）也归 nnrt，由 NnrtBackend 内部再分发 ✓"""
        with tempfile.TemporaryDirectory() as d:
            _dirs(d, "dec0", "pre0", "weights")
            self.assertEqual(detect_backend(d), "nnrt")

    # ---- 官方布局 ----------------------------------------------------------

    def test_official_hiai(self):
        with tempfile.TemporaryDirectory() as d:
            _files(d, "api_config.json")
            self.assertEqual(detect_backend(d), "hiai")

    def test_official_cann_executor(self):
        with tempfile.TemporaryDirectory() as d:
            _files(d, "executor.json")
            self.assertEqual(detect_backend(d), "cann")

    def test_official_cann_context(self):
        with tempfile.TemporaryDirectory() as d:
            _files(d, "context.json")
            self.assertEqual(detect_backend(d), "cann")

    def test_hiai_wins_over_cann_when_both_markers(self):
        """两个都有时按既有优先级：hiai 在前 ✓（保持原行为 ✓）"""
        with tempfile.TemporaryDirectory() as d:
            _files(d, "api_config.json", "executor.json")
            self.assertEqual(detect_backend(d), "hiai")

    # ---- 认不出 ------------------------------------------------------------

    def test_empty_dir_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(detect_backend(d))

    def test_missing_dir_is_none(self):
        self.assertIsNone(detect_backend("/definitely/not/here/nope"))

    def test_none_is_none(self):
        self.assertIsNone(detect_backend(None))

    def test_only_tokenizer_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            _files(d, "tokenizer.json")
            self.assertIsNone(detect_backend(d))


if __name__ == "__main__":                                   # pragma: no cover
    unittest.main()
