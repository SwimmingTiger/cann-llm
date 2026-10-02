"""纯 Python 的 Qwen 分词器（字节级 BPE）—— 不依赖任何第三方库。

为什么需要它
------------
``cann`` / ``hiai`` 两个后端把分词交给华为引擎（prompt 直接传文本，引擎自己分）。
而 ``nnrt`` 后端跑的是**我们自己转出来的图**，引擎不参与 ⇒ 分词、解码循环都得自己做。
鸿蒙设备上既没有 ``tokenizers`` 也没有 ``transformers``，所以这里用标准库实现一份。

实现要点
--------
* 读 HuggingFace 的 ``tokenizer.json``：``model.vocab`` / ``model.merges`` /
  ``added_tokens``；
* GPT-2 风格的**字节级** BPE：先把文本按字节映射到可见字符，再做合并；
* 预切分用 Qwen 的正则。原版用的是 ``\\p{L}`` / ``\\p{N}`` 这类 Unicode 属性转义，
  Python 标准库 ``re`` 不支持 —— 用等价写法替代（``re.UNICODE`` 下）：

  ==================  =========================
  原版                 这里
  ==================  =========================
  ``\\p{L}``            ``[^\\W\\d_]``（Unicode 字母）
  ``\\p{N}``            ``\\d``
  ``[^\\s\\p{L}\\p{N}]`` ``[^\\s\\w]``
  ==================  =========================

* ``added_tokens``（``<|im_start|>`` 这类特殊 token）按**最长优先**先切出来。

用法::

    from cann_llm.tokenizer_bpe import QwenTokenizer
    tok = QwenTokenizer.from_file("/path/to/tokenizer.json")
    ids = tok.encode("你好，世界")
    text = tok.decode(ids)
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["QwenTokenizer"]


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 的字节→可见字符映射（保证每个字节都能放进 vocab）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}

#: Qwen 的预切分模式（\\p{L}/\\p{N} 用 re 的等价写法）
_PAT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\w]?[^\W\d_]+"        # 可选前缀（可含空格！）+ 字母串
    #   ★ 前缀只排除 CR/LF/字母/数字 —— 空格是允许的，这样 " world" 才能并成一个 token
    r"|\d"                           # 单个数字
    r"| ?[^\s\w]+[\r\n]*"            # 标点串
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE,
)


class QwenTokenizer:
    """字节级 BPE 分词器（Qwen / Qwen2 / Qwen2.5 通用）。"""

    def __init__(self, vocab: Dict[str, int], merges: Sequence[str],
                 added: Optional[Dict[str, int]] = None) -> None:
        self.vocab = dict(vocab)
        self.ranks: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ") if isinstance(m, str) else list(m)
            if len(parts) == 2:
                self.ranks[(parts[0], parts[1])] = i
        # added tokens：长的优先匹配
        self.added = dict(added or {})
        self._added_sorted = sorted(self.added.keys(), key=len, reverse=True)
        self._added_re = (re.compile("(" + "|".join(re.escape(t) for t in self._added_sorted) + ")")
                          if self._added_sorted else None)
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ 加载
    @classmethod
    def from_file(cls, path: str) -> "QwenTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model") or {}
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        if not vocab:
            # 兜底：有些导出把 vocab 放在顶层
            vocab = data.get("vocab") or {}
        added: Dict[str, int] = {}
        for t in data.get("added_tokens") or []:
            c = t.get("content")
            if c is not None and t.get("id") is not None:
                added[c] = int(t["id"])
        return cls(vocab, merges, added)

    # ------------------------------------------------------------------ 编码
    def _bpe(self, token: str) -> List[str]:
        """对一段（已字节映射的）文本做 BPE 合并。"""
        hit = self._cache.get(token)
        if hit is not None:
            return hit
        word = list(token)
        if not word:
            return []
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        while pairs:
            best = None
            best_rank = None
            for p in pairs:
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            first, second = best
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        self._cache[token] = word
        return word

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """文本 → token id 列表。``add_special`` 保留（Qwen 默认不加 BOS）。"""
        ids: List[int] = []
        if not text:
            return ids
        # 1) 先把 added/special token 切出来
        chunks: List[Tuple[bool, str]] = []
        if self._added_re is not None:
            pos = 0
            for m in self._added_re.finditer(text):
                if m.start() > pos:
                    chunks.append((False, text[pos:m.start()]))
                chunks.append((True, m.group(0)))
                pos = m.end()
            if pos < len(text):
                chunks.append((False, text[pos:]))
        else:
            chunks.append((False, text))

        for is_added, chunk in chunks:
            if is_added:
                tid = self.added.get(chunk)
                if tid is not None:
                    ids.append(tid)
                continue
            for piece in _PAT.findall(chunk):
                mapped = "".join(_BYTE_ENCODER[b] for b in piece.encode("utf-8"))
                for tok in self._bpe(mapped):
                    tid = self.vocab.get(tok)
                    if tid is None:      # 理论上不会发生；兜底成逐字节
                        for ch in tok:
                            ids.append(self.vocab.get(ch, 0))
                    else:
                        ids.append(tid)
        return ids

    # ------------------------------------------------------------------ 解码
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """token id 列表 → 文本。"""
        special_ids = set(self.added.values()) if skip_special else set()
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in special_ids:
                continue
            tok = self._idx2tok.get(i)
            if tok is None:
                continue
            buf.extend(self._tok_to_bytes(tok))
        return buf.decode("utf-8", errors="replace")

    def _tok_to_bytes(self, tok: str) -> bytes:
        return bytes(_BYTE_DECODER.get(ch, 0) for ch in tok)

    # 建反向表（懒加载，省启动时间）
    @property
    def _idx2tok(self) -> Dict[int, str]:
        if not hasattr(self, "_idx2tok_cache"):
            self._idx2tok_cache = {v: k for k, v in self.vocab.items()}  # type: ignore[attr-defined]
        return self._idx2tok_cache  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ 方便用
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def special_ids(self) -> Dict[str, int]:
        return dict(self.added)
