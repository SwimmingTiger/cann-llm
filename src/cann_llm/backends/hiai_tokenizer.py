# -*- coding: utf-8 -*-
"""从 HuggingFace `tokenizer.json` 实现的最小 Qwen 分词器（纯 Python，无依赖）。

为什么需要它：内部引擎的 ``HIAI_LLMEngine_Prompt_SetText`` 走 ``std::string``，
其堆分配路径在本机环境下不可用（实测边界精确在 22→23 字节，即 libc++ 的 SSO 容量），
超过就只给引擎留下 1 个 token。系统服务因此**自己在外面分词**，再用
``Prompt_SetTokenIds`` 传整数数组 —— 本模块就是那条路上的分词环节。

管线（与 tokenizer.json 的描述一致）：
    NFC 规范化 → 正则切分 → ByteLevel 字节↔unicode → BPE 合并 → 查 vocab
另外先做一遍 added_tokens 的字面匹配（<|im_start|> 等特殊 token）。
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

#: Qwen 的 pre_tokenizer 正则。Python 的 re 不支持 \p{L}/\p{N}，
#: 用等价写法：\p{L} → [^\W\d_]，\p{N} → \d
_QWEN_PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\w]?[^\W\d_]+"
    r"|\d"
    r"| ?[^\s\w]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2/ByteLevel 的字节 → unicode 映射（可见字符保持原样，其余平移到 256+）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_BYTE_TO_UNI = _bytes_to_unicode()


def _bpe(token: str, ranks: Dict[Tuple[str, str], int]) -> List[str]:
    """对一个 ByteLevel 片段做 BPE 合并。"""
    if not token:
        return []
    word = list(token)
    while len(word) > 1:
        best: Optional[Tuple[int, int, int]] = None      # (rank, i, j)
        for i in range(len(word) - 1):
            r = ranks.get((word[i], word[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i, i + 1)
        if best is None:
            break
        _, i, j = best
        word[i] = word[i] + word[j]
        del word[j]
    return word


class QwenTokenizer:
    """只依赖 `tokenizer.json` 的 Qwen 分词器。"""

    def __init__(self, tokenizer_json_path: str):
        with open(tokenizer_json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"只支持 BPE，得到 {model.get('type')!r}")
        self.vocab: Dict[str, int] = model["vocab"]
        # merges 可能是 "a b" 字符串或 ["a","b"] 数组，两种都兼容
        self.ranks: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(model.get("merges", [])):
            if isinstance(m, str):
                a, _, b = m.partition(" ")
            else:
                a, b = m[0], m[1]
            if (a, b) not in self.ranks:
                self.ranks[(a, b)] = i
        # added_tokens：按内容长度降序，先做字面匹配
        self.special: List[Tuple[str, int]] = sorted(
            ((t["content"], t["id"]) for t in data.get("added_tokens", [])
             if t.get("content")),
            key=lambda x: -len(x[0]),
        )
        self._special_re = (
            re.compile("|".join(re.escape(s) for s, _ in self.special))
            if self.special else None
        )
        self._special_map = dict(self.special)
        # 反向表（解码用）
        self.inv: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    # ------------------------------------------------------------------ encode

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        text = unicodedata.normalize("NFC", text)
        ids: List[int] = []

        # ① added_tokens 的字面匹配（<|im_start|> 等），其余部分走 BPE
        if allow_special and self._special_re is not None:
            pos = 0
            for m in self._special_re.finditer(text):
                if m.start() > pos:
                    ids.extend(self._encode_plain(text[pos:m.start()]))
                ids.append(self._special_map[m.group()])
                pos = m.end()
            if pos < len(text):
                ids.extend(self._encode_plain(text[pos:]))
            return ids
        return self._encode_plain(text)

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in _QWEN_PATTERN.findall(text):
            if not chunk:
                continue
            # ② ByteLevel：把每个字节映射成 unicode 字符
            bl = "".join(_BYTE_TO_UNI[b] for b in chunk.encode("utf-8"))
            # ③ BPE 合并
            for piece in _bpe(bl, self.ranks):
                tid = self.vocab.get(piece)
                if tid is not None:
                    out.append(tid)
                else:                       # 理论上不该发生；退化为逐字符
                    for ch in piece:
                        t = self.vocab.get(ch)
                        if t is not None:
                            out.append(t)
        return out

    def decode(self, ids: List[int]) -> str:
        """整数 → 文本（byte-level 逆变换）。"""
        uni_to_byte = {v: k for k, v in _BYTE_TO_UNI.items()}
        buf = bytearray()
        for i in ids:
            tok = self.inv.get(i)
            if not tok:
                continue
            for ch in tok:
                b = uni_to_byte.get(ch)
                if b is not None:
                    buf.append(b)
        return buf.decode("utf-8", "replace")
