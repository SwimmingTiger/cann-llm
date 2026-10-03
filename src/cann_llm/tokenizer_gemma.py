"""Gemma 4 分词器（纯 Python，可在设备上跑）。

结构（来自 tokenizer.json 实测）：BPE + merges(514906) + byte_fallback，
pre_tokenizer = Split(" ", MergedWithPrevious)，decoder = ▁→空格 + ByteFallback + Fuse。
"""
import json, os

class GemmaTokenizer:
    def __init__(self, path):
        self.dir = path if os.path.isdir(path) else os.path.dirname(path)
        tj = json.load(open(os.path.join(self.dir, "tokenizer.json"), encoding="utf-8"))
        m = tj["model"]
        self.vocab = m["vocab"] if isinstance(m["vocab"], dict) else {t: i for i, t in enumerate(m["vocab"])}
        self.ranks = {}
        for i, mg in enumerate(m["merges"]):
            a, b = (mg[0], mg[1]) if isinstance(mg, list) else tuple(mg.split(" ", 1))
            self.ranks[(a, b)] = i
        self.added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
        self.inv = {i: t for t, i in self.vocab.items()}
        self.inv.update({i: t for t, i in self.added.items()})
        self.byte_tok = {}                      # <0xXX> 的反查
        for t, i in self.vocab.items():
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                self.byte_tok[int(t[3:5], 16)] = i
        self.bos_id = self.added.get("<bos>", 2)
        self.eos_ids = {1, self.added.get("<turn|>", 106), 50}

    def _bpe(self, piece):
        if len(piece) <= 1:
            return [piece]
        parts = list(piece)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        return parts

    def _tok(self, s):
        """单个 piece → id 列表（含 byte_fallback）"""
        out = []
        # 先按单个 token 直查（覆盖特殊/整体命中）
        cur = [s]
        cur = self._bpe(s) if s not in self.vocab else [s]
        for p in cur:
            if p in self.vocab:
                out.append(self.vocab[p]); continue
            if p in self.added:
                out.append(self.added[p]); continue
            for ch in p.encode("utf-8"):        # ★ byte_fallback
                out.append(self.byte_tok.get(ch, self.added.get("<unk>", 3)))
        return out

    def encode(self, text, add_bos=False):
        """按 Gemma 的 pre_tokenizer：空格用 ▁ 表示并合并到前一个 piece ✓"""
        spec = sorted(self.added.keys(), key=len, reverse=True)
        ids = []
        # 先把特殊标记切开（长优先）
        i, buf = 0, ""
        while i < len(text):
            hit = None
            for s in spec:
                if s.startswith("<") and text.startswith(s, i):
                    hit = s; break
            if hit:
                if buf: ids += self._plain(buf); buf = ""
                ids.append(self.added[hit]); i += len(hit)
            else:
                buf += text[i]; i += 1
        if buf: ids += self._plain(buf)
        return ([self.bos_id] if add_bos else []) + ids

    def _plain(self, text):
        """SentencePiece 风格：把空格换成 ▁，并把 piece 的第一个字符前也加 ▁"""
        if not text: return []
        t = "▁" + text.replace(" ", "▁") if not text.startswith(" ") else "▁" + text[1:].replace(" ", "▁")
        # 按 ▁ 切段（每段独立做 BPE，▁ 归属段首）
        out = []
        for seg in t.split("▁"):
            if seg == "": continue
            out += self._tok("▁" + seg)
        return out

    def decode(self, ids, skip_special=True):
        bs = bytearray(); pieces = []
        for i in ids:
            t = self.inv.get(i, "")
            if skip_special and t.startswith("<") and t.endswith(">") and not t.startswith("<0x"):
                continue
            if len(t) == 6 and t.startswith("<0x"):
                bs.append(int(t[3:5], 16)); continue
            if bs:
                pieces.append(bs.decode("utf-8", "replace")); bs = bytearray()
            pieces.append(t)
        if bs: pieces.append(bs.decode("utf-8", "replace"))
        return "".join(pieces).replace("▁", " ").lstrip(" ")
