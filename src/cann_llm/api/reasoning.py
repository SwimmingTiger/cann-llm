# -*- coding: utf-8 -*-
"""把模型的思考段拆成 (思考, 正文) 两路。

模型自带的 chat template 长这样::

    '<|im_start|>' + role + '\\n<think>\\n' + reasoning + '\\n</think>\\n\\n' + content

★ 判定规则：**标记必须独立成行**才算标记。

    开始：`<think>` 在【文本开头】或 `\\n` 之后，且其后紧跟 `\\n`（或文本结束）
    结束：`</think>` 同理

为什么必须这么严：用户的消息里可能内联写着 "<think> </think>"（讨论这个标签），
模型思考时也可能在正文里引用它。实测踩过：内联出现的 `</think>` 被当成结束标记，
思考段从中间被截断、后半段错当成正文露出去。模型真正用的标记永远独占一行，
所以按行判定既准确又不会误伤。
"""
from __future__ import annotations

from typing import List, NamedTuple, Tuple

__all__ = ["THINK_OPEN", "THINK_CLOSE", "ThinkPieces", "ThinkSplitter", "split_reasoning"]

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_BEFORE = "before"      # 还没确认思考开始：这一段先按正文对待
_INSIDE = "inside"      # 在思考段里（已吃掉 <think>，还没见到 </think>）
_AFTER = "after"        # 已见过 </think>：之后全是正文，不再找标记


class ThinkPieces(NamedTuple):
    """一次 `feed` 切出来的两路增量。"""

    reasoning: str = ""
    content: str = ""


def _standalone(buf: str, tag: str, i: int) -> bool:
    """`tag` 出现在 buf[i:] 时，是否满足"独立成行"。"""
    if i != 0 and buf[i - 1] != "\n":
        return False
    end = i + len(tag)
    return end == len(buf) or buf[end] == "\n"


def _find(buf: str, tag: str, *, needs_close: bool) -> Tuple[int, bool]:
    """找第一个独立成行的 `tag`。

    返回 ``(位置, 是否已确定)``：位置为 -1 表示没找到。末尾处"可能还差一个换行"
    的候选返回 ``(-1, False)``，表示**还得再等更多文本**，不能就此收尾。
    """
    start = 0
    while True:
        i = buf.find(tag, start)
        if i < 0:
            return -1, True
        end = i + len(tag)
        if (i == 0 or buf[i - 1] == "\n"):
            if end == len(buf):
                # 标记落在缓冲区末尾：后面的字符还没到，无法判定
                return -1, False
            if buf[end] == "\n":
                return i, True
        start = i + 1


class ThinkSplitter:
    """增量状态机：喂模型输出，吐 ``(思考, 正文)``。

    只认**第一个**独立成行的 ``<think>…</think>``；见过 ``</think>`` 之后一律当正文，
    之后再出现标记也不再当标记 —— 否则模型在正文里讨论这个标签会被吞掉。
    没等到 ``</think>`` 就被 ``max_tokens`` 截断时，已生成的部分全算思考。
    """

    #: 尾部最多压住这么多字符不吐：要能容纳一个标记外加两侧换行
    _HOLD = max(len(THINK_OPEN), len(THINK_CLOSE)) + 2

    def __init__(self, open_tag: str = THINK_OPEN,
                 close_tag: str = THINK_CLOSE) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._buf = ""
        self._state = _BEFORE

    def feed(self, text: str) -> ThinkPieces:
        self._buf += text
        reasoning: List[str] = []
        content: List[str] = []

        while True:
            if self._state == _AFTER:
                content.append(self._buf)
                self._buf = ""
                break

            tag = self.open_tag if self._state == _BEFORE else self.close_tag
            i, settled = _find(self._buf, tag, needs_close=(self._state == _INSIDE))
            if i < 0:
                if not settled:
                    break                     # 缓冲区末尾可能是半个标记 → 等更多文本
                # 没有标记：只留可能成为标记的尾巴，其余按当前状态吐出去
                keep = self._keep_len()
                out, self._buf = self._buf[:len(self._buf) - keep], self._buf[len(self._buf) - keep:]
                (content if self._state == _BEFORE else reasoning).append(out)
                break

            head, self._buf = self._buf[:i], self._buf[i + len(tag):]
            (content if self._state == _BEFORE else reasoning).append(head)
            self._state = _INSIDE if self._state == _BEFORE else _AFTER
            # 吃掉标记后面【所有】连续换行：规则是 `<think>\n+ … \n+</think>`，
            # 模型到底给几个换行无法保证，所以不写死个数（它们都是分隔符）。
            self._buf = self._buf.lstrip("\n")

        return ThinkPieces("".join(reasoning), "".join(content))

    def _keep_len(self) -> int:
        """当前状态下，尾部至少要压住多少字符才能保证不漏判标记。"""
        return self._HOLD

    def flush(self) -> ThinkPieces:
        """流结束：吐出缓冲里剩下的（此后本对象作废）。"""
        tail, self._buf = self._buf, ""
        if not tail:
            return ThinkPieces()
        # 在思考段里没等到 </think> → 按调用方要求，已生成的部分都算思考
        return ThinkPieces(reasoning=tail) if self._state == _INSIDE else ThinkPieces(content=tail)


def split_reasoning(text: str) -> Tuple[str, str]:
    """一次性切分（非流式用）。返回 ``(思考, 正文)``。"""
    sp = ThinkSplitter()
    first = sp.feed(text)
    last = sp.flush()
    return first.reasoning + last.reasoning, first.content + last.content
