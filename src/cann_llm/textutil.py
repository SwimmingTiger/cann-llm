"""文本层面共用的判断（目前只有流式标记的「半截」判定）。

放在包根而不是某个子模块里：``agent``（``<tool_call>``）与 ``api``（``<think>``）
两边都要用同一个判据，各自抄一份很容易在后面改动时漏改其中一个。
"""

from __future__ import annotations


def prefix_hold_len(text: str, tag: str) -> int:
    """返回 ``text`` 末尾「是 ``tag`` 真前缀」的最长长度。

    用途：流式过滤标记时，标记可能被切在两个 chunk 之间（``"<thi"`` + ``"nk>"``）。
    这段尾巴**既不能当正文发出去、也不能丢掉**，只能留着等下一个 chunk 到齐再判。

    上限刻意取 ``len(tag) - 1``：整个 tag 已经算「匹配成功」，该由调用方处理，
    不属于"需要继续等"的范围。没有任何后缀是 tag 的前缀时返回 0（可以立刻放行）。
    """
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
