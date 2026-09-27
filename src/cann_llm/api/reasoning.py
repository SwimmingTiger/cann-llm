"""把模型自带的思考段（``<think>…</think>``）从正文里拆出来。

**为什么需要**：Qwen3-8B 自带思考模式，输出形如::

    <think>
    （一段思考过程）
    </think>

    （真正的回答）

服务此前把整段原样放进 ``delta.content``，客户端（DSH 用的
``@earendil-works/pi-ai``）于是把它当**正文**渲染 —— 用户看到的是一堆
``<think>`` 标签，而不是"思考"。

**为什么是 ``reasoning_content`` 这个字段名**：查过 pi-ai 的
``dist/api/openai-completions.js``，它按 ``["reasoning_content", "reasoning",
"reasoning_text"]`` 的顺序取**第一个非空**的字段作为 thinking_delta，正文仍读
``delta.content``；源码注释原话是 *Some endpoints return reasoning in
reasoning_content (llama.cpp), or reasoning (other openai compatible
endpoints)*。也就是 llama.cpp / vLLM 那一套既有做法，照做即可 —— 不需要新增
任何请求参数（OpenAI 协议里没有"关思考"这种东西）。

**为什么不用 ``text.split("<think>")``**：流式下一个标记会被切在两个 chunk 之间
（``"<thi"`` + ``"nk>"``），简单切分会把两个半截都当正文漏出去。这里用
「保留可能是标记真前缀的尾巴」的办法增量判定，与 ``agent/loop.py`` 里处理
``<tool_call>`` 的 :class:`~cann_llm.agent.loop.StreamFilter` 同一套思路
（判据本身已抽到 :func:`cann_llm.textutil.prefix_hold_len`）。

**对不带标记的模型没有影响**：整个流都走同一个状态机，但只要输出里没出现
``<think>``，它至多把"可能是标记前缀"的尾巴多留一拍（最多 6 个字符），正文的
字符一个不少、顺序也不变；而且响应里**不会出现 reasoning_content 字段**
（空字符串不写），7B 与自转模型的响应结构与改动前完全一致。
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

from ..textutil import prefix_hold_len

#: Qwen3 思考段的标记。模型只会原样吐这两个字符串（见模型自带 chat template 里的
#: ``'<|im_start|>' + role + '\n<think>\n' + reasoning + '\n</think>\n\n' + content``），
#: 没有别的变体，所以不做正则、不认别名。
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# 状态机的三个状态。用字符串而不是 Enum：只在 feed 内部比较，日志里也直接可读。
_BEFORE = "before"      # 还没见到 <think>：这一段是「待定」，见到标记则前面的算正文
_INSIDE = "inside"      # 在思考段里（已吃掉 <think>，还没见到 </think>）
_AFTER = "after"        # 已见过 </think>：之后全是正文，此后不再找标记


class ThinkPieces(NamedTuple):
    """一次 :meth:`ThinkSplitter.feed` 切出来的两路增量。"""

    reasoning: str = ""
    content: str = ""


class ThinkSplitter:
    """增量的 ``<think>`` 状态机：喂入模型输出，吐出 (思考, 正文)。

    只认**第一个** ``<think>…</think>``：见到 ``</think>`` 之后一律当正文，
    之后再出现的 ``<think>`` 也不再当标记 —— 否则模型在正文里讨论这个标记
    （很常见：用户问"你怎么输出思考"）会被我们当成思考吞掉。

    边界都按「不丢模型说过的话」处理：

    * ``<think>`` 之后一直没等到 ``</think>``（被 ``max_tokens`` 截断）→
      已生成的部分全算思考；
    * 流结束时缓冲里只剩半个标记（如 ``"<thi"``）→ 它不构成完整的 ``<think>``，
      当正文交出去。
    """

    def __init__(self, open_tag: str = THINK_OPEN,
                 close_tag: str = THINK_CLOSE) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        #: 还没能判定的尾巴（可能是半个标记）
        self._tail = ""
        self._state = _BEFORE

    def feed(self, delta: str) -> ThinkPieces:
        """喂入一段增量，返回这一段里属于思考 / 正文的部分。"""
        if not delta:
            return ThinkPieces()
        self._tail += delta
        reasoning: List[str] = []
        content: List[str] = []
        while self._tail:
            if self._state == _AFTER:
                # 已经没有标记可找了：整段透传，连缓冲都不用留
                content.append(self._tail)
                self._tail = ""
                break
            # 在思考段里就找闭标记（找到的算思考），否则找开标记（找到的前面算正文）
            tag, out = ((self.close_tag, reasoning) if self._state == _INSIDE
                        else (self.open_tag, content))
            i = self._tail.find(tag)
            if i < 0:
                # 没有完整标记：能确定归属的部分立刻吐出去，
                # 只留「可能是标记真前缀」的尾巴等下一个 chunk
                hold = prefix_hold_len(self._tail, tag)
                if hold:
                    out.append(self._tail[:len(self._tail) - hold])
                    self._tail = self._tail[len(self._tail) - hold:]
                else:
                    out.append(self._tail)
                    self._tail = ""
                break
            out.append(self._tail[:i])
            self._tail = self._tail[i + len(tag):]
            self._state = _INSIDE if self._state == _BEFORE else _AFTER
        return ThinkPieces("".join(reasoning), "".join(content))

    def flush(self) -> ThinkPieces:
        """流结束时吐出缓冲里剩下的文本（此后本对象作废）。

        * 卡在 ``_INSIDE``：被截断、没等到 ``</think>`` —— 按调用方的要求，
          已生成的部分都算思考。
        * 卡在 ``_BEFORE``：剩下的一定只是"半个标记"，当正文。丢掉它等于把
          模型确实说过的话吞了。
        """
        tail, self._tail = self._tail, ""
        if not tail:
            return ThinkPieces()
        if self._state == _INSIDE:
            return ThinkPieces(reasoning=tail)
        return ThinkPieces(content=tail)


def split_reasoning(text: str) -> Tuple[str, str]:
    """非流式：一次性切出 ``(reasoning, content)``。

    刻意复用同一个状态机，而不是在这里另写一遍 ``split()``：流式与非流式必须给出
    **一致**的结果，否则「同样的输出、流式与非流式对不上」是最难查的一类 bug
    （典型例子就是截断在半个标记上）。
    """
    splitter = ThinkSplitter()
    head = splitter.feed(text)
    tail = splitter.flush()
    return head.reasoning + tail.reasoning, head.content + tail.content
