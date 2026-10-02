"""采样：把 logits 变成下一个 token。

``cann`` / ``hiai`` 两个后端的采样是**引擎**做的（参数只是下发下去），
所以项目里原本没有 Python 侧实现。``nnrt`` 后端跑的是我们自己转的图，
采样得自己做 —— 这里放共用的一份（分段与整图两个 runner 都用它）。

顺序与常见实现一致：**重复惩罚 → 温度 → top-k → top-p → 按概率抽样**。
``temperature <= 0`` 直接退化成贪心（取 argmax），便于复现问题。

用 ``random.Random(seed)``：给了 ``seed`` 就完全可复现；没给就每次不同。
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence

__all__ = ["sample_token"]


def _apply_repetition_penalty(logits: List[float], history: Sequence[int],
                              penalty: float) -> None:
    """对**已经出现过**的 token 打折：正数除以 penalty、负数乘以 penalty。

    这是 CTRL 论文里的做法（也是 HF 的 ``repetition_penalty``），
    只压"已经说过"的词，不动其余的分布形状。
    """
    if penalty == 1.0:
        return
    for t in set(int(x) for x in history):
        if 0 <= t < len(logits):
            v = logits[t]
            logits[t] = v / penalty if v > 0 else v * penalty


def sample_token(logits: Sequence[float], params, history: Sequence[int] = (),
                 rng: Optional[random.Random] = None) -> int:
    """从 ``logits`` 里挑一个 token。

    :param params: :class:`~cann_llm.types.GenerationParams`（只用 temperature /
        top_k / top_p / repetition_penalty / seed；``stop`` 由调用方处理）。
    :param history: prompt + 已生成的 token（重复惩罚用）。
    :param rng: 传入可复用的 :class:`random.Random`（同一轮内保持状态）。
    """
    n = len(logits)
    if n == 0:
        raise ValueError("logits 为空")
    rng = rng or random.Random(getattr(params, "seed", None))
    temp = float(getattr(params, "temperature", 1.0) or 0.0)

    work: List[float] = [float(x) for x in logits]
    # ★ 重复惩罚是 logits processor：在【采样判定之前】就要生效 ——
    #   包括 temperature=0 的贪心路径（HF 也是这么做的）。
    #   踩过：原来把它放在 temp<=0 的 return 之后，贪心时惩罚被整段跳过 ✗。
    _apply_repetition_penalty(work, history, float(getattr(params, "repetition_penalty", 1.0) or 1.0))

    if temp <= 0.0:                      # 贪心（可复现）
        return max(range(n), key=work.__getitem__)

    work = [x / temp for x in work]

    # 先按分数取候选（后面在候选子集上做 softmax，省掉全词表的 exp）
    order = sorted(range(n), key=work.__getitem__, reverse=True)

    top_k = int(getattr(params, "top_k", 0) or 0)
    if top_k > 0:
        order = order[:max(1, min(top_k, n))]

    top_p = float(getattr(params, "top_p", 1.0) or 1.0)
    if 0.0 < top_p < 1.0:
        m = max(work[i] for i in order)
        probs = [math.exp(work[i] - m) for i in order]
        total = sum(probs)
        acc, keep = 0.0, []
        for idx, p in zip(order, probs):
            keep.append(idx)
            acc += p / total
            if acc >= top_p:
                break
        order = keep

    # 候选子集上 softmax
    m = max(work[i] for i in order)
    weights = [math.exp(work[i] - m) for i in order]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return order[0]
    r = rng.random() * total
    acc = 0.0
    for idx, w in zip(order, weights):
        acc += w
        if r <= acc:
            return idx
    return order[-1]
