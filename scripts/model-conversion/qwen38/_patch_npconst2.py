"""一次性替换 npu_attention 里"造张量"的三个函数为纯 numpy 常量版。"""
import io
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/npu_attention.py"
s = io.open(P, encoding="utf-8").read()

start = s.index("def _perm_indices(")
end = s.index("def _rotate_half(")
new = '''_PERM_CACHE: dict = {}
_CAUSAL_CACHE: dict = {}


def _perm_indices(seq: int, heads: int, device) -> torch.Tensor:
    """`[B, S*H, D]` ➜ `[B, H*S, D]` 的行置换索引（★纯 numpy 常量★）。

    原布局行 = s*H + h；目标行 = h*S + s。
    ★必须用 numpy 造好再 from_numpy★：用 Python 循环造张量会让旧导出器把它
      展开成一堆算子（图爆炸到 Constant 18576 ✗，§59 实测）
    """
    key = (seq, heads, str(device))
    if key not in _PERM_CACHE:
        idx = np.arange(seq * heads, dtype=np.int64).reshape(seq, heads).T.reshape(-1)
        _PERM_CACHE[key] = torch.from_numpy(idx.copy()).to(device)
    return _PERM_CACHE[key]


def _unperm_tensor(seq: int, heads: int, device) -> torch.Tensor:
    """上式的逆置换 ✓（同样纯 numpy）。"""
    key = ("inv", seq, heads, str(device))
    if key not in _PERM_CACHE:
        base = np.arange(seq * heads, dtype=np.int64).reshape(seq, heads).T.reshape(-1)
        _PERM_CACHE[key] = torch.from_numpy(np.argsort(base).astype(np.int64).copy()).to(device)
    return _PERM_CACHE[key]


def _causal_bias(seq: int, device, dtype=torch.float32) -> torch.Tensor:
    """常量因果掩码（上三角 -inf）—— ★numpy 造，只产出一个 Constant 节点★。

    torch.triu / torch.where 会让旧导出器产出 Trilu / ScatterND ✗（DDK 缺这两个 ✗）
    """
    key = (seq, str(device), str(dtype))
    if key not in _CAUSAL_CACHE:
        m = np.zeros((seq, seq), dtype=np.float32)
        m[np.triu_indices(seq, k=1)] = -np.inf
        _CAUSAL_CACHE[key] = torch.from_numpy(m).to(device=device, dtype=dtype)
    return _CAUSAL_CACHE[key]


'''
s = s[:start] + new + s[end:]
s = s.replace("_perm_indices(b, s, n_head, x.device)", "_perm_indices(s, n_head, x.device)")
s = s.replace("_unperm_tensor(b, s, n_head, x.device)", "_unperm_tensor(s, n_head, x.device)")
io.open(P, "w", encoding="utf-8").write(s)
print("npu_attention 已改为纯 numpy 常量 ✓")
