"""把常量掩码全部改成 numpy 常量（消灭 Trilu ✗ —— OMG 的 pre-check 拒绝它）。"""
import io
P = "npu_gated_delta.py"
s = io.open(P, encoding="utf-8").read()
old_start = s.index("def tril_ones(")
old_end = s.index("def _l2norm(")
new = '''_MASK_CACHE: dict = {}


def tril_ones(c: int, dtype: torch.dtype, device) -> torch.Tensor:
    """★前缀和用的常量矩阵★：`M[j,t] = 1 ⟺ j <= t`，即 `triu(ones)` ✓

    ★坑★：直觉会写 `tril(ones)`，但 `tril(ones)[j,t] = 1 ⟺ j >= t` ⇒ 算出来是【后缀和】✗
    （实测正好反了 ✓，§51）⇒ cumsum 是前缀和 ⇒ 必须用 triu(ones) ✓。

    ★用 numpy 造★：`torch.triu` 会导出成 `Trilu` 节点 ✗，而 OMG 的 pre-check 拒收它 ✗
    （§68 实测：全图只剩 Trilu 一个 fail ✓）
    """
    key = ("ones", c, str(dtype), str(device))
    if key not in _MASK_CACHE:
        arr = np.triu(np.ones((c, c), dtype=np.float32))
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return _MASK_CACHE[key]


def strict_lower(c: int, device) -> torch.Tensor:
    """严格下三角（布尔 ✓，numpy 常量 ✓）。"""
    key = ("lo", c, str(device))
    if key not in _MASK_CACHE:
        arr = np.tril(np.ones((c, c), dtype=bool), -1)
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device)
    return _MASK_CACHE[key]


def strict_upper(c: int, device) -> torch.Tensor:
    """严格上三角（布尔 ✓，numpy 常量 ✓）。"""
    key = ("up", c, str(device))
    if key not in _MASK_CACHE:
        arr = np.triu(np.ones((c, c), dtype=bool), 1)
        _MASK_CACHE[key] = torch.from_numpy(arr).to(device=device)
    return _MASK_CACHE[key]


'''
s = s[:old_start] + new + s[old_end:]
if "import numpy as np" not in s:
    s = s.replace("import torch", "import numpy as np\nimport torch", 1)
io.open(P, "w", encoding="utf-8").write(s)
print("掩码已 numpy 化 ✓")

P2 = "export_hiai_q35.py"
t = io.open(P2, encoding="utf-8").read()
t = t.replace('''    out_names = ["lm_logits"]''',
'''    out_names = ["hidden_states" if args.no_embed_head else "lm_logits"]   # ★名字要对上★ ✓''')
io.open(P2, "w", encoding="utf-8").write(t)
print("输出名已修 ✓")
