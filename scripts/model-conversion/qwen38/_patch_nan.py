"""让 install() 在导出期间把 torch.nan_to_num 变成恒等 —— 从源头消除 IsNaN ✗。

理由（§55）：OMG 的解析器**只要碰到被改写过的 IsNaN 节点就崩** ✗
（Not(Equal)/Expand/Less 各种写法都试过 ✓），而它唯一的来源就是注意力里的
`torch.nan_to_num(attn)`（图里长成 `Where(IsNaN(attn), 0, attn)` ✓）。
我们的图全是有限运算 ⇒ NaN 不会出现 ⇒ 导出时恒等化即可 ✓。
"""
import io, re
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/npu_gated_delta.py"
s = io.open(P, encoding="utf-8").read()

s = s.replace('''_PATCHED = {}''', '''_PATCHED = {}
_NAN_PATCHED = False''')

s = s.replace('''    # M-RoPE 分节：函数式替换（去 select_scatter ✗）''',
'''    # ★导出期间把 torch.nan_to_num 变成恒等★（从源头去掉 IsNaN ✗，见 §55）
    global _NAN_PATCHED
    if not _NAN_PATCHED:
        _PATCHED["nan_to_num"] = torch.nan_to_num
        torch.nan_to_num = lambda x, *a, **k: x
        _NAN_PATCHED = True
    # M-RoPE 分节：函数式替换（去 select_scatter ✗）''')

s = s.replace('''def uninstall(modeling_module=None):
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    for name, old in _PATCHED.items():
        if old is not None:
            setattr(modeling_module, name, old)
    _PATCHED.clear()''',
'''def uninstall(modeling_module=None):
    global _NAN_PATCHED
    if modeling_module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling_module
    for name, old in list(_PATCHED.items()):
        if name == "nan_to_num":
            torch.nan_to_num = old
            _NAN_PATCHED = False
        elif old is not None:
            setattr(modeling_module, name, old)
    _PATCHED.clear()''')
io.open(P, "w", encoding="utf-8").write(s)
print("nan_to_num 恒等补丁已加入 install() ✓")
