"""把 legacy tracer 下会变成 Tensor 的 shape 取用，改成显式 Python int 参数。

§53 已踩过同类坑（L.shape[-1] ⇒ Tensor ⇒ bit_length 报错 ✗）。
本文件把 delta rule 与线性层里的 batch/seq 也改成显式传入 ✓。
"""
import io
P = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/npu_gated_delta.py"
s = io.open(P, encoding="utf-8").read()
s = s.replace('''    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):''', '''    use_qk_l2norm_in_kernel: bool = False,
    seq_len: int = 0,
    batch: int = 0,
    **kwargs,
):''')
s = s.replace('''    initial_dtype = query.dtype
    batch_size, seq_len, hk, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]''',
'''    initial_dtype = query.dtype
    # ★shape 取值一律走显式 Python int★：legacy TorchScript 追踪下
    #   `key.shape[1]` 可能是 Tensor ✗（§53/§66 实测）⇒ 这里优先用传入的 int ✓
    _bs, _sl, hk, k_head_dim = key.shape
    batch_size = batch or _bs
    seq_len = seq_len or _sl
    num_v_heads, v_head_dim = value.shape[-2:]''')
io.open(P, "w", encoding="utf-8").write(s)
print("delta rule 已支持显式 seq_len/batch ✓")

P2 = "/storage/Users/currentUser/work/llm/cann-llm/scripts/model-conversion/qwen38/npu_layers.py"
t = io.open(P2, encoding="utf-8").read()
t = t.replace('''def linear_attention_layer(layer, hidden, conv_state, rec_state, heads, kv_heads, hd, seq: int = 0,
                           trace: list | None = None):''',
'''def linear_attention_layer(layer, hidden, conv_state, rec_state, heads, kv_heads, hd, seq: int = 0,
                           batch: int = 0, trace: list | None = None):''')
t = t.replace('''    la = layer.linear_attn
    b, s, _ = hidden.shape''', '''    la = layer.linear_attn
    # ★同样只用显式 int★（避免 legacy 追踪把 shape 变 Tensor ✗）
    b = batch or hidden.shape[0]
    s = seq or hidden.shape[1]''')
t = t.replace('''    core, new_rec = npu_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=64, initial_state=rec_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)''',
'''    core, new_rec = npu_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=64, initial_state=rec_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True, seq_len=s, batch=b)''')
t = t.replace('''        h, nk, nv = linear_attention_layer(layer, h, k_slot, v_slot, heads, kv_heads, hd, seq)''',
'''        h, nk, nv = linear_attention_layer(layer, h, k_slot, v_slot, heads, kv_heads, hd, seq, batch=batch)''')
t = t.replace('''def layer_forward(layer, hidden, mask, cos, sin, k_slot, v_slot, idx, layer_type,
                  kv_max, heads, kv_heads, hd, seq: int = 0):''',
'''def layer_forward(layer, hidden, mask, cos, sin, k_slot, v_slot, idx, layer_type,
                  kv_max, heads, kv_heads, hd, seq: int = 0, batch: int = 0):''')
io.open(P2, "w", encoding="utf-8").write(t)
print("线性层已支持显式 batch ✓")
