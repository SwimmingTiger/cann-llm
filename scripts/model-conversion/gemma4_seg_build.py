"""Gemma 4 E2B 文本侧的分段构建（3 维 NPU 友好实现）。

为什么必须这样写（全部来自实测，细节见 docs/offline-model-nnrt.md §13）：
  ① 分段是硬要求：embed_tokens_per_layer = [262144, 8960] = 2.35e9 元素，
     超过 OMG 的单张量 INT_MAX 上限 ⇒ 只能按层切开，每段只带自己那几层；
  ② 段内 config 一律不动（num_hidden_layers / num_kv_shared_layers 都要保持全局，
     否则 first_kv_shared_layer_idx 算错、KV 共享语义全错 —— 实测输出会差 350）；
  ③ per_layer 要预先切片传进来，并把 project_per_layer_inputs 换成"直通"
     （它被无条件调用且按全局层数 reshape）；段内末尾的 self.norm 要关掉；
  ④ rotate_half 必须用【常量矩阵乘法】实现（x @ P）：
     原实现会把 head_dim 切成两半 ⇒ 4 维 StridedSlice ⇒ OMG 直接拒收；
  ⑤ 注意力全部用【3 维】张量（[heads, seq, head_dim]）：NPU-CL 对 ≥4 维支持很差；
  ⑥ P 必须注册成 module buffer：若在 forward 里构造，ONNX 追踪会把它展开成
     上万个节点（实测 1 层 21521 节点 ⇒ 注册后 599 节点）。
"""
"""NPU 友好的 Gemma 段图（3 维数学），先只做 prefill、不带 KV，验证数值与 HF 一致。

复刻自 modeling_gemma4.py（已逐行读过）：
  DecoderLayer: input_norm → attn → post_attn_norm → +res
                → pre_ffn_norm → mlp → post_ffn_norm → +res
                → (per_layer 分支) → × layer_scalar
  Attention  : q=q_proj→q_norm→rope→(transpose)；k/v 同（v 只 v_norm）
               打分 sdpa(q,k,v, mask, scaling) → reshape → o_proj
差异：全部用【3 维】张量（[heads, seq, head_dim]），不用 4 维 ⇒ 绕开 NPU-CL 的限制 ✓
"""
import torch, torch.nn as nn, torch.nn.functional as F, time
from transformers import AutoModelForCausalLM

M = "/home/hu60/work/llm/ddk-llm/models/gemma-4-E2B-it"
START, NO, SEQ = 0, 4, 4

m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
tm = m.model.language_model
print("  载入 ✓", flush=True)

ids  = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
mask4 = torch.zeros([1, 1, SEQ, SEQ], dtype=torch.float32)
ref = {}
h = tm.layers[3].register_forward_hook(
        lambda mo, i, o: ref.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
with torch.no_grad():
    tm(input_ids=ids, attention_mask=mask4, use_cache=False)
h.remove()
print("  HF 参考 layer3 输出:", tuple(ref["h"].shape), flush=True)

def rmsnorm(x, w, eps):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v.to(x.dtype)) * w

def rope3(x, cos, sin):
    """x: [heads, seq, d] · cos/sin: [seq, d] ⇒ 用矩阵版 rotate_half（无切片）✓"""
    d = x.shape[-1]; h = d // 2
    P = torch.zeros(d, d)
    for i in range(h):
        P[i, i + h] = -1.0; P[i + h, i] = 1.0
    rh = x @ P.to(x.dtype)
    c = cos.unsqueeze(0); s = sin.unsqueeze(0)
    return x * c + rh * s

class Seg3D(nn.Module):
    def __init__(self, tm, start, no):
        super().__init__()
        self.layers = nn.ModuleList(list(tm.layers)[start:start + no])
        self.eps = 1e-6
        self.scaling = None
    def attn(self, L, x, cos, sin, mask3):
        A = L.self_attn
        B, S, _ = x.shape
        heads = A.q_proj.out_features // A.head_dim
        kvh   = A.k_proj.out_features // A.head_dim
        D = A.head_dim
        q = A.q_proj(x).view(heads, S, D)
        q = rmsnorm(q, A.q_norm.weight, self.eps)
        q = rope3(q, cos, sin)
        k = A.k_proj(x).view(kvh, S, D)
        k = rmsnorm(k, A.k_norm.weight, self.eps)
        k = rope3(k, cos, sin)
        v = A.v_proj(x).view(kvh, S, D) if A.v_proj is not None else k
        v = rmsnorm(v, A.v_norm.weight, self.eps)          # ★ v 只做 norm，不做 rope ✓
        sc = self.scaling if self.scaling is not None else (D ** -0.5)
        att = torch.matmul(q, k.transpose(-1, -2)) * sc     # [heads, S, S] ✓ 3 维
        att = att + mask3                                    # mask3: [1, S, S] 广播 ✓
        att = F.softmax(att, dim=-1)
        o = torch.matmul(att, v)                             # [heads, S, D]
        o = o.reshape(B, S, heads * D)
        return A.o_proj(o)
    def ffn(self, L, x):
        return L.mlp(x)
    def forward(self, x, cos, sin, mask3, *per_layers):
        for i, L in enumerate(self.layers):
            residual = x
            xn = L.input_layernorm(x)
            xn = self.attn(L, xn, cos, sin, mask3)
            xn = L.post_attention_layernorm(xn)
            x = residual + xn
            residual = x
            xn = L.pre_feedforward_layernorm(x)
            xn = self.ffn(L, xn)
            xn = L.post_feedforward_layernorm(xn)
            x = residual + xn
            if getattr(L, "hidden_size_per_layer_input", 0):
                residual = x
                xn = L.per_layer_input_gate(x)
                xn = L.act_fn(xn)
                xn = xn * per_layers[i]
                xn = L.per_layer_projection(xn)
                xn = L.post_per_layer_input_norm(xn)
                x = residual + xn
            x = x * L.layer_scalar
        return x

w = Seg3D(tm, START, NO).eval()
# 取 HF 的 cos/sin 与 per_layer（复用它的 helper ✓）
with torch.no_grad():
    hidden = tm.embed_tokens(ids)
    ple = tm.get_per_layer_inputs(ids, hidden)
    ple = tm.project_per_layer_inputs(hidden, ple)          # [1,4,35,256]
    cos, sin = tm.rotary_emb(hidden, torch.arange(SEQ).unsqueeze(0), tm.config.layer_types[0])
    print("  cos/sin:", tuple(cos.shape), tuple(sin.shape), "· ple:", tuple(ple.shape), flush=True)
    mask3 = torch.zeros([1, SEQ, SEQ])
    out = w(hidden, cos[0], sin[0], mask3,
            *[ple[:, :, i, :] for i in range(START, START + NO)])
print("  3D 段输出:", tuple(out.shape), flush=True)
d = (out - ref["h"]).abs().max().item()
print("  ★ 与 HF layer3 输出最大差 = %.3e ⇒ %s" %
      (d, "★★ 3D 版数学一致 ★★" if d < 1e-3 else "✗ 有差异，要查"), flush=True)
