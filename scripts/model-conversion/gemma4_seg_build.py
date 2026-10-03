"""NPU 友好的 Gemma 4 分段实现（3 维数学 + KV 共享）。

复刻自 modeling_gemma4.py，关键差异（全部来自实测）：
  · 全部 3 维张量 [heads, seq, head_dim] —— NPU-CL 对 ≥4 维支持很差
  · rotate_half 用常量矩阵乘法 x@P —— 原实现切 head_dim ⇒ 4 维 StridedSlice ⇒ OMG 拒收
  · P 注册成 module buffer —— forward 里构造会被 ONNX 追踪展开成上万节点
  · head_dim 逐层不同（sliding 256 / full 512）⇒ 逐层取用
  · ★KV 共享★：层 15~34 没有 k_proj/v_proj，复用共享槽；
    只有层 13（sliding）与层 14（full）store_full_length_kv=True ⇒ 只有它们写槽
    ⇒ 段间传 2 组 KV（sliding / full 各一）✓
"""
import os, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM

M = "/home/hu60/work/llm/ddk-llm/models/gemma-4-E2B-it"

def rmsnorm(x, mod):
    e = getattr(mod, "eps", 1e-6)
    v = x.float()
    v = v * torch.pow(v.pow(2).mean(-1, keepdim=True) + e, -0.5)
    if getattr(mod, "with_scale", True) and hasattr(mod, "weight"):
        v = v * mod.weight.float()
    return v.type_as(x)

def rope3(x, cos, sin, P):
    return x * cos.unsqueeze(0) + (x @ P.to(x.dtype)) * sin.unsqueeze(0)

class Seg3D(nn.Module):
    def __init__(self, tm, start, no):
        super().__init__()
        self.layers = nn.ModuleList(list(tm.layers)[start:start + no])
        self.start, self.no = start, no
        self.layer_types = list(tm.config.layer_types)[start:start + no]
        self.dims = [int(tm.layers[start + i].self_attn.head_dim) for i in range(no)]
        for d in sorted(set(self.dims)):
            Pm = torch.zeros(d, d); hh = d // 2
            for i in range(hh):
                Pm[i, i + hh] = -1.0; Pm[i + hh, i] = 1.0
            self.register_buffer("rot_P_%d" % d, Pm.t().contiguous())

    def attn(self, L, x, cos, sin, mask3, P, kv=None):
        A = L.self_attn
        B, S, _ = x.shape
        D = int(A.head_dim)
        heads = A.q_proj.out_features // D
        kvh = (A.k_proj.out_features // D) if A.k_proj is not None else 1
        q = A.q_proj(x).view(S, heads, D).transpose(0, 1)
        q = rope3(rmsnorm(q, A.q_norm), cos, sin, P)
        if kv is None:
            k = A.k_proj(x).view(S, kvh, D).transpose(0, 1)
            k = rope3(rmsnorm(k, A.k_norm), cos, sin, P)
            v = A.v_proj(x).view(S, kvh, D).transpose(0, 1) if A.v_proj is not None else k
            v = rmsnorm(v, A.v_norm)
            new_kv = (k, v)
        else:
            k, v = kv; new_kv = None
        sc = getattr(A, "scaling", None) or (D ** -0.5)
        att = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * sc + mask3, dim=-1)
        o = torch.matmul(att, v).transpose(0, 1).reshape(B, S, heads * D)
        return A.o_proj(o), new_kv

    def forward(self, x, mask3, sk, sv, fk, fv, *args):
        """sk/sv = sliding 共享槽 · fk/fv = full 共享槽（来自上游段；无则传 0 张量）
        args = cos_0,sin_0,...,cos_{n-1},sin_{n-1}, per_layer_0,...,per_layer_{n-1}"""
        n = len(self.layers)
        cossin, ples = args[:2 * n], args[2 * n:]
        for i, L in enumerate(self.layers):
            A = L.self_attn
            lt = self.layer_types[i]
            P = getattr(self, "rot_P_%d" % self.dims[i])
            residual = x
            xn = L.input_layernorm(x)
            if getattr(A, "is_kv_shared_layer", False):
                kv = (sk, sv) if lt == "sliding_attention" else (fk, fv)
            else:
                kv = None
            xn, nkv = self.attn(L, xn, cossin[2*i], cossin[2*i+1], mask3, P, kv)
            if nkv is not None and getattr(A, "store_full_length_kv", False):
                if lt == "sliding_attention": sk, sv = nkv
                else: fk, fv = nkv
            x = residual + L.post_attention_layernorm(xn)
            residual = x
            xn = L.mlp(L.pre_feedforward_layernorm(x))
            x = residual + L.post_feedforward_layernorm(xn)
            if getattr(L, "hidden_size_per_layer_input", 0):
                residual = x
                xn = L.act_fn(L.per_layer_input_gate(x)) * ples[i]
                x = residual + L.post_per_layer_input_norm(L.per_layer_projection(xn))
            x = x * L.layer_scalar
        return x, sk, sv, fk, fv

def load():
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    return m, m.model.language_model

if __name__ == "__main__":
    import sys
    NO = int(os.environ.get("NO", "4")); SEQ = 4
    m, tm = load()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask4 = torch.zeros([1, 1, SEQ, SEQ])
    NTOT = len(tm.layers)
    caps = {}
    hk = [tm.layers[i].register_forward_hook(
            (lambda i: lambda mo, inp, o: caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().clone()))(i))
          for i in range(NTOT)]
    with torch.no_grad():
        tm(input_ids=ids, attention_mask=mask4, use_cache=False)
    for h in hk: h.remove()
    with torch.no_grad():
        x = tm.embed_tokens(ids)
        ple = tm.project_per_layer_inputs(x, tm.get_per_layer_inputs(ids, x))
        pos = torch.arange(SEQ).unsqueeze(0)
        sk = sv = fk = fv = torch.zeros(1)
        st = 0; bad = 0
        while st < NTOT:
            no = min(NO, NTOT - st)
            w = Seg3D(tm, st, no).eval()
            cs = []
            for i in range(st, st + no):
                c, s2 = tm.rotary_emb(x, pos, tm.config.layer_types[i])
                cs += [c[0], s2[0]]
            x, sk, sv, fk, fv = w(x, torch.zeros([1, SEQ, SEQ]), sk, sv, fk, fv,
                                  *cs, *[ple[:, :, i, :] for i in range(st, st + no)])
            last = st + no - 1
            d = (x - caps[last]).abs().max().item(); b = caps[last].abs().max().item()
            ok = d / b < 1e-3
            bad += 0 if ok else 1
            print("  段 [%2d:%2d] head_dim=%-22s → 与 HF layer%-2d 相对差 %.2e %s"
                  % (st, st + no, str(w.dims), last, d / b, "✓" if ok else "✗"), flush=True)
            st += no
        print("  ★ 全链 %d 层：%s" % (NTOT, "★★ 全部一致 ★★" if bad == 0 else "✗ %d 段不符" % bad), flush=True)
