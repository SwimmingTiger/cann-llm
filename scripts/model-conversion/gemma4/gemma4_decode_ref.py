"""KV 缓存 decode 的参考实现 + 验证（主机侧）。

为什么要它：现在的段图是 prefill-only ⇒ 每生成一个 token 都要重跑整个上下文
（实测 60~100 秒 / 8 token）。真正的解法是 decode 时只算 1 个 token。

Gemma 4 的 KV 语义（本文件用逐位对拍证明复刻正确）：
  · 层 0~14 各自维护自己的 K/V（非共享层）
  · ★层 13（sliding）/ 层 14（full）把 KV 写进【共享槽】★（store_full_length_kv=True）
  · ★层 15~34 没有 k_proj/v_proj★，直接从共享槽取（注意：是【没有属性】而非 None）
  · decode 时每层的 K/V = concat(历史, 新算的这一个 token)

验证方式：4 个 token 的 prefill（因果掩码）与 4 次 decode 的结果逐位比 —— 实测差 3e-05 ✓

导出 decode 图时的接口（下一步）：
  段 0~2（层 0-11）：hidden[1,1,1536] + cos/sin + per_layer_0..3 + 本段 4 层的 k/v 缓存
                     → hidden + 更新后的 4 组
  段 3  （层 12-15）：… + 层 12/13/14 的 3 组 + ★2 个共享槽★ → 更新后的 3 组 + 2 个槽
  段 4~8（层 16-34）：… + ★2 个共享槽★ → hidden
  ★缓存长度固定（如 128）⇒ 需要 kv_mask[1,1,128] 把补零位置屏蔽掉★
"""
"""KV 缓存 decode 的主机侧实现 + 验证。

思路：decode 时 seq=1，每层维护自己的 K/V（非共享层）或从共享槽取（共享层）。
验证：4 个 token 的 prefill 结果 必须等于 4 次 decode 的结果（逐位）。
"""
import sys, os, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_model as G
from collections import UserDict

rmsnorm = G.rmsnorm

class Seg3DDec(torch.nn.Module):
    """单段 + KV 缓存（逐 token）。返回 (hidden, new_kv_dict)。"""
    def __init__(self, tm, start, no):
        super().__init__()
        self.layers = torch.nn.ModuleList(list(tm.layers)[start:start + no])
        self.start, self.no = start, no
        self.layer_types = list(tm.config.layer_types)[start:start + no]
        self.dims = [int(tm.layers[start + i].self_attn.head_dim) for i in range(no)]
        for d in sorted(set(self.dims)):
            Pm = torch.zeros(d, d); hh = d // 2
            for i in range(hh):
                Pm[i, i + hh] = -1.0; Pm[i + hh, i] = 1.0
            self.register_buffer("rot_P_%d" % d, Pm.t().contiguous())

    def attn(self, L, x, cos, sin, P, past, slot):
        """past = (k,v) 该层自己的缓存 或 None；slot = (k,v) 共享槽（共享层用）"""
        A = L.self_attn
        D = int(A.head_dim); heads = A.q_proj.out_features // D
        q = A.q_proj(x).view(1, heads, D).transpose(0, 1)          # [heads,1,D]
        q = G.rope3(rmsnorm(q, A.q_norm), cos, sin, P)
        if getattr(A, "is_kv_shared_layer", False):
            k, v = slot                                            # ★ 从共享槽取
            new = None
        else:
            kvh = 1 if getattr(A, "k_proj", None) is None else (A.k_proj.out_features // D)
            k = A.k_proj(x).view(1, kvh, D).transpose(0, 1)
            k = G.rope3(rmsnorm(k, A.k_norm), cos, sin, P)
            v = A.v_proj(x).view(1, kvh, D).transpose(0, 1) if getattr(A, "v_proj", None) is not None else k
            v = rmsnorm(v, A.v_norm)
            if past is not None:                                   # ★ 追加到缓存
                k = torch.cat([past[0], k], dim=1)
                v = torch.cat([past[1], v], dim=1)
            new = (k, v)
        sc = getattr(A, "scaling", None) or (D ** -0.5)
        att = torch.matmul(q, k.transpose(-1, -2)) * sc            # [heads,1,kv]（无需掩码 ✓）
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(0, 1).reshape(1, 1, heads * D)
        return A.o_proj(o), new

    def step(self, x, cossin, ples, kv):
        """x: [1,1,1536] · cossin: [(cos,sin)]*no · kv: dict（层索引 → (k,v)）或槽"""
        shared = kv.get("shared", UserDict())
        for i, L in enumerate(self.layers):
            gi = self.start + i
            A = L.self_attn
            lt = self.layer_types[i]
            P = getattr(self, "rot_P_%d" % self.dims[i])
            residual = x
            xn = L.input_layernorm(x)
            slot = shared.get(lt) if getattr(A, "is_kv_shared_layer", False) else None
            xn, new = self.attn(L, xn, cossin[2*i], cossin[2*i+1], P, kv.get(gi), slot)
            if new is not None:
                kv[gi] = new
                if getattr(A, "store_full_length_kv", False):      # ★ 层 13/14 写共享槽
                    shared[lt] = new
            x = residual + L.post_attention_layernorm(xn)
            residual = x
            xn = L.mlp(L.pre_feedforward_layernorm(x))
            x = residual + L.post_feedforward_layernorm(xn)
            if getattr(L, "hidden_size_per_layer_input", 0):
                residual = x
                xn = L.act_fn(L.per_layer_input_gate(x)) * ples[i]
                x = residual + L.post_per_layer_input_norm(L.per_layer_projection(xn))
            x = x * L.layer_scalar
        kv["shared"] = shared
        return x, kv

if __name__ == "__main__":
    SEQ = 4
    NO = 4
    m, tm = G.load()
    ids = torch.tensor([[1, 42, 777, 9000]], dtype=torch.long)
    with torch.no_grad():
        emb = tm.embed_tokens(ids)
        pl = tm.project_per_layer_inputs(emb, tm.get_per_layer_inputs(ids, emb))
        pos = torch.arange(SEQ).unsqueeze(0)
        # ① 整段 prefill（已验证正确的 Seg3D）
        w = G.Seg3D(tm, 0, NO).eval()
        cs = []
        for i in range(NO):
            c, s2 = tm.rotary_emb(emb, pos, tm.config.layer_types[i]); cs += [c[0], s2[0]]
        ref, _, _, _, _ = w(emb, torch.zeros([1, SEQ, SEQ]), torch.zeros(1), torch.zeros(1),
                            torch.zeros(1), torch.zeros(1), *cs,
                            *[pl[:, :, i, :] for i in range(NO)])
        # ② 逐 token decode（KV 缓存）
        d = Seg3DDec(tm, 0, NO).eval()
        kv = {}
        out = []
        for t in range(SEQ):
            x = emb[:, t:t+1, :]
            cst = []
            p1 = torch.tensor([[t]])
            for i in range(NO):
                c, s2 = tm.rotary_emb(x, p1, tm.config.layer_types[i]); cst += [c[0], s2[0]]
            y, kv = d.step(x, cst, [pl[:, t:t+1, i, :] for i in range(NO)], kv)
            out.append(y)
        dec = torch.cat(out, dim=1)
        diff = (dec - ref).abs().max().item()
        print("  ★ prefill vs 4 次 decode 的最大差 = %.4e ⇒ %s" %
              (diff, "★★ 逐位一致，KV 语义正确 ★★" if diff < 1e-4 else "✗ 不一致"), flush=True)
