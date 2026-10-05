import io
P = "export_hiai_q35.py"
s = io.open(P, encoding="utf-8").read()
# ① 输入构造：no-embed-head 时第一路是 input_embed [B,S,H] ✓（对齐官方 embedding_separate 约定 ✓）
s = s.replace('''    input_ids = torch.ones(b, s, dtype=torch.int64)
    attention_mask = torch.ones(b, 1, s, kv, dtype=torch.float32)''',
'''    if args.no_embed_head:
        # ★官方 embedding_separate 约定：第一路输入是 input_embed [B,S,H]★ ✓（不是 input_ids ✗）
        first = torch.zeros(b, s, tc.hidden_size, dtype=torch.float32)
        first_name = "input_embed"
    else:
        first = torch.ones(b, s, dtype=torch.int64)
        first_name = "input_ids"
    attention_mask = torch.ones(b, 1, s, kv, dtype=torch.float32)''')
s = s.replace('''    inputs = [input_ids, attention_mask, position_ids, pos_new]
    in_names = ["input_ids", "attention_mask", "position_ids", "new_kv_cache_pos"]''',
'''    inputs = [first, attention_mask, position_ids, pos_new]
    in_names = [first_name, "attention_mask", "position_ids", "new_kv_cache_pos"]''')
# ② forward 里对应改名 ✓
s = s.replace('''        def forward(self, input_ids, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = self.body.embed_tokens(input_ids) if not args.no_embed_head else input_ids''',
'''        def forward(self, first, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = first if args.no_embed_head else self.body.embed_tokens(first)''')
io.open(P, "w", encoding="utf-8").write(s)
print("已修输入约定 ✓")
