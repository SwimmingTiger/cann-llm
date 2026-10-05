import io
P = "export_hiai_q35.py"
s = io.open(P, encoding="utf-8").read()
s = s.replace('''        def forward(self, input_ids, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = self.body.embed_tokens(input_ids) if not args.no_embed_head else input_ids
            hidden = inputs_embeds''',
'''        def forward(self, input_ids, attention_mask, position_ids, new_kv_cache_pos, *states):
            """states 按层给出：每层两个张量（第 i 层的 key 槽 / value 槽 ✓）。"""
            inputs_embeds = self.body.embed_tokens(input_ids) if not args.no_embed_head else input_ids
            # ★M-RoPE 的 position_ids 是 [3,B,S]★ ✓（官方接口给 [B,S] ⇒ 这里展开 ✓）
            pos3 = position_ids
            if pos3.dim() == 2:
                pos3 = pos3.unsqueeze(0).expand(3, -1, -1)
            cos, sin = self.body.rotary_emb(inputs_embeds, pos3)
            hidden = inputs_embeds''')
s = s.replace('''                hidden, new_k, new_v = call_layer(layer, hidden, attention_mask, position_ids,
                                                  new_kv_cache_pos, k_slot, v_slot, idx)''',
'''                hidden, new_k, new_v = call_layer(layer, hidden, attention_mask, cos, sin,
                                                  k_slot, v_slot, idx)''')
s = s.replace('''    def call_layer(layer, hidden, mask, pos, pos_new, k_slot, v_slot, idx):''',
'''    def call_layer(layer, hidden, mask, cos, sin, k_slot, v_slot, idx):''')
s = s.replace('''        return npu_layers.layer_forward(layer, hidden, mask, pos, pos_new, k_slot, v_slot,
                                        idx, layer_types[idx], kv, heads, kv_heads, hd)''',
'''        return npu_layers.layer_forward(layer, hidden, mask, cos, sin, k_slot, v_slot,
                                        idx, layer_types[idx], kv, heads, kv_heads, hd)''')
io.open(P, "w", encoding="utf-8").write(s)
print("导出脚本已接入 cos/sin ✓")
