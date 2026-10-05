import numpy as np
from torch import nn

from .modeling_qwen3 import Qwen3ForCausalLM

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Qwen3ForCausalLMWrapper(nn.Module):
    def __init__(self, model_path, device, dtype, embedding_config):
        super(Qwen3ForCausalLMWrapper, self).__init__()
        self.model = Qwen3ForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype, trust_remote_code=False)
        self.model.eval()
        self.forward_count = 0
        self.embedding_in_omc = embedding_config.get("embedding_in_omc", False)
        self.mul_twice = embedding_config.get("mul_twice", False)
        self.scales = None
    
    def set_int_embedding(self, ckpt):
        we = ckpt["model.embed_tokens.weight"]
        sa = ckpt["model.embed_tokens.quant_op.weight_quantizer.s"]
        qw = we / sa
        dd = qw - torch.round(qw)
        qw = torch.round(qw)
        self.model.model.embed_tokens.weight = torch.nn.Parameter(qw.to(torch.int8), requires_grad=False)
        self.scales = sa
    
    def forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos=None,
            embed_scale=None,
            output_pos=None,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
    ):
        self.forward_count += 1
        if self.embedding_in_omc:
            if self.scales is not None:
                inputs_embeds = self.model.model.embed_tokens(input_ids)
                scale = self.scales[input_ids]
                inputs_embeds = inputs_embeds * scale
            else:
                inputs_embeds = self.model.model.embed_tokens(input_ids)

        else:
            assert embed_scale is not None
            inputs_embeds = input_ids * embed_scale
            if self.mul_twice:
                inputs_embeds = inputs_embeds * embed_scale

        hidden_states = inputs_embeds
        kv_caches_out = []
        for idx, decoder_layer in enumerate(self.model.model.layers):
            past_key_value = past_key_values[idx] if past_key_values else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                new_kv_cache_pos=new_kv_cache_pos,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                kv_caches_out.extend(layer_outputs[2 if output_attentions else 1],)

        bsz, q_len, hidden_state_len = hidden_states.size()
        if output_pos is not None: hidden_states = hidden_states[:, output_pos, :]
        hidden_states = hidden_states.view(-1, hidden_state_len)
        hidden_states = self.model.model.norm(hidden_states)
        lm_logits = self.model.lm_head(hidden_states).view(bsz, -1, self.model.lm_head.weight.shape[0])
        return lm_logits, *kv_caches_out


class Qwen3ForCausalLMWithoutEmbeddingWrapper(nn.Module):
    def __init__(self, model_path, device, dtype):
        super(Qwen3ForCausalLMWithoutEmbeddingWrapper, self).__init__()
        self.model = Qwen3ForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype).eval()
        self.forward_count = 0

    def get_embedding_weight(self):
        embed_weight = self.model.model.embed_tokens.weight.detach().numpy().astype(np.float16)
        print(f"embed_weight shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")
        return embed_weight

    def forward(
            self,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
            new_kv_cache_pos=None,
            output_pos=None,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
    ):
        hidden_states = inputs_embeds
        kv_caches_out = []
        for idx, decoder_layer in enumerate(self.model.model.layers):
            past_key_value = past_key_values[idx] if past_key_values else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                new_kv_cache_pos=new_kv_cache_pos,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            kv_caches_out.extend(layer_outputs[2 if output_attentions else 1], )

        bsz, q_len, hidden_state_len = hidden_states.size()
        if output_pos is not None:
            hidden_states = hidden_states[:, output_pos, :]
        hidden_states = hidden_states.view(-1, hidden_state_len)
        hidden_states = self.model.model.norm(hidden_states)
        lm_logits = self.model.lm_head(hidden_states).view(bsz, -1, 151936)
        return lm_logits, *kv_caches_out


if __name__ == "__main__":
    pass
