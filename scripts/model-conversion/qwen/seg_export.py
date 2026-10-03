#!/usr/bin/env python3
"""分段导出 Qwen2 的某几层 —— 为「按层切段」准备（见 docs/offline-model-nnrt.md §11）。

为什么需要它
------------
整模型上不了 NPU：设备侧 KV 张量不能大、OMG 侧对「层数 × KV」另有限制，
两者叠加使 24 层无解。实测 **4 层 + kv_cache_max_len=1024** 能 Predict 成功
（含中间段），所以把模型按层切开、逐段编译、段间用 KV 串起来。

厂商的 npu_tuned_model 封装是【整模型】的（既做 embedding 又做 lm_head/norm），
中间层拿不出来。这里用 monkey-patch 换成"分段 forward"，**厂商代码保持原样**：

    第 0 段：input_ids   → hidden_states   (+ KV 出)
    中间段：hidden       → hidden_states   (+ KV 出)
    末段：  hidden       → lm_logits       (+ KV 出)

用法
----
    seg_export.py <总层数> <kv_len> <起始层> <层数> <输出目录> [seq_len]

例：24 层的模型按 4 层切段 ——
    seg_export.py 24 1024  0 4 ~/out/seg0
    seg_export.py 24 1024  4 4 ~/out/seg4
    ...
    seg_export.py 24 1024 20 4 ~/out/seg20

★ 四个必须按段切换/替换的点（都是实测踩出来的）：
  1. **换 forward 要用子类**，只改类的 forward 属性不生效（跑的仍是原来那份）；
     而且厂商脚本是 `from npu_tuned_model import build_model`（导入时绑定），
     所以 `NT.build_model` 与 `EX.build_model` **都要**替换；
  2. `export_model` 读的是 `__main__` 里 `for seq_len in …` 设下的**全局** seq_len，
     直接调用要自己补 `EX.seq_len = …`；
  3. `embedding_separate`：首段 False（导出器给的 dummy 是 token id、输入名 input_ids）；
     其余段 True（导出器把 dummy 查表成 [1,1,896] hidden、输入名变 input_embed 并带 embed_scales）；
  4. `embedding_as_fp16`：中间段的输入是 hidden states，给 False 会被铸成 **int8**（错），
     必须 True 得 FP32。
"""
import os, sys, json
import torch

LAYERS, KV, START, COUNT, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
SEQ = int(sys.argv[6]) if len(sys.argv) > 6 else 1
E = os.path.expanduser("~/work/llm/ddk-llm/cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export")
sys.path.insert(0, E)

import npu_tuned_model.qwen2.export_model_wrapper as W

IS_FIRST = (START == 0)
IS_LAST = (START + COUNT == LAYERS)

# ---- ① 载入后把 layers 截成 [START, START+COUNT) ----
_orig_init = W.Qwen2ForCausalLMWrapper.__init__
def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    all_layers = list(self.model.model.layers)
    self.model.model.layers = torch.nn.ModuleList(all_layers[START:START + COUNT])
    self.model.config.num_hidden_layers = COUNT
    print("  截取层: [%d, %d) ⇒ %d 层" % (START, START + COUNT, COUNT), flush=True)
W.Qwen2ForCausalLMWrapper.__init__ = _init

# ---- ② 换成分段 forward ----
def _seg_forward(self, input_ids, attention_mask, position_ids, past_key_values,
                 new_kv_cache_pos=None, embed_scale=None, output_pos=None,
                 output_attentions=False, output_hidden_states=False, use_cache=True):
    self.forward_count += 1
    if IS_FIRST:
        # 第一段：token id → 嵌入（embedding 在图里，不需要外部 scale）
        if self.embedding_in_omc:
            if self.scales is not None:
                inputs_embeds = self.model.model.embed_tokens(input_ids) * self.scales[input_ids]
            else:
                inputs_embeds = self.model.model.embed_tokens(input_ids)
        else:
            assert embed_scale is not None
            inputs_embeds = input_ids * embed_scale
    else:
        # 中间/末段：输入就是上一段吐出来的 hidden states
        inputs_embeds = input_ids
    hidden_states = inputs_embeds
    kv_out = []
    for idx, layer in enumerate(self.model.model.layers):
        pkv = past_key_values[idx] if past_key_values else None
        out = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=pkv, output_attentions=output_attentions,
                    new_kv_cache_pos=new_kv_cache_pos, use_cache=use_cache)
        hidden_states = out[0]
        if use_cache:
            kv_out.extend(out[2 if output_attentions else 1])
    if not IS_LAST:
        return hidden_states, *kv_out          # ★ 中间段：吐 hidden states
    bsz, q_len, h = hidden_states.size()
    if output_pos is not None:
        hidden_states = hidden_states[:, output_pos, :]
    hidden_states = hidden_states.view(-1, h)
    hidden_states = self.model.model.norm(hidden_states)
    logits = self.model.lm_head(hidden_states).view(bsz, -1, self.model.lm_head.weight.shape[0])
    return logits, *kv_out
# ★ 只改类的 forward 属性实测没生效（跑的还是厂商那份）⇒ 改成【子类】，
#   并 hook build_model 让导出器拿到我们的子类，这样实例一定用新 forward。
import npu_tuned_model as NT
_orig_build = NT.build_model
def _build(arch):
    Base = _orig_build(arch)
    return type("SegWrapper", (Base,), {"forward": _seg_forward})
NT.build_model = _build


def dims_override(wrapper, input_names, output_names, dynamic_axes):
    """★ 把中间段的输入 dtype 从 int32 改回 float32（它收的是 hidden states）。"""
    pass


# ---- ③ 用厂商的导出入口跑，但改掉输入输出的名字/类型 ----
import export_model_single_qwen2 as EX
# ★ 关键：厂商脚本是 `from npu_tuned_model import build_model` —— 导入那一刻就把
#   原函数绑进了它自己的模块命名空间 ⇒ 只改 NT.build_model 没用，必须连它一起改。
EX.build_model = _build

_orig_export = EX.export_model
def _export(cfg):
    import numpy as np
    # 先让厂商脚本照常载入+建图，然后在导出时替换掉第一段的"输入"语义
    cfg = dict(cfg)
    cfg["_seg"] = (IS_FIRST, IS_LAST, START, COUNT)
    return _orig_export(cfg)
EX.export_model = _export

cfg = {
    "hf_model_path": os.path.expanduser("~/work/llm/ddk-llm/models/Qwen2.5-0.5B-Instruct"),
    "config_file": None, "quant_pth": None,
    "output_dir": OUT, "dump_path": OUT + "/dump",
    # ★ 首段吃 token id ⇒ embedding_separate=False（导出器 dummy 保持 ids，输入名 input_ids）
    #   其余段吃上一段的 hidden ⇒ True（导出器会把 dummy 查表成 [1,1,896] hidden，
    #   输入名变 input_embed 并附带 embed_scales —— 我们的分段 forward 非首段直接用输入、
    #   不乘 scale，所以正好对上）。
    "embedding_config": {"embedding_in_omc": True, "embedding_separate": (not IS_FIRST),
                         # 中间段的输入是 hidden states ⇒ 必须 FP32（否则导出器会把它铸成 int8 ✗）
                         "embedding_quant": False, "embedding_as_fp16": (not IS_FIRST), "mul_twice": False},
    "no_gemm": True, "model_arch": "qwen2",
    "hf_model_device": "cpu", "hf_model_data_type": "torch.float32",
    "onnx_output_model_name": "seg", "onnx_opset": 12, "batch": 1,
    "kv_cache_max_len": KV, "layers": COUNT, "seq_len": [SEQ],
    "run_mode": "hardware", "lora": {"enable": False},
}
os.makedirs(OUT, exist_ok=True)
print("  段: 首段=%s 末段=%s · %d..%d · KV=%d · seq=%d" % (IS_FIRST, IS_LAST, START, START+COUNT, KV, SEQ), flush=True)
# ★ 厂商脚本的 __main__ 里是 `for seq_len in config["seq_len"]: export_model(...)`，
#   而 export_model 内部读的是这个【全局】 seq_len —— 直接调它就 NameError。
EX.seq_len = SEQ
EX.export_model(cfg)
print("  ✓ 导出结束", flush=True)
