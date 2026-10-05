import os, sys
import gc
import shutil
import torch
import onnx
from onnxsim import simplify
from npu_tuned_model import build_model


def refresh_lora_ckpt(ckpt, lora_ckpt):
    key_map = dict()
    for i in range(22):
        key_map[f"language_model.encoder.layers.{i}.attention.wq.lora.lora_A"] = f"model.layers.{i}.self_attn.q_proj.lora.lora_A.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wq.lora.lora_B"] = f"model.layers.{i}.self_attn.q_proj.lora.lora_B.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wk.lora.lora_A"] = f"model.layers.{i}.self_attn.k_proj.lora.lora_A.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wk.lora.lora_B"] = f"model.layers.{i}.self_attn.k_proj.lora.lora_B.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wv.lora.lora_A"] = f"model.layers.{i}.self_attn.v_proj.lora.lora_A.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wv.lora.lora_B"] = f"model.layers.{i}.self_attn.v_proj.lora.lora_B.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wo.lora.lora_A"] = f"model.layers.{i}.self_attn.o_proj.lora.lora_A.weight"
        key_map[f"language_model.encoder.layers.{i}.attention.wo.lora.lora_B"] = f"model.layers.{i}.self_attn.o_proj.lora.lora_B.weight"
        if i < 15:
            key_map[f"language_model.encoder.layers.{i}.feed_forward.gate_proj.lora.lora_A"] = f"model.layers.{i}.mlp.gate_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.gate_proj.lora.lora_B"] = f"model.layers.{i}.mlp.gate_proj.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.gate_proj2.lora.lora_A"] = f"model.layers.{i}.mlp.gate_proj2.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.gate_proj2.lora.lora_B"] = f"model.layers.{i}.mlp.gate_proj2.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.down_proj.lora.lora_A"] = f"model.layers.{i}.mlp.down_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.down_proj.lora.lora_B"] = f"model.layers.{i}.mlp.down_proj.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.up_proj.lora.lora_A"] = f"model.layers.{i}.mlp.up_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.up_proj.lora.lora_B"] = f"model.layers.{i}.mlp.up_proj.lora.lora_B.weight"
        else:
            key_map[f"language_model.encoder.layers.{i}.attn_linear.0.lora.lora_A"] = f"model.layers.{i}.attn_linear.0.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.attn_linear.0.lora.lora_B"] = f"model.layers.{i}.attn_linear.0.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.attn_linear.2.lora.lora_A"] = f"model.layers.{i}.attn_linear.2.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.attn_linear.2.lora.lora_B"] = f"model.layers.{i}.attn_linear.2.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w1.lora.lora_A"] = f"model.layers.{i}.mlp.gate_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w1.lora.lora_B"] = f"model.layers.{i}.mlp.gate_proj.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w2.lora.lora_A"] = f"model.layers.{i}.mlp.down_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w2.lora.lora_B"] = f"model.layers.{i}.mlp.down_proj.lora.lora_B.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w3.lora.lora_A"] = f"model.layers.{i}.mlp.up_proj.lora.lora_A.weight"
            key_map[f"language_model.encoder.layers.{i}.feed_forward.w3.lora.lora_B"] = f"model.layers.{i}.mlp.up_proj.lora.lora_B.weight"
    for k in key_map:
        ckpt[key_map[k]] = lora_ckpt[k]
    return ckpt


def dec_time(func):
    def _wrap(*args, **argv):
        import time
        st = time.time()
        obj = func(*args, **argv)
        print(f"withdraw cost time ", time.time() - st)
        return obj
    return _wrap
@dec_time
def export_llama(
    export_config
):
    hf_model_path   = os.path.realpath(export_config["hf_model_path"])
    if export_config["quant_pth"] is not None: quant_pth = os.path.realpath(export_config["quant_pth"])
    else: quant_pth = None

    onnx_output_dir = export_config["output_dir"]
    embedding_config = export_config.get("embedding_config", dict())
    embedding_separate = embedding_config.get("embedding_separate", True)
    mul_twice = embedding_config.get("mul_twice", True)
    if onnx_output_dir is None:
        if quant_pth: onnx_output_dir = os.path.join(os.path.dirname(quant_pth), "output", "models")
        else: onnx_output_dir = os.paht.join("output", "models")
    if embedding_separate:
        onnx_output_dir = onnx_output_dir + "_embedding_out"
        if mul_twice: onnx_output_dir = onnx_output_dir + "_mul2"
        if embedding_config.get("embedding_as_fp16", True): onnx_output_dir = onnx_output_dir + "_fp16"
    elif embedding_config.get("embedding_quant", False):
        onnx_output_dir = onnx_output_dir + "_qembed"

    if not export_config.get("output_pos", None):
        onnx_output_dir = onnx_output_dir + "_no_output_pos"

    lora_export = export_config.get("lora", dict()).get("enable", False)
    if lora_export: onnx_output_dir = onnx_output_dir + "_lora"

    os.makedirs(onnx_output_dir, exist_ok=True)
    print("saving to ", onnx_output_dir)
    print("model_arch ", export_config["model_arch"])
    model_cls = build_model(export_config["model_arch"])

    hf_model_device = "cpu"
    hf_model_dtype = torch.float32
    print(f"using {hf_model_device}, dtype {hf_model_dtype}")
    onnx_output_model_name = export_config["onnx_output_model_name"]
    onnx_opset = export_config["onnx_opset"]
    batch = export_config["batch"]
    kv_cache_max_len = export_config["kv_cache_max_len"]

    print(f"Begin load model from {hf_model_path}, "
          f"hf_model_device: {hf_model_device}, hf_model_dtype: {hf_model_dtype}.")
    model_wrapper = model_cls(
        model_path=hf_model_path, device=hf_model_device, dtype=hf_model_dtype,
        embedding_config=embedding_config,
    )
    print(f"Finish load model from {hf_model_path}.")
    print(model_wrapper)

    if quant_pth is not None:
        # ckpt = torch.load(quant_pth, map_location="cpu")
        # from dopt.do_opt import process_embedding_weights
        # process_embedding_weights(ckpt, mul_twice, onnx_output_dir, f"{onnx_output_model_name}_{seq_len}_{kv_cache_max_len}")
        # from dopt.do_opt import load_state_dict; load_state_dict(model_wrapper.model, ckpt)

        # if embedding_config.get("embedding_separate", True) and embedding_config.get("embedding_quant", False): model_wrapper.set_int_embedding(ckpt)
        model_wrapper.model.load_state_dict(torch.load(quant_pth, map_location="cpu"), strict=False)

    if lora_export:
        assert export_config["lora"]["lora_config"] is not None and os.path.exists(export_config["lora"]["lora_config"])
        from dopt.do_opt import optimize_model_add_lora_layers
        model_wrapper.model = optimize_model_add_lora_layers(model_wrapper.model, export_config["lora"]["lora_config"])

        if export_config["lora"].get("lora_ckpt", None):
            lora_ckpt = torch.load(export_config["lora"]["lora_ckpt"], map_location="cpu")
            if export_config["lora"].get("megatron", False):
                lora_ckpt = refresh_lora_ckpt(ckpt, lora_ckpt)
            if export_config["lora"].get("lora_scale", 1.) != 1.:
                lora_scale = export_config["lora"].get("lora_scale", 1.)
                print(f"using lora alpha {lora_scale} in lora_B")
                for k in lora_ckpt:
                    if "lora_B" in k:
                        lora_ckpt[k] *= lora_scale

            from dopt.do_opt import load_state_dict; load_state_dict(model_wrapper.model, lora_ckpt)
        else:
            print("warning!!!!!!!!!!!!!!!!!!!!!!!")
            print("warning!!!!!!!!!!!!!!!!!!!!!!!")
            print("warning!!!!!!!!!!!!!!!!!!!!!!!")
            print("no lora ckpt supplied")
    elif export_config.get("no_gemm", None):
        from dopt.do_opt import optimize_model_gemm2matmul; model_wrapper.model = optimize_model_gemm2matmul(model_wrapper.model)

    if export_config.get("lm_head"):
        lm_head_init = model_wrapper.model.lm_head.weight.shape[0]
        assert export_config["lm_head"] > lm_head_init
        weight = torch.zeros((export_config["lm_head"], model_wrapper.model.lm_head.weight.shape[1]))
        weight[:lm_head_init, :] = model_wrapper.model.lm_head.weight
        model_wrapper.model.lm_head.weight = torch.nn.Parameter(weight)

    print(f"Export pagu model arch to txt.")
    with open(onnx_output_dir + "/" + onnx_output_model_name + ".txt", "w") as f:
        f.write(str(model_wrapper.model))

    print(f"Export pagu model to onnx tmp.")
    output_dir_tmp = os.path.join(onnx_output_dir, "tmp")
    shutil.rmtree(output_dir_tmp, ignore_errors=True); os.mkdir(output_dir_tmp)
    onnx_file_name_tmp = os.path.join(output_dir_tmp, f"{onnx_output_model_name}.onnx")

    if export_config.get("layers", -1) > 0:
        model_wrapper.model.config.num_hidden_layers = export_config["layers"]
        model_wrapper.model.model.layers = model_wrapper.model.model.layers[:export_config["layers"]]
    config = model_wrapper.model.config
    print(f"Config:{config}")

    layer_num = model_wrapper.model.config.num_hidden_layers
    config.kv_cache_max_len = kv_cache_max_len

    layer_num = config.num_hidden_layers
    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = 128
    kv_head_num = config.num_key_value_heads

    input_ids_shape = [batch, seq_len]
    attention_mask_shape = [batch, 1, seq_len, kv_cache_max_len]
    position_ids_shape = [batch, seq_len]

    input_ids = torch.ones(input_ids_shape, dtype=torch.int64).to(hf_model_device)
    attention_mask = torch.randn(attention_mask_shape, dtype=hf_model_dtype).to(hf_model_device)
    position_ids = torch.ones(position_ids_shape, dtype=torch.int64).to(hf_model_device)

    in_names = ["input_ids", "attention_mask", "position_ids"]
    out_names = ["lm_logits"]

    kv_cache_in_shape = [kv_cache_max_len, kv_head_num, batch, head_dim]
    past_key_values = []
    print(f"Layer_num: {layer_num}")
    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=hf_model_dtype).to(hf_model_device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=hf_model_dtype).to(hf_model_device)
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])
        past_key_values.append((past_key_in, past_value_in))

    new_kv_cache_pos_shape = seq_len
    new_kv_cache_pos = torch.ones(new_kv_cache_pos_shape, dtype=torch.int64).to(hf_model_device)
    in_names.append("new_kv_cache_pos")

    if embedding_separate:
        input_ids = model_wrapper.model.model.embed_tokens(input_ids)
        if embedding_config.get("embedding_as_fp16", True):
            input_ids = input_ids.to(torch.float32)
        else:
            input_ids = input_ids.to(torch.int8)
    example_input = [input_ids, attention_mask, position_ids, past_key_values, new_kv_cache_pos]

    print(f"Net In Names: {in_names}")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"position_ids shape: {position_ids.shape}")
    for idx, past_key_value in enumerate(past_key_values):
        print(f"past_key_in{idx} shape: {past_key_value[0].shape}")
        print(f"past_value_in{idx} shape: {past_key_value[1].shape}")
    print(f"Net Out Names: {out_names}")

    embed_scales = None
    if embedding_separate:
        embed_scales = torch.randn([batch, seq_len, 1], dtype=hf_model_dtype).to(hf_model_device)
        in_names.append("embed_scales")
        in_names[0] = "input_embed"
    example_input.append(embed_scales)

    output_pos = None
    if export_config.get("output_pos", None):
        output_pos_shape = 1
        output_pos = torch.zeros(output_pos_shape, dtype=torch.int32).to(hf_model_device)
    in_names.append("output_pos"); example_input.append(output_pos)

    torch.onnx.export(
        model_wrapper,
        tuple(example_input),
        f = onnx_file_name_tmp,
        opset_version=onnx_opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
    )

    del model_wrapper
    gc.collect()
    print(f"Merge pagu model to onnx and sim.")

    from dopt.do_opt import process_onnx; process_onnx(onnx_file_name_tmp, lora_export=True)
    onnx_model = onnx.load(onnx_file_name_tmp); shutil.rmtree(output_dir_tmp, ignore_errors=True)

    onnx_file_name = os.path.join(onnx_output_dir, f"{onnx_output_model_name}.onnx")
    if os.path.exists(onnx_file_name): os.unlink(onnx_file_name)
    if os.path.exists(onnx_file_name[:-4] + "pb"): os.unlink(onnx_file_name[:-4] + "pb")
    onnx.save(onnx_model,
              onnx_file_name,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=os.path.basename(onnx_file_name)[:-4] + "pb",
              convert_attribute=False)

    print("generating finished")
    target_dir = os.path.dirname(onnx_file_name)
    os.system(f"md5sum {onnx_file_name} >> {onnx_output_dir}/logs.log")
    os.system(f"md5sum {onnx_file_name[:-5] + '.pb'} >> {onnx_output_dir}/logs.log")


if __name__ == "__main__":
    # info_path内填写model_info_target.yaml实际路径
    info_path = "model_info_target.yaml"
    if len(sys.argv) > 1: info_path = sys.argv[1]
    import yaml
    with open(info_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    if config.get("qlibs", None) is not None: import sys; sys.path.insert(0, config["qlibs"])
    for seq_len in config["seq_len"]:
        export_llama(
            export_config = config
        )

