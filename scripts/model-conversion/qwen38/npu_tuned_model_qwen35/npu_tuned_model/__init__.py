def build_model(model_arch):
    if model_arch == "qwen2":
        from .qwen2.export_model_wrapper import Qwen2ForCausalLMWrapper
        return Qwen2ForCausalLMWrapper

    if model_arch == "qwen3":
        from .qwen3.export_model_wrapper import Qwen3ForCausalLMWrapper
        return Qwen3ForCausalLMWrapper

    if model_arch == "zhipu":
        from .glm.export_model_wrapper import GLMForCausalLMWrapper
        return GLMForCausalLMWrapper
        
    raise Exception(f"not support {model_arch}")