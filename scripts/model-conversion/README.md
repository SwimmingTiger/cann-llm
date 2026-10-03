# 模型转换脚本

把 HuggingFace 的模型转成能上设备 NPU 的 `.ms`（MindSpore Lite 离线模型）。

## 目录

| 目录/文件 | 内容 |
|---|---|
| **`gemma4/`** | ★Gemma 4 的全部脚本★：5 个图的导出器 + `gemma4_batch.sh` + 对拍/验证工具 ⇒ 见 `gemma4/README.md` |
| **`int8/`** | int8 量化链路（ONNX → dopt → OMG → .ms）★模型无关★；★本设备跑不通，原因在设备侧✗★，详见其 README 与 `docs/maintainer-notes.md` |
| **`qwen/`** | Qwen 系列（`build_model.py` 一条命令从 HF 检查点产出模型目录；`patch_qwen3_*.py` 为官方示例打补丁） |
| `omg_convert.py` | 通用：生成并执行华为 OMG 的转换命令（ONNX → `.omc`） |
| `onnx_weights_to_fp16.py` | 通用：权重转 FP16（就地插入 Cast，保持拓扑顺序） |
| `normalize_tokenizer_merges.py` | 把新版 HF `tokenizer.json` 的 merges 规范成 NPU 引擎认的旧格式 |
| `check_weights2.py` · `rebuild_weights.py` | 校验 / 重建导出 ONNX 的权重 |
| `ort_check.py` · `patch_ort.py` | 用 ONNXRuntime 做数值检查（含给其加载器打补丁） |
| `check_quant_clamp.py` · `set_quant_strategy.py` · `split_downproj_fixed.py` · `confirm_relu.py` | 量化相关辅助 |

## 通用流程

```
HF 检查点 ──导出──▶ ONNX ──OMG──▶ .omc ──converter_lite──▶ .ms ──▶ 设备（NNRt 后端）
```

各模型的导出入口见对应子目录的 README ✓；**运行**只需 `scripts/start_chat.sh -d <模型目录>` ✓。

> 历史上 Gemma 4 有两代导出器；**旧的一代已被 `gemma4/` 取代并删除** ✓
> （其构建流程与参数已并入 `gemma4/README.md` ✓）。
