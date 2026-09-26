# 模型转换脚本

配套文档：**[../../docs/model-conversion.md](../../docs/model-conversion.md)** ——
完整流程与每一步的说明都在那里，这里只列脚本。

| 脚本 | 作用 | 用在哪一步 |
|---|---|---|
| `rebuild_weights.py` | 从 HF 检查点重建 ONNX 权重 | **★ 必做** —— dopt 量化会把权重负半轴钳成 0，不修就是垃圾输出 |
| `confirm_relu.py` | 诊断：确认权重是否被量化破坏（比对 `corr(ONNX, ReLU(HF))`） | 怀疑中招时 |
| `check_weights2.py` | 逐权重比对 ONNX 与 HF | 同上 |
| `split_downproj_fixed.py` | 沿 K 切分过大的 MatMul（OMG 不支持 K=8960） | 转换前 |
| `omg_convert.py` | 生成并执行 OMG 转换命令（按层数/KV/隐藏维自动拼参数） | 转换 |
| `patch_ort.py` | 给 ScatterND 的 indices 插 `Cast(int32→int64)` | 用 ORT 验证前 |
| `ort_check.py` | 在 ONNXRuntime 里跑导出图，判定模型本身对不对 | 验证 |

## 典型顺序

```bash
cd scripts/model-conversion

# 0) 先确认权重是不是坏的
python3 confirm_relu.py /path/to/qwen2_1p5b_w4.onnx

# 1) 修复权重（从 HF 重建）
python3 rebuild_weights.py in.onnx repaired.onnx /path/to/Qwen2.5-1.5B-Instruct

# 2) 切分大矩阵
python3 split_downproj_fixed.py repaired.onnx repaired_split.onnx 28 2048

# 3) 用 ORT 验证模型本身是对的（可选但强烈建议）
python3 patch_ort.py repaired_split.onnx repaired_ort.onnx
ONNX_MODEL=repaired_ort.onnx EMB_WEIGHTS=... EMB_SCALES=... python3 ort_check.py

# 4) OMG 转换（先 --dry-run 看一眼）
python3 omg_convert.py --onnx repaired_split.onnx --out ./om_out \
    --layers 28 --kv-len 2048 --hidden 1536 --weight-data-type FP16 --dry-run
python3 omg_convert.py --onnx repaired_split.onnx --out ./om_out \
    --layers 28 --kv-len 2048 --hidden 1536 --weight-data-type FP16
```

## 依赖

`onnx` / `onnxruntime` / `numpy` / `safetensors`（`pip install onnx onnxruntime numpy safetensors`）。
这些脚本**不属于运行时依赖** —— `cann-llm` 本身仍是零依赖，跑模型不需要它们。

## 路径

脚本里没有写死的机器路径。需要外部路径的地方都用参数或环境变量：

| 环境变量 | 用途 |
|---|---|
| `HF_MODEL` | HuggingFace 检查点目录 |
| `ONNX_MODEL` | 待检查的 ONNX 路径 |
| `EMB_WEIGHTS` / `EMB_SCALES` | 导出时产生的 embedding 文件 |
| `OMG_DIR` / `ASC_DIR` | DDK 的 `tools_omg` / `tools_ascendc` 目录 |
| `NLAYERS` / `K_CHUNK` | 层数 / 切分块大小（也可用位置参数） |

## 相关

- `confirm_relu.py` / `check_quant_clamp.py` —— 判断量化产物有没有被钳位
  （`quant_param_2` 写错的症状），见 [../docs/model-conversion.md](../docs/model-conversion.md) 附录 B
