# Qwen3.8（qwen3_5）· DDK hiai C++ 直跑路线 · 现状与交接

> 原始实验记录见 `docs/maintainer-notes.md` **§60~§113**（每一步都有实测数据 ✓）。
> 本文是**结论与操作手册** ✓，不含过程。

## 1. 路线与结论

**路线**：不依赖 hiai LLM 引擎，直接用 DDK 的 hiai C++ API（`CreateBuiltModel` +
`RestoreFromFile` + `CreateModelManager` + `ModelManager::Init`）加载我们自己编的 `.omc` 并在设备上跑。

**为什么不用引擎**：hiai LLM 引擎要求「动态 seq（`--dynamic_dims` 两档 S=1/S=64）+
图内不含 KV（KV 由引擎自带 `executor_attention_op` 管理 ⇒ 官方图 17 进 / 5 出）」，
与 qwen3_5 的「**静态 S**（delta rule 分块算法要求）+ **混合架构**（18 层线性注意力没有 KV）」
不匹配（§84/§86 ✓）。

**当前结论**（§113 ✓）：

| 对象 | 结果 |
|---|---|
| 官方 Qwen3-8B（36 层 · 17 进/5 出 · 图内无 KV） | `Init rc=0` ✓ |
| **我们的单层**（完整 delta-rule 层 · 3 进/1 出 · 有状态） | `Init rc=0` ✓ |
| **我们的 ≥2 层**（8~89 进/8~85 出 · 有状态） | **`Init rc=1` ✗** |

⇒ **复现面极窄**：单层能过、≥2 层不过；DDK 不吐失败原因（细节在 hilog，读不到）。

**已排除的假设**（全部有实测，详见 §110~§113）：串联本身 ✗ · 2 维张量 ✗ · RealDiv ✗ ·
`--hiai_version`(IR/v310/v300) ✗ · 权重文件缺失 ✗ · 层数/状态数/权重内容 ✗ ·
`formatMode=USE_NCHW` ✗ · 子图数（整模型 13→1 后仍失败）✗ · ONNX 合法性（非法→合法后仍失败）✗ ·
强制 `ExecuteDevice::CPU` ✗ · 显式 `inputTensorDescs` + `Format::ND` ✗。

## 2. 完整链路（可复现 ✓）

```bash
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl

# ① 导出（旧 TorchScript 导出器 + 3 维化改写 ✓）
python export_hiai_q35.py --hf <hf模型目录> --seq 64 --kv-len 2048 \
       --no-embed-head --legacy --out q35_24f.onnx
# ② 图级 lower（Softplus / ConstantOfShape ✓ + ★dtype 保留★ + 类型守卫 ✓）
python lower_hiai.py q35_24f.onnx q35_24f_low.onnx
# ③ OMG（★必须带 --save_weights_as_external_data=true★ ⇒ 官方形态 omc + SubGraph_0.weight）
omg --model q35_24f_low.onnx --framework 5 --output omg/seg \
    --input_shape="<从 onnx 生成，逐输入 name:d0,d1,...>" \
    --input_type="<同上，INT32 的输入写 INT32，其余 FP32>" \
    --output_type="hidden_states:FP32" --weight_data_type FP16 \
    --save_weights_as_external_data=true --platform=kirinx90 --target=omc
# ④ 组装模型包（int8 embedding + 每行 scale + tokenizer + 4 个 json ✓）
python build_hiai_pkg.py --hf <hf模型目录> --name qwen38_2b --seq 64 --kv 2048 \
       --omc omg/seg/seg.omc --out pkg/
# ⑤ DDK 侧验证（判据是 Init rc ✓；★在包目录内用相对路径★，否则找不到 SubGraph_0.weight ✗）
hiai_runner <omc>          # 见 scripts/model-conversion/qwen38/hiai_runner/
```

一键脚本：`scripts/model-conversion/qwen38/run_hiai_ddk_chain.sh`（①~④ ✓）。

## 3. 本次修掉的真 bug（★最有价值的产出✓★）

`onnx_lower.py` 的 `ConstantOfShape → Expand` 降级**把原本 int64 的常量强转成 float32**：

```python
# 旧（✗）：val = float(...); np.array([val], dtype=np.float32)
# 新（✓）：val_arr = numpy_helper.to_array(attr.t); from_array(val_arr.astype(val_arr.dtype))
```

后果与收益：
- 旧写法让图变成**非法 ONNX**（`Concat(int64, float32)` 类型冲突）⇒ ONNXRuntime **直接拒收** ✗；
- 且 `int64(8B) → float32(4B)` 尺寸变化 ⇒ **正是 DDK 报 `param[size] is less than[dataSize]` 的来源** ✓；
- 修后：图合法 ✓（ORT 能加载并执行 ✓）· 整模型 OMG 的**子图代理数 13 → 1** ✓。

配套：`onnx_lower.py::fix_mixed_dtypes`（自做类型传播 + 修混合类型算子 + **清空全部
`value_info`**，避免任何陈旧类型声明 ✗）。

## 4. 工具

| 工具 | 用途 |
|---|---|
| `export_hiai_q35.py` | 导出 hiai 接口图（3 维化 ✓ prefill/decode 同一套 ✓ `--layers` 可切片 ✓） |
| `lower_hiai.py` / `onnx_lower.py` | 图级 lower + 类型守卫 ✓（单外置权重文件 ✓） |
| `check_hiai_parity.py` | ONNX vs HF 参考对拍（★配置必须匹配导出：4 层 / KV=256★ ✓） |
| `hiai_runner/hiai_runner.cpp` | DDK 侧验证器：Load ✓ / Init rc ✓；开关 `INIT_NCHW` · `INIT_CPU` · `HW_INPUTS_FILE` |
| `probe_*.py` | 逐算子 / 逐外形 / 分级链探针（最小复现的载体 ✓） |

## 5. 建议的下一步

1. **拿本文第 1 节的复现面去问厂商**（附我们的 omc + 输入规格）—— 最快拿到确定性答案 ✓；
2. 换执行路径（CPU/GGUF 已跑通 ✓；或 NNRt）；
3. 若继续在 DDK 上深挖：可从「单层能过 / 两层不过」的**唯一差异**入手 —— 用
   `probe_2l_variants.py` 的 `indep`（并联 ✓ 过）与 `dep`（串联 ✗ 不过）做差分 ✓。
