# qwen3_5（Qwen3.8）的 NPU 友好改写

目标：让 **Qwen3.8（`qwen3_5` 架构）** 能上设备 NPU。
官方转换链只支持 `qwen2/qwen3/glm` ✗（见 `docs/model-conversion.md` 附录 D ✓），
所以这里自己动手：**把线性注意力改写成"只用 DDK 支持算子"的版本** ✓。

## 背景（为什么要改写）

`qwen3_5` 的线性注意力（Gated DeltaNet）导出的 ONNX 需要 4 个 **DDK 平台库没有**的算子 ✗
（实测见 `docs/maintainer-notes.md` §49）：

```
✗ CumSum · Trilu · ScatterElements · ScatterND      （Conv1D 也只有 Conv2D ✓）
```

## 改写要点（`npu_gated_delta.py`）

| 原写法 | 缺失算子 | 替换 |
|---|---|---|
| `decay.cumsum(dim=3)` | CumSum | ★常量 `triu(ones)` 矩阵乘法★（注意：`tril(ones)` 算出来是**后缀和** ✗，踩过 ✓）|
| `torch.ones(C,C).triu(1)` / `.tril(-1)` | Trilu | ★常量布尔掩码 + `Where`★ |
| 循环里 `out[:, :, i] = …`（切片赋值） | Scatter* | ★列表收集 + `torch.cat`★ |
| `freqs_thw[..., idx] = freq[dim, ..., idx]`（M-RoPE 分节） | Scatter* | ★常量掩码 + `Where`★（`npu_recomposition_frequencies`）|
| `F.conv1d`（depthwise 因果卷积） | Conv1D | ★移位切片 + Mul/Add★ |
| UT 变换求 `(I−L)⁻¹` | — | ★分块前代★（小块 16 用平方-乘积，块间 matmul ✓）|

★数值教训★：整体用"平方-乘积恒等式" `(I−L)⁻¹ = Π(I+L^{2^k})` 数学上精确 ✓，
但会把条件数**平方** ✗ —— 真实模型里 `beta≈0.995` 的层误差冲到 **6.6e+16** ✗；
改成分块前代后降到 **2.4e-06** ✓（§51）。

另外：`IsNaN`（注意力里的 NaN 保护）用**图级 lowering** 换成 `Not(Equal(x,x))` ✓（`onnx_lower.py`）。
`CumSum/GatherND`（从 attention_mask 反推 position_ids 的路径 ✗）通过**显式传全 1 mask** 绕开 ✓。

## 脚本

| 文件 | 作用 |
|---|---|
| `npu_gated_delta.py` | ★核心★：线性注意力 + 因果卷积 + M-RoPE 的 NPU 友好实现，`install()` 一行打补丁 ✓ |
| `parity_check.py` | 函数级对拍（多种形状 ✓） |
| `debug_delta.py` | 逐步对拍中间量（定位改写偏差 ✓） |
| `model_parity.py` | ★整模型对拍★（argmax 一致率 ✓） |
| `tiny_parity.py` | 微型模型对拍 + 补丁命中统计 ✓ |
| `export_npu.py` | 带补丁导出 ONNX + 与 DDK 算子集比对（目标：**零缺失** ✓） |
| `onnx_lower.py` | 图级 lowering（`IsNaN → Not(Equal)` ✓） |
| `make_quant_cfg.py` | 从 ONNX 生成 converter_lite 的 int8 量化配置 ✓ |

## 实测结果

```
① 函数级对拍：cum_decay/pairwise/ut/inv/new_values 全部 ≤5e-07 ✓（debug_delta.py）
② 整模型对拍：logits 最大相对差 2.9e-05 · ★argmax 一致率 1.0000★ ✓（model_parity.py）
③ 算子清点：Trilu/Scatter*/CumSum/GatherND/IsNaN 全部消掉 ⇒ ★零缺失算子★ ✓（export_npu.py）
④ 真实宽度 2 层切片：ONNX 2.51 GB（int32 输入 ✓）⇒ 容器内 converter_lite（int8 W8）转 .ms ✓
```

## 跑法（hu60tx）

```bash
~/q38env/bin/python model_parity.py                     # 整模型对拍 ✓
~/q38env/bin/python export_npu.py --tiny --attn-mask    # 秒级：看算子集 ✓
~/q38env/bin/python export_npu.py --full --layers 2 --attn-mask --out q35_L2.onnx
~/q38env/bin/python make_quant_cfg.py q35_L2.onnx wq_q35.cfg
docker exec mslite-dev bash -c 'B=/mslite/mindspore-lite-2.7.0-linux-x64; \
  export LD_LIBRARY_PATH=$B/tools/converter/lib:$B/runtime/lib:$B/runtime/third_party/glog; \
  cd /q38 && $B/tools/converter/converter/converter_lite --fmk=ONNX \
  --modelFile=q35_L2.onnx --outputFile=q35_L2 --configFile=wq_q35.cfg'
```
