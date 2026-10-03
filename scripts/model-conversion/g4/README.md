# Gemma 4 分段导出（NNRt 离线路径）

这套脚本把 HuggingFace 的 **gemma-4-E2B-it** 切成能交给
`OMG → .omc → converter_lite → .ms → 设备 NNRt` 的若干张图。

> 它**不是运行所需**的脚本 ✓ —— 只是把当初实际用的导出流程留档，供复现/参考 ✓。
> 运行时只需 `scripts/start_chat.sh -d <模型目录>`（模型目录里放现成的 `.ms`）✓。

## 脚本

| 文件 | 作用 |
|---|---|
| `g4_seg3d.py` | 公共部分：加载 HF 模型、把注意力折成 3 维、常量化的 rotate 矩阵等 |
| `g4_export.py` | **段图**：`START`/`NO` 指定层区间，输出 `seg.onnx`（含 KV 共享槽） |
| `g4_exp_p.py`  | **图 P**：per-layer 前处理（[8960,1536] 投影，纯 Python 太慢 ✗ 必须进图） |
| `g4_exp_lm.py` | **lm_head 分块**：`J=0..3`，每块输出 65536 列（整表 262144 会顶到设备上限 ✗） |
| `g4_exp_dec.py`| **KV 解码**图（`MODE`/`KVMAX`/`KVND` 控制形态） |

## 接口（全部用环境变量指定，无硬编码路径 ✓）

```
MODEL_DIR   HF 模型目录（gemma-4-E2B-it）★必填★
OUTDIR      输出目录（缺省 ./out/g4seg）
DTYPE       fp32（缺省）/ fp16 —— ★fp16 见下面说明★
SEQ         序列长度（段图常用 128，图 P / lm 用 1）
START / NO  段图：起始层与层数（如 START=16 NO=8）
J           lm 分块序号 0..3
```

## 产出怎么变成 `.ms`

```
python g4_export.py            # ① 导出 ONNX（同时生成 omg.txt / c.cfg）
omg --model seg.onnx --framework 5 --output q/seg \
    --input_shape=<omg.txt 第1行> --input_type=<第2行> --output_type=<第3行> \
    --weight_data_type FP16 --platform=kirinx90 --target=omc      # ② OMG
converter_lite --fmk=THIRDPARTY --modelFile=q/seg.omc \
    --outputFile=seg --configFile=c.cfg                            # ③ 转 .ms
```

★ `omg.txt` / `c.cfg` 里的 dtype 一律**从 ONNX 实读** ✓ ——
写死成 FP32 会让 fp16 的图在 OMG 报 `InputOutputGraphComplete fail` ✗。

## ★关于 fp16（结论）★

* `DTYPE=fp16` 会同时把**权重**和**I/O** 转成 fp16 ✓
* 设备对 fp16 是**逐图**的，不是一律不可用 ✓：
  * 图 P（embedding + Linear + Mul 那类）⇒ **能跑** ✓
  * lm（[1,1,1536]×[1536,65536]）⇒ **core dump** ✗
  * 段图（注意力那类）⇒ `Build` 返回非 0 ✗
* 另有一档"**权重 fp16 + I/O 保持 fp32**"能跑 ✓，但**推理更慢** ✗ ⇒ 不适合作为加速手段
* ⇒ 详细证据与排查过程见 `docs/maintainer-notes.md` ✓

## 依赖

```
torch / transformers（能加载 gemma-4-E2B-it 的版本）· onnx · onnxruntime
OMG 与 converter_lite 来自设备厂商 DDK（不随本仓库分发 ✗）
```
