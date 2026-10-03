# Gemma 4 导出与验证脚本

把 HuggingFace 的 **gemma-4-E2B-it** 切成能交给
`OMG → .omc → converter_lite → .ms → 设备 NNRt` 的若干张图，并附带对拍/验证工具。

> **导出脚本不是运行所需** ✓ —— 运行时只要 `scripts/start_chat.sh -d <模型目录>`，
> 目录里放现成的 `.ms` 即可 ✓。这里留档是为了复现与参考 ✓。

## 导出器（产出可上 NPU 的图）

| 文件 | 作用 |
|---|---|
| `gemma4_model.py` | 公共部分：加载 HF 模型、把注意力折成 3 维、常量化的 rotate 矩阵、KV 共享 |
| `gemma4_export_seg.py` | **段图（prefill）**：`START`/`NO` 指定层区间，输出 `seg.onnx`（含 KV 共享槽） |
| `gemma4_export_graphp.py` | **图 P**：per-layer 前处理（[8960,1536] 投影，纯 Python 太慢 ✗ 必须进图） |
| `gemma4_export_lm.py` | **lm_head 分块**：`J=0..3`，每块 65536 列（整表 262144 会顶到设备上限 ✗） |
| `gemma4_export_decode.py` | **decode 图**（seq=1 + KV）：`MODE`/`KVMAX`/`KVND` 控制形态 |
| `gemma4_batch.sh` | 9 段整条流水线：ONNX → OMG → converter_lite |

## 验证与对拍工具（主机侧 / 设备侧）

| 文件 | 作用 |
|---|---|
| `gemma4_chain_check.py` | 把段**串联**起来，在每个边界与 HF 对应层输出对比 |
| `gemma4_decode_ref.py` | KV 缓存 decode 的参考实现 + 验证（主机侧） |
| `gemma4_device_cmp.py` | 设备端对拍：用真实输入跑段图，与 HF 参考输出比较 |
| `gemma4_dump_io.py` | 导出**真实输入**与 HF 参考输出，供设备端对拍 |
| `gemma4_dump_chain_io.py` | 为设备端多段串联准备输入（hidden / per_layer 切片 / cos-sin / mask） |
| `gemma4_seg_run_device.py` | 跑**单个**段：读 hidden(+KV) → Predict → 写 hidden_out(+KV) |
| `gemma4_final_cmp.py` | 设备输出 → 主机侧 norm + lm_head + softcap → 与 HF 的 top-5 对比 |
| `gemma4_logits_check.py` | 全链 → logits → 与 HF 做 top-k 对拍（验收标准） |

## 接口（全部环境变量，无硬编码路径 ✓）

```
MODEL_DIR   HF 模型目录（gemma-4-E2B-it）★必填★
OUTDIR      输出目录（缺省 ./out/gemma4）
DTYPE       fp32（缺省）/ fp16 —— 见下面 fp16 说明
SEQ         序列长度（段图常用 128；图 P / lm 用 1）
START / NO  段图：起始层与层数（如 START=16 NO=8）
J           lm 分块序号 0..3
MODE / KVMAX / KVND   decode 图形态
DDK         厂商 DDK 根目录（OMG / converter_lite 所在）
```

## 产出怎么变成 `.ms`

```
python gemma4_export_seg.py          # ① 导出 ONNX（同时生成 omg.txt / c.cfg）
omg --model seg.onnx --framework 5 --output q/seg \
    --input_shape=<omg.txt 第1行> --input_type=<第2行> --output_type=<第3行> \
    --weight_data_type FP16 --platform=kirinx90 --target=omc          # ② OMG
converter_lite --fmk=THIRDPARTY --modelFile=q/seg.omc \
    --outputFile=seg --configFile=c.cfg                                # ③ 转 .ms
```

★ `omg.txt` / `c.cfg` 里的 dtype 一律**从 ONNX 实读** ✓ ——
写死成 FP32 会让 fp16 的图在 OMG 报 `InputOutputGraphComplete fail` ✗。

## ★fp16 的结论（实测）★

`DTYPE=fp16` 会同时把**权重**和**I/O** 转成 fp16 ✓。设备对 fp16 是**逐图**的 ✓：

| 图 | 结果 |
|---|---|
| 图 P（embedding + Linear + Mul 那类） | ★**能跑** ✓★（Build 0 · Predict 0） |
| lm（[1,1,1536]×[1536,65536]） | ✗ core dump |
| 段图（注意力那类） | ✗ Build 返回非 0 |

另有一档「**权重 fp16 + I/O 保持 fp32**」能跑 ✓，但**推理更慢** ✗ ⇒ 不适合作为加速手段。
⇒ 详细数据与排查过程见 `docs/maintainer-notes.md` ✓。

## 依赖

```
torch / transformers（能加载 gemma-4-E2B-it 的版本）· onnx · onnxruntime
OMG 与 converter_lite 来自设备厂商 DDK（不随本仓库分发 ✗）
```
