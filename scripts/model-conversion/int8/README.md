# int8 量化链路（ONNX → dopt → OMG → .ms）

把任意 ONNX 模型量化成 **int8** 并尝试在 NPU 上运行。
这套脚本是**模型无关**的 ✓（不限于 Gemma 4）。

> ## ⚠️ 结论先说：**本设备上跑不通，原因在设备侧** ✗
>
> 设备固件的 hiai foundation 明确声明 **"not support extension config"** ✗，
> 而厂商的 int8 量化链路（dopt）**恰恰必须通过扩展配置把量化参数传给它** ✓
> ⇒ 编译 int8 模型必然失败（`Build -1`）✓。
>
> 因此这套脚本**目前不能产出可用的 int8 模型** —— 但格式与流程都已验证正确 ✓，
> **换设备 / 换固件版本后可直接复用** ✓。
> 完整证据链与排查过程见 `docs/maintainer-notes.md` 的「int8 探索记录」✓。

## 流程（四步）

```
① 造校准数据      make_calib_bin.py      ⇒ *.bin（★带 magic 头★）
② 写校准配置      make_cal_conf.py       ⇒ config.prototxt
③ 量化            run_dopt.sh            ⇒ 量化 ONNX + ★compress_conf★
④ 转设备模型      build_with_omg.sh      ⇒ .omc ⇒ .ms
   （运行时注入量化配置：见 probe_quant.py）
```

## 脚本

| 文件 | 作用 |
|---|---|
| `make_calib_bin.py` | 把校准数据写成 dopt 认的 `.bin`。★格式在文件头注释里★（magic 510/610 + shape + float32 裸数据） |
| `make_cal_conf.py` | 按 ONNX 输入生成 `config.prototxt`（★每输入一个块 · `input_type: BINARY` · 绝对路径★） |
| `run_dopt.sh` | 调用 dopt 做量化，产出**量化 ONNX** 与 **`compress_conf`** |
| `build_with_omg.sh` | 走 OMG（`--compress_conf`）⇒ `.omc` ⇒ `converter_lite` ⇒ `.ms` |
| `probe_quant.py` | 在设备上 Build/Predict，并把量化配置作为**扩展项**注入 |

## 环境变量

```
DOPT_PY     python3.10 解释器（★dopt 的 .so 只认 3.10★，否则报 _PyUnicode_Ready 未定义）
DOPT_DIR    dopt_onnx_py3 目录
MODEL       输入 ONNX          CAL_CONF   config.prototxt
OUT         量化后 ONNX        CONF       输出的 compress_conf
OMG / CONVERTER / OMGTXT / CFG / OUT      见 build_with_omg.sh
MSLITE_LIB  MindSpore Lite NDK 库（缺省 /system/lib64/ndk/libmindspore_lite_ndk.so）
```

## ★三个最容易踩的坑（都是实测出来的）★

```
① 校准 bin 必须有 magic 头 ✗ 裸数据会被拒 ⇒ 报 "Invalid bin file!"
② prototxt 里 ★每个输入一个 preprocess_parameter 块★（不能重复同一个键 ✗）
   · input_type 必须显式写 BINARY
   · input_file_path 必须绝对路径
③ dopt 的环境：python 必须 3.10；protobuf 太新要设
   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## ★排查建议：先读 hilog★

MindSpore-Lite 的 `MS_LOG` **不走 stderr** ✗，而是写 **hilog** ✓。
任何 NNRt 侧的 `Build` 失败，**先做这一步**就能看到确切原因：

```sh
hilog -x | grep -aiE "MS_LITE|NNRt|CANN|AI_FMK|hiai"
```

本次正是靠它一步看到 `hiai foundation not support extension config` ✗。

## 另一条（更简单但只对 CPU 有效）的路线

`converter_lite` **自带量化器** ✓，可直接吃 ONNX：

```sh
converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=out --configFile=quant.cfg
```

* 配置模板见源码树 `tools/converter/quantizer/config/`（`dynamic_quant.cfg` 无需校准集 ✓）
* ★第三方模型要把 `[third_party_model]` 与 `[common_quant_param]` 合到**同一个** configFile★ ✓
* 产出的 `.ms` **能在 CPU 后端跑** ✓，但 **NNRt 下 `Build -1`** ✗
