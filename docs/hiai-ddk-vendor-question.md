# 给厂商/论坛的提问材料：DDK `ModelManager::Init` 对多层图返回 1

## 一句话问题

用 DDK（HiAI Foundation V2 C++ 接口）加载我们自己用 **OMG `--target=omc`** 编出来的模型：
`CreateBuiltModel` + `RestoreFromFile` **成功**（`rc=0`，`CheckCompatibility` 报"兼容"），
但 `ModelManager::Init` 在 **≥2 层**的图上返回 **1**，**单层**图返回 **0**。
`Init` 失败时 DDK 不输出任何原因（hilog 读不到）。想知道：**`Init` 对模型的哪些约束没满足？**

## 环境

```
设备：Kirin NPU（NPU_ohos.boot.hardware.KirinX90_v2_0）
DDK：tools_omg（--target=omc）· DDK_Demo/V2（libhiai.so aarch64）
SDK 头：HiAIDemo/DDK_Demo/V2/include（model_builder_types.h / model_manager_types.h / nd_tensor_desc.h）
```

## 复现步骤（最小）

```
① 用旧 TorchScript 导出器（opset 14）导出一个**纯 3 维张量**的 ONNX：
     input_embed:1,64,2048 (FP32) · attention_mask:1,1,64,2048 · position_ids:1,64 (INT32)
     + 每层状态：conv 状态 (1,6144,3) / 递归状态 (1,16,128,128)（都是 FP32）
   输出：hidden_states:1,64,2048  + 各层新状态
② OMG：
     omg --model x.onnx --framework 5 --output out/seg \
         --input_shape="<逐输入 name:d0,d1,...>" --input_type="<逐输入 name:FP32|INT32>" \
         --output_type="hidden_states:FP32" --weight_data_type FP16 \
         --save_weights_as_external_data=true --platform=kirinx90 --target=omc
   ⇒ 成功，产出 seg.omc + SubGraph_0.weight
③ DDK 侧：
     built = CreateBuiltModel();
     built->RestoreFromFile("seg.omc", nullptr);          // rc = 0 ✓
     built->CheckCompatibility(...);                      // 报"兼容" ✓
     man = CreateModelManager(...);
     ModelInitOptions opt;
     opt.buildOptions.formatMode = FormatMode::USE_ORIGIN;
     opt.buildOptions.precisionMode = PRECISION_MODE_FP16;
     man->Init(opt, built, nullptr);                      // ★ 单层=0 ✓ ；两层=1 ✗ ★
```

## 我们实测的对照表

| 模型 | 输入/输出 | 图内是否有 KV/状态输入 | Init |
|---|---|---|---|
| 厂商发布的 Qwen3-8B（`qwen3_8b_ceval_g256.omc`） | 17 / 5 | 无（KV 由引擎管） | **0 ✓** |
| 我们 · **单层**（完整 delta-rule 层） | 3 / 1 | 有（2 个状态） | **0 ✓** |
| 我们 · **2 层** | 8 / 8 | 有（4 个状态） | **1 ✗** |
| 我们 · 4 层 / 24 层 | 19/15 · 89/85 | 有 | **1 ✗** |
| 两个普通 MatMul **串联**（对照实验） | 1 / 1 | 无 | **0 ✓** |
| 两个 delta-rule 层**并联**（对照实验） | 8 / 10 | 有 | **0 ✓** |
| 两个 delta-rule 层**串联**（对照实验） | 8 / 8 | 有 | **1 ✗** |

⇒ 现象收敛为：**「把一层的输出喂给下一层」+ 我们这种层结构** ⇒ `Init` 失败；
但纯 MatMul 串联没问题，单层也没问题。

## 我们已经试过并排除的（都是实测）

```
· --hiai_version 的 IR / v310 / v300 三个值（都编不过）
· formatMode = USE_NCHW（默认）与 USE_ORIGIN（都失败）
· 显式填 inputTensorDescs（Format::ND ✓ + 正确 dims/dtype ✓）—— 仍失败
· 强制 modelDeviceConfig.modelDeviceOrder = { ExecuteDevice::CPU } —— 仍失败
· 权重文件：SubGraph_*.weight 全部拷齐（仍失败）；weight_merge true/false（都试过）
· 图合法性：ONNXRuntime 现在能正常加载并执行该 ONNX ✓（此前有 int64/float32 类型问题，已修）
· 子图数：OMG 产物的子图代理计数已从 13 降到 1（与官方相同）—— 仍失败
· input/output 张量个数：从 3/1 到 89/85 都试过 —— 只有单层能过
```

## 希望得到的信息

1. `ModelManager::Init` 返回 1 的**具体错误码含义**与**日志打开方式**（hilog 里应该记了什么？）；
2. `omc` 被 `Init` 接受需要满足的**模型约束清单**（例如：是否要求图内不含中间状态 IO？
   是否要求每层的 op 组合落在某个白名单？是否与 `SubGraph` 数量/边界有关？）；
3. 有没有**校验工具**能在主机侧检查 omc 是否满足 `Init` 的前置条件；
4. 官方那批 omc（如 `qwen3_8b_ceval_g256.omc`）是用**哪条链路**编出来的
   （OMG 版本 + 完整参数）—— 我们想对齐。
