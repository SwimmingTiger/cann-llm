# mslite-nnrt —— 用 NNRt 后端跑 MindSpore Lite 的离线模型（`.ms`）

一个**单文件 C 程序**：在鸿蒙设备上加载 `.ms` 并做一次推理，
默认走 **NNRt 后端**，用于验证「厂商离线模型 → `.ms` → NPU」这条链路。

完整背景与转换步骤见 **[docs/offline-model-nnrt.md](../../docs/offline-model-nnrt.md)**。

## 编译与运行

设备上（自带 `libmindspore_lite_ndk.so` 与 SDK 头文件）：

```sh
./build.sh                      # 编译到 ./bin（可用 SDK= 指定头文件目录）
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/mslite_run <model.ms> nnrt
LD_LIBRARY_PATH=/system/lib64/ndk ./bin/mslite_run <model.ms> cpu    # 对照
```

## 两个后端的结果不一样，这是对的

| 后端 | 输出 |
|---|---|
| `cpu` | 通常是 `0.000 …` —— 离线模型被包成 **custom 算子**，CPU 侧没有它的实现，输出未初始化 |
| `nnrt` | **正确数值** —— 真正算它的是 NPU 侧的厂商实现 |

这正是官方文档说的「离线模型仅支持在 NNRt 后端推理」。
所以：**在 `nnrt` 下拿到正确数值，就说明整条链通了。**

## 关键 API 速查

```c
OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
OH_AI_ContextAddDeviceInfo(ctx, OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT)); /* = 60 */

OH_AI_ModelHandle m = OH_AI_ModelCreate();
OH_AI_ModelBuildFromFile(m, path, OH_AI_MODELTYPE_MINDIR, ctx);   /* .ms 就是 MindIR */

OH_AI_TensorHandleArray ins  = OH_AI_ModelGetInputs(m);
OH_AI_TensorHandleArray outs = OH_AI_ModelGetOutputs(m);
/* 往 ins 的数据里填输入 → */
OH_AI_ModelPredict(m, ins, &outs, NULL, NULL);
/* ← 从 outs 的数据里读结果 */
```

## 两个程序

| 文件 | 用途 |
|---|---|
| `mslite_run.c` | 通用：加载任意 `.ms`，打印输入输出，把所有输入填 1.0 后跑一次。适合看"能不能加载/边界长什么样" |
| `mlp_run.c` | **带权重模型的完整验证**：把输入与 CPU 参考值写死，跑完自动逐元素比对（我们用它验证 `Gelu(x@W1+b1)@W2`，最大偏差 0.00025） |

## 已知问题

程序在**退出阶段**可能 `core dumped`（`exit code 139`）——
推理结果已经正确，崩在析构/动态库卸载阶段，属独立问题。
