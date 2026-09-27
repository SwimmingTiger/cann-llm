# npu-probe —— 绕过 python，用原生 ELF 直接驱动 NPU

回答一个问题：**原生 ELF 有没有权限打开 NPU、能不能跑完一次推理？**
结论：**有，能。**（不需要 python、不需要 ctypes）

两个版本，分别对应 `cann-llm` 的两个后端：

| 程序 | 后端 | 引擎库 | API 风格 |
|---|---|---|---|
| `npu_infer_hiai.c` | hiai | `/system/lib64/libhiai_llm_engine.so` | 逐个 `SetOption` 组装配置（`HIAI_LLMEngine_*`）|
| `npu_infer_cann.c` | cann | `/system/lib64/ndk/libcann_llm_engine.so` | **吃 JSON**（`HMS_LLMEngine*`）|

## 编译与运行

```bash
# ★ 别编到 /tmp —— 本项目设备上 /tmp 是只读的（会报 Read-only file system）
cc -O1 npu_infer_hiai.c -o npu_infer_hiai -ldl
cc -O1 npu_infer_cann.c -o npu_infer_cann -ldl

LD_LIBRARY_PATH=/system/lib64/ndk ./npu_infer_hiai /path/to/model_dir
LD_LIBRARY_PATH=/system/lib64/ndk ./npu_infer_cann /path/to/model_dir
```

实测（Qwen2.5-1.5B，本机 Kirin X90）：

```
npu_infer_hiai : 你好！你好！很高兴能为你服务。有什么我可以帮助你吗？

npu_infer_cann : [1] Executor_CreateFromExecutorJson -> 0x64ea27b300  (2600 ms)
                 [2] Context_CreateFromContextJson -> 0x6636afda00
                 [3] Generate rc=0  (1211 ms)
                 [4] gen_len=69 · in=9 tok · out=14 tok
                     prefill 390 ms · decode 3245 ms · total 3636 ms
                 你好！我是你的智能助手。有什么我可以帮助你吗？
```

## 两个后端的差别（都在 python 侧验证过，这里照抄）

* **hiai** 逐个组装：`InitOption_Create` → `SetInferType` → `SetTokenizer` →
  `ModelInfo_SetWeightDir/SetModelPath/SetModelType` → `SetModel` →
  `Executor_Create` → `Executor_Init_Use_Option`。
* **cann** 只吃两个 JSON：`Executor_CreateFromExecutorJson` 与
  `Context_CreateFromContextJson` —— ⚠ **两个参数都是【文件名】，不是 JSON 文本**
  （传文本只会拿到 NULL；python 里传的是它自己写的
  `os.path.join(model_dir, ".context.live.json")`）。
* 两者都**必须先 `chdir` 到模型目录**：hiai 的 `modelPath` 写裸文件名
  （`xxx.omc`）+ `weightDir="./"`；cann 的 `executor.json` 也是按相对当前目录找的。
  传绝对路径时 hiai 的 `Init_Use_Option` 会直接返回 1。
* `Generate` 的第 3 个参数都是 prompt **文本**（`std::string::c_str()`），
  引擎自己分词；取输出都用 `GetAllGenerationLen` + `GetAllGeneration`，拿到明文。
* **两边的 `Destroy` 都不要调** —— `cann.py` 的注释写明会崩。

## 坑（都踩过）

1. **引擎库别弄错**：hiai 那套 `HIAI_LLMEngine_*` 符号只在
   `/system/lib64/libhiai_llm_engine.so` 里；`ndk/libcann_llm_engine.so` 导出的是
   另一套 `HMS_LLMEngine*`。弄错的表现是 `dlsym` 全部 missing symbol。
2. **`modelPath` / `executor.json` 都按相对路径找**，所以要先 `chdir` 到模型目录。
3. **`/tmp` 只读**：编到工作区里，别编到 `/tmp`。
4. **cann 的 context 参数是文件名**（见上），传 JSON 文本会得到 NULL。

## 这跟「终端权限」的关系

能跑出上面结果，说明**这个终端有 NPU 权限**。反例是某些应用的内置终端（见仓库 README 的「终端权限问题」），在那里连
`libneural_network_runtime.so` 都映射不上。**这句报错不在进程的 stdout/stderr 上，
而是引擎打到 hilog 里的** —— 所以要按 pid 去 hilog 里看：

```bash
hilog -x | grep <pid>        # 或用 hilog 持续抓，再 grep 本次进程的 pid
# 里面会看到：
#   Error loading header libneural_network_runtime.so: failed to map header
```

拿这两个小程序各跑一圈，能最快区分「是权限问题」还是「模型/配置问题」。
