# npu-probe —— 绕过 python，用原生 ELF 直接驱动 NPU

用来回答一个问题：**原生 ELF 有没有权限打开 NPU、能不能完成一次推理？**
结论：**有，能。** 不需要 python、不需要 ctypes。

```bash
cc -O1 npu_infer.c -o npu_infer -ldl
LD_LIBRARY_PATH=/system/lib64/ndk ./npu_infer /path/to/model_dir
```

实测（Qwen2.5-1.5B，本机 Kirin X90）：

```
[2] Executor_Init_Use_Option = 0  (3060 ms)   ← 引擎加载成功
[4] 推理完成 2895 ms · out=16 tok · gen_len=78
===== 引擎输出（明文）=====
你好！你好！很高兴能为你服务。有什么我可以帮助你吗？
```

## 三个必须知道的点（都踩过）

1. **引擎库是 `/system/lib64/libhiai_llm_engine.so`**，不是
   `/system/lib64/ndk/libcann_llm_engine.so`（后者导出的是另一套
   `CreateFromExecutorJson`，`HIAI_LLMEngine_*` 全在里面找不到）。
2. **`modelPath` 只写文件名**（`xxx.omc`），`weightDir` 写 `"./"`，
   并且**先 chdir 到模型目录**。引擎按 `modelPath` 去掉扩展名 + `".json"`
   去找模型配置；传绝对路径时 `Init_Use_Option` 直接返回 1。
3. **`GenerateAsync(exec, ctx, prompt)` 的第 3 个参数是 prompt 文本**
   （`std::string::c_str()`），引擎自己分词 —— 不用自己 encode，
   取输出用 `GetAllGenerationLen` + `GetAllGeneration`，拿到的是明文。

调用序列与 `src/cann_llm/backends/hiai.py` 完全一致（那套签名是实测/反编译确认过的）。
