# 部署指南

## 一、结论：仍然是 100% 纯 Python

| 项 | 现状 |
|---|---|
| 运行时依赖 | `dependencies = []` —— **只用标准库 + ctypes** ✓ |
| 是否需要编译 | **不需要** ✓（没有 ext_modules、没有 CMake、没有编译目标，且**不含任何 C/C++ 源文件**）|
| C/C++ 代码 | **没有** ✓ —— 曾有一个实验用的 `hiai_shim.c`，因最终走 `CreateFromJson` 而用不到，**已删除** |
| 原生库从哪来 | **系统自带**：`/system/lib64/libhiai_llm_engine.so`（hiai 后端）与 `/system/lib64/ndk/libcann_llm_engine.so`（cann 后端）|

**所以"部署"= 把 Python 代码 + 模型文件放到设备上，然后用设备上的 Python 跑。**

---

## 二、硬约束：只能在鸿蒙设备上跑

这两条是**物理约束**，绕不开：

1. **引擎是系统的**：`/system/lib64/libhiai_llm_engine.so` 随 HarmonyOS 提供，
   内部还要用 NPU 驱动 —— **x86 / Docker / 桌面 Linux 上都没有** ✗
2. **必须用系统自带的 musl Python 3.12**：

   ```
   /data/service/hnp/python.org/python_3.12/bin/python3.12
   （或软链 /data/service/hnp/bin/python3）
   ```

   **用 brew / glibc 的 Python 加载引擎会 segfault** ✗ —— 因为引擎依赖设备上的
   musl 与 `libmusl_compat`（`cann.py` 里有 `_is_musl()` 检测就是为了这个）。

   > 设备上的 Python **自带 pip 24.3.1** ✓，可以直接 `pip install`。

---

## 三、部署三步

### 步骤 1：准备模型目录

**先拿模型包**：官方模型在 Matrix 模型库，网页点击步骤见
👉 **[docs/get-models.md](get-models.md)**（关键：详情页点 **「模型文件」** 标签取直链）。

拿到 zip 后转换：

```bash
# 导入官方包（解压 + 转成后端认识的目录）
scripts/import_model.sh <官方zip> -d /path/to/models

# 先看一眼会生成什么（不写盘）
scripts/import_model.sh <官方zip> --dry-run
```

产物目录（官方布局，`detect_layout` 认它为 `official`）：

```
qwen25_coder_7b_omc1024/
├── api_config.json      # 引擎参数（modelType/tokenizerType/inferType/…）
├── qwen7b.json          # 模型结构配置
├── qwen7b.omc           # 编译好的模型
├── tokenizer.json       # 分词器（本项目用纯 Python 实现读取它）
└── *.weight / *.embedding_weights   # 权重
```

**约 4.3 GB** ✓。后端会自己从 `api_config.json` + `qwen7b.json` **合成**
executor / context JSON（`build_configs()`），**不需要**任何手工调参文件。

> 复现性提示：`models/<name>/` 下若出现 `executor_super.json` 之类，
> 那是调查期间的实验产物，**部署时不需要**。

### 步骤 2：放置代码

把整个仓库拷到设备上任意可写目录即可（**不需要安装、不需要编译**）。
脚本自己会找到合适的解释器，并从仓库位置推算出源码路径。

> 想装成 pip 包、或用 `make` / `python -m` 直接跑模块，属于维护者用法，
> 见 [README 的「维护者」一节](../README.md#维护者)。

### 步骤 3：启动

```bash
# ① 交互式对话（默认就是逐字流式输出）
./scripts/start_chat.sh -d <模型目录> -b hiai

# ② OpenAI 兼容推理服务
./scripts/start_server.sh -d <模型目录> -b hiai --port 8000
```

就这两个脚本 —— 用户侧不需要其它启动方式（`make` / `python -m` / pip 安装的命令
都属于维护者用法，见 [README 的「维护者」一节](../README.md#维护者)）。

服务脚本还支持后台运行与运维：

```bash
# 注意：-b 是【后端】，-B 才是【后台】！
scripts/start_server.sh -d <模型目录> -b hiai -B   # 后台 + 等就绪
scripts/start_server.sh --status
scripts/start_server.sh --stop
```

---

## 四、验证部署是否成功

```bash
# ① 非流式
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hiai","messages":[{"role":"user","content":"def add(a, b): return a + b"}],"max_tokens":60}'
# 期望：{"object":"chat.completion", ..., "finish_reason":"stop", "usage":{...}} ✓

# ② 流式（SSE）
curl -sN -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hiai","messages":[{"role":"user","content":"请写一个加法函数"}],"max_tokens":40,"stream":true}'
# 期望：多行 data: {"object":"chat.completion.chunk",...} 最后 data: [DONE] ✓

# ③ 分词器自检（不加载模型，秒级；维护者用）
PYTHONPATH=src python3 -c "
from cann_llm.backends.hiai_tokenizer import QwenTokenizer
t = QwenTokenizer('<模型目录>/tokenizer.json')
for s in ['def add(a, b): return a + b', '你好，请写一个加法函数']:
    ids = t.encode(s); print(len(ids), t.decode(ids) == s)
"
```

---

## 五、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| **segfault（崩溃）** | 用了 glibc 的 Python ✗ → 换 `/data/service/hnp/bin/python3`（musl）|
| `未知后端 'hiai'` | 没装/没设 `PYTHONPATH`，或忘了 `-b hiai`（默认是 cann）|
| `BackendUnavailableError: … 未加载` | 设备上没有该引擎库，或 `CANN_LLM_HIAI_LIB` 指错了 |
| `不是官方结构（需要 <model>.json + api_config.json）` | 模型目录用的是旧版 K0200 包或自己拼的 ✗ → 用官方 OMC 包重跑步骤 1 |
| 输出成段胡话（`ampie...`） | 历史问题：走了 `Prompt_SetText` ✗ —— 当前后端已改走 token ids ✓，若复现请查 `git log` 与 `docs/hiai-backend-handoff.md` |

### 可覆盖的环境变量

| 变量 | 用途 |
|---|---|
| `CANN_LLM_HIAI_LIB` | 覆盖 hiai 引擎库路径（默认 `/system/lib64/libhiai_llm_engine.so`）|
| `CANN_LLM_MODEL_DIR` | `start_server.sh` 的默认模型目录 |
| `PYTHON` | `start_server.sh` 显式指定解释器 |
