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
```

---

## 五、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| **segfault（崩溃）** | 用了 glibc 的 Python ✗ → 换 `/data/service/hnp/bin/python3`（musl）|
| `未知后端 'xxx'` | 没装/没设 `PYTHONPATH`；或后端名写错（`-b` 可用 `hiai` / `cann`，默认 `hiai`）|
| `BackendUnavailableError: … 未加载` | 设备上没有该引擎库，或 `CANN_LLM_LIB`（cann）/ `CANN_LLM_HIAI_LIB`（hiai）指错了 |
| `缺少 api_config.json（或 <model>.json）` | 模型目录是旧格式 ✗ → 补一次即可：`python -m cann_llm.modelpkg <模型目录>` |
| 输出成段胡话（`ampie...`） | **极可能是 prompt 没套对话模板**：裸 prompt（如只有“你好”）没有 `<\|im_start\|>` 结构，模型会跑偏。CLI / HTTP 会自动套模板，正常使用不会出现；直接调后端 API 时请自行套模板 |
| 推理失败，且换终端后就好 / 换终端后依旧 | **无法自动判断是不是权限问题** —— 实测 `/dev/npu*` 的 `stat`/`open` 在**有权限与没权限的终端上都失败**（errno 13），没有区分度，所以本项目不做这种预检、也不替引擎断言原因。失败时给出的是可能原因列表：当前终端没有访问 NPU 的权限 / 输入超出 KV 缓存 / 无法分词的字符 / 引擎内部错误。**要确认是不是权限问题，唯一的办法是换一个系统终端实际跑一次** |
| **满屏 `Unknown class perfgenius_interface`** | 引擎为给 NPU 设温控会 `dlopen` 华为的 perfgenius 客户端，它连带加载 `libselinux`，而系统的 SELinux 策略里 `perfgenius_interface` 这个 class 在解析阶段不可见 → 解析器把警告写到 stderr。**不是本项目的错误，也不影响功能**。确实嫌吵就在命令末尾加 `2>/dev/null`：<br>`./scripts/start_chat.sh -d … 2>/dev/null`<br>⚠️ `2>/dev/null` 会屏蔽**所有** stderr（含真正的报错），只在确认其它一切正常时用 |

> 关于那条 SELinux 警告的完整调查（含它为什么会出现、以及"用 stub 屏蔽它"为什么不划算——会连温控一起屏蔽）见
> `docs/hiai-backend-handoff.md` 的对应章节。

### 可覆盖的环境变量

| 变量 | 用途 |
|---|---|
| `CANN_LLM_HIAI_LIB` | 覆盖 hiai 引擎库路径（默认 `/system/lib64/libhiai_llm_engine.so`）|
| `CANN_LLM_MODEL_DIR` | `start_server.sh` 的默认模型目录 |
| `PYTHON` | `start_server.sh` 显式指定解释器 |
