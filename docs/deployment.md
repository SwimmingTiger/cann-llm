# 部署指南

## 一、结论：仍然是 100% 纯 Python

| 项 | 现状 |
|---|---|
| 运行时依赖 | `dependencies = []` —— **只用标准库 + ctypes** ✓ |
| 是否需要编译 | **不需要** ✓（没有 ext_modules、没有 CMake、没有编译目标，且**不含任何 C/C++ 源文件**）|
| C/C++ 代码 | **没有** ✓ —— 曾有一个实验用的 `hiai_shim.c`，**已删除**。它当初的用途是"自己拼一个真正的 `std::string` 塞进 ModelInfo"；后来发现官方 API（`LMEngine_ModelInfo_SetModelPath/SetWeightDir`）本来就收 C 字符串，用不着它 |
| 原生库从哪来 | **系统自带**：`/system/lib64/libhiai_llm_engine.so`（hiai 后端）与 `/system/lib64/ndk/libcann_llm_engine.so`（cann 后端）|

**所以"部署"= 把 Python 代码 + 模型文件放到设备上，然后用设备上的 Python 跑。**

---

## 二、硬约束：只能在鸿蒙设备上跑

这两条是**物理约束**，绕不开：

1. **引擎是系统的**：`/system/lib64/libhiai_llm_engine.so` 随 HarmonyOS 提供，
   内部还要用 NPU 驱动 —— **x86 / Docker / 桌面 Linux 上都没有** ✗
2. **必须用 musl 构建的 Python 3.12**：
   **鸿蒙 PC 不自带 Python**，需要先从应用市场安装「**Python安装器**」
   （`com.develop.opensource.ohdpc.python.launcherforpython312`），
   装好后落在：

   ```
   /data/service/hnp/python.org/python_3.12/bin/python3.12
   （或软链 /data/service/hnp/bin/python3）
   ```

   **用 brew / glibc 构建的 Python 加载引擎会 segfault** ✗ —— 原因见下面的
   [「为什么 glibc 的 Python 不行」](#为什么-glibc-的-python-不行)。
   `scripts/start_chat.sh` / `scripts/start_server.sh` 会自动把 hnp 的
   Python 加到 `PATH` 末尾并选中它，无需手工配置。

   > 它是**本项目唯一的第三方依赖**（Python 包层面仍然是零依赖）。
   > 这个 Python **自带 pip 24.3.1** ✓。装包一律用
   > `python3 -m pip install <包>` —— 这条对「Python安装器」版和 harmonybrew 版
   > **都适用**；直接写 `pip` 有可能会落到不是你想用的那个解释器上。

### 为什么 glibc 的 Python 不行

**一句话**：这一个进程里只能有一个 `libc.so`，而引擎要的是 musl 那个。

本机（鸿蒙）的系统 libc 是 **musl**：

```
/lib/ld-musl-aarch64.so.1      ← 系统动态加载器就是 musl 的
/system/lib/libc.so            ← 引擎 NEEDED 的 libc.so 解析到它
```

引擎自己的依赖表（`readelf -d` 实测）：

```
NEEDED  libz.so · libhilog_ndk.z.so · libc++_shared.so · libc.so
                                                         ↑ musl 版
```

而 brew 装出来的 Python 是 **glibc** 构建的 —— 它要的是 **glibc 版** `libc.so`，
靠一个叫 `libmusl_compat.so` 的**转发层**在 musl 系统上跑起来：

```
brew python3 的 NEEDED：libmusl_compat.so · libintl.so.8 · libpython3.14.so.1.0 · libc.so
                                                                                   ↑ glibc 版
```

`libmusl_compat.so` 做的是「**让 glibc 构建的程序在 musl 系统上跑**」——
它提供 musl 那边缺失或名字不同的符号（实测导出 58 个，如 `aio_*`、`crypt`、
`crypt_r`、`__res_state`）并转译到 glibc。**方向是单向的**：它让 glibc 程序
去用 musl 系统，**并不能让 glibc 进程去加载一个 musl 库**。

于是 `dlopen` 引擎时：

| 步骤 | 实测结果 |
|---|---|
| `ctypes.CDLL("/system/lib64/libhiai_llm_engine.so")` | ✅ **成功**（不崩） |
| 第一次调用引擎的函数（当时是 `Executor_CreateFromJson`，现在是 `InitOption_Create`） | ❌ **Segmentation fault** |

> 上面那条 traceback 是**当时的实测记录**，入口后来换成了 `InitOption_Create`
> ＋ `Executor_Init_Use_Option`（见 [hiai-backend-handoff.md](hiai-backend-handoff.md)
> 最后一节）—— 结论不变：**glibc 构建的 Python 跑不动引擎**。

**为什么第一步能过、第二步才崩**：`dlopen` 只做符号绑定，此时引擎还没真正
用 libc；等它一执行到 `malloc` / `pthread_*` / `std::string` 分配，调用的就是
**进程里那个 glibc 的 `libc.so`** ✗ —— 而引擎是按 musl 的 ABI 和结构体布局
编译的，两边对不上，直接段错误。

实测崩点（`python -X faulthandler`）：

```
Fatal Python error: Segmentation fault
  File "…/src/cann_llm/backends/hiai.py", line 337 in load
      self._exec = self._bind.lib.HIAI_LLMEngine_Executor_CreateFromJson(…)
```

> 所以脚本里的检测是**看 `/proc/self/maps` 里有没有 `libmusl_compat`** ——
> 有就说明这个解释器是 glibc 构建的，直接跳过。这是一条**保守规则**：
> 不是"一 `dlopen` 就崩"，而是"用它跑不通"。**宁可早跳过并说清原因，
> 也不要让用户拿到一句没头没尾的 segfault。**

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

**约 4.3 GB** ✓。后端**不需要**任何手工调参文件，也不再往引擎里塞合成的 JSON：

* 建 Executor 用**官方服务那套逐项 setter**（`InitOption_*` + `LMEngine_ModelInfo_*`），
  只吃 `api_config.json` 里的 `inferType` / `tokenizerType` / `tokenizerPath` /
  `modelPath` / `weightDir` 五项；
* 模型的结构超参（`num_hidden_layers` / `hidden_size` / `kv_cache_max_len` /
  embedding 权重文件名 …）由**引擎自己**按 `modelPath` 去掉扩展名 + `.json` 去读
  —— 也就是上面那个 `qwen7b.json`。所以**这个文件必须与 `.omc` 同名**。

> 复现性提示：`models/<name>/` 下若出现 `executor_super.json` 之类，
> 那是调查期间的实验产物，**部署时不需要**（引擎也不会读它）。

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
