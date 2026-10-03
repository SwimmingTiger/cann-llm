# 维护者笔记

给**改这个项目的人**看的：环境怪癖、平台限制、踩过的坑。
面向使用者的用法一律不写在这里，写在该功能自己的文档里 ——
比如 `offline-model-nnrt.md`。

## ★★ 基准测试的铁律：同时只能有【一个】推理进程 ★★

**这块 NPU 没有能力并行跑多个推理。** 只要同时存在第 2 个推理进程：

* 两个都会变慢 ⇒ **所有数据都是垃圾**，不能拿来比较任何东西；
* 每个进程要把段图（合计约 10 GB）连同缓存装进内存 ⇒ **2~3 个进程就能把 32 GB 撑爆并进 swap** ✗。

**2026-10-04 的实测反例**：维护者一边跑"布局 A/B"、一边用户自己在跑 CLI，
结果内存和 swap 全部被打满，**双方数据都不可用**（用户当场指出）。

⇒ 因此：

1. 跑任何对照实验**之前**，先确认没有别的推理在跑 —— **自己挂的后台任务也算**；
2. 所有对照组**串行**执行，**绝不允许**用 `nohup` / `&` 并发跑推理；
3. 每组跑完确认 `free -m` 已回落，再跑下一组；
4. 宁可**少测几组**，也不要并发 —— 并发出来的数字比不测更糟（会导出错误结论）。

> 这条规则是**用户明确要求**的，不是建议。

## 本机调试器（lldb）的可用性

* 鸿蒙 7 **不带**系统自带的 `lldb` / `lldb-server`。
* `/data/service/hnp/bin/lldb-server` **是存在的**，但它是 **DevBox 装的**，
  不是系统自带；它与 Harmonybrew、OHOS-SDK 提供的那几份一样，
  在当前身份下会 `ptrace failed: Permission denied`（应用沙箱禁 ptrace）。
* 唯一实测可用的：应用商店 **CodeArts IDE**（`com.huawei.codearts`，
  注意与白名单里的 `com.huawei.codearts.agent` 是**两个应用**）自带那份
  只依赖 musl libc 的自包含 `huawei-debug-lldb-server`。它躺在 IDE 自己的沙箱里，
  必须在 **CodeArts IDE 的终端**里拷出来（源文件在 `/data/storage/el2/base/files/`）。
  ```
  $ mkdir -p ~/.local/bin
  $ cp /data/storage/el2/base/files/huawei-debug-lldb-server ~/.local/bin/
  ```
* 本机 lldb 客户端直接 `lldb -- <prog>` 会报
  `error: 'A' packet returned an error: 8` —— 这是**平台限制**，
  与"是否带参数"无关（不带参数也一样）。必须由它 `gdbserver` 先拉起进程，
  再用 lldb `gdb-remote` 接上去（`src/cann_llm/lldb_launch.py` 已经封装）。
* 查找这份 server 时**别只试一个位置**：`CANN_LLM_LLDB_SERVER` 是显式覆盖点；
  没有它时把 `~/.local/bin` 与 `/data/storage/el2/base/files` **追加**到 `PATH`
  末尾（追加而非前插，免得盖掉用户已有的同名程序）再按名字找。

### 想调试 Python 的话

* `/data/service/hnp/` 里那份 Python（`/data/service/hnp/bin/python3`）**无法被 lldb 调试**。
* 把它**拷出来并重新签名**之后也**不行** —— cann-llm 的推理跑不起来，
  会报内存分配相关的错误：

  ```
  MemoryError:
  ```

  ⇒ **拷出来这条路也不通，别在上面花时间。**
* ⇒ 想调试 Python，**得自己编译一份不使用 `libmusl_compat` 垫片的 Python**。
  原因：引擎 `libcann_llm_engine.so` / `libhiai_llm_engine.so` 是按 musl 编译的，
  而 glibc 构建的 Python 靠 `libmusl_compat` 垫片运行，
  把 musl 版引擎加载进来就会崩（README「依赖与构建」里那段"不要用 glibc 构建的
  Python"说的就是这件事）—— 换个壳（拷贝 / 重新签名）解决不了。

