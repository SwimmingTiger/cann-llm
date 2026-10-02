# 维护者笔记

给**改这个项目的人**看的：环境怪癖、平台限制、踩过的坑。
面向使用者的用法一律不写在这里，写在该功能自己的文档里 ——
比如 `offline-model-nnrt.md`。

## 本机调试器（lldb）的可用性

* 鸿蒙 7 **不带**系统自带的 `lldb` / `lldb-server`。
* `/data/service/hnp/bin/lldb-server` **是存在的**，但它是 **DevBox 装的**，
  不是系统自带；它与 Harmonybrew、OHOS-SDK 提供的那几份一样，
  在当前身份下会 `ptrace failed: Permission denied`（应用沙箱禁 ptrace）。
* 唯一实测可用的：应用商店 **CodeArts IDE**（`com.huawei.codearts`，
  注意与白名单里的 `com.huawei.codearts.agent` 是**两个应用**）自带那份
  只依赖 musl libc 的自包含 `huawei-debug-lldb-server`。它躺在 IDE 自己的沙箱里，
  必须在 **CodeArts IDE 的终端**里拷出来（源文件在 `/data/storage/el2/base/files/`）。
* 本机 lldb 客户端直接 `lldb -- <prog>` 会报
  `error: 'A' packet returned an error: 8` —— 这是**平台限制**，
  与"是否带参数"无关（不带参数也一样）。必须由它 `gdbserver` 先拉起进程，
  再用 lldb `gdb-remote` 接上去（`src/cann_llm/lldb_launch.py` 已经封装）。
* 查找这份 server 时**别只试一个位置**：`CANN_LLM_LLDB_SERVER` 是显式覆盖点；
  没有它时把 `~/.local/bin` 与 `/data/storage/el2/base/files` **追加**到 `PATH`
  末尾（追加而非前插，免得盖掉用户已有的同名程序）再按名字找。
