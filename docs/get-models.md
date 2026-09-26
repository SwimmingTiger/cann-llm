# 怎么拿到官方模型（网页点击步骤）

> 所有鸿蒙可用模型都在 **Matrix 模型库**：<https://matrix.openharmony.cn>
> 专门用于 **鸿蒙 PC** 的模型入口：
> <https://matrix.openharmony.cn/#/model/main?tag=OpenHarmonyPC>

---

## 一、点击步骤（以 Qwen2.5-Coder-7B-Instruct 为例）

**第 1 步 · 打开模型库**

浏览器打开 <https://matrix.openharmony.cn> ，点顶部导航的 **「模型库」**。

**第 2 步 · 按标签筛出鸿蒙 PC 模型**

在筛选区点 **「按标签」**，再点 **「鸿蒙PC」**（括号里显示 **8**，即 8 个模型）。

> 也可以用「按分类」「按组织」筛；「鸿蒙PC」标签下的才是本机 NPU 能跑的。

**第 3 步 · 点开想用的模型**

在模型卡片列表里**点模型名字**，例如 **`Qwen2.5-Coder-7B-Instruct`**，进入详情页。

**第 4 步 · 点「模型文件」标签** ★ 关键

详情页有 5 个标签：

```
模型库介绍 | 合规检测 | 模型文件 | 模型下载 | 用户评论
                     ↑
                   点这里
```

> ⚠ 注意点 **「模型文件」**，不是「模型下载」。
> 「模型下载」只给 `git clone` / AtomGit SDK 两种方式；
> **「模型文件」才是带文件名、SHA256、直链和大小的一览表**。

**第 5 步 · 在表格里取下载地址**

表格列如下，**点「文件下载地址」那一列的链接**即可下载（右键可复制链接）：

| 文件名称 | SHA256 | 文件下载地址 | 文件大小 | 文件浏览地址 | 文件相对路径 |
|---|---|---|---|---|---|

**第 6 步 · 下载下来的文件名是一串 hash —— 要自己加 `.zip`** ★

点「文件下载地址」保存后，**文件名往往不是 `xxx.zip`，而是一长串十六进制**、**没有扩展名**，例如：

```
96f1956fc3a9f6aee9528ffd817937fd1fbc463c889c144c0e24d300a197
```

> ⚠️ 这串名字**和文件的 SHA256 没有关系**，不要拿它去和表格的 SHA256 列对照 —— 那对不上。

**要做的只有一件事：手动把文件名补上 `.zip` 后缀**，例如：

```bash
mv 96f1956fc3a9f6aee9528ffd817937fd1fbc463c889c144c0e24d300a197 \
   Qwen25-Coder-7B-Instruct-OMC-20251024.zip
```

导入脚本**按 `.zip` 扩展名识别**：不加后缀时它会把这个文件当成目录，
报出「**不是目录**」—— 那不是文件坏了，只是缺后缀。

**想知道自己下的是哪个包（OMC / K0200 / 其它）？** 对下载到的文件**自己算一遍 SHA256**，
再和表格的「SHA256」列比对（文件名帮不上忙）：

```bash
shasum -a 256 ./96f1956fc3a9f6aee9528ffd817937fd1fbc463c889c144c0e24d300a197
```

两端一致即确认无误。

**第 7 步 · 选对文件（很重要）**

以 Qwen2.5-Coder-7B-Instruct 为例，表里实际有 4 行：

| 文件名称 | 大小 | 能不能用于本项目 |
|---|---|---|
| `Qwen25-Coder-7B-Instruct-OMC-20251024.zip` | 3.39 GB | ✅ **本项目用的就是这个** |
| `Qwen2.5-Coder-7B-Instruct-OMC-20260514.zip` | 3.25 GB | ✅ 同为 OMC 版，可用 |
| `Qwen25-Coder-7B-Instruct-K0200-20260624.zip` | 3.23 GB | ❌ **跑不通**，见下方说明 |
| `trained.pth` | 14.81 GB | ❌ 未量化的原始权重，NPU 用不了 |

> **❌ 为什么不选 K0200 版**：该包里 `is_kv_cache_merge = false`，
> 会触发引擎断言 `!configOptionList_[i].isKvCacheMerge && GetKvCacheLenNum(i) > 1` 而失败。
> **认准文件名里的 `OMC`**。

**第 8 步 · 校验一下（可选但推荐）**

表格的 **SHA256** 列就是校验值。例如 OMC-20251024 那个包：

```
5d81580e625df8fc593a683e9351b073cc102317f19be7193aa62f50db629883
```

---

## 二、下载完成后：装成模型目录

```bash
# 直接把 zip 交给导入脚本，它会解压并整理成后端认识的目录
python3 scripts/import_omc_package.py Qwen25-Coder-7B-Instruct-OMC-20251024.zip --dest models/

# 先看一眼会生成什么（不写盘）
python3 scripts/import_omc_package.py Qwen25-Coder-7B-Instruct-OMC-20251024.zip --dry-run
```

产物形如 `models/qwen25_coder_7b_omc1024/`，含 `api_config.json`、`qwen7b.omc`、
`qwen7b.json`、`tokenizer.json` 与权重（**约 4.3 GB**）。

启动：

```bash
make server MODEL=models/qwen25_coder_7b_omc1024 BACKEND=hiai PORT=8000
```

---

## 三、其它入口

| 想找什么 | 去哪 |
|---|---|
| 鸿蒙 PC 模型（8 个）| <https://matrix.openharmony.cn/#/model/main?tag=OpenHarmonyPC> |
| 全部模型 | <https://matrix.openharmony.cn> → 「模型库」 |
| 盘古 / 开源 Qwen / 开源 DeepSeek | 模型库 →「按分类」 |
| 模型对应的代码仓库 | 详情页顶部的 **「地址」** 就是 GitCode 仓库，如 `https://gitcode.com/openharmony-models/<模型名>` |
| 只想用 git 拉 | 详情页 **「模型下载」** 标签里有 `git clone` 与 AtomGit SDK 两种写法 |

> 提示：详情页里 **「模型文件」** 的表格是把「哪个文件、多大、SHA256、直链」一次列全的地方，
> 找包优先看它。
