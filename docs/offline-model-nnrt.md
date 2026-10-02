# 离线模型 + NNRt：让 NPU 跑"引擎装不下"的模型

> 这是一条**与 [模型转换](model-conversion.md) 完全不同的路**。
> 那条是把模型转成引擎要的 `.omc`；这条是**把厂商离线模型包成 MindSpore Lite 的 `.ms`，
> 再由 NNRt 在设备上直接加载执行** —— 中间**不经过华为的 LLM 引擎**。

本文记录的是**我们实际跑通过**的一条链路，供你参考尝试。

> **术语**：本文里的**转换机**指装了华为 DDK 的 Linux 机器（我们用的是 x86_64），
> **设备**指鸿蒙设备本身。转换在转换机上做，产物拷到设备上跑。

> [!IMPORTANT]
> **什么时候需要这条路**：华为 LLM 引擎要求模型是「逐层一对 K/V + 每层 `past_key_value`
> 进出」的结构。像 Gemma 4 这类新架构（每层不同配置、K=V、K/V 跨层共享、每层额外输入通路）
> **引擎根本装不下**，图导得出来、OMG 也转得动，照样进不去。
>
> 而**离线模型对 MindSpore Lite 是个黑盒** —— 输入输出张量由扩展配置声明，模型内部结构
> 上层完全不解析，真正算它的是硬件厂商的工具链。**所以这条路不受任何结构约束。**

> [!NOTE]
> **验证程度**：我们用一个 `y = Add(x1, x2)` 的极小图**端到端验证通过**
> （转换成功 + 设备上 NNRt 后端推理结果正确）。**真实 LLM 尚未在这条路上验证过**，
> 请把它当作"路是通的、模型自己试"。

---

## 0. 流程总览

```
原始模型 (ONNX 等)
  │ OMG（DDK 自带）                      --platform=kirinx90
  ▼
厂商离线模型 (.om / .omc)
  │ converter_lite --fmk=THIRDPARTY      ★ 关键是这个参数
  │ + [third_party_model] 扩展配置
  ▼
MindSpore Lite 模型 (.ms)
  │ 设备上 libmindspore_lite_ndk.so + NNRt 后端
  ▼
推理结果
```

三个环节各自的前提：

| 环节 | 前提 |
|---|---|
| OMG | 华为 DDK（含对应平台的 platform 插件） |
| `converter_lite --fmk=THIRDPARTY` | **必须源码编译**（公开的预编译包里没有这个功能） |
| 设备侧推理 | 系统自带 `libmindspore_lite_ndk.so`（鸿蒙设备通常自带） |

---

## 1. 为什么必须自己编转换器

公开的预编译包（例如 `mindspore-lite-2.7.0-linux-x64.tar.gz`）**不支持**这条路：

* 它的 `--fmk` 只认 `TF | TFLITE | CAFFE | MINDIR | ONNX | OM | PYTORCH | MSLITE`
  —— **没有 `THIRDPARTY`**；
* 它内部的 `--fmk=OM` 走的是 **Ascend(ACL) 的自定义算子**路径
  （源码里写死 `"ACL_om_data"`），在麒麟上跑不了；
* 全仓库 grep `third_party_model` **零命中**。

而 OpenHarmony 用的那份 MindSpore 源码里有：

```
https://gitcode.com/openharmony/third_party_mindspore.git

mindspore-src/source/mindspore-lite/tools/converter/parser/third_party/third_party_model_parser.cc
mindspore-src/source/mindspore-lite/tools/converter/converter_lite/converter_flags.cc
    → {"THIRDPARTY", kFmkTypeThirdParty}
mindspore-src/source/mindspore-lite/test/ut/test_data/third_party_model.cfg   ← 官方配置样例
```

> 判断你手上的转换器支不支持，一条命令即可：
> `converter_lite --fmk=THIRDPARTY --modelFile=/nonexistent --outputFile=/tmp/x`
> 若报错里出现 `third_party_param_parser`，说明支持（报"文件不存在"是对的，
> 因为这一步只是让参数校验通过）。

---

## 2. 编译带 THIRDPARTY 的 converter_lite（容器里做）

> **别在宿主机上直接编。** 这份代码（2022 年前后）对工具链很挑，我们用 Arch 的最新
> 工具链一路报错；**换 Debian 12 容器后一次通过**。宿主机要避开的坑：
>
> | 宿主机现象 | 容器里为什么没有 |
> |---|---|
> | CMake 4.x：`Compatibility with CMake < 3.5 has been removed` | Debian 12 是 CMake 3.25 |
> | gcc 16 编老代码报一堆错 | Debian 12 是 gcc 12 |
> | jpeg 等依赖装进 `lib64/`，CMake 却找 `lib/` | Debian 的 install libdir 惯例不同 |
>
> 另外 `mindspore-lite/cmake/compile_link_option.cmake` 里有 `-Werror`，
> 会把 gcc 12 的**假阳性**警告当错误（我们踩到的是
> `-Wfree-nonheap-object` 在 `Tensor::shape()` 内联时的误报）。
> 官方自己也有绕过的先例（`src/litert/cxx_api/kernel_executor/CMakeLists.txt` 里
> `string(REPLACE "-Werror" "" ...)`），照做即可。

```bash
# ① 取源码
git clone --depth 1 https://gitcode.com/openharmony/third_party_mindspore.git
cd third_party_mindspore

# ② 起一个常驻构建容器（只装一次工具链，之后都用 docker exec）
docker run -d --name mslite-dev \
    -v "$PWD":/src -w /src/mindspore-src/source \
    debian:12 bash -c 'tail -f /dev/null'

docker exec mslite-dev bash -c '
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential cmake python3 python3-dev python3-numpy python3-yaml \
        python-is-python3 git ca-certificates curl patch unzip wget
    git config --global --add safe.directory "*"'
```

> 那几个 Python 包不是可选项：`gencode` 步骤要用 `yaml`
> （缺了会以 `ModuleNotFoundError: No module named '\''yaml'\''` 失败），
> 而 Debian 没有 `python` 命令，脚本里回退用的就是 `python`，
> 所以要装 `python-is-python3`。

```bash
# ③ 把 -Werror 换成 -Wno-error
docker exec mslite-dev bash -c '
    F=/src/mindspore-src/source/mindspore-lite/cmake/compile_link_option.cmake
    cp -n $F ${F}.bak && sed -i "s/ -Werror/ -Wno-error/g" $F'

# ④ 配置 + 编译（不用官方 build.sh：它每次会删掉 build/ 重下依赖）
docker exec mslite-dev bash -lc '
    set -e
    S=/src/mindspore-src/source
    mkdir -p $S/build && cd $S/build
    cmake -DCMAKE_BUILD_TYPE=Release -DVERSION_STR=2.7.0 -DENABLE_ASAN=off \
          -DCMAKE_INSTALL_PREFIX=$S/output/tmp -DPLATFORM_X86_64=on \
          -DMSLITE_ENABLE_TRAIN=off \
          $S/mindspore-lite/
    echo $(git -C /src rev-parse HEAD) > .commit_id
    cmake --build . --target install -- -j8'
```

两点说明：

* `.commit_id` 是 `install` 阶段要的文件，平时由官方 `build.sh` 生成。
  直接跑 cmake 时要自己补，否则 install 会在最后一步失败
  （`file INSTALL cannot find ".../build/.commit_id"`）—— **编译其实已经成功了**。
* 第一次配置会下载并编译 protobuf / flatbuffer / glibc 等一堆依赖（几十分钟）。
  **只要不删 `build/`，以后重编只走 `cmake --build . --target install -- -j8`，不会再下依赖。**

产物：

```
<source>/output/tmp/mindspore-lite-2.7.0-linux-x64/tools/converter/converter/converter_lite
```

---

## 3. 厂商离线模型 → `.ms`

### 3.1 先得到离线模型

用 DDK 自带的 `omg`。注意平台名要和你的设备对上：

```bash
export LD_LIBRARY_PATH=<ddk>/tools/tools_ascendc/package/lib64:<ddk>/tools/tools_ascendc/package/lib64/plugin:<ddk>/tools/tools_omg/master/lib64
omg --model=model.onnx --framework=5 --output=model --platform=kirinx90
# → model.om   （kirinx90 平台的产物后缀是 .om）
```

> 平台插件在 `<ddk>/tools/platform/<平台名>/`。只有插件里有的平台名才可用；
> 用错会报 `platform/<名字> not exists! Please install platform plugin first!`。

### 3.2 写扩展配置（**格式很容易搞错**）

离线模型是黑盒，**输入输出张量必须手写**在这个配置文件里：

```ini
[third_party_model]
input_names=in_0;in_1
input_dtypes=float32;float32
input_shapes=1,4;1,4
output_names=out_0
output_dtypes=float32
output_shapes=1,4
```

> [!WARNING]
> **三个必须记住的点**（我们全踩过）：
>
> 1. **节名是 `[third_party_model]`**，不是 `[om_converter]`；
> 2. **多个条目用分号 `;` 分隔**，不是冒号；形状内部才用逗号 `,`；
> 3. **类型名是小写**（`float32` / `float16` / `int32` …），不是大写。
>
> `[om_converter]` + 冒号 + 大写 是**另一条路**（`--fmk=OM`，走 Ascend ACL）的格式，
> 两者互不通用。最可靠的参照是官方样例
> `mindspore-lite/test/ut/test_data/third_party_model.cfg`。

### 3.3 转换

```bash
B=<source>/output/tmp/mindspore-lite-2.7.0-linux-x64
export LD_LIBRARY_PATH=$B/tools/converter/lib:$B/runtime/lib:$B/runtime/third_party/glog

$B/tools/converter/converter/converter_lite \
    --fmk=THIRDPARTY \
    --modelFile=model.om \
    --outputFile=model_ms \
    --configFile=third_party.cfg
# → model_ms.ms
```

成功时输出 `CONVERT RESULT SUCCESS:0`（"这条配置已被保存，将被覆盖"之类的
WARNING 是无害的）。

> `.ms` 可能是以 **root** 身份生成的（如果你的转换在容器里做），
> 拷出来前记得 `chmod 644`，否则宿主机读不到。

---

## 4. 设备侧：用 NNRt 后端跑 `.ms`

系统自带 `libmindspore_lite_ndk.so` 与配套头文件（OpenHarmony SDK 的
`native/sysroot/usr/include/mindspore/`）。关键枚举：

```c
OH_AI_DEVICETYPE_NNRT = 60      /* types.h */
OH_AI_MODELTYPE_MINDIR = 0      /* .ms 就是 MindIR */
```

最小示例见 [`examples/mslite-nnrt/`](../examples/mslite-nnrt/)：

```bash
cc -O1 -I<sysroot>/usr/include mslite_run.c -o mslite_run \
   -L/system/lib64/ndk -lmindspore_lite_ndk
LD_LIBRARY_PATH=/system/lib64/ndk ./mslite_run model_ms.ms nnrt
```

我们那次实测输出：

```
后端: NNRT (devicetype=60)
OH_AI_ModelBuildFromFile -> 0 (SUCCESS)
输入 2 个 / 输出 1 个
  in[0] x1       dtype=43 shape=[1,4] 元素=4
  in[1] x2       dtype=43 shape=[1,4] 元素=4
OH_AI_ModelPredict -> 0 (SUCCESS)
  out[0] y        shape=[1,4] → 2.000 2.000 2.000 2.000   （期望 2.0 ✓）
```

---

## 5. 一个很有说服力的交叉验证

同一个 `.ms`，换后端结果完全不同：

| 后端 | Build | Predict | 输出 |
|---|---|---|---|
| **CPU** | SUCCESS | SUCCESS | `0.000 0.000 0.000 0.000` ✗ |
| **NNRt** | SUCCESS | SUCCESS | `2.000 2.000 2.000 2.000` ✓ |

**CPU 后端出 0 不是 bug，而正好印证了官方那句「离线模型仅支持在 NNRt 后端推理」**：
离线模型被包成一个 **custom 算子**交给底层，CPU 侧没有它的实现，输出就是未初始化的；
真正算它的是 NPU 侧的厂商实现。

> 反过来说：**如果你在 NNRt 后端拿到了正确数值，就说明整条链（转换 → 包成 `.ms` →
> 厂商算子 → NPU）真的通了。**

---

## 6. 已知问题

* **进程退出阶段会 core dump**（`exit code 139`）。推理**结果已经正确**，
  崩在析构/动态库卸载阶段，是独立问题，尚未定位。
* 只在**单算子极小图**上验证过。真实 LLM（Transformer 级图）尚未在这条路上试过 ——
  需要厂商工具链能把该图编成 `.om`，且 `.om` 体量与设备内存匹配。
* `converter_lite --help` 打印的 `--fmk` 列表**不含 THIRDPARTY**（那段帮助文本没更新），
  以实际能否接受该参数为准。
