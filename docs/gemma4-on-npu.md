# Gemma 4 E2B 在麒麟 NPU 上跑通并接入聊天：实测记录与性能诊断

> 面向维护者的实测记录。设备：HarmonyOS 7.0.0 aarch64 / KirinX90 / 32 GB；
> 路径：`OMG → .omc → converter_lite → .ms → 设备 NNRt`（即 `-b nnrt` 后端）。
> 模型：`google/gemma-4-E2B-it`，**只取文本部分**。


## 0. ★★读本文所有计时数字之前必读★★

```
★★ 本文 §5 / §5.7 里由【维护者自己】测出的那些秒数（125s/85s、61s/32s、93s/53s、
   62s/24s、69s/22s 等）★【不可信，请勿引用】★ ✗

原因：维护者的测试方法与并发控制有严重问题 ——
  · 这块 NPU ★没有能力并行跑多个推理★；
  · 维护者多次在【自己挂着后台任务】的同时又跑新的推理，还与用户自己的运行重叠；
  · 结果把 32 GB 内存和 swap 全部打满 ⇒ 那些运行本身就是被拖慢的 ✗。
  （2026-10-04 由用户当场指出；维护者已把铁律写进 docs/maintainer-notes.md ✓）

★ 目前【可信】的只有用户自己测的数据，见 §8（含加载/推理分离的完整明细 ✓）：
      gemma4_5seg_s72   ⇒ 49.3 s
      gemma4_5seg_s128  ⇒ 57.9 s
      gemma4_9seg_s128  ⇒ 59.5 s
   （prompt 均为 '1+1='，同一台设备，time 实测 ✓）
⇒ ★要下任何关于性能的结论，必须重新串行测；本文其余部分的【结论性判断】
   （例如"瓶颈在哪"）只能当作线索，不能当证据 ★
```

## 1. 结论

```
✓ 全 35 层分段在设备上 Build 0 / Predict 0，9 段串起来能跑
✓ 与 transformers 对拍：设备 seg0 vs host 相对差 6.1e-05；全链 host 侧 top-5 完全一致
✓ ★已接入 cann-llm 后端★，通过 create_backend('nnrt') 能聊天：
     '1+1='                          ⇒ '1+1= **2**'
     'What is the capital of France?' ⇒ 'The capital of France is **Paris**.'
✗ 速度仍慢（8 个 token 约 24~60 秒）——原因见 §5（★不是"重跑全上下文"的计算量★）
```

## 2. 为什么必须分段（三条硬限制，全部实测）

| 限制 | 实测结果 | 影响 |
|---|---|---|
| OMG 单张量 ≤ INT_MAX | `embed_tokens_per_layer` = [262144, 8960] = **2.35e9 元素** ✗ | 必须按层切开；每段只带自己那几层的列切片 |
| ~~设备单张量 < 1 MB~~ ★★此结论已被推翻，见 §5.1★★ | 早期二分（1 层 Qwen，48 个逐层 KV 张量）：≤1920 ✓ / 2048 ✗ ⇒ 当时【误判】成"单张量 <1MB" ✗ | 当时据此把 lm_head 分了块、KV 限在 960 |
| NPU-CL 对 ≥4 维支持差 | `StridedSlice`/`ExpandDims` 直接拒收 4 维 | 注意力全部折成 **3 维** `[heads,S,D]` |

## 3. 分段设计（每条都是被实测逼出来的）

```
① config 一律不动 —— 动它会算错 first_kv_shared_layer_idx（实测输出差 350 ✗）
② 段内 layers 截断 + per_layer 预先切片 + project_per_layer_inputs 直通 + 段内不做末尾 norm
③ ★rotate_half 改成常量矩阵乘法 x@P★ —— 原实现切 head_dim ⇒ 4 维 StridedSlice ⇒ OMG 拒收；
   矩阵化后零误差 ✓
④ ★P 注册成 module buffer★ —— 在 forward 里构造会被 ONNX 追踪展开：85682 节点 → 599 ✓
⑤ head_dim 逐层不同（sliding 256 / full 512；full 层在 4/9/14/…/34）
⑥ ★KV 共享★：层 0~14 各自 K/V；层 13(sliding)/14(full) 写共享槽；层 15~34 **没有 k_proj/v_proj**
   （是"没有属性"而非 None ⇒ 必须 getattr）；段间只传 2 组 KV
⑦ 验证必须【串联】做 —— 逐段单独测会喂错输入（我为此浪费过一轮）
```

## 4. 分词器与对话格式

```
· Gemma 4 是 BPE + merges(514906) + byte_fallback；normalizer 负责 " "→"▁"（★不要自己加前导 ▁★）
· 连续的一串 ▁ 必须与后面文本作为同一片段做 BPE，否则 ▁▁ 这类 token 永远合不出来
· 对话格式：<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
  eos_token_id = [1, 106, 50]（106 即 <turn|>）
· 纯 Python 实现与 HF 对拍 ★7/7 一致★（含中文 / 连续空格 / 代码缩进）
```

## 5. ★性能诊断（本节是本次优化最重要的产出）★

```
★ 决定性发现：瓶颈【不是】每 token 的计算量，而是【每 token 的图调用次数 × 每次调用的固定开销】★

证据：
  · 每个 token 要调 9 段 + 4 块 lm + 1 次图P = ★14 次 Predict★
  · 做了完整的 KV 缓存 / decode（seq=1，每步只算 1 个 token，计算量降到 1/32）
    ⇒ 功能完全正确（回答不变 ✓）但★速度没有任何提升★（62s/24s vs 61s/32s ✗）
  · "图P 分块"（把 S 次调用降为 ceil(S/24) 次）也【变慢】了 ✗
    ⇒ 因为总调用次数没减、单次计算量反而变大 ✓

⇒ 结论：这个架构有约 10~14 次图调用/token 的"地板"，单靠减少计算量无法提速。
```

### 各条优化路线的实测结果

| 路线 | 结果 |
|---|---|
| ① 多尺寸段图 + 自动选最小够用尺寸（S=32/128） | ✓ **有效**：基线 125s/85s → 61s/32s（2.0~2.7×） |
| ② 图P 分块（S=24） | ✗ **反向**：93s/53s ⇒ 已停用 |
| ③ lm_head 只算最后 token（S=1） | ✓ 一直保持 |
| ④ KV 缓存 / decode（18 张新图） | ✓ **功能正确、数学已验证**，但 ✗ **不提速**（原因见上） |

> 计时噪声说明：不同批次之间波动很大（同为 8 token，观测到 24 / 32 / 61 / 85 / 98 / 125 秒）。
> 上述"2.0~2.7×"来自一次同条件前后对比；**没有**做多次取中位数的严谨统计 ✗。

## 6. 仍未做 / 可继续的方向

```
· lm_head 4 块 → 2 块（每块 131072×4B = 512KB < 1MB）⇒ 省 2/14 ≈ 14%（收益确定但不大）
· 把 9 段合并成更少的图 —— 受 OMG"高层数只吃特定 KV"的限制，风险高
· 图P 折进首段 ⇒ 省 1 次
· 真正的多流 / 流水线：把 9 段的 NPU 执行重叠起来（架构性改动）
★ 结论：在"每段一个 .ms"的架构下，最多再挤出约 20% ✗；要数量级提升需要换执行模型。★
```

## 7. 复现入口

```
导出脚本（x570，需 venv-g4 + DDK + mslite-dev 容器）：
  scripts/model-conversion/gemma4_seg_build.py       分段实现（3 维）
  scripts/model-conversion/gemma4_seg_export.py      段图导出（含 KV 模式化）
  scripts/model-conversion/gemma4_prefill_kv_export.py  prefill+KV 图（decode 路线用）
  scripts/model-conversion/gemma4_decode_export.py      decode 图（seq=1）
  scripts/model-conversion/gemma4_batch.sh            9 段批量：导出 → OMG → converter_lite
设备侧：
  src/cann_llm/backends/gemma4_runner.py   Gemma4SegRunner / ChatRunner / KvRunner
  src/cann_llm/tokenizer_gemma.py          GemmaTokenizer（与 HF 7/7 一致）
模型目录：~/work/llm/models/gemma4_chat/（seg{0..32}[/_s32]、dec{0..32}、pre{..}_s32、lm0..3、weights、io、tokenizer.json）
```


### 5.1 ★★对"单张量 < 1MB"的更正（后续实验推翻）★★

后续用 Gemma 的 dec 图做了更干净的对照（同一张图、同一个 KVMAX，只改 KV 的 dtype）：

| 配置 | slot_fu_k/v 大小 | 设备 Predict |
|---|---|---|
| KVMAX=512  fp32 | 262,656 元素 = 1026 KB | ★成功★ |
| KVMAX=4096 fp32 | 2,097,664 元素 = ★8194 KB（约 8 MB）★ | ★成功★ |
| KVMAX=4096 int8 | 同样的元素数，输入 dtype 改 INT8 | ★同样成功★（与 fp32 无差别） |

⇒ ⇒ ★★"单张量必须 < 1MB"是错的★★ —— 8 MB 的 fp32 张量都能过 ✗
⇒ 而且 ★int8 并不能提升 KV 上限★：限制根本不是按"单张量字节数"卡的 ✗
   （KVMAX=4096 时 fp32 与 int8 都通过，没有区分度）

那早先 Qwen kv=2048 为什么失败 ✗？目前★仍未查明★ ✗，但已可排除"张量字节数" ✓。
两个案例的差异在【结构】：

    · 失败：1 层，但 ★48 个逐层 KV 张量★（每层一对 K/V）
    · 成功：4 层，但只有 ★4 个共享槽张量★

⇒ 嫌疑是"图的 I/O 张量个数 / 逐层 KV 这种结构"触发了某种预算 ✗（未验证）。

### 5.2 由此产生的一个未解问题

★ 我们现在【无法预测】一个给定的分段图能否在设备上 Predict ✗ ——
既不是"层数"（24 层小 KV 连 OMG 都不过 ✗，而 4 层 8MB KV 却能过 ✓），
也不是"张量大小"（8MB 能过 ✓）✗。
⇒ 实用做法仍是【小步试】：每段导出后先在设备上 Build+Predict 一次，再往下走 ✓

### 5.3 ★★限制到底跟什么有关：证据指向"4 维 KV 张量"，不是大小★★

把已知案例按【张量维度】排一下，规律立刻显出来：

| 案例 | KV 张量形状 | 维度 | 单张量 | 设备结果 |
|---|---|---|---|---|
| Qwen 1 层 kv=1920 | `[1920,2,1,64]` fp32 | **4 维** | 960 KB | ✓ 通过 |
| Qwen 1 层 kv=2048 | `[2048,2,1,64]` fp32 | **4 维** | 1 MB | ✗ **失败** |
| Qwen 4 层 kv=1024 | `[1024,2,1,64]` fp32 | **4 维** | 512 KB | ✓ 通过 |
| Gemma dec16 KVMAX=512 | `[1,513,512]` fp32 | **3 维** | 1026 KB | ✓ 通过 |
| Gemma dec16 KVMAX=4096 | `[1,4097,512]` fp32 | **3 维** | **8194 KB（8 MB）** | ✓ **通过** |
| Gemma dec16 KVMAX=4096 | `[4097,2,1,256]` fp32 | **4 维** | 8194 KB | OMG 通过 ✓（设备侧未测完 ✗） |

```
⇒ ★结论（强证据，非最终确证）★：卡住的是【4 维 KV 张量】这条路径，而不是张量大小 ✗。
   · 4 维：[2048,2,1,64] 只有 1 MB 就失败 ✗
   · 3 维：[1,4097,512] 高达 8 MB 却通过 ✓
★ 这与本项目【最早遇到的 OMG 拒绝完全一致】：
     "strided_slice_get_format.cc IsDimThreeNdCase ... inputDim.size() 4" ✗
     —— NPU 的很多快速路径是【按 3 维写死】的 ✓
★ 当初"单张量 < 1 MB"的错觉，来自 Qwen 的 4 维 KV 恰好在 kv=2048 时到 1 MB —— 纯属巧合 ✗
```

**尚未完成的一步** ✗：4 维版的**设备侧 Predict 对照**没测完（我的 cfg 生成脚本出错了 ✗）。
要确证只需补一次：把同一张图的两个版本（3 维 vs 4 维、都是 8 MB）都在设备上 Predict 一遍 ——
若 3 维过、4 维挂，本节的结论即被钉死 ✓。

### 5.4 实用结论

```
· 设计分段图时，★KV/中间张量一律保持 3 维★ ✓ —— 这既是过 OMG 的要求（≥4 维常被拒 ✗），
  也很可能就是设备侧真正的约束来源 ✓（8 MB 的 3 维张量没问题 ✓）。
· 由此：★INT8 对 KV 上限没有帮助★ ✓ —— 限制不按字节卡 ✓（KVMAX=4096 时 fp32 与 int8 都过 ✓）。
  INT8 唯一还站得住的理由是【模型体积减半】✓（见 §6）。
· ★不再需要"KV ≤ 960"这个自我限制★ ✓ —— 3 维布局下可以放大得多 ✓（本会话实测到 4096 ✓）。
```

### 5.5 ★★§5.3 的"4 维"假说【已被实验推翻】★★

做完那最后一步对照后，结论变了 —— 4 维同样是 **Predict 成功** ✗：

| 案例 | KV 形状 | 维度 | 张量数 | 单张量 | 设备结果 |
|---|---|---|---|---|---|
| Qwen 1 层 kv=1920 | `[1920,2,1,64]` | 4 维 | 2 | 960 KB | ✓ |
| Qwen 1 层 kv=2048 | `[2048,2,1,64]` | 4 维 | **2** | **1 MB** | ✗ **失败** |
| Gemma KVMAX=512 | `[1,513,512]` | 3 维 | 4 | 1026 KB | ✓ |
| Gemma KVMAX=4096 | `[1,4097,512]` | 3 维 | 4 | **8 MB** | ✓ |
| Gemma KVMAX=4096 | `[4097,2,1,256]` | **4 维** | 4 | **8 MB** | ✓ |

```
⇒ ⇒ 钉死的结论：
   ★这个限制【不是】单张量大小 ✗（8 MB 能过）
   ★也【不是】张量维度 ✗（4 维 8 MB 同样能过）
   ★更【不是】张量个数 ✗（2 个 1 MB 失败 vs 4 个 8 MB 成功）
⇒ 那个 Qwen kv=2048 的失败是【另外的原因】✗ —— 与 KV 张量本身无关，
   最可能是那张图特有的算子/形状推断问题（未查明 ✗）。
```

**因此 §5.3 里"限制来自 4 维 KV"的说法应视为作废** ✗；本节取代它。

### 5.6 实用结论（这才是真正稳的部分）

```
✓ ★"单张量 < 1 MB"和"KV ≤ 960"都不必再遵守★ ✓ ——
   实测 3 维与 4 维布局下，单张量 8 MB（KVMAX=4096）都能正常 Build + Predict ✓✓
✓ ★INT8 对 KV 上限没有帮助★ ✓（同尺寸下 fp32 与 int8 都通过，无区分度）
✗ ★无法从图的结构预测它能否 Predict★ ✗ ——
   既不是大小、也不是维度、也不是个数（都已排除 ✓）
⇒ 工程做法：★每段导出后先在设备上 Build+Predict 一次再往下走★ ✓（小步验证）
⇒ 待办：那个 Qwen kv=2048 的失败值得单独查（它是唯一一个反例 ✗），
   但在 Gemma 这条链路上它【不复现】✓ ⇒ 不阻塞我们放大 KV 与减少分段 ✓
```

### 5.7 ★合并分段（9 段 → 5 段）：功能成功，但速度没变★

**做了什么**：把 9 段合并成 5 段 `(0-7)(8-15)(16-23)(24-29)(30-34)`，
每段 ≤ 8 层 ⇒ 每段 `.ms` 都 < 2 GB ✓。跑通了两个关键坑：

```
① ★ONNX 的 external data 必须合并成【单个 .data】★ ——
   torch 在 >2GB 时会把权重写成【成千上万个小分片】（onnx__MatMul_xxxx ✗），
   OMG 读到这种目录会产出一个★权重全丢的模型，却仍然报 success★ ✗
   （表现：omc 只有 1~2 MB ✗）⇒ 用 onnx.external_data_helper 合并 ✓
② ★converter_lite 有约 2 GB 的模型上限★（third_party_model_parser … init tensor ✗）
   ⇒ 含 full-attention 层的段要更少层数（12 层 = 2.78 GB ✗；8 层 ✓）
③ ★槽可以跨段传递★：含层 13/14 的段输出 sk/sv/fk/fv，后面的共享段消费 ✓
   （先把 KV_OUT 放宽成 has_store 即输出 ✓）
```

**结果**：

```
✓ 功能：5 段全部 Build/Predict 正常，回答不变（'1+1= **2**' · '… **Paris**.' ✓）
✓ 后端 extra 显示 segments=5 ✓
✗ 速度：★69s / 22s，与 9 段的 62s / 24s 在噪声内 —— 没有变化★ ✗
```

**这推翻了 §5 里"瓶颈是图调用次数"的判断** ✗：

```
· 调用次数 15 → 11（-27%）⇒ 时间没变 ⇒ 每次调用的固定开销不是主要瓶颈 ✗
· 加上此前 KV decode 把计算量降到 1/32 也毫无变化 ✗
⇒ 两个看似必然的优化都无效 ⇒ ★真正的瓶颈仍未定位★
· 剩下的嫌疑：每 token 的 11 次调用里，★4 次是 lm_head 分块、2 次是图P★ ⇒ 占 6/11 ✓
```

## 8. ★用户实测（唯一可信的计时）★

设备空闲、单进程、串行；prompt 均为 `1+1=`，`time` 实测 ✓。
开启 `CANN_LLM_TIMING=1` 后 runner 会打印"加载 / 推理"分开的明细 ✓。

| 模型目录 | 布局 | **加载 .ms** | **推理** | 墙钟 |
|---|---|---|---|---|
| `gemma4_5seg_s72` | 5 段合并 · S=72 | **45.0s（80%）** | 11.2s | 61.2s |
| `gemma4_5seg_s128` | 5 段合并 · S=128 | **32.2s（63%）** | 19.2s | 56.7s |
| `gemma4_9seg_s128` | 9 段 · S=128 | **33.5s（60%）** | 21.9s | 60.5s |

逐图明细（`5seg_s128`）：`graphP=load 2.5/run 0.7` · `lm=load 3.4/run 0.1` ·
`mseg0=3.7/1.7` · `mseg8=4.1/1.8` · `mseg16=5.7/6.2` · `mseg24=6.3/4.7` · `mseg30=6.4/3.9`
逐段（`9seg_s128`）：`seg0=2.8s seg4=3.0s seg8=4.0s seg12=4.7s seg16=6.6s seg20=6.4s seg24=7.4s seg28=7.6s seg32=6.1s`

### ★结论（与本文早期判断相反，以此为准）★

```
① ★时间大头是【把 .ms 加载/构建进 NPU】，占 60~85%★ ✗ —— 不是推理，也不是计算量。
   ★ 约 2 GB / 约 10 s ≈ 200 MB/s ⇒ 这是【磁盘 I/O 瓶颈】★ ✓
② 加载【每个进程只发生一次】✓（模型缓存有效：逐图都只 load 一次 ✓）
   ⇒ ★用 CLI 每次都是新进程 ⇒ 每次都重付这笔钱；多轮对话与常驻服务会把它摊掉★ ✓
③ ★S=72 是负优化★ ✗：加载比 S=128 还贵（45.0s vs 32.2s），推理也没省
   ⇒ 这一档应当弃用 ✓
④ lm_head 分 4 块其实很便宜（run 合计 0.4s ✓）⇒ 不值得为此改结构
⑤ 推理部分仍有很重的每层开销（约 0.78 s/层 ✗）⇒ 合并成更大的段仍有意义 ✓
⑥ ★本文 §5 里维护者自测的那些秒数全部作废★ ✗（原因见 §0：并发测试 + 内存/sawp 打满）
```

### 由此得到的优化优先级

```
① ★权重转 FP16★ ⇒ .ms 体积减半 ⇒ 加载时间约减半（省 15~22 秒）✓✓
   顺带突破 converter 的 2 GB 限制 ⇒ 可以做成更大的段（推理也省）✓
② 用常驻服务（start_server.sh）或连续多轮 ⇒ 加载只付一次 ✓
③ 弃用 S=72 ✓
④ 合并更大的段（依赖 ① 先把体积压下来）✓
```


### 8.1 ★把权重转 FP16：体积减半（已跑通 ✓）★

**动机**（来自 §8 的实测）：时间大头是【加载 .ms，约 200 MB/s 的磁盘 I/O】✗ ⇒
**把 .ms 体积减半，加载时间就差不多减半**；顺带也能突破 converter 的 2 GB 上限，
让"更大的段"成为可能 ✓。

工具：`scripts/model-conversion/onnx_weights_to_fp16.py`

```
用法：python onnx_weights_to_fp16.py <输入.onnx> <输出.onnx>
      （读入 external data ✓，输出写成【单个 .data】✓）
```

**关键细节（第一次做错的地方 ✗）**：

```
★ 只转【消费者全是 MatMul/Gemm】的权重 ★ ✓
✗ 错误做法：给所有大权重都插 Cast ⇒ 连 RMSNorm 的权重也被插上 ⇒
   ReduceMean 变成混合精度 ⇒ OMG 直接拒收 ✗
   （报错长这样：reduce_check_support.cc CheckElemSupportV0 … check reduce shape support fail ✗）
✓ 正确做法：只动 MatMul/Gemm 的权重 —— 它们是体积的 ~99%，且不经过任何 reduce ✓
   做法：先统计每个 initializer 被哪些 op_type 消费；ops <= {"MatMul","Gemm"} 才转 ✓
```

**实测结果（5 段 · S=128）**：

| 段 | fp32 | **fp16** |
|---|---|---|
| 0-7 | 1188 MB | **595 MB** |
| 8-15 | 1326 MB | **664 MB** |
| 16-23 | 2065 MB | **1033 MB** |
| 24-29 | 1581 MB | **791 MB** |
| 30-34 | 1301 MB | **650 MB** |
| 合计 | 7.46 GB | **3.74 GB** ✓ |

部署位置：`~/work/llm/models/f16segs/mseg{st}_s128/` ✓
对比目录：★`~/work/llm/models/gemma4_5seg_s128_f16/`★（软链 ✓，与 fp32 版逐项对照 ✓）

### 8.2 ★FP16 的实测结论：这条路在这块设备上【走不通】✗★

花了几轮把 FP16 试透了，结论是**否定的**，但**是实测出来的**，值得记下来 ✓。

| 形态 | 权重 | 激活 / I/O | 设备能否跑 | 实测 |
|---|---|---|---|---|
| **fp32**（现状） | fp32 | fp32 | ✓ | 加载 32.2s / 推理 19.2s / 墙钟 56.7s |
| **v1**：权重 fp16 + Cast 回 fp32 | fp16 | **fp32** | ✓ | ★加载 11.1s（3 倍快 ✓）/ 推理 46.2s（2.4 倍慢 ✗）★ |
| **v3**：权重 fp16 + 激活侧 Cast | fp16 | 中间 fp16 | ✓ | ★结果错误★（输出乱码 ✗） |
| **整图 fp16** | fp16 | **fp16** | ✗ | ★5 个段全部 Build -> -1★ ✗ |

```
★ 结论：★设备接受 fp16 权重，但不接受 fp16 的激活 / I/O★ ✓
   （用探针直接调 NDK 逐个验证过：graphP/lm 是 fp32 版 ⇒ Build 0 ✓；
     整图 fp16 的 5 个段 ⇒ 全部 Build -1 ✗）
⇒ 所以"全程 FP16"这条路不是我们代码的问题，而是★设备不支持★ ✗
⇒ 唯一能跑的 fp16 形态是 v1：收益真实（加载 3 倍快 ✓）但代价更大（推理 2.4 倍慢 ✗）
   ⇒ ★净损失（56.7 → 62.4 秒）★ ✗ ⇒ 不采用 ✓
★ 推测 v1 变慢的原因：权重上多了一个 Cast ⇒ NPU 无法把它当"常量权重"优化 ⇒
  每次调用都要重新搬/转 ✗（体积只有几百 MB，按内存带宽算不该这么贵 ⇒
  更像是打断了 NPU 的权重常驻路径 ✓）
```

**踩过的坑（都写下来省得重踩 ✗）**：

```
· OMG 失败时会写 ★check_result.json★（逐算子 pass/fail + 原因 ✓）⇒ 排错先读它 ✓
  （这次靠它定位到：Gated MLP 的 gate_proj/up_proj 共用激活 ⇒ 我插了两个同名 Cast ✗）
· 转 fp16 不能碰 RMSNorm 的权重 ✗（会产生 ReduceMean 混合精度 ⇒ OMG 拒收 ✗）
  ⇒ 只转"消费者全是 MatMul/Gemm"的权重 ✓
· torch 的 ONNX 导出里，`torch.zeros(..., dtype=fp16)` 与 `model.half()` 都要显式写 ✓
  （用正则去改脚本会漏 ✗ —— `L(WJ).eval()` 这种写法正则匹配不到 ✓）
· Python 的 array 模块★不支持 fp16（typecode "e"）★ ✗ ⇒ 用 struct ✓
· MindSpore Lite 的 dtype 编号：★Float16 = 42 · Float32 = 43★ ✓（不是 1/2 ✗）
· 每段 .ms 必须 < 2 GB（converter 的 int32 限制 ✓）；ONNX 的 external data
  必须★合并成单个 .data★ ✓，否则 OMG 会静默产出权重全丢的模型（omc 只有 1~2 MB ✗）
```

### 8.3 ★那么真正的杠杆是什么（基于实测）★

```
★ 既然 fp16 走不通，就回到 §8 的实测：★加载占 60~85%★ ✓ 而它★每个进程只发生一次★ ✓
⇒ ⇒ ★最大的实际收益不在图，而在"别让每个进程都重付加载费"★ ✓✓：
     ① 用常驻服务（start_server.sh）✓ 或连续多轮对话 ✓ ⇒ 加载只付一次
        ⇒ 之后每轮只付推理（约 19 秒 ✓），而不是每次 56.7 秒 ✓✓
     ② 减少段数（5 → 3）⇒ 每段更大、加载更少次 ✓（受 2 GB/段 限制 ✓）
     ③ lm_head 4 块很便宜（run 合计 0.4s ✓）⇒ 不必动
```

### 8.4 ★FP16 的最终铁证：最小图就崩（与结构无关）★

为了排除"图太大 / 结构太复杂 / 跨图 dtype 不匹配"这些解释，做了一个**最小对照**：

```
两张图结构完全相同，只有一个 MatMul，各 4.6 MB：
  C32：输入 FP32 + 权重 FP32   ⇒ ★Build 0 · Predict 0 ✓★
  C16：输入 FP16 + 权重 FP16   ⇒ ★core dump（段错误）✗★

对照：整图 fp16 的大段图（594 MB 等）⇒ Build -> -1 ✗
```

**⇒ 结论（排除法，逐项验证过）**：

| 可能的解释 | 是否成立 |
|---|---|
| graphP/lm 是 fp32，dtype 不匹配 | ✗ 最小图里没有 graphP/lm |
| 图太大 / 太复杂 | ✗ 只有 1 个 MatMul、4.6 MB |
| 某个算子不支持 fp16 | ✗ 只有 MatMul |
| **★设备 NNRt 路径的 fp16 支持本身有问题★** | **✓ 唯一成立的解释** |

```
★ 注意失败方式还不止一种：小图 ⇒ 段错误 ✗；大图 ⇒ Build -1 ✗
   ⇒ 不是"某种受限的 fp16 支持"，而是★不可用★ ✗
★ 因此：全 fp16 / 权重 fp16 + 激活 fp16 / 权重 fp16 + Cast 回 fp32 —— 前两种不可能跑通；
   第三种（v1）能跑但推理慢 2.4 倍 ⇒ 净损失 ✗
⇒ ★FP16 在这块设备上到此为止★ ✓（是否支持要看以后换设备/换系统版本 ✓）
```

### 8.5 ★fp16 崩溃的确切位置（lldb 实测）★

> ⚠️ **本节末尾那个"机制"结论已被 §9 更正 ✗**：不是"NPU 后端没有 fp16 kernel" ✗ ——
> §9 反编译出的映射表证明 **fp16 在 NNRt 侧是有映射的** ✓（TypeId 30 → OH_NN_FLOAT16）。
> **调用栈本身准确 ✓，但机制解释请以 §9 为准 ✓。**

用 lldb 调全 fp16 的最小图（单 MatMul，4.6 MB），拿到**完整符号化栈**：

```
SIGSEGV · fault address = 0x0（空指针解引用 ✗）

#1  libmindspore-lite.so`Scheduler::FindBackendKernel(...)      ★崩在这里★
#2  Scheduler::ScheduleNodeToKernel(...) + 172
#3  Scheduler::ScheduleSubGraphToKernels(...)
#4  Scheduler::ScheduleMainSubGraphToKernels() + 116
#5  Scheduler::ScheduleGraphToKernels(...)
#6  Scheduler::Schedule(...)
#7  LiteSession::CompileGraph(...) + 1200
#8  LiteSession::LoadModelAndCompileByPath(...)
#9  ModelImpl::Build(...)
#10 Model::Build(...)
#11 OH_AI_ModelBuildFromFile + 1160        ← 我们 ctypes 直接调的那个 API
```

**机制**：`Scheduler` 在**建图阶段**为每个算子挑 NPU 后端 kernel ✓
⇒ fp16 图里没有对应的 kernel 项（NPU 后端没注册 fp16 的 MatMul ✓）
⇒ 拿到空表项后**直接解引用** ⇒ SIGSEGV ✗

```
★ 这是 ★MindSpore Lite 的缺陷★ ✓：本该"报错说 fp16 不支持"，
  却变成了段错误 ✗（大图那批还能报 Build -1 ✓，小图直接崩 ⇒ 两条路径不一致 ✓）
★ 结论不变但现在是【机制级】的：★这块设备的 NPU 后端没有 fp16 kernel★ ✓
   ⇒ 与我们的代码无关 ✓，也不是"图太大/结构问题" ✓
```

### 8.6 ★附带修好的一件真事：python 3.14 的"兼容性问题"★

排查 fp16 崩溃时，先在 python-3.14 下遇到 `CDLL` 就段错误 ✗ —— 一度误以为是 ABI 问题 ✗。
**逐层定位**（每一层都有实测对照 ✓）：

```
· 绕过 CDLL 直接 _ctypes.dlopen ⇒ 3.14 崩 ✗ / 3.12 正常 ✓
· 换个库（libc.so）⇒ 3.14 也能加载 ✓ ⇒ 不是 _ctypes 通用坏 ✗
· 查该库的 DT_NEEDED ⇒ libmindspore-lite.so / libmindspore-lite-train.so /
    libhilog.so / libsec_shared.z.so ★都不在 /system/lib64/ndk★ ✓
    而在 ★/system/lib64/platformsdk★ ✓
· LD_LIBRARY_PATH 只给 ndk ⇒ 段错误 ✗；给 ndk:platformsdk ⇒ ★立刻正常 ✓★
⇒ ★根因：搜索路径缺 platformsdk 时，musl 的 dlopen 在"依赖解析失败"的
   分支上段错误★ ✗（一个很隐蔽的 loader 行为 ✓）
⇒ 已修：launcher / launcher_server 统一补 ENGINE_LIB_DIRS =
    (/system/lib64/ndk, /system/lib64/platformsdk) ✓
```

### 8.7 ★"python 3.14 段错误"的真身：我漏了一个 ctypes 声明★

**症状**：同一个 runner，python-3.12 正常 ✓，python-3.14 在第一次建图时 SIGSEGV ✗。

**定位（每步都有对照，没有猜）**：

```
① 最小复现（CDLL + ModelBuildFromFile）在 3.14 下 ★成功 ✓★ ⇒ 不是加载/Build 的问题
② 3.14 下建★真实的大图★（594MB 段图 / lm / graphP）⇒ ★全部 Build 0 ✓★
③ 但走 runner 就崩 ⇒ ★问题在 runner 的 ctypes 调用方式★
④ 对照 nnrt.py 的正确绑定，发现 `gemma4_runner._bind()` 有两处错：
     · OH_AI_TensorGetElementNum 声明成 c_size_t ✗（应为 c_int64）
     · ★OH_AI_TensorGetDataType 压根没声明★ ✗ —— 而新加的大小检查里调用了它
```

**机制**：

```
★ ctypes 对【未声明 argtypes】的函数，会把指针参数按默认的 32 位 int 传 ✗
  ⇒ 64 位下指针被截断 ⇒ NPU 解引用坏指针 ⇒ SIGSEGV ✓
★ 3.12 之所以"没事"：地址布局不同，指针恰好没被截断坏 ⇒ ★侥幸★ ✓
  ⇒ 所以这不是"3.14 特有的问题"，而是★一直存在的隐患★，被 3.14 暴露了 ✓
```

**修法**：补齐/改正声明（`GetDataType`、`GetDataSize`、`GetElementNum=c_int64`）✓
**验证**：python-3.14 下跑真实聊天 ⇒ `bot> 2` ✓✓

> 结论：遇到"某版本才崩"的段错误，先怀疑**自己的 ctypes/FFI 声明**，别急着怪版本或 ABI ✓ ——
> 这次就是漏一个 `argtypes` 造成的 ✓。

## 9. ★dtype 支持的逆向结论：fp16（存在但设备上不可用）/ fp8（不存在）/ int8（存在且很可能可用）★

**方法**（都可复现 ✓）：
* 材料：固件解包 `/media/hu60/SSD/work/hmos/firmware/unpack_result_010554/system` 里的**运行库** ✓
  （`platformsdk/libmindspore-lite.so`、`libnnrt_proxy_*.z.so`、
   `vendor/anco_spec/vendor/lib64/libai_npucore_*.so` ✓）
  + DDK 的 `tools/platform/kirinx90/lib64/libai_npucore_*.so` ✓
* 手段：`idalib`（IDA 9.3，`~/idaenv/bin/python`）+ `strings` 快筛 ✓
* 注意：先把库**拷到可写目录**再开库 ✓（只读介质上 `open_database` 会失败 ✗）

### 9.1 结论表

| dtype | MS-Lite / OMG | **NNRt 映射** | **NPU kernel** | **设备实测** |
|---|---|---|---|---|
| fp32 | ✓ | ✓ `TypeId 31 → OH_NN_FLOAT32` | ✓ | ✓ 正常 |
| **fp16** | ✓ | ★**✓ `TypeId 30 → OH_NN_FLOAT16`**★ | ✓（有 fp16 转换设施） | ★**✗ 崩**★ |
| **int8** | ✓ | ★**✓ `TypeId 32 → OH_NN_INT8`**★ | ★**✓ 量化设施非常完整**★ | 未实测（**很可能可用**） |
| **fp8** | ✗ | ✗ | ✗ | — |

### 9.2 fp16：**映射存在，但设备上实际不可用** ✗

**映射确实存在** ✓ —— 反编译 `mindspore::lite::CastToNNRtDataType`：

```c
__int64 mindspore::lite::CastToNNRtDataType(int a1) {
  if ((unsigned int)(a1 - 30) > 0xE) return 0;   // 接受 TypeId 30..44
  return dword_A63E0[a1 - 30];                   // 查表
}
// 表内容（dump 自 0xA63E0）：
//   TypeId 30 ⇒ 1  ★OH_NN_FLOAT16★      TypeId 31 ⇒ 0  OH_NN_FLOAT32
//   TypeId 32 ⇒ 2  ★OH_NN_INT8★          TypeId 33 ⇒ 3  OH_NN_INT32
//   TypeId 34 ⇒ 4  OH_NN_UINT8           TypeId 35 ⇒ 5  OH_NN_INT64
//   TypeId 37 ⇒ 6  OH_NN_BOOL            TypeId 39 ⇒ 8  OH_NN_FLOAT64
//   TypeId 40 ⇒ 9  OH_NN_INT16           TypeId 44 ⇒ 12（另有含义）
```

NNRt 的 HDI 桥接层也**确实有** fp16 能力协商 ✓：

```
libnnrt_proxy_1.0.z.so / libnnrt_proxy_2.1.z.so：
  ★IsFloat16PrecisionSupported★（NnrtDeviceProxy 的方法 ✓）
  v2.1 还有 ★enableFloat16★ 字段（"read/write dataBlock.enableFloat16" ✓）
```

**但设备上跑不起来** ✗：

```
· 全 fp16 的最小图（单个 MatMul、4.6 MB）⇒ SIGSEGV ✗
    栈：#1 Scheduler::FindBackendKernel → … → LiteSession::CompileGraph
        （fault address = 0x0 ⇒ 空指针解引用）
· 试开 `OH_AI_DeviceInfoSetEnableFP16(dev, 1)` ⇒ ★回读仍是 False★ ✗
    ⇒ 这个开关对 NNRt 设备不生效（它主要给 CPU/GPU 后端用）
· 大图那批（594 MB ~ 2.8 GB）⇒ 报 `Build -> -1` ✗（与小图的"直接崩"是两条路径）
```

⇒ ⇒ **修正后的结论**：**fp16 的通路在 MS-Lite 侧是通的（有映射、有协商接口）** ✓，
**但在这台设备的 NNRt 上实际跑不起来** ✗ —— 卡在**更下层**（能力协商的结果 / 厂商插件），
目前无法在我们的代码侧绕过 ✓。

### 9.3 ★更正 §8.5 的机制结论★

§8.5 我写的是"NPU 后端**没有** fp16 算子 kernel" ✗ —— **这是错的** ✗：
那是从"最小 fp16 图崩溃"外推的，**没有去看映射表** ✗。映射表摆明 fp16 有映射 ✓。
**现象是对的（fp16 跑不起来 ✓），机制是错的** ✗ —— 以本节为准 ✓。

### 9.4 int8：**存在，且很可能可用** ✓（值得试）

```
✓ 映射存在：TypeId 32 ⇒ OH_NN_INT8 ✓（表里还有 UINT8 ✓）
✓ NPU kernel 侧的量化设施非常完整 ✓（strings 统计）：
     libai_npucore_elementary.so       1733 处量化相关
     libai_npucore_ascendc.so          4373 处量化相关
  具体符号举例：
     TransFilterConvForInt8 · TransFixpipeUtilInt8 ·
     TransTensorNDToFractalNZNInt8 · UpdateBias_WeightInt8_Gen ·
     UpdateDeqBias_WeightInt8_Gen · PreProcessingMeanVar<npucl::tagFp16> ·
     TransFilterConvToFp16 / TransFilterLBConvToFp16 / npucl::tagFp16 全套 ✓
✓ 另一侧证据：OMG 也曾明确报错 "FC do not support UINT8 weight on this chip version" ✓
     ⇒ 说明 int8 权重是**这条路径上的常规选项** ✓（只是 uint8 权重不支持 ✓）
✗ 但 MS-Lite 的 `--weight_data_type` 只支持 FP16/FP32 ✗
  ⇒ 真要上 int8，得走 ★dopt 量化（校准）★ 那条链路 ✓（见 §6：卡在校准数据格式 ✓）
```

### 9.5 fp8：**没有证据** ✗

```
· 映射表只有 TypeId 30..44 那 15 项，没有 fp8 ✓
· 运行库/kernel 库里没有 fp8 / hif8 / e4m3 / e5m2 的痕迹 ✓
  （只在 libai_npucore_ascendc.so 里见到一处 BF16 ✓）
⇒ ★fp8 在这套栈上不存在★ ✓
```

### 9.6 fp16 的 kernel 为什么在 CPU 侧？

```
libmindspore-lite.so 里所有 fp16 kernel 的源码路径都是：
  litert/kernel/★cpu★/fp16/*.cc（convolution_fp16.cc、fullconnection_fp16.cc、cast_fp16.cc …）
⇒ ★MS-Lite 自带的 fp16 kernel 全在 CPU 后端★ ✓
   ★NNRt 侧不是"逐算子 kernel"，而是★把整段子图交给 NNRt 跑（NNRtModel Kernel）★ ✓：
     NNRTWrapper::GetInstance / LoadLibrary / IsSupportAIPP ·
     CastToNNRtDataType / CastToNNRtFormat ·
     字符串 "Schedule NNRt kernel failed:" · "Running NNRtModel Kernel..." ✓
```

### 9.7 ★必须先分清的两件事（很容易被混为一谈 ✗）★

| | A. **fp16 的"图"** | B. **`enableFP16` 开关** |
|---|---|---|
| 做什么 | 把权重/激活/IO 都导成 fp16 | 图保持 fp32 ✓，只是**允许 NPU 内部用 fp16 算** |
| 设备结果 | ★**跑不起来**✗★（大图 Build -1 · 小图 SIGSEGV） | ★**功能正常**✓★（Build 0 · Predict 0 都成功 ✓） |
| 性能 | — | ★**没有提升**✗★（开关前后 Build/Predict 完全一致 ✓） |

* §8.2/§8.4 说的"FP16 走不通"指的是 **A** ✓
* **B 是后来才发现的第二条路** ✓ —— 它由源码 `nnrt_delegate.cc:717` 的
  `enable_fp16_` → `OH_NNCompilation_EnableFloat16` 而来 ✓
* 已接进 runner（默认开 ✓，`CANN_LLM_NO_FP16=1` 可关 ✓）

**B 的实测**（预热页缓存后交叉测 4 次，排除冷热干扰 ✓）：

```
enableFP16=False ⇒ Build 1.3s · Predict 0.63 / 0.57 / 0.58s
enableFP16=True  ⇒ Build 1.3s · Predict 0.62 / 0.58 / 0.57s
⇒ ★功能正常，但没有性能提升★ ✗（这张图 / 这台设备上 NPU 内部并没有因此变快 ✓）
★ 教训：我先前看到的 "Build 4.0s → 1.3s" 是【页面缓存假象】✗
  （先跑 False 冷、再跑 True 热 ⇒ 把冷热当成开关效果 ✗）—— 对照实验必须消除冷热变量 ✓
```

## 10. ★多轮对话的真实开销（用户最关心的指标）★

**为什么关心它**：单轮里"加载"占大头（§8），但**加载每个进程只付一次** ✓
⇒ 一旦进入多轮，真正决定体验的就是**每轮耗时** ✓。

实测（同一进程连续 3 轮，`CANN_LLM_TIMING=1` ✓，单进程 ✓）：

```
轮1  95.9s  = 首次加载 35.3s + 推理 57.1s
轮2  48.9s  （加载累计仍是 35.3s ⇒ ★只加载了一次 ✓ 模型缓存有效 ✓★）
轮3  50.3s
⇒ ★每轮 ~49 秒，几乎全是推理，而且【不随上下文增长】★ ✓
  （因为每次前向都 padding 到 S=128 ⇒ 计算量恒定 ✓；上下文超 128 才会变 ✗）
```

**逐段明细（第 3 轮）**：

```
seg16=15.2s  seg24=11.8s  seg30=9.5s  seg8=4.1s  seg0=3.7s   ⇒ 段合计 ≈44s（90%）
graphP=4.5s（9%）  lm=0.4s（1%）
```

### 10.1 ★这里藏着一个 20~30 倍的异常★

```
★★ 探针直接调 NDK 测同一张图（mseg0，594MB 预热后）：
     → ★Predict 只要 0.6 秒★ ✓
★★ 而 runner 里同一段要 ★3.3 ~ 15.2 秒★ ✗✗ —— 差 20~30 倍
⇒ ⇒ ★★ 所以每轮那 49 秒【不是 NPU 在算】，而是【runner 在喂数据】★★ ✓✓
★ 具体嫌疑：`forward` 里为**每个段、每个 token、每个 per-layer 输入**做纯 Python 的
  切片与拼接 ✓（`b"".join(per_layer[off + …] for t in range(S))` ✓，S=128 ✓）
   ⇒ ★O(段数 × 层数 × S) 的 Python 循环，每轮重复一遍★ ✗
⇒ ⇒ ★★ 这是纯主机侧的活，且可以大幅优化（缓存/预拼一次复用）——
   ★这才是"多轮 token 产生速度"的真正瓶颈★★ ✓✓
```

**§8.3 里"真正的杠杆是别重付加载费"的说法需要修正** ✗：
加载确实只付一次 ✓，但**每轮 49 秒的瓶颈在主机侧喂数据** ✗，不在加载、也不在 NPU ✓。
