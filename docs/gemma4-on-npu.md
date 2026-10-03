# Gemma 4 E2B 在麒麟 NPU 上跑通并接入聊天：实测记录与性能诊断

> 面向维护者的实测记录。设备：HarmonyOS 7.0.0 aarch64 / KirinX90 / 32 GB；
> 路径：`OMG → .omc → converter_lite → .ms → 设备 NNRt`（即 `-b nnrt` 后端）。
> 模型：`google/gemma-4-E2B-it`，**只取文本部分**。

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
