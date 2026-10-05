#!/bin/bash
# 逐算子 tiny 探针：每个构造一个小模型 ⇒ 导出(legacy) ⇒ OMG(同流水线) ⇒ 记 rc
set -u
cd ~/q38 || exit 9
mkdir -p probes && cd probes
~/q38env/bin/python - <<'PY'
import torch, torch.nn as nn, os
torch.set_grad_enabled(False)
H, S, B = 64, 8, 1
class P(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.kind = kind
        s.w = nn.Parameter(torch.randn(H, H) * 0.05)
        s.w2 = nn.Parameter(torch.randn(2 * H, H) * 0.05)
        s.conv = nn.Conv1d(H, H, 4, padding=0, groups=1)
        s.emb = nn.Parameter(torch.randn(100, H))
    def forward(s, x):
        k = s.kind
        if k == "log_min":                       # Softplus 的 lowering ✓
            y = torch.log(1.0 + torch.exp(torch.minimum(x, torch.tensor(20.0))))
        elif k == "expand":                      # ConstantOfShape 的 lowering ✓
            m = torch.ones(1, 1, S, dtype=x.dtype, device=x.device)
            y = x * m.expand(B, 1, S).transpose(0, 2).reshape(B, S, 1)
        elif k == "gather":                      # 3 维化换序 ✓
            idx = torch.arange(H - 1, -1, -1, dtype=torch.int64)
            y = x.index_select(2, idx)
        elif k == "slice1d":                     # 1 维切片 ✓
            flat = x.reshape(-1)
            y = flat[: H].reshape(B, 1, H) * 1.0
        elif k == "slice3d":                     # 3 维切片 ✓
            y = x[:, :, : H // 2]
            y = torch.cat([y, y], dim=2)
        elif k == "concat":                      # Concat 窗口拼接 ✓
            y = torch.cat([x, x], dim=2)[:, :, :S] if False else torch.cat([x[:, :, : S // 2], x[:, :, S // 2:]], dim=2)
        elif k == "where":                       # 掩码选择 ✓
            mask = (x > 0)
            y = torch.where(mask, x, torch.zeros_like(x))
        elif k == "conv1d":                      # 卷积 ✓
            y = s.conv(x.transpose(1, 2)).transpose(1, 2)
            y = torch.nn.functional.pad(y, (0, S - y.shape[2]))
        elif k == "4d":                          # 4 维乘加（rec 状态 ✓）
            st = torch.zeros(B, 4, 8, 8)
            y = (x[:, :, :32].reshape(B, 4, 8) @ st.reshape(B, 4, 8, 8).sum(2).unsqueeze(-1)).reshape(B, 32)[:, None, :]
            y = torch.nn.functional.pad(y, (0, H - 32))
        elif k == "softplus":                    # 已知不支持 ✗（对照组 ✓）
            y = torch.nn.functional.softplus(x)
        else:
            y = x
        return y @ s.w
for kind in ["plain", "log_min", "expand", "gather", "slice1d", "slice3d", "concat",
             "where", "conv1d", "4d", "softplus"]:
    try:
        torch.onnx.export(P(kind).eval(), (torch.randn(B, S, H),), "p_%s.onnx" % kind,
                          input_names=["input_embed"], output_names=["hidden_states"],
                          opset_version=14, dynamo=False)
        print("  导出 %-9s ✓" % kind)
    except Exception as e:
        print("  导出 %-9s ✗ %s" % (kind, str(e)[:60]))
PY
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
printf "%-10s %-6s\n" "探针" "OMG"
for f in p_*.onnx; do
  k=$(basename "$f" .onnx | sed 's/^p_//')
  rm -rf out_$k && mkdir -p out_$k && rm -f check_result.json
  timeout 600 $D/tools/tools_omg/omg --model "$f" --framework 5 --output out_$k/seg \
    --input_shape="input_embed:1,8,64" --input_type="input_embed:FP32" --output_type="hidden_states:FP32" \
    --weight_data_type FP16 --save_weights_as_external_data=true \
    --platform=kirinx90 --target=omc > out_$k/omg.log 2>&1
  printf "%-10s %-6s\n" "$k" "$(grep -ac 'OMG generate offline model success' out_$k/omg.log)"
done
