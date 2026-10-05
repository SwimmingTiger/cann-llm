#!/bin/bash
# 外形探针：状态进出 / 多进多出 / 我们模型的尺寸 ⇒ 看 Init rc
set -u
cd ~/q38 || exit 9
mkdir -p shp && cd shp
~/q38env/bin/python - <<'PY'
import torch, torch.nn as nn
torch.set_grad_enabled(False)

class S1(nn.Module):          # 1 个状态进出（in→out 同名不同 ✓）
    def __init__(s):
        super().__init__(); s.w = nn.Parameter(torch.randn(64, 64) * 0.05)
    def forward(s, x, state):
        return (x + state * 0.0) @ s.w, state * 1.0

class S2(nn.Module):          # 2 个状态进出（模拟 KV 形状 ✓）
    def __init__(s):
        super().__init__(); s.w = nn.Parameter(torch.randn(64, 64) * 0.05)
    def forward(s, x, k, v):
        return (x + k[:, :, :8].sum(2)[:, None, :] * 0.0 + v[:, :1, :] * 0.0) @ s.w, k * 1.0, v * 1.0

class S3(nn.Module):          # 我们模型的尺寸（hidden 2048 ✓ 大张量 ✓）
    def __init__(s):
        super().__init__(); s.w = nn.Parameter(torch.randn(2048, 2048) * 0.02)
    def forward(s, x, mask):
        return (x * mask.sum(3, keepdim=True).transpose(1, 2) * 0.0 + x) @ s.w

class S4(nn.Module):          # 8 个状态进出（模拟 4 层 ✓）
    def __init__(s):
        super().__init__(); s.w = nn.Parameter(torch.randn(64, 64) * 0.05)
    def forward(s, x, *st):
        y = x
        for t in st:
            y = y + t[:, :, :8].sum(2)[:, None, :] * 0.0
        return (y @ s.w,) + tuple(t * 1.0 for t in st)

cases = {
 "state1": (S1(), [torch.randn(1, 8, 64), torch.randn(1, 8, 64)],
            ["input_embed", "state"], ["hidden_states", "state_out"]),
 "state2": (S2(), [torch.randn(1, 8, 64), torch.randn(8, 2, 1, 64), torch.randn(8, 2, 1, 64)],
            ["input_embed", "past_key_in0", "past_value_in0"],
            ["hidden_states", "past_key0", "past_value0"]),
 "big2048": (S3(), [torch.randn(1, 8, 2048), torch.randn(1, 1, 8, 64)],
             ["input_embed", "attention_mask"], ["hidden_states"]),
 "state8": (S4(), [torch.randn(1, 8, 64)] + [torch.randn(1, 8, 64)] * 8,
            ["input_embed"] + ["s%d" % i for i in range(8)],
            ["hidden_states"] + ["s%d_out" % i for i in range(8)]),
}
import re
for name, (m, args, ins, outs) in cases.items():
    try:
        dyn = {k: {2: "S"} for k in ins if k in ("input_embed",)}
        torch.onnx.export(m.eval(), tuple(args), "s_%s.onnx" % name, input_names=ins, output_names=outs,
                          opset_version=14, dynamo=False)
        print("  导出 %-8s ✓ 输入 %d 输出 %d" % (name, len(ins), len(outs)))
    except Exception as e:
        print("  导出 %-8s ✗ %s" % (name, str(e)[:70]))
PY
D=~/ddk; export SOC_VERSION=kirinx90
export LD_LIBRARY_PATH=$D/tools/tools_omg/master/lib64:$D/tools/platform/kirinx90/lib64
export PYTHONPATH=$D/tools/platform/kirinx90/ops/impl
for f in s_*.onnx; do
  k=$(basename "$f" .onnx | sed 's/^s_//')
  rm -rf o_$k && mkdir -p o_$k && rm -f check_result.json
  timeout 900 $D/tools/tools_omg/omg --model "$f" --framework 5 --output o_$k/seg \
    --input_shape="$(~/q38env/bin/python -c "
import onnx,sys
m=onnx.load('$f',load_external_data=False)
ps=[]
for i in m.graph.input:
    t=i.type.tensor_type
    if t.HasField('shape'):
        d=[x.dim_value for x in t.shape.dim]
        if all(v>0 for v in d): ps.append('%s:%s'%(i.name,','.join(map(str,d))))
print(';'.join(ps))")" \
    --input_type="$(~/q38env/bin/python -c "
import onnx
m=onnx.load('$f',load_external_data=False)
ts=[]
for i in m.graph.input:
    t=i.type.tensor_type
    if t.HasField('shape'):
        d=[x.dim_value for x in t.shape.dim]
        if all(v>0 for v in d): ts.append('%s:%s'%(i.name,'INT32' if t.elem_type==6 else 'FP32'))
print(';'.join(ts))")" \
    --output_type="$(~/q38env/bin/python -c "
import onnx
m=onnx.load('$f',load_external_data=False)
print(';'.join('%s:%s'%(o.name,'INT32' if o.type.tensor_type.elem_type==6 else 'FP32') for o in m.graph.output))")" \
    --weight_data_type FP16 --save_weights_as_external_data=true \
    --platform=kirinx90 --target=omc > o_$k/omg.log 2>&1
  printf "  %-10s OMG成功=%s\n" "$k" "$(grep -ac 'OMG generate offline model success' o_$k/omg.log)"
done
