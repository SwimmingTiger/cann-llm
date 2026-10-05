"""对 hiai 导出的 onnx 做图级 lowering，并存成【单个外置权重文件】✓。

用法：~/q38env/bin/python lower_hiai.py <in.onnx> <out.onnx> [权重文件名]
"""
import os, sys
import onnx
import onnx_lower
from onnx_lower import fix_mixed_dtypes

src, dst = sys.argv[1], sys.argv[2]
wfile = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(dst) + ".weights"
m = onnx.load(src)                       # ★会加载外置数据★ ✓
print("  lowering:", onnx_lower.lower_model(m))
# ★类型一致性修正（§108 ✓）★：修掉"混合类型的算术算子"（如 Concat(int64, float32) ✗）
#   ⇒ ORT 能加载 ✓ 且 DDK 不再按错误类型分配内存 ✓
_nf = fix_mixed_dtypes(m)
print("  类型修正：%d 处 int 输入已 Cast→FP32 ✓" % _nf)
onnx_lower.fix_static_shapes(m, verbose=False)
if os.path.exists(dst):
    os.unlink(dst)
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=wfile, size_threshold=1024)
print("  已写 %s（外置 %s ✓）" % (dst, wfile))
