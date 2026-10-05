"""对 hiai 导出的 onnx 做图级 lowering，并存成【单个外置权重文件】✓。

用法：~/q38env/bin/python lower_hiai.py <in.onnx> <out.onnx> [权重文件名]
"""
import os, sys
import onnx
import onnx_lower

src, dst = sys.argv[1], sys.argv[2]
wfile = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(dst) + ".weights"
m = onnx.load(src)                       # ★会加载外置数据★ ✓
print("  lowering:", onnx_lower.lower_model(m))
onnx_lower.fix_static_shapes(m, verbose=False)
if os.path.exists(dst):
    os.unlink(dst)
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
          location=wfile, size_threshold=1024)
print("  已写 %s（外置 %s ✓）" % (dst, wfile))
