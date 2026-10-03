"""设备端对拍：用真实输入跑 g4seg3d.ms，与 HF 的 layer3 参考输出比较。"""
import ctypes as C, struct, sys, os

LIB = "/system/lib64/ndk/libmindspore_lite_ndk.so"
D   = os.environ.get("MODEL_DIR", "")
IO  = os.path.join(D, "io")
lib = C.CDLL(LIB)

class TensorArray(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]

lib.OH_AI_ModelCreate.restype = C.c_void_p
lib.OH_AI_ContextCreate.restype = C.c_void_p
lib.OH_AI_DeviceInfoCreate.restype = C.c_void_p
lib.OH_AI_DeviceInfoCreate.argtypes = [C.c_int]
lib.OH_AI_ContextAddDeviceInfo.argtypes = [C.c_void_p, C.c_void_p]
lib.OH_AI_DeviceInfoSetProvider.argtypes = [C.c_void_p, C.c_char_p]
lib.OH_AI_ModelBuildFromFile.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p]
lib.OH_AI_ModelBuildFromFile.restype = C.c_int
lib.OH_AI_ModelGetInputs.argtypes = [C.c_void_p];  lib.OH_AI_ModelGetInputs.restype = TensorArray
lib.OH_AI_ModelGetOutputs.argtypes = [C.c_void_p]; lib.OH_AI_ModelGetOutputs.restype = TensorArray
lib.OH_AI_TensorGetMutableData.argtypes = [C.c_void_p]; lib.OH_AI_TensorGetMutableData.restype = C.c_void_p
lib.OH_AI_TensorGetElementNum.argtypes = [C.c_void_p];  lib.OH_AI_TensorGetElementNum.restype = C.c_size_t
lib.OH_AI_TensorGetShape.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]; lib.OH_AI_TensorGetShape.restype = C.POINTER(C.c_int64)
lib.OH_AI_ModelPredict.argtypes = [C.c_void_p, TensorArray, C.POINTER(TensorArray), C.c_void_p, C.c_void_p]
lib.OH_AI_ModelPredict.restype = C.c_int

dev = lib.OH_AI_DeviceInfoCreate(60)                 # OH_AI_DEVICETYPE_NNRT = 60
ctx = lib.OH_AI_ContextCreate(); lib.OH_AI_ContextAddDeviceInfo(ctx, dev)
m = lib.OH_AI_ModelCreate()
st = lib.OH_AI_ModelBuildFromFile(m, os.path.join(D, "g4seg3d.ms").encode(), 0, ctx)  # MINDIR=0
print("  Build -> %d" % st, flush=True)

ins = lib.OH_AI_ModelGetInputs(m)
NAMES = ["hidden", "cos", "sin", "mask3", "per_layer_0", "per_layer_1", "per_layer_2", "per_layer_3"]
for i in range(ins.handle_num):
    t = ins.handle_list[i]
    n = lib.OH_AI_TensorGetElementNum(t)
    p = lib.OH_AI_TensorGetMutableData(t)
    f = os.path.join(IO, NAMES[i] + ".bin")
    data = open(f, "rb").read()
    assert len(data) == n * 4, "%s: 字节数 %d vs 元素 %d" % (NAMES[i], len(data), n)
    C.memmove(p, data, len(data))
    print("   %-12s 元素 %5d ← %s" % (NAMES[i], n, os.path.basename(f)), flush=True)

outs = TensorArray()
st = lib.OH_AI_ModelPredict(m, ins, C.byref(outs), None, None)   # 5 参
print("  ★ Predict -> %d" % st, flush=True)
if st != 0:
    sys.exit(1)

ot = outs.handle_list[0]
n = lib.OH_AI_TensorGetElementNum(ot)
nd = C.c_size_t(0); sp = lib.OH_AI_TensorGetShape(ot, C.byref(nd))
shape = [sp[i] for i in range(nd.value)]
buf = C.string_at(lib.OH_AI_TensorGetMutableData(ot), n * 4)
dev_out = struct.unpack("<%df" % n, buf)
ref = struct.unpack("<%df" % n, open(os.path.join(IO, "ref_layer3.bin"), "rb").read())
print("  输出 shape=%s 元素=%d" % (shape, n), flush=True)
print("  设备前 6 个: %s" % [round(x, 4) for x in dev_out[:6]], flush=True)
print("  参考前 6 个: %s" % [round(x, 4) for x in ref[:6]], flush=True)
mx = max(abs(a - b) for a, b in zip(dev_out, ref))
rel = mx / (max(abs(x) for x in ref) + 1e-9)
print("  ★★ 最大绝对差 = %.4e · 相对最大 = %.4e ⇒ %s" %
      (mx, rel, "★★ 对拍一致！★★" if rel < 5e-2 else "✗ 有偏差，要查"), flush=True)
