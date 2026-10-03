"""跑【单个】段：读 hidden(+KV) → Predict → 写 hidden_out(+KV)。
每段一个独立进程（照已跑通的最小写法：按【序号】喂输入、不调 destroy ✓）。
用法: python3 g4_seg_run.py <root> <io> <start>
"""
import ctypes as C, os, struct, sys

LIB = "/system/lib64/ndk/libmindspore_lite_ndk.so"
lib = C.CDLL(LIB)
class TA(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]
lib.OH_AI_ModelCreate.restype = C.c_void_p
lib.OH_AI_ContextCreate.restype = C.c_void_p
lib.OH_AI_DeviceInfoCreate.restype = C.c_void_p
lib.OH_AI_DeviceInfoCreate.argtypes = [C.c_int]
lib.OH_AI_ContextAddDeviceInfo.argtypes = [C.c_void_p, C.c_void_p]
lib.OH_AI_ModelBuildFromFile.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p]
lib.OH_AI_ModelBuildFromFile.restype = C.c_int
lib.OH_AI_ModelGetInputs.argtypes = [C.c_void_p];  lib.OH_AI_ModelGetInputs.restype = TA
lib.OH_AI_TensorGetMutableData.argtypes = [C.c_void_p]; lib.OH_AI_TensorGetMutableData.restype = C.c_void_p
lib.OH_AI_TensorGetElementNum.argtypes = [C.c_void_p];  lib.OH_AI_TensorGetElementNum.restype = C.c_size_t
lib.OH_AI_TensorGetName.argtypes = [C.c_void_p]; lib.OH_AI_TensorGetName.restype = C.c_char_p
lib.OH_AI_ModelPredict.argtypes = [C.c_void_p, TA, C.POINTER(TA), C.c_void_p, C.c_void_p]
lib.OH_AI_ModelPredict.restype = C.c_int

ROOT, IO, ST = sys.argv[1], sys.argv[2], int(sys.argv[3])
no = 4 if ST < 32 else 3
ctx = lib.OH_AI_ContextCreate()
lib.OH_AI_ContextAddDeviceInfo(ctx, lib.OH_AI_DeviceInfoCreate(60))
m = lib.OH_AI_ModelCreate()
st = lib.OH_AI_ModelBuildFromFile(m, os.path.join(ROOT, "seg%d" % ST, "seg.ms").encode(), 0, ctx)
print("  [%2d] Build -> %d" % (ST, st), flush=True)
if st != 0: sys.exit(1)
ins = lib.OH_AI_ModelGetInputs(m)
kvbuf = {}
def get(name):
    for cand in (os.path.join(IO, "state_" + name + ".bin"), os.path.join(IO, name + ".bin"),
                 os.path.join(IO, "pl_%d_%s.bin" % (ST, name.split("_")[-1]))):
        if os.path.exists(cand): return open(cand, "rb").read()
    return None
for i in range(ins.handle_num):
    t = ins.handle_list[i]
    nm = lib.OH_AI_TensorGetName(t).decode()
    n = lib.OH_AI_TensorGetElementNum(t)
    p = lib.OH_AI_TensorGetMutableData(t)
    if nm == "hidden":
        d = open(os.path.join(IO, "state_hidden.bin"), "rb").read()
    elif nm.startswith("per_layer_"):
        d = open(os.path.join(IO, "pl_%d_%s.bin" % (ST, nm.split("_")[-1])), "rb").read()
    else:
        # ★ KV 槽等状态文件优先用 state_ 前缀（上游段产出的）✓
        cand = os.path.join(IO, "state_" + nm + ".bin")
        d = open(cand if os.path.exists(cand) else os.path.join(IO, nm + ".bin"), "rb").read()
    assert len(d) == n * 4, "%s: %d vs %d" % (nm, len(d), n * 4)
    C.memmove(p, d, len(d))
outs = TA()
st = lib.OH_AI_ModelPredict(m, ins, C.byref(outs), None, None)
print("  [%2d] Predict -> %d" % (ST, st), flush=True)
if st != 0: sys.exit(1)
for i in range(outs.handle_num):
    t = outs.handle_list[i]
    nm = lib.OH_AI_TensorGetName(t).decode()
    n = lib.OH_AI_TensorGetElementNum(t)
    d = C.string_at(lib.OH_AI_TensorGetMutableData(t), n * 4)
    name = "state_hidden.bin" if nm == "hidden_out" else ("state_" + nm.replace("_out", "") + ".bin")
    open(os.path.join(IO, name), "wb").write(d)
    print("     %-10s -> %s (%d 字节)" % (nm, name, len(d)), flush=True)
