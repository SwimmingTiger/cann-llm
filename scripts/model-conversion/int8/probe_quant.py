#!/usr/bin/env python3
"""探针：把量化配置作为 NNRt 扩展项传给设备，并 Build/Predict。

★这一条是"运行时注入量化配置"的完整写法★（设备端是否支持见 README 的结论 ✗）：
    OH_AI_DeviceInfoAddExtension(dev, "QuantConfigData" | "QuantBuffer", <数据>, <长度>)

★排查建议★：失败时先读 hilog（MS_LOG 不走 stderr ✓）：
    hilog -x | grep -aiE "MS_LITE|NNRt|CANN|AI_FMK|hiai"

用法：python probe_quant.py <model.ms> [compress_conf] [QuantConfigData|QuantBuffer] [device_type]
"""
import ctypes as C
import os
import sys

path = sys.argv[1]
conf = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
ext_name = (sys.argv[3] if len(sys.argv) > 3 else "QuantConfigData").encode()
dev_type = int(sys.argv[4]) if len(sys.argv) > 4 else 60   # 60=NNRt · 0=CPU

lib = C.CDLL(os.environ.get("MSLITE_LIB", "/system/lib64/ndk/libmindspore_lite_ndk.so"))
H = C.c_void_p


class TA(C.Structure):
    _fields_ = [("n", C.c_size_t), ("l", C.POINTER(C.c_void_p))]


lib.OH_AI_ModelCreate.restype = H
lib.OH_AI_ContextCreate.restype = H
lib.OH_AI_DeviceInfoCreate.restype = H
lib.OH_AI_DeviceInfoCreate.argtypes = [C.c_int]
lib.OH_AI_ContextAddDeviceInfo.argtypes = [H, H]
lib.OH_AI_DeviceInfoAddExtension.argtypes = [H, C.c_char_p, C.c_char_p, C.c_size_t]
lib.OH_AI_DeviceInfoAddExtension.restype = C.c_int
lib.OH_AI_ModelBuildFromFile.restype = C.c_int
lib.OH_AI_ModelBuildFromFile.argtypes = [H, C.c_char_p, C.c_int, H]
lib.OH_AI_ModelGetInputs.restype = TA
lib.OH_AI_ModelGetInputs.argtypes = [H]
lib.OH_AI_TensorGetDataSize.restype = C.c_size_t
lib.OH_AI_TensorGetDataSize.argtypes = [H]
lib.OH_AI_TensorGetMutableData.restype = H
lib.OH_AI_TensorGetMutableData.argtypes = [H]
lib.OH_AI_ModelPredict.restype = C.c_int
lib.OH_AI_ModelPredict.argtypes = [H, TA, C.POINTER(TA), H, H]

ctx = lib.OH_AI_ContextCreate()
dev = lib.OH_AI_DeviceInfoCreate(dev_type)
if conf and os.path.exists(conf):
    data = open(conf, "rb").read()
    r = lib.OH_AI_DeviceInfoAddExtension(dev, ext_name, data, len(data))
    print("  AddExtension(%s, %d 字节) -> %d" % (ext_name.decode(), len(data), r))
lib.OH_AI_ContextAddDeviceInfo(ctx, dev)
m = lib.OH_AI_ModelCreate()
st = lib.OH_AI_ModelBuildFromFile(m, path.encode(), 0, ctx)
if st != 0:
    print("  ★ Build -> %d ✗  （读 hilog 看原因 ✓）" % st)
    sys.exit(0)
ins = lib.OH_AI_ModelGetInputs(m)
for i in range(ins.n):
    t = ins.l[i]
    C.memset(lib.OH_AI_TensorGetMutableData(t), 0, int(lib.OH_AI_TensorGetDataSize(t)))
outs = TA()
p = lib.OH_AI_ModelPredict(m, ins, C.byref(outs), None, None)
print("  Build 0 · Predict -> %d %s" % (p, "✓" if p == 0 else "✗"))
