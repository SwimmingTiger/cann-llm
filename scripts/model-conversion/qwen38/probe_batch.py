"""批量试建：把一批 .ms 逐个 Build + Predict，报告 rc ✓（用于定位哪个算子被 NPU 拒收 ✗）。

用法（设备上）：
    LD_LIBRARY_PATH=/system/lib64/ndk:/system/lib64/platformsdk python3 probe_batch.py <目录>
"""
import ctypes as C
import glob
import os
import sys

_NDK = "/system/lib64/ndk/libmindspore_lite_ndk.so"


class _TA(C.Structure):
    _fields_ = [("handle_num", C.c_size_t), ("handle_list", C.POINTER(C.c_void_p))]


def bind(lib):
    for fn, rt, at in [
        ("OH_AI_ModelCreate", C.c_void_p, []),
        ("OH_AI_ContextCreate", C.c_void_p, []),
        ("OH_AI_DeviceInfoCreate", C.c_void_p, [C.c_int]),
        ("OH_AI_DeviceInfoSetEnableFP16", None, [C.c_void_p, C.c_bool]),
        ("OH_AI_ContextAddDeviceInfo", None, [C.c_void_p, C.c_void_p]),
        ("OH_AI_ModelBuildFromFile", C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p]),
        ("OH_AI_ModelGetInputs", _TA, [C.c_void_p]),
        ("OH_AI_ModelGetOutputs", _TA, [C.c_void_p]),
        ("OH_AI_TensorGetMutableData", C.c_void_p, [C.c_void_p]),
        ("OH_AI_TensorGetDataSize", C.c_size_t, [C.c_void_p]),
        ("OH_AI_ModelPredict", C.c_int, [C.c_void_p, _TA, C.POINTER(_TA), C.c_void_p, C.c_void_p]),
        # ★NNRt device id 必须显式设置★（§41 实测：int8 模型不设就 Build -1 ✗）
        ("OH_AI_GetAllNNRTDeviceDescs", C.c_void_p, [C.POINTER(C.c_size_t)]),
        ("OH_AI_GetElementOfNNRTDeviceDescs", C.c_void_p, [C.c_void_p, C.c_size_t]),
        ("OH_AI_GetNameFromNNRTDeviceDesc", C.c_char_p, [C.c_void_p]),
        ("OH_AI_GetDeviceIdFromNNRTDeviceDesc", C.c_size_t, [C.c_void_p]),
        ("OH_AI_DeviceInfoSetDeviceId", None, [C.c_void_p, C.c_size_t]),
    ]:
        f = getattr(lib, fn)
        if rt:
            f.restype = rt
        if at:
            f.argtypes = at


def npu_device_id(lib):
    num = C.c_size_t(0)
    descs = lib.OH_AI_GetAllNNRTDeviceDescs(C.byref(num))
    for i in range(num.value):
        d = lib.OH_AI_GetElementOfNNRTDeviceDescs(descs, i)
        nm = lib.OH_AI_GetNameFromNNRTDeviceDesc(d)
        if nm and nm.startswith(b"NPU_"):
            return lib.OH_AI_GetDeviceIdFromNNRTDeviceDesc(d), nm.decode()
    return None, None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "*.ms")))
    if not files:
        print("没有 .ms ✗")
        return 1
    lib = C.CDLL(_NDK)
    bind(lib)
    did, dname = npu_device_id(lib)
    print("设备: %s (id=%s)" % (dname, did))
    ok = fail = 0
    print("%-22s %-8s %-8s %s" % ("模型", "Build", "Predict", "输入数"))
    for f in files:
        ctx = lib.OH_AI_ContextCreate()
        dev = lib.OH_AI_DeviceInfoCreate(60)
        lib.OH_AI_DeviceInfoSetEnableFP16(dev, C.c_bool(True))
        if did is not None:
            lib.OH_AI_DeviceInfoSetDeviceId(dev, did)
        lib.OH_AI_ContextAddDeviceInfo(ctx, dev)
        m = lib.OH_AI_ModelCreate()
        rc_b = lib.OH_AI_ModelBuildFromFile(m, f.encode(), 0, ctx)
        n_in = 0
        rc_p = -99
        if rc_b == 0:
            ins = lib.OH_AI_ModelGetInputs(m)
            n_in = ins.handle_num
            for i in range(n_in):                     # 输入清零 ✓
                ds = lib.OH_AI_TensorGetDataSize(ins.handle_list[i])
                p = lib.OH_AI_TensorGetMutableData(ins.handle_list[i])
                C.memset(p, 0, ds)
            outs = lib.OH_AI_ModelGetOutputs(m)
            rc_p = lib.OH_AI_ModelPredict(m, ins, C.byref(outs), ctx, None)
        name = os.path.basename(f)
        flag = "✓" if (rc_b == 0 and rc_p == 0) else "✗"
        print("%-22s %-8d %-8d %-4d %s" % (name, rc_b, rc_p, n_in, flag))
        if rc_b == 0 and rc_p == 0:
            ok += 1
        else:
            fail += 1
    print("\n通过 %d · 失败 %d" % (ok, fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
