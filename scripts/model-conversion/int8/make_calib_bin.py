#!/usr/bin/env python3
"""把校准数据写成 dopt 认的 .bin 格式。

★格式来自官方示例★（tools_dopt/dopt_pytorch_py3/demo/quant8-8/notrain/bin_data_preprocessing.py）：

    magic = 510（4 维张量）：
        <magic 4B LE> + <每维 shape 4B LE × 4> + tensor.tobytes()
    magic = 610（非 4 维）：
        <magic 4B LE> + <rank 4B LE> + <每维 shape 4B LE × rank> + tensor.tobytes()

★没有这个头部的裸数据会被拒，报 `Invalid bin file!` ✗★ —— 这是最容易踩的坑 ✓。

用法（Python）：
    from make_calib_bin import dump_tensor
    dump_tensor(arr, [1, 128, 1536], "hidden.bin")     # arr 是 numpy float32 数组
"""
import struct


def dump_tensor(tensor, shape, path):
    """tensor: numpy float32 数组（会先 ascontiguousarray 保证内存连续 ✓）"""
    import numpy as np
    a = np.ascontiguousarray(tensor, dtype=np.float32)
    with open(path, "wb") as f:
        if len(shape) == 4:
            f.write(struct.pack("<I", 510))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
        else:
            f.write(struct.pack("<I", 610))
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
        f.write(a.tobytes())
    return path


def expected_size(shape):
    """该 shape 的 .bin 应有的字节数（便于自检 ✓）"""
    hdr = 4 + 4 * len(shape) if len(shape) != 4 else 4 + 16
    n = 1
    for d in shape:
        n *= int(d)
    return hdr + n * 4
