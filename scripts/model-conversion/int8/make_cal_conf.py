#!/usr/bin/env python3
"""按 ONNX 的输入，生成 dopt 的 cal_conf.prototxt。

★三条硬规则（都是实测踩出来的）★：
  ① ★每个输入写一个 `preprocess_parameter { … }` 块★ —— 不能在一个块里重复
     `input_file_path` 键 ✗（protobuf 字典会互相覆盖 ⇒ 报
     "Input nodes number does not match preprocess parameters!"）
  ② `input_type` ★必须显式写 BINARY★ —— 默认值/IMAGE 会拒 bin ✗
     （报 "Binary calibration set is invalid when input_type is set to Image or default!"）
  ③ `input_file_path` ★必须绝对路径★ —— 相对路径会以 dopt 自己的工作目录为基准 ✗

用法：
    python make_cal_conf.py model.onnx <bin 目录> <输出 prototxt>
"""
import os
import sys

import onnx


def main():
    model, bindir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    bindir = os.path.abspath(bindir)
    m = onnx.load(model, load_external_data=False)
    lines = ["strategy: 'Quant_INT8-8'", "device: USE_CPU"]
    for i in m.graph.input:
        name = i.name
        lines += [
            "preprocess_parameter:",
            "{",
            "    input_type: BINARY",
            '    input_file_path: "%s"' % os.path.join(bindir, name + ".bin"),
            "}",
        ]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  写入 %s（%d 个 preprocess_parameter 块 ✓）" % (out, len(m.graph.input)))


if __name__ == "__main__":
    main()
