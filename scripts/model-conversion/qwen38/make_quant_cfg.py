"""从 ONNX 生成 converter_lite 的 int8 量化配置（IO 规格 + W8 权重 ✓）。"""
import sys
import onnx
from onnx import TensorProto

T2 = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16",
      TensorProto.INT32: "int32", TensorProto.INT64: "int64", TensorProto.BOOL: "bool"}


def dims(t):
    return ",".join(str(d.dim_value) if d.HasField("dim_value") else "-1" for d in t.type.tensor_type.shape.dim)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    g = onnx.load(src, load_external_data=False).graph
    ins = [(i.name, T2.get(i.type.tensor_type.elem_type, "float32"), dims(i)) for i in g.input]
    outs = [(o.name, T2.get(o.type.tensor_type.elem_type, "float32"), dims(o)) for o in g.output]
    with open(dst, "w") as f:
        f.write("[common_quant_param]\nquant_type=WEIGHT_QUANT\nbit_num=8\n"
                "min_quant_weight_size=0\nmin_quant_weight_channel=1\n\n[third_party_model]\n")
        f.write("input_names=" + ";".join(n for n, _, _ in ins) + "\n")
        f.write("input_dtypes=" + ";".join(d for _, d, _ in ins) + "\n")
        f.write("input_shapes=" + ";".join(s for _, _, s in ins) + "\n")
        f.write("output_names=" + ";".join(n for n, _, _ in outs) + "\n")
        f.write("output_dtypes=" + ";".join(d for _, d, _ in outs) + "\n")
        f.write("output_shapes=" + ";".join(s for _, _, s in outs) + "\n")
    print("已写 %s" % dst)
    for n, d, s in ins:
        print("  in  %-14s %-8s %s" % (n, d, s))
    for n, d, s in outs:
        print("  out %-14s %-8s %s" % (n, d, s))


if __name__ == "__main__":
    main()
