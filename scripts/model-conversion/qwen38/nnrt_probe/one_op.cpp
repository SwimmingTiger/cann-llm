// ★单算子 NPU 编译测试★（干净版 ✓）用法：./one_op <op枚举号> [名字]
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "neural_network_runtime.h"
#include "neural_network_core.h"
int main(int argc, char **argv) {
    if (argc < 2) { printf("用法: one_op <op号> [名字]\n"); return 1; }
    OH_NN_OperationType op = (OH_NN_OperationType)atoi(argv[1]);
    const char *nm = argc > 2 ? argv[2] : "op";
    size_t npu = 0; const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        if (n && strstr(n, "NPU")) npu = ids[i];
    }
    OH_NNModel *m = OH_NNModel_Construct();
    static int32_t d3[3] = {1, 64, 64};
    static OH_NN_Tensor tx, ty;                       // ★持久化✓★
    tx.dataType = OH_NN_FLOAT32; tx.dimensionCount = 3; tx.dimensions = d3;
    tx.quantParam = nullptr; tx.type = OH_NN_TENSOR;
    ty = tx;
    int ax = OH_NNModel_AddTensor(m, &tx), ay = OH_NNModel_AddTensor(m, &ty);
    uint32_t in0 = 0, out0 = 1;
    OH_NN_UInt32Array IN, OUT; IN.data = &in0; IN.size = 1; OUT.data = &out0; OUT.size = 1;
    int ra = OH_NNModel_AddOperation(m, op, nullptr, &IN, &OUT);
    uint32_t mi0 = 0, mo0 = 1;
    OH_NN_UInt32Array MI, MO; MI.data = &mi0; MI.size = 1; MO.data = &mo0; MO.size = 1;
    int rs = OH_NNModel_SpecifyInputsAndOutputs(m, &MI, &MO);
    int rf = OH_NNModel_Finish(m);
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    int rd = c ? OH_NNCompilation_SetDevice(c, npu) : -1;
    int rb = c ? OH_NNCompilation_Build(c) : -1;
    printf("  %-12s(op=%d) AddTensor=%d,%d AddOp=%d Specify=%d Finish=%d SetDev=%d ★Build=%d★ %s\n",
           nm, (int)op, ax, ay, ra, rs, rf, rd, rb, (rf == 0 && rb == 0) ? "★可用★" : "✗");
    return 0;
}
