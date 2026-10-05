// ★交叉验证★：完全照 §127 能跑通的写法（纯 main 局部变量 ✓），逐个算子建一个模型 ✓
// 用法: ./two_ops <op号> [param枚举号] [参数个数]
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

static size_t NPU = 0;
static int run_one(int op, int ptype, int nparam, const char *nm, int pdt = 4) {  // pdt: 4=I32 1=BOOL 11=F32
    static int32_t d3[3] = {1, 64, 64};
    static OH_NN_Tensor tx, tw, ty, p1, p2;
    tx.dataType = OH_NN_FLOAT32; tx.dimensionCount = 3; tx.dimensions = d3;
    tx.quantParam = nullptr; tx.type = OH_NN_TENSOR;
    tw = tx; ty = tx;
    static float W[64 * 64];
    for (int i = 0; i < 64 * 64; ++i) W[i] = 1.0f;
    static bool fb = false;
    static int32_t iv = 0;
    OH_NNModel *m = OH_NNModel_Construct();
    int r0 = OH_NNModel_AddTensor(m, &tx);
    int r1 = OH_NNModel_AddTensor(m, &tw);
    int r2 = OH_NNModel_AddTensor(m, &ty);
    OH_NNModel_SetTensorData(m, 1, W, sizeof(W));
    uint32_t prm[2]; int np = 0;
    if (nparam >= 1) {
        p1.dataType = (OH_NN_DataType)pdt; p1.dimensionCount = 0; p1.dimensions = nullptr;
        p1.quantParam = nullptr; p1.type = (OH_NN_TensorType)ptype;
        OH_NNModel_AddTensor(m, &p1);
        if (pdt == 1) { static bool fb0 = false; OH_NNModel_SetTensorData(m, 3, &fb0, sizeof(fb0)); }
        else if (pdt == 11) { static float fv0 = 1.0f; OH_NNModel_SetTensorData(m, 3, &fv0, sizeof(fv0)); }
        else OH_NNModel_SetTensorData(m, 3, &iv, sizeof(iv));
        prm[np++] = 3;
    }
    if (nparam >= 2) {
        p2 = p1; p2.dataType = OH_NN_BOOL; p2.type = (OH_NN_TensorType)(ptype + 1);  // MATMUL 的两个都是 BOOL ✓
        OH_NNModel_AddTensor(m, &p2);
        OH_NNModel_SetTensorData(m, 4, &fb, sizeof(fb));
        prm[np++] = 4;
    }
    uint32_t ins[2] = {0, 1}, outs[1] = {2};
    OH_NN_UInt32Array P{prm, (uint32_t)np}, I{ins, 2}, O{outs, 1};
    int ra = OH_NNModel_AddOperation(m, (OH_NN_OperationType)op, &P, &I, &O);
    uint32_t mi = 0, mo = 2;
    OH_NN_UInt32Array MI{&mi, 1}, MO{&mo, 1};
    int rs = OH_NNModel_SpecifyInputsAndOutputs(m, &MI, &MO);
    int rf = OH_NNModel_Finish(m);
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    int rd = c ? OH_NNCompilation_SetDevice(c, NPU) : -1;
    int rb = c ? OH_NNCompilation_Build(c) : -1;
    printf("  %-12s op=%-3d params=%d ⇒ AddOp=%-2d AddTensor=%d,%d,%d Finish=%d ★Build=%d★ %s\n",
           nm, op, np, ra, r0, r1, r2, rf, rb, (ra == 0 && rf == 0 && rb == 0) ? "★可用★" : "✗");
    return ra;
}
int main() {
    const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) { const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        if (n && strstr(n, "NPU")) NPU = ids[i]; }
    run_one(19, 33, 2, "MATMUL", 1);            // §127 已验证 ✓ 作对照
    run_one(19, 33, 1, "MATMUL-1p", 1);
    run_one(22, 41, 1, "MUL", 4);
    run_one(22, 41, 0, "MUL-0p");
    run_one(1,  1,  1, "ADD", 4);
    run_one(1,  1,  0, "ADD-0p");
    run_one(38, 59, 1, "SUB", 4);
    run_one(11, 29, 1, "DIV", 4);
    run_one(28, 0,  0, "SIGMOID");
    run_one(60, 0,  0, "EXP");
    run_one(47, 0,  0, "RELU");
    run_one(24, 116,1, "PAD", 4);
    run_one(25, 107,2, "POW", 11);
    return 0;
}
