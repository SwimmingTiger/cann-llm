// ★参数化算子探针★（结构照抄 §127 能跑通的 nnrt_online.cpp ✓）
// 用法: ./ap <op号> <mode>
//   mode 0 = 不给参数张量        （params = nullptr）
//   mode 1 = 给 1 个 FLOAT 参数张量（type 由 -DPARAM_TYPE 指定 ✓）
//   mode 2 = 给 1 个 INT32 参数张量
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

int main(int argc, char **argv) {
    if (argc < 3) { printf("用法: ap <op号> <mode 0|1|2>\n"); return 1; }
    OH_NN_OperationType op = (OH_NN_OperationType)atoi(argv[1]);
    int mode = atoi(argv[2]);
    size_t npu = 0; const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) { const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        if (n && strstr(n, "NPU")) npu = ids[i]; }

    static int32_t d3[3] = {1, 64, 64};
    static OH_NN_Tensor tx, tw, ty, tp;
    tx.dataType = OH_NN_FLOAT32; tx.dimensionCount = 3; tx.dimensions = d3;
    tx.quantParam = nullptr; tx.type = OH_NN_TENSOR;
    tw = tx; ty = tx;
    OH_NNModel *m = OH_NNModel_Construct();
    int rx = OH_NNModel_AddTensor(m, &tx);
    int rw = OH_NNModel_AddTensor(m, &tw);
    int ry = OH_NNModel_AddTensor(m, &ty);
    static float W[1 * 64 * 64], X[1 * 64 * 64], Y[1 * 64 * 64];
    for (int i = 0; i < 64 * 64; ++i) { W[i] = 1.0f; X[i] = 0.5f; }
    OH_NNModel_SetTensorData(m, 1, W, sizeof(W));

    uint32_t prm = 0;
    if (mode > 0) {
        tp.dataType = (mode == 1) ? OH_NN_FLOAT32 : OH_NN_INT32;
        tp.dimensionCount = 0; tp.dimensions = nullptr; tp.quantParam = nullptr;
        tp.type = (OH_NN_TensorType)PARAM_TYPE;
        int rp = OH_NNModel_AddTensor(m, &tp);
        static int32_t iv = 0; static float fv = 0.0f;
        if (mode == 1) OH_NNModel_SetTensorData(m, 3, &fv, sizeof(fv));
        else           OH_NNModel_SetTensorData(m, 3, &iv, sizeof(iv));
        printf("  AddTensor x=%d w=%d y=%d param=%d\n", rx, rw, ry, rp);
        prm = 3;
    } else {
        printf("  AddTensor x=%d w=%d y=%d (无参数)\n", rx, rw, ry);
    }
    uint32_t ins[2] = {0, 1}, outs[1] = {2};
    OH_NN_UInt32Array I, O, P; I.data = ins; I.size = 2; O.data = outs; O.size = 1;
    uint32_t pv[1] = {prm}; P.data = pv; P.size = (mode > 0) ? 1 : 0;
    int ra = OH_NNModel_AddOperation(m, op, &P, &I, &O);
    uint32_t mi = 0, mo = 2;
    OH_NN_UInt32Array MI, MO; MI.data = &mi; MI.size = 1; MO.data = &mo; MO.size = 1;
    int rs = OH_NNModel_SpecifyInputsAndOutputs(m, &MI, &MO);
    int rf = OH_NNModel_Finish(m);
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    int rb = c ? OH_NNCompilation_Build(c) : -1;
    printf("  op=%d mode=%d AddOp=%d Specify=%d Finish=%d ★Build=%d★ %s\n",
           (int)op, mode, ra, rs, rf, rb, (ra == 0 && rf == 0 && rb == 0) ? "★可用★" : "✗");
    return 0;
}
