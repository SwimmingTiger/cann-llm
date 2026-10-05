// ★NNRt 在线建图：极小 MatMul ⇒ 在 NPU 上编译 + 执行 + 数值校验★
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

#define B 1
#define S 64
#define H 2048

static void dump_devices(size_t *npuId, int *ok) {
    const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NN_ReturnCode rc = OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    printf("  GetAllDevicesID rc=%d count=%u\n", (int)rc, cnt);
    *ok = 0;
    for (uint32_t i = 0; i < cnt; ++i) {
        const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        printf("    [%u] id=%zu name=%s\n", i, ids[i], n ? n : "?");
        if (n && strstr(n, "NPU")) { *npuId = ids[i]; *ok = 1; }
    }
    if (!*ok && cnt) { *npuId = ids[0]; *ok = 1; }
}

int main() {
    size_t npu = 0; int ok = 0;
    dump_devices(&npu, &ok);
    if (!ok) { printf("✗ 没有可用设备\n"); return 1; }
    printf("  选用 device id=%zu\n", npu);

    OH_NNModel *m = OH_NNModel_Construct();
    int32_t dx[3] = {B, S, H}, dw[2] = {H, H}, dy[3] = {B, S, H};
    OH_NN_Tensor tx; tx.dataType = OH_NN_FLOAT32; tx.dimensionCount = 3; tx.dimensions = dx;
    tx.quantParam = nullptr; tx.type = OH_NN_TENSOR;
    OH_NN_Tensor tw = tx; tw.dimensionCount = 2; tw.dimensions = dw;
    OH_NN_Tensor ty = tx;
    printf("  AT0 dt=%d rank=%d | AT1 dt=%d rank=%d | AT2 dt=%d rank=%d\n", (int)tx.dataType, (int)tx.dimensionCount, (int)tw.dataType, (int)tw.dimensionCount, (int)ty.dataType, (int)ty.dimensionCount);
    printf("  AddTensor x=%d w=%d y=%d\n",
           (int)OH_NNModel_AddTensor(m, &tx), (int)OH_NNModel_AddTensor(m, &tw),
           (int)OH_NNModel_AddTensor(m, &ty));

    static float w[H * H];
    for (int i = 0; i < H * H; ++i) w[i] = 0.01f * ((i % 7) - 3);
    printf("  SetTensorData(w)=%d\n", (int)OH_NNModel_SetTensorData(m, 1, w, sizeof(w)));

    // ★MatMul 的两个参数张量（transposeA / transposeB ✓ 必须显式给 ✗ 传 nullptr 会 INVALID_PARAMETER ✓）
    static bool fa = false, fb = false;
    OH_NN_Tensor pa; pa.dataType = OH_NN_BOOL; pa.dimensionCount = 0; pa.dimensions = nullptr;
    pa.quantParam = nullptr; pa.type = OH_NN_MATMUL_TRANSPOSE_A;
    OH_NN_Tensor pb = pa; pb.type = OH_NN_MATMUL_TRANSPOSE_B;
    printf("  AT3 dt=%d rank=%d | AT4 dt=%d rank=%d\n", (int)pa.dataType, (int)pa.dimensionCount, (int)pb.dataType, (int)pb.dimensionCount);
    printf("  AddTensor pa=%d pb=%d\n", (int)OH_NNModel_AddTensor(m, &pa), (int)OH_NNModel_AddTensor(m, &pb));
    printf("  SetTensorData(pa)=%d pb=%d\n", (int)OH_NNModel_SetTensorData(m, 3, &fa, sizeof(fa)),
           (int)OH_NNModel_SetTensorData(m, 4, &fb, sizeof(fb)));
    uint32_t prmD[2] = {3, 4};
    OH_NN_UInt32Array prm; prm.data = prmD; prm.size = 2;
    uint32_t insD[2] = {0, 1}, outsD[1] = {2};
    OH_NN_UInt32Array ins, outs; ins.data = insD; ins.size = 2; outs.data = outsD; outs.size = 1;
    printf("  OPARGS params=[3,4] ins=[0,1] outs=[2]\n");
    printf("  AddOperation(MATMUL)=%d\n", (int)OH_NNModel_AddOperation(m, OH_NN_OPS_MATMUL, &prm, &ins, &outs));
    // ★必须先声明模型的输入/输出✓★（否则 Finish 返回 OPERATION_FORBIDDEN ✗）
    uint32_t mInsD[1] = {0}, mOutsD[1] = {2};
    OH_NN_UInt32Array mIns, mOuts; mIns.data = mInsD; mIns.size = 1; mOuts.data = mOutsD; mOuts.size = 1;
    printf("  SpecifyInputsAndOutputs=%d\n", (int)OH_NNModel_SpecifyInputsAndOutputs(m, &mIns, &mOuts));
    printf("  Finish=%d\n", (int)OH_NNModel_Finish(m));

    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    printf("  Compilation=%p\n", (void *)c);
    printf("  SetDevice=%d\n", (int)OH_NNCompilation_SetDevice(c, npu));
    printf("  ★Build=%d★\n", (int)OH_NNCompilation_Build(c));

    OH_NNExecutor *e = OH_NNExecutor_Construct(c);
    printf("  Executor=%p\n", (void *)e);
    if (e) {
        size_t inCnt = 0, outCnt = 0;
        OH_NNExecutor_GetInputCount(e, &inCnt); OH_NNExecutor_GetOutputCount(e, &outCnt);
        printf("  in=%zu out=%zu\n", inCnt, outCnt);
        static float x[B * S * H];
        for (int i = 0; i < B * S * H; ++i) x[i] = 0.001f * (i % 13);
        printf("  SetInput=%d\n", (int)OH_NNExecutor_SetInput(e, 0, &tx, x, sizeof(x)));
        static float y[B * S * H];
        memset(y, 0, sizeof(y));
        printf("  SetOutput=%d\n", (int)OH_NNExecutor_SetOutput(e, 0, y, sizeof(y)));
        printf("  ★Run=%d★\n", (int)OH_NNExecutor_Run(e));
        double ref = 0.0;
        for (int k = 0; k < H; ++k) ref += (double)x[k] * (double)w[k * H + 0];
        printf("  ★NPU y[0]=%.6f  参考=%.6f  差=%.3e★\n", y[0], ref, fabs((double)y[0] - ref));
    }
    return 0;
}
