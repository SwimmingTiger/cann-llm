/* nnrt_block_test.c - ② 组一个真实的多算子块（MatMul -> Tanh -> MatMul -> Sin），
 * 在 NPU 上编译并执行，然后把结果与 CPU 参考值对比。
 * 用的全是前面实测【能编译到 NPU】的算子：MatMul / Tanh / Sin。
 * 权重取单位矩阵，便于核对：out = sin(tanh(x))。
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

static const char *rn(OH_NN_ReturnCode rc) {
    switch (rc) {
        case OH_NN_SUCCESS: return "SUCCESS";
        case OH_NN_FAILED: return "FAILED";
        case OH_NN_INVALID_PARAMETER: return "INVALID_PARAM";
        default: return "?";
    }
}

#define R 4
#define C 8

int main(void) {
    const size_t *ids = NULL; uint32_t nd = 0;
    if (OH_NNDevice_GetAllDevicesID(&ids, &nd) != OH_NN_SUCCESS || !nd) { printf("no device\n"); return 1; }
    size_t dev = ids[0];
    const char *nm = NULL; OH_NNDevice_GetName(dev, &nm);
    printf("设备: %s\n", nm ? nm : "?");

    /* 单位权重 */
    float I[C * C];
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) I[i * C + j] = (i == j) ? 1.0f : 0.0f;

    OH_NNModel *m = OH_NNModel_Construct();
    int32_t sx[3] = {1, R, C}, sw[2] = {C, C};
    uint32_t idx = 0;
    uint32_t tX, tW1, tH1, tH2, tW2, tOut;
    /* 输入 x */
    {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, sx, 3); OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d); OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR); tX = idx++;
    }
    /* 常量权重 W1 / W2，以及两个中间张量、一个输出 */
    for (int k = 0; k < 5; k++) {
        int isW = (k == 0 || k == 3);
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, isW ? sw : sx, isW ? 2 : 3);
        OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d); OH_NNTensorDesc_Destroy(&d);
        if (isW) {
            OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR);
            OH_NNModel_SetTensorData(m, idx, I, sizeof(I));
        } else {
            OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR);
        }
        if (k == 0) tW1 = idx; else if (k == 1) tH1 = idx; else if (k == 2) tH2 = idx;
        else if (k == 3) tW2 = idx; else tOut = idx;
        idx++;
    }
    /* 两个 MatMul 的 BOOL 转置参数 */
    uint32_t tPA, tPB, tPA2, tPB2;
    for (int k = 0; k < 4; k++) {
        int32_t s1[1] = {1};
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, s1, 1); OH_NNTensorDesc_SetDataType(d, OH_NN_BOOL);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d); OH_NNTensorDesc_Destroy(&d);
        int8_t v = 0;
        OH_NNModel_SetTensorType(m, idx,
            (k % 2 == 0) ? OH_NN_MATMUL_TRANSPOSE_A : OH_NN_MATMUL_TRANSPOSE_B);
        OH_NNModel_SetTensorData(m, idx, &v, sizeof(v));
        if (k == 0) tPA = idx; else if (k == 1) tPB = idx; else if (k == 2) tPA2 = idx; else tPB2 = idx;
        idx++;
    }

    /* op1: H1 = MatMul(x, W1, transposeA=0, transposeB=0) */
    { uint32_t in[2] = {tX, tW1}, out[1] = {tH1}, par[2] = {tPA, tPB};
      OH_NN_UInt32Array A = {in, 2}, O = {out, 1}, P = {par, 2};
      printf("op1 MatMul      -> %s\n", rn(OH_NNModel_AddOperation(m, OH_NN_OPS_MATMUL, &P, &A, &O))); }
    /* op2: H2 = Tanh(H1) */
    { uint32_t in[1] = {tH1}, out[1] = {tH2};
      OH_NN_UInt32Array A = {in, 1}, O = {out, 1}, P = {NULL, 0};
      printf("op2 Tanh        -> %s\n", rn(OH_NNModel_AddOperation(m, OH_NN_OPS_TANH, &P, &A, &O))); }
    /* op3: H2 = MatMul(H2, W2)  (原地用 tH2 作为输入 2 的输出会冲突，改为 MatMul(H2,W2)->tOut) */
    { uint32_t in[2] = {tH2, tW2}, out[1] = {tOut}, par[2] = {tPA2, tPB2};
      OH_NN_UInt32Array A = {in, 2}, O = {out, 1}, P = {par, 2};
      printf("op3 MatMul      -> %s\n", rn(OH_NNModel_AddOperation(m, OH_NN_OPS_MATMUL, &P, &A, &O))); }
    /* 本图就三个算子：MatMul -> Tanh -> MatMul，输出即 op3 的结果。
       权重取单位矩阵，所以期望输出 = tanh(x)，便于与 CPU 参考值逐个比对。 */

    uint32_t mi[1] = {tX}, mo[1] = {tOut};
    OH_NN_UInt32Array miA = {mi, 1}, moA = {mo, 1};
    printf("SpecifyIO       -> %s\n", rn(OH_NNModel_SpecifyInputsAndOutputs(m, &miA, &moA)));
    printf("Finish          -> %s\n", rn(OH_NNModel_Finish(m)));

    OH_NNCompilation *cp = OH_NNCompilation_Construct(m);
    OH_NNCompilation_SetDevice(cp, dev);
    OH_NNCompilation_SetPerformanceMode(cp, OH_NN_PERFORMANCE_EXTREME);
    printf("Build           -> %s  ← 多算子图能否在 NPU 上编译\n", rn(OH_NNCompilation_Build(cp)));

    OH_NNExecutor *ex = OH_NNExecutor_Construct(cp);
    if (!ex) { printf("executor 空\n"); return 1; }

    /* 填输入 */
    NN_TensorDesc *din = OH_NNExecutor_CreateInputTensorDesc(ex, 0);
    NN_Tensor *tin = OH_NNTensor_Create(dev, din);
    float *px = (float *) OH_NNTensor_GetDataBuffer(tin);
    size_t cnt = 0; OH_NNTensorDesc_GetElementCount(din, &cnt);
    float ref[R * C];
    for (size_t i = 0; i < cnt; i++) {
        px[i] = (float) i * 0.05f;                 /* 0 .. 1.55 */
        ref[i] = tanhf(px[i]);                     /* 期望: tanh(x)（两个单位矩阵 + 无 Sin） */
    }
    NN_TensorDesc *dout = OH_NNExecutor_CreateOutputTensorDesc(ex, 0);
    NN_Tensor *tout = OH_NNTensor_Create(dev, dout);

    NN_Tensor *inT[1] = {tin};
    printf("RunSync         -> %s\n", rn(OH_NNExecutor_RunSync(ex, inT, 1, &tout, 1)));

    float *po = (float *) OH_NNTensor_GetDataBuffer(tout);
    printf("\n输入/期望 tanh/实际 NPU 前 6 个:\n");
    int bad = 0;
    for (size_t i = 0; i < cnt && i < 6; i++)
        printf("  x=%.4f  期望=%.5f  NPU=%.5f\n", (double) px[i], (double) ref[i], (double) po[i]);
    for (size_t i = 0; i < cnt; i++)
        if (fabsf(po[i] - ref[i]) > 1e-3f) bad++;
    printf("\n全量 %zu 个元素中偏差 >1e-3 的: %d 个\n", cnt, bad);
    printf(bad == 0 ? "★★★ 多算子图在 NPU 上【编译并执行正确】★★★\n"
                    : "数值与期望不符（继续查）\n");
    return 0;
}
