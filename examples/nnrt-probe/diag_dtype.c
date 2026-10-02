/* nnrt_fp16_test.c - 猜想：NPU 的 elementary_lib 里那些数学算子只支持 FP16。
 * 做法：同一批算子，用 FP16 输入/输出再试一次 Build；并顺带修正 MatMul 的参数集
 *       （TransposeX BOOL + TransposeY BOOL + activationType）。
 */
#include <stdio.h>
#include <string.h>
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

static size_t dev = 0;

/* 单入单出：dt 指定数据类型，shape 由 s/nd 给 */
static OH_NN_ReturnCode one_io(OH_NN_OperationType op, OH_NN_DataType dt, int32_t *s, int nd) {
    OH_NNModel *m = OH_NNModel_Construct();
    uint32_t idx = 0;
    uint32_t ins[1], outs[1];
    for (int k = 0; k < 2; k++) {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, s, (size_t) nd);
        OH_NNTensorDesc_SetDataType(d, dt);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR);
        if (k == 0) ins[0] = idx; else outs[0] = idx;
        idx++;
    }
    OH_NN_UInt32Array A = {ins, 1}, O = {outs, 1}, P = {NULL, 0};
    OH_NN_ReturnCode rc = OH_NNModel_AddOperation(m, op, &P, &A, &O);
    if (rc != OH_NN_SUCCESS) { OH_NNModel_Destroy(&m); return rc; }
    OH_NN_UInt32Array mi = {ins, 1}, mo = {outs, 1};
    OH_NNModel_SpecifyInputsAndOutputs(m, &mi, &mo);
    if (OH_NNModel_Finish(m) != OH_NN_SUCCESS) { OH_NNModel_Destroy(&m); return OH_NN_FAILED; }
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    OH_NNCompilation_SetDevice(c, dev);
    OH_NNCompilation_SetPerformanceMode(c, OH_NN_PERFORMANCE_EXTREME);
    OH_NN_ReturnCode brc = OH_NNCompilation_Build(c);
    OH_NNCompilation_Destroy(&c);
    OH_NNModel_Destroy(&m);
    return brc;
}

/* MatMul：FP16，参数 = TransposeX(BOOL) + TransposeY(BOOL) + activationType(INT8) */
static OH_NN_ReturnCode matmul(OH_NN_DataType dt, int npar) {
    OH_NNModel *m = OH_NNModel_Construct();
    int32_t a[3] = {1, 4, 8}, b[2] = {8, 8}, o[3] = {1, 4, 8}, s1[1] = {1};
    struct { int32_t *s; int nd; OH_NN_DataType dt; OH_NN_TensorType tt; } t[6] = {
        { a, 3, dt, OH_NN_TENSOR }, { b, 2, dt, OH_NN_TENSOR }, { o, 3, dt, OH_NN_TENSOR },
        { s1, 1, OH_NN_BOOL, OH_NN_MATMUL_TRANSPOSE_A },
        { s1, 1, OH_NN_BOOL, OH_NN_MATMUL_TRANSPOSE_B },
        { s1, 1, OH_NN_INT8, OH_NN_MATMUL_ACTIVATION_TYPE },
    };
    int nt = 3 + npar;
    for (int i = 0; i < nt; i++) {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, t[i].s, (size_t) t[i].nd);
        OH_NNTensorDesc_SetDataType(d, t[i].dt);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, (uint32_t) i, t[i].tt);
        if (i >= 3) {
            int64_t v = 0;
            OH_NNModel_SetTensorData(m, (uint32_t) i, &v, t[i].dt == OH_NN_BOOL ? 1 : 1);
        }
    }
    uint32_t ins[2] = {0, 1}, outs[1] = {2}, pars[3] = {3, 4, 5};
    OH_NN_UInt32Array A = {ins, 2}, O = {outs, 1}, P = {pars, (uint32_t) npar};
    OH_NN_ReturnCode rc = OH_NNModel_AddOperation(m, OH_NN_OPS_MATMUL, &P, &A, &O);
    if (rc != OH_NN_SUCCESS) { OH_NNModel_Destroy(&m); return rc; }
    OH_NN_UInt32Array mi = {ins, 2}, mo = {outs, 1};
    OH_NNModel_SpecifyInputsAndOutputs(m, &mi, &mo);
    if (OH_NNModel_Finish(m) != OH_NN_SUCCESS) { OH_NNModel_Destroy(&m); return OH_NN_FAILED; }
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    OH_NNCompilation_SetDevice(c, dev);
    OH_NN_ReturnCode brc = OH_NNCompilation_Build(c);
    OH_NNCompilation_Destroy(&c);
    OH_NNModel_Destroy(&m);
    return brc;
}

int main(void) {
    const size_t *ids = NULL;
    uint32_t n = 0;
    if (OH_NNDevice_GetAllDevicesID(&ids, &n) != OH_NN_SUCCESS || !n) return 1;
    dev = ids[0];

    struct { const char *n; OH_NN_OperationType op; } ops[] = {
        {"RSqrt", OH_NN_OPS_RSQRT}, {"Sqrt", OH_NN_OPS_SQRT}, {"Tanh", OH_NN_OPS_TANH},
        {"Sin", OH_NN_OPS_SIN}, {"Cos", OH_NN_OPS_COS}, {"Exp", OH_NN_OPS_EXP},
        {"Neg", OH_NN_OPS_NEG}, {"Abs", OH_NN_OPS_ABS}, {"Log", OH_NN_OPS_LOG},
        {"Square", OH_NN_OPS_SQUARE}, {"Erf", OH_NN_OPS_ERF}, {"Reciprocal", OH_NN_OPS_RECIPROCAL},
        {"Floor", OH_NN_OPS_FLOOR}, {"Ceil", OH_NN_OPS_CEIL}, {"Relu", OH_NN_OPS_RELU},
        {"Sigmoid", OH_NN_OPS_SIGMOID}, {"Softmax", OH_NN_OPS_SOFTMAX},
    };
    int32_t s3[3] = {1, 4, 8};

    printf("%-14s %-14s %s\n", "算子", "FP32", "FP16");
    printf("--------------------------------------------\n");
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
        OH_NN_ReturnCode a = one_io(ops[i].op, OH_NN_FLOAT32, s3, 3);
        OH_NN_ReturnCode b = one_io(ops[i].op, OH_NN_FLOAT16, s3, 3);
        printf("%-14s %-14s %s\n", ops[i].n, rn(a), rn(b));
    }
    printf("\n--- MatMul（参数数 0/1/2/3，FP16 与 FP32）---\n");
    for (int np = 0; np <= 3; np++) {
        printf("  MatMul 参数数=%d  FP32=%-12s FP16=%s\n", np,
               rn(matmul(OH_NN_FLOAT32, np)), rn(matmul(OH_NN_FLOAT16, np)));
    }
    return 0;
}
