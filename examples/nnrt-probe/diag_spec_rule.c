/* nnrt_op_rule.c - 隔离实验：OH_NNModel_AddOperation 在什么条件下成功？
 * 假设：paramIndices 为空（size=0）时 AddOperation 会失败。
 * 做法：对同几个算子，分别用【0 个参数】和【1 个参数】建图，比较 AddOperation 的返回码。
 */
#include <stdio.h>
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

static const char *rn(OH_NN_ReturnCode rc) {
    switch (rc) {
        case OH_NN_SUCCESS: return "SUCCESS";
        case OH_NN_FAILED: return "FAILED";
        case OH_NN_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case OH_NN_OPERATION_FORBIDDEN: return "OPERATION_FORBIDDEN";
        default: return "?";
    }
}

/* 加一个张量，返回其索引；失败返回 -1 */
static int add_tens(OH_NNModel *m, int32_t *shape, int nd, OH_NN_DataType dt,
                    OH_NN_TensorType tt, const void *data, int nval) {
    NN_TensorDesc *d = OH_NNTensorDesc_Create();
    if (!d) return -1;
    if (OH_NNTensorDesc_SetShape(d, shape, (size_t) nd) != OH_NN_SUCCESS) return -1;
    if (OH_NNTensorDesc_SetDataType(d, dt) != OH_NN_SUCCESS) return -1;
    if (OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE) != OH_NN_SUCCESS) return -1;
    if (OH_NNModel_AddTensorToModel(m, d) != OH_NN_SUCCESS) return -1;
    static int idx = 0;              /* 调用方按顺序添加，索引即序号 */
    idx = 0;
    OH_NNTensorDesc_Destroy(&d);
    return 0;
}

/* 试验：n_in 个输入 + n_param 个参数 + 1 个输出，全部 f32 {2,3} */
static OH_NN_ReturnCode try_op(OH_NN_OperationType op, int n_in, int n_param,
                               OH_NN_TensorType ptt, int32_t pval) {
    OH_NNModel *m = OH_NNModel_Construct();
    if (!m) return OH_NN_FAILED;
    int32_t s2[2] = {2, 3};
    int32_t s1[1] = {1};
    uint32_t idx = 0;
    uint32_t ins[4], pars[4];
    for (int i = 0; i < n_in; i++) {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, s2, 2);
        OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR);
        ins[i] = idx++;
    }
    for (int i = 0; i < n_param; i++) {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, s1, 1);
        OH_NNTensorDesc_SetDataType(d, OH_NN_INT8);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, idx, ptt);
        int8_t v = (int8_t) pval;
        OH_NNModel_SetTensorData(m, idx, &v, sizeof(v));
        pars[i] = idx++;
    }
    uint32_t outIdx;
    {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, s2, 2);
        OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        OH_NNModel_SetTensorType(m, idx, OH_NN_TENSOR);
        outIdx = idx++;
    }
    OH_NN_UInt32Array A = {ins, (uint32_t) n_in};
    OH_NN_UInt32Array P = {pars, (uint32_t) n_param};
    OH_NN_UInt32Array O = {&outIdx, 1};
    OH_NN_ReturnCode rc = OH_NNModel_AddOperation(m, op, &P, &A, &O);
    OH_NNModel_Destroy(&m);
    return rc;
}

int main(void) {
    printf("%-14s %-8s %-8s %s\n", "算子", "输入数", "参数数", "AddOperation");
    printf("--------------------------------------------------\n");
    struct { const char *n; OH_NN_OperationType op; int in; OH_NN_TensorType pt; } t[] = {
        { "RSqrt",  OH_NN_OPS_RSQRT, 1, 0 },
        { "Sqrt",   OH_NN_OPS_SQRT,  1, 0 },
        { "Tanh",   OH_NN_OPS_TANH,  1, 0 },
        { "Neg",    OH_NN_OPS_NEG,   1, 0 },
        { "Exp",    OH_NN_OPS_EXP,   1, 0 },
    };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        printf("%-14s %-8d %-8d %s\n", t[i].n, t[i].in, 0, rn(try_op(t[i].op, t[i].in, 0, 0, 0)));
        printf("%-14s %-8d %-8d %s\n", t[i].n, t[i].in, 1,
               rn(try_op(t[i].op, t[i].in, 1, OH_NN_ADD_ACTIVATIONTYPE, 0)));
    }
    printf("\n--- 二元算子 ---\n");
    printf("%-14s %-8d %-8d %s\n", "Add", 2, 0, rn(try_op(OH_NN_OPS_ADD, 2, 0, 0, 0)));
    printf("%-14s %-8d %-8d %s\n", "Add", 2, 1, rn(try_op(OH_NN_OPS_ADD, 2, 1, OH_NN_ADD_ACTIVATIONTYPE, 0)));
    printf("%-14s %-8d %-8d %s\n", "Mul", 2, 1, rn(try_op(OH_NN_OPS_MUL, 2, 1, OH_NN_MUL_ACTIVATION_TYPE, 0)));
    return 0;
}
