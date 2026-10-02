/* nnrt_variant.c - RSQRT（枚举文档说：1 输入 0 参数）为什么 AddOperation 失败？
 * 试多种结构变体，找出真正被校验的判据。
 */
#include <stdio.h>
#include <string.h>
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

static const char *rn(OH_NN_ReturnCode rc) {
    switch (rc) {
        case OH_NN_SUCCESS: return "SUCCESS";
        case OH_NN_FAILED: return "FAILED";
        case OH_NN_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case OH_NN_OPERATION_FORBIDDEN: return "OPERATION_FORBIDDEN";
        case OH_NN_NULL_PTR: return "NULL_PTR";
        default: return "?";
    }
}

#define MAXT 6

typedef struct {
    int add;                 /* 是否添加该张量 */
    int32_t s[3]; int nd;
    OH_NN_DataType dt;
    int settype;             /* 是否调用 SetTensorType */
    OH_NN_TensorType tt;
    int hasdata; int32_t dv;
} TS;

/* 通用试验：tensors[] 描述，op 的输入/输出/参数索引由调用方给 */
static OH_NN_ReturnCode run(const char *tag, OH_NN_OperationType op, TS *ts, int nt,
                            uint32_t *ins, int nins, uint32_t *outs, int nouts,
                            uint32_t *pars, int npars) {
    OH_NNModel *m = OH_NNModel_Construct();
    if (!m) return OH_NN_FAILED;
    for (int i = 0; i < nt; i++) {
        if (!ts[i].add) continue;
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        OH_NNTensorDesc_SetShape(d, ts[i].s, (size_t) ts[i].nd);
        OH_NNTensorDesc_SetDataType(d, ts[i].dt);
        OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
        OH_NNModel_AddTensorToModel(m, d);
        OH_NNTensorDesc_Destroy(&d);
        if (ts[i].settype) OH_NNModel_SetTensorType(m, (uint32_t) i, ts[i].tt);
        if (ts[i].hasdata) {
            int8_t v = (int8_t) ts[i].dv;
            OH_NNModel_SetTensorData(m, (uint32_t) i, &v, sizeof(v));
        }
    }
    OH_NN_UInt32Array A = {ins, (uint32_t) nins};
    OH_NN_UInt32Array O = {outs, (uint32_t) nouts};
    OH_NN_UInt32Array P = {pars, (uint32_t) npars};
    OH_NN_ReturnCode rc = OH_NNModel_AddOperation(m, op, &P, &A, &O);
    printf("  %-52s -> %s\n", tag, rn(rc));
    OH_NNModel_Destroy(&m);
    return rc;
}

#define TENS(dt_, nd_, ...) { 1, {__VA_ARGS__}, nd_, dt_, 1, OH_NN_TENSOR, 0, 0 }

int main(void) {
    printf("=== RSQRT 变体（f32 {2,3} -> f32 {2,3}）===\n");
    uint32_t in0[1] = {0}, out1[1] = {1}, out2[1] = {2};
    uint32_t in01[2] = {0, 1};

    {   /* V1 基线：in(0) out(1)，都 SetTensorType=TENSOR，无参数 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3) };
        run("V1 基线 in/out +SetTensorType, 无参数", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V2 完全不调 SetTensorType */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3) };
        t[0].settype = t[1].settype = 0;
        run("V2 不调 SetTensorType", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V3 只给 input 设类型 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3) };
        t[1].settype = 0;
        run("V3 只给 input 设类型", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V4 只给 output 设类型 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3) };
        t[0].settype = 0;
        run("V4 只给 output 设类型", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V5 一维输入 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 1, 6), TENS(OH_NN_FLOAT32, 1, 6) };
        run("V5 一维 {6}", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V6 多一个未被引用的张量 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3),
                       TENS(OH_NN_FLOAT32, 2, 2, 3) };
        run("V6 多一个未引用张量", OH_NN_OPS_RSQRT, t, 3, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V7 两个输入（只把第一个当 op 输入） */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3),
                       TENS(OH_NN_FLOAT32, 2, 2, 3) };
        run("V7 加 2 输入 + 1 输出，op 只吃第 1 个", OH_NN_OPS_RSQRT, t, 3, in0, 1, out2, 1, NULL, 0);
    }
    {   /* V8 output 用 f16 */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT16, 2, 2, 3) };
        run("V8 output 用 float16", OH_NN_OPS_RSQRT, t, 2, in0, 1, out1, 1, NULL, 0);
    }
    {   /* V9 用老的 OH_NNModel_AddTensor 接口（与 V1 同结构） */
        OH_NNModel *m = OH_NNModel_Construct();
        int32_t s[2] = {2, 3};
        for (int i = 0; i < 2; i++) {
            NN_TensorDesc *d = OH_NNTensorDesc_Create();
            OH_NNTensorDesc_SetShape(d, s, 2);
            OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
            OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
            OH_NNModel_AddTensorToModel(m, d);
            OH_NNTensorDesc_Destroy(&d);
            OH_NNModel_SetTensorType(m, (uint32_t) i, OH_NN_TENSOR);
        }
        OH_NN_UInt32Array A = {in0, 1}, O = {out1, 1}, P = {NULL, 0};
        printf("  %-52s -> %s\n", "V9 参数数组传 NULL,0",
               rn(OH_NNModel_AddOperation(m, OH_NN_OPS_RSQRT, &P, &A, &O)));
        OH_NNModel_Destroy(&m);
    }
    {   /* V10 参数数组 size=0 但 data 指向有效缓冲（与 V1 相同，显式） */
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3) };
        uint32_t dummy = 0;
        OH_NNModel *m = OH_NNModel_Construct();
        int32_t s[2] = {2, 3};
        for (int i = 0; i < 2; i++) {
            NN_TensorDesc *d = OH_NNTensorDesc_Create();
            OH_NNTensorDesc_SetShape(d, s, 2);
            OH_NNTensorDesc_SetDataType(d, OH_NN_FLOAT32);
            OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
            OH_NNModel_AddTensorToModel(m, d);
            OH_NNTensorDesc_Destroy(&d);
            OH_NNModel_SetTensorType(m, (uint32_t) i, OH_NN_TENSOR);
        }
        (void) t;
        OH_NN_UInt32Array A = {in0, 1}, O = {out1, 1}, P = {&dummy, 0};
        printf("  %-52s -> %s\n", "V10 参数数组 data 非空 size=0",
               rn(OH_NNModel_AddOperation(m, OH_NN_OPS_RSQRT, &P, &A, &O)));
        OH_NNModel_Destroy(&m);
    }
    printf("\n=== 对照：ADD（2 输入 + 1 activation 参数）===\n");
    {
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 2, 3), TENS(OH_NN_FLOAT32, 2, 2, 3),
                       TENS(OH_NN_FLOAT32, 2, 2, 3),
                       { 1, {1}, 1, OH_NN_INT8, 1, OH_NN_ADD_ACTIVATIONTYPE, 1, 0 } };
        uint32_t p3[1] = {3};
        run("ADD 2in + activation + out", OH_NN_OPS_ADD, t, 4, in01, 2, out2, 1, p3, 1);
    }
    printf("\n=== 试：GATHER 补上第 3 个输入 axis ===\n");
    {
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 2, 16, 8),      /* input */
                       TENS(OH_NN_INT32, 2, 1, 4),          /* inputIndices */
                       { 1, {1}, 1, OH_NN_INT32, 1, OH_NN_TENSOR, 1, 0 },  /* axis=0 */
                       TENS(OH_NN_FLOAT32, 3, 1, 4, 8) };   /* output */
        uint32_t gi[3] = {0, 1, 2}, go[1] = {3};
        run("GATHER 3 输入（含 axis）+ out", OH_NN_OPS_GATHER, t, 4, gi, 3, go, 1, NULL, 0);
    }
    printf("\n=== 试：MATMUL 补上第 3 个参数 activationType ===\n");
    {
        TS t[MAXT] = { TENS(OH_NN_FLOAT32, 3, 1, 4, 8), TENS(OH_NN_FLOAT32, 2, 8, 8),
                       TENS(OH_NN_FLOAT32, 3, 1, 4, 8),
                       { 1, {1}, 1, OH_NN_INT8, 1, OH_NN_MATMUL_TRANSPOSE_A, 1, 0 },
                       { 1, {1}, 1, OH_NN_INT8, 1, OH_NN_MATMUL_TRANSPOSE_B, 1, 0 },
                       { 1, {1}, 1, OH_NN_INT8, 1, OH_NN_MATMUL_ACTIVATION_TYPE, 1, 0 } };
        uint32_t mi[2] = {0, 1}, mo[1] = {2}, mp[3] = {3, 4, 5};
        run("MATMUL 2in + 3 参数（含 activationType）", OH_NN_OPS_MATMUL, t, 6, mi, 2, mo, 1, mp, 3);
    }
    return 0;
}
