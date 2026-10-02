/* nnrt_ops_probe.c - ① 问出这块麒麟 NPU 实际支持哪些算子。
 *
 * 做法：对每个算子单独建一个最小合法模型 -> OH_NNModel_Finish ->
 *       OH_NNModel_GetAvailableOperations(dev) 问"底层设备支持这个算子吗"。
 *       只为探测，不做编译执行。
 *
 * Build: cc -O1 -I<sysroot>/usr/include nnrt_ops_probe.c -o nnrt_ops_probe \
 *          -L/system/lib64/ndk -lneural_network_runtime -lneural_network_core
 * Run:   LD_LIBRARY_PATH=/system/lib64/ndk ./nnrt_ops_probe
 */
#include <stdio.h>
#include <string.h>

#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

#define R_IN 0
#define R_OUT 1
#define R_CONST 2

typedef struct {
    int32_t d[4];
    int nd;
    OH_NN_DataType dt;
    int role;
    OH_NN_TensorType ttype;
    int64_t iv[8];
    float fv[8];
    int nval;
} Tens;

typedef struct {
    const char *name;
    OH_NN_OperationType op;
    Tens t[8];
    int nt;
    int in[4];  int nin;
    int out[2]; int nout;
    int par[4]; int npar;
} Case;

#define IN_(dt, ...)  { .dt = dt, .role = R_IN,  .nd = 0 }
#define T(dt, nd, role) { .dt = dt, .nd = (nd), .role = (role) }

static size_t dwidth(OH_NN_DataType dt) {
    switch (dt) {
        case OH_NN_FLOAT32: case OH_NN_INT32: return 4;
        case OH_NN_FLOAT16: case OH_NN_INT16: case OH_NN_UINT16: return 2;
        case OH_NN_INT64: return 8;
        default: return 1;
    }
}

/* 建一个单算子模型；成功返回 model，失败返回 NULL */
static const char *g_err = "?";
static OH_NNModel *build(const Case *c) {
    OH_NNModel *m = OH_NNModel_Construct();
    if (!m) { g_err = "Construct"; return NULL; }
    uint32_t mi[4], mo[2];
    int nmi = 0, nmo = 0;
    for (int i = 0; i < c->nt; i++) {
        const Tens *x = &c->t[i];
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        if (!d) { g_err = "DescCreate"; return NULL; }
        if (OH_NNTensorDesc_SetShape(d, x->d, (size_t) x->nd) != OH_NN_SUCCESS) { g_err = "SetShape"; return NULL; }
        if (OH_NNTensorDesc_SetDataType(d, x->dt) != OH_NN_SUCCESS) { g_err = "SetDataType"; return NULL; }
        if (OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE) != OH_NN_SUCCESS) { g_err = "SetFormat"; return NULL; }
        if (OH_NNModel_AddTensorToModel(m, d) != OH_NN_SUCCESS) { g_err = "AddTensorToModel"; return NULL; }
        OH_NNTensorDesc_Destroy(&d);
        if (x->role == R_CONST) {
            if (OH_NNModel_SetTensorType(m, (uint32_t) i, x->ttype) != OH_NN_SUCCESS) { g_err = "SetTensorType(const)"; return NULL; }
            if (x->nval > 0) {
                const void *p = (x->dt == OH_NN_FLOAT32) ? (const void *) x->fv
                                                         : (const void *) x->iv;
                if (OH_NNModel_SetTensorData(m, (uint32_t) i, p,
                                             (size_t) x->nval * dwidth(x->dt)) != OH_NN_SUCCESS) { g_err = "SetTensorData"; return NULL; }
            }
        } else {
            if (OH_NNModel_SetTensorType(m, (uint32_t) i, OH_NN_TENSOR) != OH_NN_SUCCESS) { g_err = "SetTensorType(in/out)"; return NULL; }
            if (x->role == R_IN) mi[nmi++] = (uint32_t) i;
            else mo[nmo++] = (uint32_t) i;
        }
    }
    uint32_t ain[4], aout[2], apar[4];
    for (int i = 0; i < c->nin; i++) ain[i] = (uint32_t) c->in[i];
    for (int i = 0; i < c->nout; i++) aout[i] = (uint32_t) c->out[i];
    for (int i = 0; i < c->npar; i++) apar[i] = (uint32_t) c->par[i];
    OH_NN_UInt32Array ins = {ain, (uint32_t) c->nin};
    OH_NN_UInt32Array outs = {aout, (uint32_t) c->nout};
    /* ★ 无参数时必须传 {NULL, 0}：传未初始化的非空指针会被当成"多传了参数" → INVALID_PARAMETER */
    OH_NN_UInt32Array pars = {c->npar ? apar : NULL, (uint32_t) c->npar};
    if (OH_NNModel_AddOperation(m, c->op, &pars, &ins, &outs) != OH_NN_SUCCESS) { g_err = "AddOperation"; return NULL; }
    OH_NN_UInt32Array miA = {mi, (uint32_t) nmi};
    OH_NN_UInt32Array moA = {mo, (uint32_t) nmo};
    if (OH_NNModel_SpecifyInputsAndOutputs(m, &miA, &moA) != OH_NN_SUCCESS) { g_err = "SpecifyIO"; return NULL; }
    if (OH_NNModel_Finish(m) != OH_NN_SUCCESS) { g_err = "Finish"; return NULL; }
    return m;
}

/* 简写：单入单出的逐元素算子 */
#define EW(NAME, OP) {                                                         \
    .name = NAME, .op = OP, .nt = 2,                                           \
    .t = { { .d = {2,3}, .nd = 2, .dt = OH_NN_BOOL, .role = R_IN },          \
           { .d = {2,3}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },       \
    .in = {0}, .nin = 1, .out = {1}, .nout = 1 }

static const Case cases[] = {
    /* ---- Transformer 关键算子 ---- */
    { "Gather(embedding)", OH_NN_OPS_GATHER, .nt = 3,
      .t = { { .d = {16,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4},  .nd = 2, .dt = OH_NN_INT32,   .role = R_IN },
             { .d = {1,4,8},.nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "MatMul", OH_NN_OPS_MATMUL, .nt = 5,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {8,8},   .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_MATMUL_TRANSPOSE_A, .iv = {0}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_MATMUL_TRANSPOSE_B, .iv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3,4}, .npar = 2 },

    { "Reshape", OH_NN_OPS_RESHAPE, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {4,8}, .nval = 2 },
             { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "Transpose", OH_NN_OPS_TRANSPOSE, .nt = 3,
      .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {1,0}, .nval = 2 },
             { .d = {8,4}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "ReduceMean(axes)", OH_NN_OPS_REDUCE_MEAN, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {2}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_REDUCE_MEAN_KEEP_DIMS, .iv = {1}, .nval = 1 },
             { .d = {1,4,1}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {3}, .nout = 1, .par = {2}, .npar = 1 },

    { "Softmax", OH_NN_OPS_SOFTMAX, .nt = 3,
      .t = { { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_SOFTMAX_AXIS, .iv = {2}, .nval = 1 },
             { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

    { "GELU", OH_NN_OPS_GELU, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_GELU_APPROXIMATE, .iv = {0}, .nval = 1 },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

    { "LayerNorm", OH_NN_OPS_LAYER_NORM, .nt = 7,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_TENSOR, .fv = {1,1,1,1,1,1,1,1}, .nval = 8 },
             { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_TENSOR, .fv = {0}, .nval = 8 },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_LAYER_NORM_BEGIN_NORM_AXIS, .iv = {2}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS, .iv = {2}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_LAYER_NORM_EPSILON, .fv = {1e-5f}, .nval = 1 },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1,2}, .nin = 3, .out = {6}, .nout = 1, .par = {3,4,5}, .npar = 3 },

    /* ---- 逐元素 / 数学 ---- */
    EW("RSqrt", OH_NN_OPS_RSQRT), EW("Sqrt", OH_NN_OPS_SQRT),
    EW("Tanh", OH_NN_OPS_TANH),   EW("Sin", OH_NN_OPS_SIN),
    EW("Cos", OH_NN_OPS_COS),     EW("Exp", OH_NN_OPS_EXP),
    EW("Neg", OH_NN_OPS_NEG),     EW("Abs", OH_NN_OPS_ABS),
    EW("Log", OH_NN_OPS_LOG),     EW("Square", OH_NN_OPS_SQUARE),
    EW("Ceil", OH_NN_OPS_CEIL),   EW("Floor", OH_NN_OPS_FLOOR),
    EW("Erf", OH_NN_OPS_ERF),     EW("Reciprocal", OH_NN_OPS_RECIPROCAL),

    { "Pow", OH_NN_OPS_POW, .nt = 5,
      .t = { { .d = {2,3}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {2,3}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_POW_SCALE, .fv = {1}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_POW_SHIFT, .fv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3,4}, .npar = 2 },

    { "Add(activation)", OH_NN_OPS_ADD, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_ADD_ACTIVATIONTYPE, .iv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3}, .npar = 1 },

    { "Mul(activation)", OH_NN_OPS_MUL, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,1}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_MUL_ACTIVATION_TYPE, .iv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3}, .npar = 1 },

    { "Sub(activation)", OH_NN_OPS_SUB, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,1}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_SUB_ACTIVATIONTYPE, .iv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3}, .npar = 1 },

    { "Div(activation)", OH_NN_OPS_DIV, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_DIV_ACTIVATIONTYPE, .iv = {0}, .nval = 1 } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3}, .npar = 1 },

    { "Concat", OH_NN_OPS_CONCAT, .nt = 4,
      .t = { { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_CONCAT_AXIS, .iv = {2}, .nval = 1 },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {3}, .nout = 1, .par = {2}, .npar = 1 },

    { "Split", OH_NN_OPS_SPLIT, .nt = 5,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_SPLIT_AXIS, .iv = {2}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_CONST, .ttype = OH_NN_SPLIT_OUTPUT_NUM, .iv = {2}, .nval = 1 },
             { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
             { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {3,4}, .nout = 2, .par = {1,2}, .npar = 2 },

    { "Cast f32->f16", OH_NN_OPS_CAST, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT16, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "Slice", OH_NN_OPS_SLICE, .nt = 5,
      .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {0,0}, .nval = 2 },
             { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {2,8}, .nval = 2 },
             { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_SLICE_AXES, .iv = {0,1}, .nval = 2 },
             { .d = {2,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1,2}, .nin = 3, .out = {4}, .nout = 1, .par = {3}, .npar = 1 },

    { "Tile", OH_NN_OPS_TILE, .nt = 3,
      .t = { { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {3}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {1,1,2}, .nval = 3 },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "ExpandDims", OH_NN_OPS_EXPAND_DIMS, .nt = 3,
      .t = { { .d = {1,4}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_TENSOR, .iv = {1}, .nval = 1 },
             { .d = {1,1,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "Squeeze", OH_NN_OPS_SQUEEZE, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_SQUEEZE_AXIS, .iv = {0}, .nval = 1 },
             { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

    { "Clip", OH_NN_OPS_CLIP, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_CLIP_MIN, .fv = {-1}, .nval = 1 },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CONST, .ttype = OH_NN_CLIP_MAX, .fv = {1}, .nval = 1 },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {3}, .nout = 1, .par = {1,2}, .npar = 2 },

    { "Maximum", OH_NN_OPS_MAXIMUM, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "Greater", OH_NN_OPS_GREATER, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_BOOL, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

    { "Select", OH_NN_OPS_SELECT, .nt = 4,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_BOOL, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1,2}, .nin = 3, .out = {3}, .nout = 1 },

    { "Flatten", OH_NN_OPS_FLATTEN, .nt = 3,
      .t = { { .d = {1,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_FLATTEN_AXIS, .iv = {1}, .nval = 1 },
             { .d = {1,32}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

    { "Stack", OH_NN_OPS_STACK, .nt = 4,
      .t = { { .d = {1,4}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1,4}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
             { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CONST, .ttype = OH_NN_STACK_AXIS, .iv = {0}, .nval = 1 },
             { .d = {1,1,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
      .in = {0,1}, .nin = 2, .out = {3}, .nout = 1, .par = {2}, .npar = 1 },
};

int main(void) {
    const size_t *ids = NULL;
    uint32_t n = 0;
    size_t dev = 0;
    if (OH_NNDevice_GetAllDevicesID(&ids, &n) != OH_NN_SUCCESS || !n) {
        printf("拿不到设备\n");
        return 1;
    }
    dev = ids[0];
    const char *nm = NULL;
    OH_NNDevice_GetName(dev, &nm);
    printf("设备: %s (id=%zu)  算子数 = %zu\n\n", nm ? nm : "?", dev, sizeof(cases) / sizeof(cases[0]));
    printf("%-24s %-12s %s\n", "算子", "建图", "NPU 支持");
    printf("------------------------------------------------\n");

    int ok = 0, bad = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const Case *c = &cases[i];
        OH_NNModel *m = build(c);
        if (!m) {
            printf("%-24s %-12s %s\n", c->name, "建图失败", g_err);
            bad++;
            continue;
        }
        /* ★ 更可靠的判据：直接在这块 NPU 上编译（能 Build 成功 = 支持） */
        OH_NNCompilation *cp = OH_NNCompilation_Construct(m);
        if (!cp) {
            printf("%-24s %-12s %s\n", c->name, "OK", "编译实例创建失败");
            bad++;
            OH_NNModel_Destroy(&m);
            continue;
        }
        OH_NNCompilation_SetDevice(cp, dev);
        OH_NNCompilation_SetPerformanceMode(cp, OH_NN_PERFORMANCE_EXTREME);
        OH_NN_ReturnCode brc = OH_NNCompilation_Build(cp);
        if (brc == OH_NN_SUCCESS) {
            printf("%-24s %-12s %s\n", c->name, "OK", "★★ 可编译 → NPU 支持");
            ok++;
        } else {
            printf("%-24s %-12s %s\n", c->name, "OK", "✗ Build 失败（NPU 不支持或规格不符）");
            bad++;
        }
        OH_NNCompilation_Destroy(&cp);
        OH_NNModel_Destroy(&m);
    }
    printf("\n汇总：支持 %d / 共 %zu（另 %d 个建图失败或查询失败）\n",
           ok, sizeof(cases) / sizeof(cases[0]), bad);
    return 0;
}
