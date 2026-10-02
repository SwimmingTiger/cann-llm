/* nnrt_axis_fix.c - 按 hilog 的指正重建这批算子的图：
 *   轴/形状一类值 = 算子的【INT64 常量输入张量】（不是参数张量）
 *   布尔类 = OH_NN_BOOL 参数
 * 判据：能否在这块 NPU 上 OH_NNCompilation_Build 成功。
 */
#include <stdio.h>
#include <string.h>
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

#define R_IN 0
#define R_OUT 1
#define R_CST 2
#define R_PAR 3

typedef struct {
    int32_t d[3]; int nd;
    OH_NN_DataType dt;
    int role;
    OH_NN_TensorType tt;          /* role==3 用 */
    int64_t iv[8]; float fv[8];   /* 常量数据 */
    int nval;
} T;

typedef struct {
    const char *name;
    OH_NN_OperationType op;
    T t[8]; int nt;
    int in[4]; int nin;
    int out[2]; int nout;
    int par[4]; int npar;
} C;

static size_t dev = 0;
static const char *g_err = "?";

static size_t wdt(OH_NN_DataType dt) {
    switch (dt) {
        case OH_NN_FLOAT32: case OH_NN_INT32: return 4;
        case OH_NN_INT64: return 8;
        case OH_NN_FLOAT16: case OH_NN_INT16: case OH_NN_UINT16: return 2;
        default: return 1;
    }
}

static OH_NN_ReturnCode try_case(const C *c) {
    OH_NNModel *m = OH_NNModel_Construct();
    if (!m) { g_err = "Construct"; return OH_NN_FAILED; }
    uint32_t mi[4], mo[2]; int nmi = 0, nmo = 0;
    for (int i = 0; i < c->nt; i++) {
        const T *x = &c->t[i];
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        if (!d) { g_err = "DescCreate"; return OH_NN_FAILED; }
        if (OH_NNTensorDesc_SetShape(d, x->d, (size_t) x->nd) != OH_NN_SUCCESS) { g_err = "SetShape"; return OH_NN_FAILED; }
        if (OH_NNTensorDesc_SetDataType(d, x->dt) != OH_NN_SUCCESS) { g_err = "SetDataType"; return OH_NN_FAILED; }
        if (OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE) != OH_NN_SUCCESS) { g_err = "SetFormat"; return OH_NN_FAILED; }
        if (OH_NNModel_AddTensorToModel(m, d) != OH_NN_SUCCESS) { g_err = "AddTensor"; return OH_NN_FAILED; }
        OH_NNTensorDesc_Destroy(&d);
        if (x->role == R_PAR) {
            if (OH_NNModel_SetTensorType(m, (uint32_t) i, x->tt) != OH_NN_SUCCESS) { g_err = "SetTensorType(param)"; return OH_NN_FAILED; }
        } else {
            if (OH_NNModel_SetTensorType(m, (uint32_t) i, OH_NN_TENSOR) != OH_NN_SUCCESS) { g_err = "SetTensorType"; return OH_NN_FAILED; }
            if (x->role == R_IN) mi[nmi++] = (uint32_t) i;
            else if (x->role == R_OUT) mo[nmo++] = (uint32_t) i;
        }
        if (x->nval > 0) {
            const void *p = (x->dt == OH_NN_FLOAT32) ? (const void *) x->fv : (const void *) x->iv;
            if (OH_NNModel_SetTensorData(m, (uint32_t) i, p, (size_t) x->nval * wdt(x->dt)) != OH_NN_SUCCESS) { g_err = "SetTensorData"; return OH_NN_FAILED; }
        }
    }
    uint32_t ai[4], ao[2], ap[4];
    for (int i = 0; i < c->nin; i++) ai[i] = (uint32_t) c->in[i];
    for (int i = 0; i < c->nout; i++) ao[i] = (uint32_t) c->out[i];
    for (int i = 0; i < c->npar; i++) ap[i] = (uint32_t) c->par[i];
    OH_NN_UInt32Array A = {ai, (uint32_t) c->nin};
    OH_NN_UInt32Array O = {ao, (uint32_t) c->nout};
    OH_NN_UInt32Array P = {c->npar ? ap : NULL, (uint32_t) c->npar};
    OH_NN_ReturnCode rc = OH_NNModel_AddOperation(m, c->op, &P, &A, &O);
    if (rc != OH_NN_SUCCESS) { g_err = "AddOperation"; OH_NNModel_Destroy(&m); return rc; }
    OH_NN_UInt32Array miA = {mi, (uint32_t) nmi}, moA = {mo, (uint32_t) nmo};
    if (OH_NNModel_SpecifyInputsAndOutputs(m, &miA, &moA) != OH_NN_SUCCESS) { g_err = "SpecifyIO"; OH_NNModel_Destroy(&m); return OH_NN_FAILED; }
    if (OH_NNModel_Finish(m) != OH_NN_SUCCESS) { g_err = "Finish"; OH_NNModel_Destroy(&m); return OH_NN_FAILED; }
    OH_NNCompilation *cp = OH_NNCompilation_Construct(m);
    OH_NNCompilation_SetDevice(cp, dev);
    OH_NNCompilation_SetPerformanceMode(cp, OH_NN_PERFORMANCE_EXTREME);
    OH_NN_ReturnCode brc = OH_NNCompilation_Build(cp);
    OH_NNCompilation_Destroy(&cp);
    OH_NNModel_Destroy(&m);
    if (brc != OH_NN_SUCCESS) g_err = "Build";
    return brc;
}

#define IN_F32(...)  { .d = {__VA_ARGS__}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_IN }
#define OUT_F32(...) { .d = {__VA_ARGS__}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT }
#define CST_I64_1(V) { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {V}, .nval = 1 }
#define PAR_BOOL(V)  { .d = {1}, .nd = 1, .dt = OH_NN_BOOL, .role = R_PAR, .iv = {V}, .nval = 1 }

int main(void) {
    const size_t *ids = NULL; uint32_t n = 0;
    if (OH_NNDevice_GetAllDevicesID(&ids, &n) != OH_NN_SUCCESS || !n) { printf("no device\n"); return 1; }
    dev = ids[0];
    const char *nm = NULL; OH_NNDevice_GetName(dev, &nm);
    printf("设备: %s\n\n", nm ? nm : "?");
    printf("%-30s %s\n", "算子（按 hilog 指正重建）", "结果");
    printf("-------------------------------------------------\n");

    const C cases[] = {
        /* Softmax: 第 2 个输入 = axis(INT64) */
        { "Softmax(axis 作输入)", OH_NN_OPS_SOFTMAX, .nt = 3,
          .t = { IN_F32(1,4,8), CST_I64_1(2), OUT_F32(1,4,8) },
          .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

        /* Concat: axis 作输入 */
        { "Concat(axis 作输入)", OH_NN_OPS_CONCAT, .nt = 4,
          .t = { IN_F32(1,4,4), IN_F32(1,4,4), CST_I64_1(2), OUT_F32(1,4,8) },
          .in = {0,1,2}, .nin = 3, .out = {3}, .nout = 1 },

        /* Squeeze: 第 2 个输入 = axis */
        { "Squeeze(axis 作输入)", OH_NN_OPS_SQUEEZE, .nt = 3,
          .t = { IN_F32(1,4,8), CST_I64_1(1),
                 { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

        /* Stack: 最后一个输入 = axis */
        { "Stack(axis 作输入)", OH_NN_OPS_STACK, .nt = 4,
          .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 CST_I64_1(0),
                 { .d = {2,4,8}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1,2}, .nin = 3, .out = {3}, .nout = 1 },

        /* Gather: 3 个输入 data/indices/axis */
        { "Gather(3 输入, axis INT64)", OH_NN_OPS_GATHER, .nt = 4,
          .t = { { .d = {16,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 { .d = {4}, .nd = 1, .dt = OH_NN_INT64, .role = R_IN },
                 CST_I64_1(0),
                 { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1,2}, .nin = 3, .out = {3}, .nout = 1 },

        /* ReduceMean: axes 作输入，keep_dims 作 BOOL 参数 */
        { "ReduceMean(axes 输入 + keep BOOL)", OH_NN_OPS_REDUCE_MEAN, .nt = 4,
          .t = { IN_F32(1,4,8), CST_I64_1(2),
                 { .d = {1}, .nd = 1, .dt = OH_NN_BOOL, .role = R_PAR, .tt = OH_NN_REDUCE_MEAN_KEEP_DIMS, .iv = {1}, .nval = 1 },
                 { .d = {1,4,1}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1}, .nin = 2, .out = {3}, .nout = 1, .par = {2}, .npar = 1 },

        /* GELU: approximate 作 BOOL 参数 */
        { "GELU(approximate BOOL)", OH_NN_OPS_GELU, .nt = 3,
          .t = { IN_F32(1,4,8), PAR_BOOL(0), OUT_F32(1,4,8) },
          .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

        /* Transpose: perm 作输入 */
        { "Transpose(perm 作输入)", OH_NN_OPS_TRANSPOSE, .nt = 3,
          .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {1,0}, .nval = 2 },
                 { .d = {8,4}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1}, .nin = 2, .out = {2}, .nout = 1 },

        /* LayerNorm: gamma/beta 作输入，3 个参数 */
        { "LayerNorm(参数 3 个)", OH_NN_OPS_LAYER_NORM, .nt = 7,
          .t = { IN_F32(1,4,8),
                 { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CST, .fv = {1,1,1,1,1,1,1,1}, .nval = 8 },
                 { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CST, .fv = {0,0,0,0,0,0,0,0}, .nval = 8 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_LAYER_NORM_BEGIN_NORM_AXIS, .iv = {2}, .nval = 1 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS, .iv = {2}, .nval = 1 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_PAR, .tt = OH_NN_LAYER_NORM_EPSILON, .fv = {1e-5f}, .nval = 1 },
                 OUT_F32(1,4,8) },
          .in = {0,1,2}, .nin = 3, .out = {6}, .nout = 1, .par = {3,4,5}, .npar = 3 },

        /* Split: axis 作输入 + outputNum 作输入 */
        { "Split(axis+num 作输入)", OH_NN_OPS_SPLIT, .nt = 5,
          .t = { IN_F32(1,4,8), CST_I64_1(2), CST_I64_1(2),
                 { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
                 { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1,2}, .nin = 3, .out = {3,4}, .nout = 2 },

        /* Slice: begin/size/axes 作输入 */
        { "Slice(begin/size/axes 输入)", OH_NN_OPS_SLICE, .nt = 5,
          .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {0,0}, .nval = 2 },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {2,8}, .nval = 2 },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {0,1}, .nval = 2 },
                 { .d = {2,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1,2,3}, .nin = 4, .out = {4}, .nout = 1 },

        /* MatMul: 3 个参数（transposeA/B BOOL + activation） */
        { "MatMul(3 参数)", OH_NN_OPS_MATMUL, .nt = 6,
          .t = { IN_F32(1,4,8), { .d = {8,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN }, OUT_F32(1,4,8),
                 { .d = {1}, .nd = 1, .dt = OH_NN_BOOL, .role = R_PAR, .tt = OH_NN_MATMUL_TRANSPOSE_A, .iv = {0}, .nval = 1 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_BOOL, .role = R_PAR, .tt = OH_NN_MATMUL_TRANSPOSE_B, .iv = {0}, .nval = 1 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT8, .role = R_PAR, .tt = OH_NN_MATMUL_ACTIVATION_TYPE, .iv = {0}, .nval = 1 } },
          .in = {0,1}, .nin = 2, .out = {2}, .nout = 1, .par = {3,4,5}, .npar = 3 },

        /* ---- 按源码规格重建：轴 = INT64 标量【参数】 ---- */
        { "Softmax(axis 作参数 INT64)", OH_NN_OPS_SOFTMAX, .nt = 3,
          .t = { IN_F32(1,4,8),
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_SOFTMAX_AXIS, .iv = {2}, .nval = 1 },
                 OUT_F32(1,4,8) },
          .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

        { "Concat(axis 作参数 INT64)", OH_NN_OPS_CONCAT, .nt = 4,
          .t = { IN_F32(1,4,4), IN_F32(1,4,4),
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_CONCAT_AXIS, .iv = {2}, .nval = 1 },
                 OUT_F32(1,4,8) },
          .in = {0,1}, .nin = 2, .out = {3}, .nout = 1, .par = {2}, .npar = 1 },

        { "Squeeze(axis 作参数 INT64)", OH_NN_OPS_SQUEEZE, .nt = 3,
          .t = { IN_F32(1,4,8),
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_SQUEEZE_AXIS, .iv = {1}, .nval = 1 },
                 { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0}, .nin = 1, .out = {2}, .nout = 1, .par = {1}, .npar = 1 },

        { "Slice(3 输入 + axes 参)", OH_NN_OPS_SLICE, .nt = 5,
          .t = { { .d = {4,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_IN },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {0,0}, .nval = 2 },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_CST, .iv = {2,8}, .nval = 2 },
                 { .d = {2}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_SLICE_AXES, .iv = {0,1}, .nval = 2 },
                 { .d = {2,8}, .nd = 2, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0,1,2}, .nin = 3, .out = {4}, .nout = 1, .par = {3}, .npar = 1 },

        { "Split(1 输入 + axis/outputNum 参)", OH_NN_OPS_SPLIT, .nt = 5,
          .t = { IN_F32(1,4,8),
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_SPLIT_AXIS, .iv = {2}, .nval = 1 },
                 { .d = {1}, .nd = 1, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_SPLIT_OUTPUT_NUM, .iv = {2}, .nval = 1 },
                 { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT },
                 { .d = {1,4,4}, .nd = 3, .dt = OH_NN_FLOAT32, .role = R_OUT } },
          .in = {0}, .nin = 1, .out = {3,4}, .nout = 2, .par = {1,2}, .npar = 2 },

        { "LayerNorm(标量参数 nd=0)", OH_NN_OPS_LAYER_NORM, .nt = 7,
          .t = { IN_F32(1,4,8),
                 { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CST, .fv = {1,1,1,1,1,1,1,1}, .nval = 8 },
                 { .d = {8}, .nd = 1, .dt = OH_NN_FLOAT32, .role = R_CST, .fv = {0,0,0,0,0,0,0,0}, .nval = 8 },
                 { .d = {1}, .nd = 0, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_LAYER_NORM_BEGIN_NORM_AXIS, .iv = {2}, .nval = 1 },
                 { .d = {1}, .nd = 0, .dt = OH_NN_INT64, .role = R_PAR, .tt = OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS, .iv = {2}, .nval = 1 },
                 { .d = {1}, .nd = 0, .dt = OH_NN_FLOAT32, .role = R_PAR, .tt = OH_NN_LAYER_NORM_EPSILON, .fv = {1e-5f}, .nval = 1 },
                 OUT_F32(1,4,8) },
          .in = {0,1,2}, .nin = 3, .out = {6}, .nout = 1, .par = {3,4,5}, .npar = 3 },
    };

    int ok = 0, tot = (int) (sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < tot; i++) {
        g_err = "?";
        OH_NN_ReturnCode rc = try_case(&cases[i]);
        int good = (rc == OH_NN_SUCCESS);
        if (good) ok++;
        printf("%-30s %s%s\n", cases[i].name,
               good ? "★★ 可编译 → NPU 支持" : "✗ ",
               good ? "" : g_err);
    }
    printf("\n汇总：%d / %d 可编译到 NPU\n", ok, tot);
    return 0;
}
