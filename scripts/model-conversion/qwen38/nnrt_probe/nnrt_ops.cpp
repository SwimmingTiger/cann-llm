// ★NNRt 算子覆盖率探针★：每个算子建一个极小在线图 ⇒ Finish + Build(NPU) 判据 ✓
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

static size_t g_npu = 0;
static OH_NNModel *g_m = nullptr;
static int g_idx = 0;

// 加一个张量，返回索引 ✓
static OH_NN_Tensor g_desc[64];          // ★持久化描述符（AddTensor 可能不拷贝 ✓ §131）★
static int32_t g_dims[64][8];            // ★持久化维度数组✓★
static uint32_t T(const char *nm, OH_NN_DataType dt, uint32_t rank, const int32_t *dims,
                  OH_NN_TensorType ty = OH_NN_TENSOR) {
    OH_NN_Tensor &t = g_desc[g_idx];
    for (uint32_t k = 0; k < rank && k < 8; ++k) g_dims[g_idx][k] = dims[k];
    t.dataType = dt; t.dimensionCount = rank; t.dimensions = g_dims[g_idx];
    t.quantParam = nullptr; t.type = ty;
    int r = OH_NNModel_AddTensor(g_m, &t);
    (void)nm;
    if (r != 0) printf("    [%s] AddTensor失败 rc=%d\n", nm, r);
    return (uint32_t)g_idx++;
}
static void DATA(uint32_t i, const void *p, size_t n) { OH_NNModel_SetTensorData(g_m, i, p, n); }

// 用给定 op/参数/输入/输出 ⇒ Finished? Build? ✓
static void chk(const char *name, OH_NN_OperationType op, uint32_t *prm, uint32_t np,
                uint32_t *ins, uint32_t ni, uint32_t *outs, uint32_t no,
                uint32_t modelIn, uint32_t modelOut) {
    OH_NN_UInt32Array P, I, O;
    P.data = prm; P.size = np; I.data = ins; I.size = ni; O.data = outs; O.size = no;
    int ra = OH_NNModel_AddOperation(g_m, op, np ? &P : nullptr, &I, &O);
    uint32_t mi[1] = {modelIn}, mo[1] = {modelOut};
    OH_NN_UInt32Array MI, MO; MI.data = mi; MI.size = 1; MO.data = mo; MO.size = 1;
    int rs = OH_NNModel_SpecifyInputsAndOutputs(g_m, &MI, &MO);
    int rf = OH_NNModel_Finish(g_m);
    OH_NNCompilation *c = OH_NNCompilation_Construct(g_m);
    int rd = c ? OH_NNCompilation_SetDevice(c, g_npu) : -1;
    int rb = c ? OH_NNCompilation_Build(c) : -1;
    printf("  %-14s AddOp=%d Specify=%d Finish=%d Build=%d  %s\n", name, ra, rs, rf, rb,
           (rf == 0 && rb == 0) ? "★可用★" : "✗");
    if (c) OH_NNCompilation_Destroy(&c);
    OH_NNModel_Destroy(&g_m);
}
static void reset() { g_m = OH_NNModel_Construct(); g_idx = 0; }

int main() {
    const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        if (n && strstr(n, "NPU")) g_npu = ids[i];
    }
    printf("NPU id=%zu\n", g_npu);
    int32_t d3[3] = {1, 64, 64}, d1[1] = {1}, d2[2] = {64, 64}, d3b[3] = {64, 64, 1};
    float one = 1.0f, half = 0.5f, zero = 0.0f;
    int32_t ax0 = 0, ax2 = 2, axm1 = -1, keep = 1;
    int32_t oneI = 1, twoI = 2;

    // 一元算子 ✓
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        uint32_t i2[2] = {x, x};
        chk("SQRT", OH_NN_OPS_SQRT, nullptr, 0, &x, 1, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        chk("EXP", OH_NN_OPS_EXP, nullptr, 0, &x, 1, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        chk("SIGMOID", OH_NN_OPS_SIGMOID, nullptr, 0, &x, 1, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        chk("NEG", OH_NN_OPS_NEG, nullptr, 0, &x, 1, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), c = T("c", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        DATA(c, &one, sizeof(one)); uint32_t i2[2] = {x, c};
        chk("DIV", OH_NN_OPS_DIV, nullptr, 0, i2, 2, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), c = T("c", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        DATA(c, &twoI /*占位不改*/ , 0); uint32_t i2[2] = {x, c};
        chk("POW", OH_NN_OPS_POW, nullptr, 0, i2, 2, &y, 1, x, y); }
    // 归约 ✓（axes 作为输入 ✓ KEEP_DIMS 参数 ✓）
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), a = T("a", OH_NN_INT32, 1, d1),
        y = T("y", OH_NN_FLOAT32, 3, d3), kp = T("kp", OH_NN_BOOL, 0, nullptr, OH_NN_REDUCE_MEAN_KEEP_DIMS);
        DATA(a, &ax2, sizeof(ax2)); DATA(kp, &keep, sizeof(keep)); uint32_t i2[2] = {x, a}, p1[1] = {kp};
        chk("REDUCE_MEAN", OH_NN_OPS_REDUCE_MEAN, p1, 1, i2, 2, &y, 1, x, y); }
    // 形状/数据搬运 ✓
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3),
        ax = T("ax", OH_NN_INT32, 1, d1, OH_NN_CONCAT_AXIS);
        DATA(ax, &ax2, sizeof(ax2)); uint32_t i2[2] = {x, x}, p1[1] = {ax};
        chk("CONCAT", OH_NN_OPS_CONCAT, p1, 1, i2, 2, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), sh = T("sh", OH_NN_INT32, 1, d2),
        y = T("y", OH_NN_FLOAT32, 3, d3); DATA(sh, d3, sizeof(d3));
        uint32_t i2[2] = {x, sh};
        chk("RESHAPE", OH_NN_OPS_RESHAPE, nullptr, 0, i2, 2, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), pm = T("pm", OH_NN_INT32, 1, d3),
        y = T("y", OH_NN_FLOAT32, 3, d3); int32_t perm[3] = {0, 2, 1}; DATA(pm, perm, sizeof(perm));
        uint32_t i2[2] = {x, pm};
        chk("TRANSPOSE", OH_NN_OPS_TRANSPOSE, nullptr, 0, i2, 2, &y, 1, x, y); }
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3),
        ax = T("ax", OH_NN_INT32, 1, d1, OH_NN_UNSQUEEZE_AXIS);
        DATA(ax, &ax0, sizeof(ax0)); uint32_t p1[1] = {ax};
        chk("UNSQUEEZE", OH_NN_OPS_UNSQUEEZE, p1, 1, &x, 1, &y, 1, x, y); }
    // SOFTPLUS（预期失败 ✓）
    reset(); { uint32_t x = T("x", OH_NN_FLOAT32, 3, d3), y = T("y", OH_NN_FLOAT32, 3, d3);
        chk("SOFTPLUS", (OH_NN_OperationType)0xFFFF, nullptr, 0, &x, 1, &y, 1, x, y); }
    return 0;
}
