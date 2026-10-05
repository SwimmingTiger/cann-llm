// ★算子覆盖率（对齐 §127 能跑通的结构 ✓）：每个算子给足输入张量 + 参数张量 + 常量张量★
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <initializer_list>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

static size_t NPU = 0;
static OH_NNModel *M = nullptr;
static OH_NN_Tensor D[64];        // 持久化描述符 ✓
static int32_t DM[64][4];
static uint32_t NI = 0;

static uint32_t add(const char *nm, OH_NN_DataType dt, uint32_t rk, int32_t a, int32_t b, int32_t c,
                    OH_NN_TensorType ty = OH_NN_TENSOR) {
    uint32_t i = NI++;
    OH_NN_Tensor &t = D[i];
    DM[i][0] = a; DM[i][1] = b; DM[i][2] = c;
    // ★秩 0 的张量必须传 nullptr 的 dimensions✓★（传非空指针 + count=0 会被拒 rc=2 ✗）
    t.dataType = dt; t.dimensionCount = rk; t.dimensions = (rk == 0) ? nullptr : DM[i];
    t.quantParam = nullptr; t.type = ty;
    int r = OH_NNModel_AddTensor(M, &t);
    if (r) printf("    [%s] AddTensor rc=%d\n", nm, r);
    return i;
}
static void data(uint32_t i, const void *p, size_t n) { OH_NNModel_SetTensorData(M, i, p, n); }

// 跑一次：报告 AddOp / Finish / Build ✓
static void go(const char *nm, OH_NN_OperationType op,
               std::initializer_list<uint32_t> prm, std::initializer_list<uint32_t> ins,
               std::initializer_list<uint32_t> outs, uint32_t mio, uint32_t moo) {
    uint32_t pv[8], iv[8], ov[8]; int np = 0, ni = 0, no = 0;
    for (uint32_t v : prm) pv[np++] = v;
    for (uint32_t v : ins) iv[ni++] = v;
    for (uint32_t v : outs) ov[no++] = v;
    OH_NN_UInt32Array P{pv, (uint32_t)np}, I{iv, (uint32_t)ni}, O{ov, (uint32_t)no};
    // ★即使没有参数也要传非空数组✓★（MATMUL 有参数才成功 ⇒ 怀疑 nullptr 被拒 ✗）
    P.size = (uint32_t)np;   // data 已指向 pv（非空 ✓）
    int ra = OH_NNModel_AddOperation(M, op, &P, &I, &O);
    uint32_t a = mio, b = moo; OH_NN_UInt32Array MI{&a, 1}, MO{&b, 1};
    int rs = OH_NNModel_SpecifyInputsAndOutputs(M, &MI, &MO);
    int rf = OH_NNModel_Finish(M);
    OH_NNCompilation *c = OH_NNCompilation_Construct(M);
    int rb = c ? OH_NNCompilation_Build(c) : -1;
    printf("  %-13s AddOp=%d Finish=%d ★Build=%d★ %s\n", nm, ra, rf, rb,
           (ra == 0 && rf == 0 && rb == 0) ? "★可用★" : "✗");
    if (c) OH_NNCompilation_Destroy(&c);
    OH_NNModel_Destroy(&M);
    M = OH_NNModel_Construct(); NI = 0;
}
#define FRESH() do { M = OH_NNModel_Construct(); NI = 0; } while (0)

int main() {
    const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) { const char *n = nullptr; OH_NNDevice_GetName(ids[i], &n);
        if (n && strstr(n, "NPU")) NPU = ids[i]; }
    printf("NPU=%zu\n", NPU);
    float one = 1.0f, zero = 0.0f; int32_t i0 = 0, i2 = 2, i1 = 1, ione = 1;

    // ★对照：MATMUL（§127 已证明能过 ✓）★
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), w = add("w", OH_NN_FLOAT32, 3, 1, 64, 64),
        y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        static bool f = false;
        uint32_t pa = add("pa", OH_NN_BOOL, 0, 0, 0, 0, OH_NN_MATMUL_TRANSPOSE_A);
        uint32_t pb = add("pb", OH_NN_BOOL, 0, 0, 0, 0, OH_NN_MATMUL_TRANSPOSE_B);
        static float WDATA[1 * 64 * 64]; for (int i = 0; i < 1 * 64 * 64; ++i) WDATA[i] = 0.01f;
        data(w, WDATA, sizeof(WDATA));                 // ★完整尺寸✓★
        data(pa, &f, sizeof(f)); data(pb, &f, sizeof(f));
        go("MATMUL", OH_NN_OPS_MATMUL, {pa, pb}, {x, w}, {y}, x, y); }
    // 一元 ✓
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        go("RELU", OH_NN_OPS_RELU, {}, {x}, {y}, x, y); }
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        go("SQRT", OH_NN_OPS_SQRT, {}, {x}, {y}, x, y); }
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        go("SIGMOID", OH_NN_OPS_SIGMOID, {}, {x}, {y}, x, y); }
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        go("NEG", OH_NN_OPS_NEG, {}, {x}, {y}, x, y); }
    // 二元 ✓
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), w = add("w", OH_NN_FLOAT32, 3, 1, 64, 64),
        y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        static float W2[1 * 64 * 64]; for (int i = 0; i < 1 * 64 * 64; ++i) W2[i] = 1.0f;
        data(w, W2, sizeof(W2));
        go("DIV", OH_NN_OPS_DIV, {}, {x, w}, {y}, x, y); }
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), w = add("w", OH_NN_FLOAT32, 3, 1, 64, 64),
        y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        static float W3[1 * 64 * 64]; for (int i = 0; i < 1 * 64 * 64; ++i) W3[i] = 0.5f;
        data(w, W3, sizeof(W3));
        go("ADD", OH_NN_OPS_ADD, {}, {x, w}, {y}, x, y); }
    // 形状类 ✓
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64),
        ax = add("ax", OH_NN_INT32, 1, 1, 0, 0, OH_NN_CONCAT_AXIS);
        data(ax, &i2, sizeof(i2));
        go("CONCAT", OH_NN_OPS_CONCAT, {ax}, {x, x}, {y}, x, y); }
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), sh = add("sh", OH_NN_INT32, 1, 3, 0, 0),
        y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64); int32_t s[3] = {1, 64, 64}; data(sh, s, sizeof(s));
        go("RESHAPE", OH_NN_OPS_RESHAPE, {}, {x, sh}, {y}, x, y); }
    // 归约 ✓
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), ax = add("ax", OH_NN_INT32, 1, 1, 0, 0),
        y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64),
        kp = add("kp", OH_NN_BOOL, 0, 0, 0, 0, OH_NN_REDUCE_MEAN_KEEP_DIMS);
        int32_t k = 1; data(ax, &i2, sizeof(i2)); data(kp, &k, sizeof(k));
        go("REDUCE_MEAN", OH_NN_OPS_REDUCE_MEAN, {kp}, {x, ax}, {y}, x, y); }
    // SOFTPLUS（不存在 ⇒ 预期失败 ✓ 用一个非法号验证报错路径 ✓）
    FRESH(); { uint32_t x = add("x", OH_NN_FLOAT32, 3, 1, 64, 64), y = add("y", OH_NN_FLOAT32, 3, 1, 64, 64);
        go("SOFTPLUS(999)", (OH_NN_OperationType)999, {}, {x}, {y}, x, y); }
    return 0;
}
