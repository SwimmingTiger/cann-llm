/* int8_probe.c —— ★最小 int8 图（在线构图 + 张量量化参数）能否在 NPU 上编译执行★
 *
 * 背景：本设备的 hiai foundation 明确不支持"扩展配置"✗ ⇒ dopt 的 QuantConfigData 那条
 *       int8 路在设备上必死 ✓。官方文档指出另一条通道：★量化参数挂在张量上★
 *       （NN_QuantParam + OH_NNModel_SetTensorQuantParams ✓），★不走扩展配置★ ✓。
 *
 * 本程序：搭一个 int8 ADD（两个 INT8 [2,2] 输入 + 一个 INT8 标量参数 + INT8 输出），
 *         给量化张量挂上 NN_QuantParam，然后★编译★。
 *
 * Build: cc -O1 -Iinc int8_probe.c -o bin/int8_probe \
 *            -L/system/lib64/ndk -lneural_network_runtime -lneural_network_core -lm
 * Run:   LD_LIBRARY_PATH=/system/lib64/ndk ./bin/int8_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "neural_network_runtime/neural_network_runtime.h"

static const char *rcn(int rc) {
    switch (rc) {
        case 0: return "SUCCESS";
        case 1: return "FAILED";
        case 2: return "INVALID_PARAMETER";
        case 3: return "MEMORY_ERROR";
        case 4: return "OPERATION_FORBIDDEN";
        case 5: return "NULL_PTR";
        default: return "?";
    }
}
#define CK(expr, what) do { int _r = (int)(expr); printf("    %-30s rc=%d(%s)\n", what, _r, rcn(_r)); \
                            if (_r != 0) goto done; } while (0)

static uint32_t g_idx = 0;   /* 张量加入顺序 = 模型内的索引 ✓ */

/* 建一个张量并加入模型；返回 0 表示成功 */
static int add_tensor(OH_NNModel *m, const char *name, OH_NN_DataType dt,
                      const int32_t *shape, size_t rank, OH_NN_TensorType ttype) {
    NN_TensorDesc *d = OH_NNTensorDesc_Create();
    if (!d) return -1;
    OH_NNTensorDesc_SetName(d, name);
    OH_NNTensorDesc_SetDataType(d, dt);
    OH_NNTensorDesc_SetShape(d, shape, rank);
    OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
    int rc = (int)OH_NNModel_AddTensorToModel(m, d);
    OH_NNTensorDesc_Destroy(&d);
    /* ★官方示例：每加一个张量后都要 SetTensorType ✓（我原先漏了 ⇒ AddOperation rc=2 ✗）★ */
    int rt = rc;
    if (rc == 0) { rt = (int)OH_NNModel_SetTensorType(m, g_idx, ttype); g_idx++; }   /* ★用真实 type ✓ */
    printf("    AddTensor %-12s rc=%d · SetTensorType rc=%d(%s)\n", name, rc, rt, rcn(rt));
    return rc ? rc : rt;
}

/* 给张量 index 挂量化参数 */
static int set_quant(OH_NNModel *m, uint32_t index, double scale, int32_t zp, uint32_t bits) {
    NN_QuantParam *qp = OH_NNQuantParam_Create();
    if (!qp) return -1;
    int r1 = (int)OH_NNQuantParam_SetNumBits(qp, &bits, 1);
    int r2 = (int)OH_NNQuantParam_SetScales(qp, &scale, 1);
    int r3 = (int)OH_NNQuantParam_SetZeroPoints(qp, &zp, 1);
    int r4 = (int)OH_NNModel_SetTensorQuantParams(m, index, qp);
    OH_NNQuantParam_Destroy(&qp);
    printf("    Quant t%-2u bits=%d/%s scales=%d/%s zp=%d/%s ★SetTensorQuantParams=%d/%s★\n",
           index, r1, rcn(r1), r2, rcn(r2), r3, rcn(r3), r4, rcn(r4));
    return r4;
}

int main(void) {
    const size_t *ids = NULL; uint32_t n = 0;
    int rc = OH_NNDevice_GetAllDevicesID(&ids, &n);
    printf("=== 0) 设备 ===\n    count=%u rc=%d(%s)\n", n, rc, rcn(rc));
    if (!n) return 1;
    size_t dev = ids[0];
    for (uint32_t i = 0; i < n; i++) {
        const char *nm = NULL; OH_NN_DeviceType t;
        OH_NNDevice_GetName(ids[i], &nm); OH_NNDevice_GetType(ids[i], &t);
        printf("    device[%u] id=%zu type=%d(%s) name=%s\n", i, ids[i], (int)t,
               (int)t == 3 ? "NPU" : "?", nm ? nm : "?");
        if ((int)t == 3) dev = ids[i];
    }

    OH_NNModel *model = OH_NNModel_Construct();
    printf("\n=== 1) 构图（int8 MATMUL: x[1,4] @ w[4,4] -> y[1,4]）===\n");
    const int32_t s_x[2] = {1, 4}, s_w[2] = {4, 4}, s_y[2] = {1, 4}, s_p[1] = {1};
    if (add_tensor(model, "x",   OH_NN_INT8,  s_x, 2, OH_NN_TENSOR)) goto done;   /* 0: 输入 */
    if (add_tensor(model, "w",   OH_NN_INT8,  s_w, 2, OH_NN_TENSOR)) goto done;   /* 1: 权重(常量) */
    if (add_tensor(model, "transX", OH_NN_INT32, s_p, 1, OH_NN_MATMUL_TRANSPOSE_A)) goto done;/* 2: 参数 */
    if (add_tensor(model, "transY", OH_NN_INT32, s_p, 1, OH_NN_MATMUL_TRANSPOSE_B)) goto done;/* 3: 参数 */
    if (add_tensor(model, "y",   OH_NN_INT8,  s_y, 2, OH_NN_TENSOR)) goto done;   /* 4: 输出 */

    printf("\n=== 2) ★挂量化参数（不走扩展配置）★ ===\n");
    set_quant(model, 1, 0.05, 0, 8);      /* 权重 */
    set_quant(model, 4, 0.05, 0, 8);      /* ★输出（文档要求不能省）★ */

    printf("\n=== 3) 数据与算子 ===\n");
    { int8_t wv[16]; for (int i = 0; i < 16; i++) wv[i] = (int8_t)(i - 8);
      CK(OH_NNModel_SetTensorData(model, 1, wv, sizeof(wv)), "SetTensorData(w)"); }
    { int32_t z = 0;
      CK(OH_NNModel_SetTensorData(model, 2, &z, sizeof(z)), "SetTensorData(transX=0)");
      CK(OH_NNModel_SetTensorData(model, 3, &z, sizeof(z)), "SetTensorData(transY=0)"); }
    { uint32_t pi[2] = {2, 3}, ii[2] = {0, 1}, oi[1] = {4};
      OH_NN_UInt32Array par = {pi, 2}, inp = {ii, 2}, outp = {oi, 1};
      CK(OH_NNModel_AddOperation(model, OH_NN_OPS_MATMUL, &par, &inp, &outp), "AddOperation(MATMUL)"); }
    { uint32_t ii[1] = {0}, oi[1] = {4};
      OH_NN_UInt32Array mi = {ii, 1}, mo = {oi, 1};
      CK(OH_NNModel_SpecifyInputsAndOutputs(model, &mi, &mo), "SpecifyInputsAndOutputs"); }
    CK(OH_NNModel_Finish(model), "Model_Finish");

    printf("\n=== 4) ★编译（判决点）★ ===\n");
    OH_NNCompilation *c = OH_NNCompilation_Construct(model);
    if (!c) { printf("    ✗ Construct 返回 NULL\n"); goto done; }
    CK(OH_NNCompilation_SetDevice(c, dev), "SetDevice");
    CK(OH_NNCompilation_SetPerformanceMode(c, OH_NN_PERFORMANCE_EXTREME), "SetPerformanceMode");
    CK(OH_NNCompilation_SetPriority(c, OH_NN_PRIORITY_HIGH), "SetPriority");
    CK(OH_NNCompilation_EnableFloat16(c, false), "EnableFloat16(false)");
    CK(OH_NNCompilation_Build(c), "★Build★");
    printf("\n★★★ 编译通过 ⇒ ★int8（张量量化参数）在这台设备上【可行】★ ✓✓\n");
done:
    printf("\n=== 结束 ===\n");
    return 0;
}
