/* avail_probe.c —— ★问 NPU：我图里的这些算子你支持吗★（OH_NNModel_GetAvailableOperations） */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "neural_network_runtime/neural_network_runtime.h"

static const char *opname(int op) {
    switch (op) {
        case 1: return "ADD"; case 2: return "AVG_POOL"; case 19: return "MATMUL";
        case 52: return "QUANT_DTYPE_CAST"; default: return "?";
    }
}
static int add1(OH_NNModel *m, const char *n, OH_NN_DataType dt, const int32_t *sh, size_t r) {
    NN_TensorDesc *d = OH_NNTensorDesc_Create();
    OH_NNTensorDesc_SetName(d, n); OH_NNTensorDesc_SetDataType(d, dt);
    OH_NNTensorDesc_SetShape(d, sh, r); OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE);
    int rc = (int)OH_NNModel_AddTensorToModel(m, d); OH_NNTensorDesc_Destroy(&d); return rc;
}
int main(void) {
    const size_t *ids = NULL; uint32_t n = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &n);
    if (!n) { printf("  没有设备\n"); return 1; }
    size_t dev = ids[0];
    for (uint32_t i = 0; i < n; i++) { OH_NN_DeviceType t; OH_NNDevice_GetType(ids[i], &t);
        if ((int)t == 3) dev = ids[i]; }
    printf("  设备 id=%zu\n", dev);

    for (int variant = 0; variant < 2; variant++) {
        OH_NNModel *m = OH_NNModel_Construct();
        const int32_t s2[2] = {2, 2}, s1[1] = {1};
        add1(m, "a", OH_NN_INT8, s2, 2);
        add1(m, "b", OH_NN_INT8, s2, 2);
        add1(m, "p", OH_NN_INT8, s1, 1);
        add1(m, "y", OH_NN_INT8, s2, 2);
        { int8_t v[4] = {1,2,3,4}; OH_NNModel_SetTensorData(m, 1, v, 4);
          int8_t a = 0; OH_NNModel_SetTensorData(m, 2, &a, 1); }
        uint32_t pi[1] = {2}, ii[2] = {0,1}, oi[1] = {3};
        OH_NN_UInt32Array par = {pi,1}, inp = {ii,2}, outp = {oi,1};
        OH_NN_OperationType op = variant ? OH_NN_OPS_MATMUL : OH_NN_OPS_ADD;
        int ra = (int)OH_NNModel_AddOperation(m, op, &par, &inp, &outp);
        printf("\n  ★ 试 %s（int8 张量）: AddOperation rc=%d\n", opname((int)op), ra);
        if (ra == 0) {
            uint32_t i1[1] = {0}, o1[1] = {3};
            OH_NN_UInt32Array mi = {i1,1}, mo = {o1,1};
            OH_NNModel_SpecifyInputsAndOutputs(m, &mi, &mo);
            int rf = (int)OH_NNModel_Finish(m);
            printf("     Finish rc=%d\n", rf);
            const bool *sup = NULL; uint32_t cnt = 0;
            int rg = (int)OH_NNModel_GetAvailableOperations(m, dev, &sup, &cnt);
            printf("     ★GetAvailableOperations rc=%d 支持状态数=%u★\n", rg, cnt);
            for (uint32_t i = 0; i < cnt; i++) printf("        第%u个算子: %s\n", i, sup[i] ? "★支持✓" : "不支持✗");
        }
        OH_NNModel_Destroy(&m);
    }
    return 0;
}
