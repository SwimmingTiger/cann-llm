/* ctrl_probe.c —— ★逐字照抄官方示例的 fp32 ADD★，用来判断"我的构造写法对不对" ✓
 * 官方示例：输入两个 FLOAT32 [1,2,2,3]，参数张量 INT8 [1]（激活类型），输出 FLOAT32。 */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "neural_network_runtime/neural_network_runtime.h"
#define CKN(rc, msg) do { int _r=(int)(rc); printf("    %-40s rc=%d\n", msg, _r); if(_r) goto done; } while(0)
int main(void) {
    const size_t *ids = NULL; uint32_t n = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &n);
    size_t dev = ids[0];
    for (uint32_t i = 0; i < n; i++) { OH_NN_DeviceType t; OH_NNDevice_GetType(ids[i], &t);
        if ((int)t == 3) dev = ids[i]; }
    printf("  设备 id=%zu\n", dev);

    OH_NNModel *model = OH_NNModel_Construct();
    int32_t dims[4] = {1, 2, 2, 3};
    NN_TensorDesc *td;
    /* 张量 0/1：两个 FLOAT32 输入（官方示例的形状是 4 维 ✓） */
    for (int idx = 0; idx < 2; idx++) {
        td = OH_NNTensorDesc_Create();
        CKN(OH_NNTensorDesc_SetShape(td, dims, 4), "SetShape(in)");
        CKN(OH_NNTensorDesc_SetDataType(td, OH_NN_FLOAT32), "SetDataType(FLOAT32)");
        CKN(OH_NNTensorDesc_SetFormat(td, OH_NN_FORMAT_NONE), "SetFormat");
        CKN(OH_NNModel_AddTensorToModel(model, td), "AddTensor(in)");
        CKN(OH_NNModel_SetTensorType(model, idx, OH_NN_TENSOR), "SetTensorType(TENSOR)");
    }
    /* 张量 2：激活参数，INT8 [1] */
    td = OH_NNTensorDesc_Create();
    int32_t adim = 1;
    CKN(OH_NNTensorDesc_SetShape(td, &adim, 1), "SetShape(act)");
    CKN(OH_NNTensorDesc_SetDataType(td, OH_NN_INT8), "SetDataType(INT8)");
    CKN(OH_NNTensorDesc_SetFormat(td, OH_NN_FORMAT_NONE), "SetFormat(act)");
    CKN(OH_NNModel_AddTensorToModel(model, td), "AddTensor(act)");
    CKN(OH_NNModel_SetTensorType(model, 2, OH_NN_TENSOR), "SetTensorType(act)");
    { int8_t av = 0; CKN(OH_NNModel_SetTensorData(model, 2, &av, sizeof(int8_t)), "SetTensorData(FUSED_NONE)"); }
    /* 张量 3：输出 FLOAT32 [1,2,2,3] */
    td = OH_NNTensorDesc_Create();
    CKN(OH_NNTensorDesc_SetShape(td, dims, 4), "SetShape(out)");
    CKN(OH_NNTensorDesc_SetDataType(td, OH_NN_FLOAT32), "SetDataType(out)");
    CKN(OH_NNTensorDesc_SetFormat(td, OH_NN_FORMAT_NONE), "SetFormat(out)");
    CKN(OH_NNModel_AddTensorToModel(model, td), "AddTensor(out)");
    CKN(OH_NNModel_SetTensorType(model, 3, OH_NN_TENSOR), "SetTensorType(out)");

    uint32_t pi[1] = {2}, ii[2] = {0, 1}, oi[1] = {3};
    OH_NN_UInt32Array par = {pi, 1}, inp = {ii, 2}, outp = {oi, 1};
    CKN(OH_NNModel_AddOperation(model, OH_NN_OPS_ADD, &par, &inp, &outp), "★AddOperation(ADD)★");
    { uint32_t i1[2] = {0, 1}, o1[1] = {3};
      OH_NN_UInt32Array mi = {i1, 2}, mo = {o1, 1};
      CKN(OH_NNModel_SpecifyInputsAndOutputs(model, &mi, &mo), "SpecifyInputsAndOutputs"); }
    CKN(OH_NNModel_Finish(model), "Model_Finish");

    OH_NNCompilation *c = OH_NNCompilation_Construct(model);
    CKN(OH_NNCompilation_SetDevice(c, dev), "SetDevice");
    CKN(OH_NNCompilation_Build(c), "★Build★");
    printf("\n  ★★ fp32 官方写法：编译通过 ✓ ⇒ 我的构造写法是对的 ✓\n");
done:
    printf("  === 结束 ===\n");
    return 0;
}
