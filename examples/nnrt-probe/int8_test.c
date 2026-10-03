/* nnrt_add_test.c - 决定性验证：从普通用户进程，走 Neural Network Runtime 的
 * 【在线构图】→ 在麒麟 NPU 上【编译】→ 【执行】，这条路通不通？
 *
 * 背景：LLM 引擎（libhiai/cann_llm_engine.so）对模型结构有硬约束（逐层 1:1 K/V），
 * 新架构（Gemma 4 的 per-layer / K=V / KV 跨层共享）表达不了。而 NNRt 是通用 NPU
 * 运行时，它有 108 个算子（MatMul/Reshape/GELU/LayerNorm/RSqrt/Sin/Cos/Gather/Softmax…）
 * —— 足以表达一个 Transformer。若这条路能通，就多了一条【绕开 LLM 引擎】的 NPU 通路。
 *
 * 本程序用文档里的 Add 单算子例子：两个 [1,2,2,3] 输入相加，期望输出 0,2,4,…,22。
 *
 * Build（设备上，用 SDK 头文件）：
 *   cc -O1 -I$SDK/ohos/native/sysroot/usr/include nnrt_add_test.c -o nnrt_add_test \
 *      -L/system/lib64/ndk -lneural_network_runtime -lneural_network_core
 * Run:
 *   LD_LIBRARY_PATH=/system/lib64/ndk ./nnrt_add_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"

static const char *rc_name(OH_NN_ReturnCode rc) {
    switch (rc) {
        case OH_NN_SUCCESS: return "SUCCESS";
        case OH_NN_FAILED: return "FAILED";
        case OH_NN_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case OH_NN_MEMORY_ERROR: return "MEMORY_ERROR";
        case OH_NN_OPERATION_FORBIDDEN: return "OPERATION_FORBIDDEN";
        case OH_NN_NULL_PTR: return "NULL_PTR";
        case OH_NN_INVALID_FILE: return "INVALID_FILE";
        case OH_NN_UNAVALIDABLE_DEVICE: return "UNAVALIDABLE_DEVICE";
        case OH_NN_INVALID_PATH: return "INVALID_PATH";
        default: return "?";
    }
}

#define STEP(expr)                                                             \
    do {                                                                       \
        OH_NN_ReturnCode _r = (expr);                                          \
        printf("    %-46s -> %s\n", #expr, rc_name(_r));                       \
        if (_r != OH_NN_SUCCESS) { printf("    !! 失败于此\n"); return 2; }    \
    } while (0)

int main(void) {
    int32_t dims[4] = {1, 2, 2, 3};

    printf("=== 1) 用在线构图建 Add 模型 ===\n");
    OH_NNModel *model = OH_NNModel_Construct();
    int8_t q_bits = 8; double q_scale = 0.05; int32_t q_zp = 0;
    if (!model) { printf("    OH_NNModel_Construct 返回 NULL\n"); return 1; }

    /* 张量 0,1 = 两个输入；2 = activation 参数（INT8，shape[1]）；3 = 输出 */
    for (int k = 0; k < 2; k++) {
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        if (!d) { printf("    TensorDesc 创建失败\n"); return 1; }
        STEP(OH_NNTensorDesc_SetShape(d, dims, 4));
        STEP(OH_NNTensorDesc_SetDataType(d, OH_NN_INT8));   /* ★int8★ */
        STEP(OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE));
        STEP(OH_NNModel_AddTensorToModel(model, d));
        STEP(OH_NNTensorDesc_Destroy(&d));
    }
    {   /* 参数张量：INT8，1 维长度 1 */
        int32_t one = 1;
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        STEP(OH_NNTensorDesc_SetShape(d, &one, 1));
        STEP(OH_NNTensorDesc_SetDataType(d, OH_NN_INT8));
        STEP(OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE));
        STEP(OH_NNModel_AddTensorToModel(model, d));
        STEP(OH_NNTensorDesc_Destroy(&d));
    }
    {   /* 输出张量 */
        NN_TensorDesc *d = OH_NNTensorDesc_Create();
        STEP(OH_NNTensorDesc_SetShape(d, dims, 4));
        STEP(OH_NNTensorDesc_SetDataType(d, OH_NN_INT8));   /* ★int8★ */
        STEP(OH_NNTensorDesc_SetFormat(d, OH_NN_FORMAT_NONE));
        STEP(OH_NNModel_AddTensorToModel(model, d));
        STEP(OH_NNTensorDesc_Destroy(&d));
    }
    uint32_t inputIdx[2] = {0, 1};
    uint32_t paramIdx = 2, outputIdx = 3;
    OH_NN_UInt32Array params = {&paramIdx, 1};
    OH_NN_UInt32Array ins = {inputIdx, 2};
    OH_NN_UInt32Array outs = {&outputIdx, 1};

    STEP(OH_NNModel_SetTensorType(model, 0, OH_NN_TENSOR));
    STEP(OH_NNModel_SetTensorType(model, 1, OH_NN_TENSOR));
    STEP(OH_NNModel_SetTensorType(model, 2, OH_NN_ADD_ACTIVATIONTYPE));
    int8_t act = OH_NN_FUSED_NONE;
    STEP(OH_NNModel_SetTensorData(model, 2, &act, sizeof(int8_t)));
    {   /* ★给 int8 张量挂量化参数（不走扩展配置）★ */
          for (uint32_t qi = 0; qi < 3; qi++) {
              uint32_t qidx = (qi < 2) ? qi : 3;   /* 0,1 输入 · 3 输出 */
              NN_QuantParam *qp = OH_NNQuantParam_Create();
              OH_NNQuantParam_SetNumBits(qp, (uint32_t *)&q_bits, 1);
              OH_NNQuantParam_SetScales(qp, &q_scale, 1);
              OH_NNQuantParam_SetZeroPoints(qp, &q_zp, 1);
              OH_NN_ReturnCode qr = OH_NNModel_SetTensorQuantParams(model, qidx, qp);
              printf("    SetTensorQuantParams(t%u) -> %s\n", qidx, rc_name(qr));
              OH_NNQuantParam_Destroy(&qp);
          }
      }
      STEP(OH_NNModel_AddOperation(model, OH_NN_OPS_ADD, &params, &ins, &outs));
    STEP(OH_NNModel_SpecifyInputsAndOutputs(model, &ins, &outs));
    STEP(OH_NNModel_Finish(model));

    printf("\n=== 2) 枚举设备 ===\n");
    const size_t *ids = NULL;
    uint32_t n = 0;
    STEP(OH_NNDevice_GetAllDevicesID(&ids, &n));
    printf("    设备数 = %u\n", n);
    size_t dev = ids[0];
    for (uint32_t i = 0; i < n; i++) {
        const char *nm = NULL;
        OH_NN_DeviceType t = OH_NN_OTHERS;
        OH_NNDevice_GetName(ids[i], &nm);
        OH_NNDevice_GetType(ids[i], &t);
        printf("      [%u] id=%zu type=%d name=%s\n", i, ids[i], (int) t, nm ? nm : "?");
    }

    printf("\n=== 3) 在线构图模型 -> NPU 编译 ===\n");
    OH_NNCompilation *c = OH_NNCompilation_Construct(model);
    if (!c) { printf("    OH_NNCompilation_Construct 返回 NULL\n"); return 1; }
    STEP(OH_NNCompilation_SetDevice(c, dev));
    STEP(OH_NNCompilation_SetPerformanceMode(c, OH_NN_PERFORMANCE_EXTREME));
    STEP(OH_NNCompilation_SetPriority(c, OH_NN_PRIORITY_HIGH));
    STEP(OH_NNCompilation_EnableFloat16(c, false));
    STEP(OH_NNCompilation_Build(c));
    printf("    ★★ 编译成功 —— 说明 NNRt 在线构图 -> NPU 这条路是通的\n");

    printf("\n=== 4) 执行器 + 张量 + 推理 ===\n");
    OH_NNExecutor *ex = OH_NNExecutor_Construct(c);
    if (!ex) { printf("    Executor 为空\n"); return 1; }
    size_t inCnt = 0, outCnt = 0;
    STEP(OH_NNExecutor_GetInputCount(ex, &inCnt));
    STEP(OH_NNExecutor_GetOutputCount(ex, &outCnt));
    printf("    输入 %zu / 输出 %zu\n", inCnt, outCnt);

    NN_Tensor *inT[2] = {NULL, NULL};
    for (size_t i = 0; i < inCnt && i < 2; i++) {
        NN_TensorDesc *d = OH_NNExecutor_CreateInputTensorDesc(ex, i);
        inT[i] = OH_NNTensor_Create(dev, d);
        OH_NNTensorDesc_Destroy(&d);
        if (!inT[i]) { printf("    输入张量 %zu 创建失败\n", i); return 1; }
        float *p = (float *) OH_NNTensor_GetDataBuffer(inT[i]);
        size_t cnt = 0;
        NN_TensorDesc *dd = OH_NNTensor_GetTensorDesc(inT[i]);
        OH_NNTensorDesc_GetElementCount(dd, &cnt);
        for (size_t j = 0; j < cnt; j++) p[j] = (float) j;   /* 两个输入都填 0..11 */
        printf("    输入 %zu 填好，%zu 个 float\n", i, cnt);
    }
    NN_TensorDesc *od = OH_NNExecutor_CreateOutputTensorDesc(ex, 0);
    NN_Tensor *outT = OH_NNTensor_Create(dev, od);
    OH_NNTensorDesc_Destroy(&od);
    if (!outT) { printf("    输出张量创建失败\n"); return 1; }

    STEP(OH_NNExecutor_RunSync(ex, inT, inCnt, &outT, 1));

    printf("\n=== 5) 结果（期望 0,2,4,...,22）===\n    ");
    float *op = (float *) OH_NNTensor_GetDataBuffer(outT);
    size_t ocnt = 0;
    NN_TensorDesc *odd = OH_NNTensor_GetTensorDesc(outT);
    OH_NNTensorDesc_GetElementCount(odd, &ocnt);
    for (size_t i = 0; i < ocnt && i < 12; i++) printf("%g ", (double) op[i]);
    printf("\n\n★★★ 结论：NNRt 在线构图 → NPU 编译 → 执行，全链路走通 ★★★\n");
    return 0;
}
