/* mslite_run.c —— 在鸿蒙设备上用 MindSpore Lite NDK（C API）加载 .ms，用 NNRt 后端推理。
 *
 * 配套文档：docs/offline-model-nnrt.md
 *
 * 用法：
 *   cc -O1 -I<sysroot>/usr/include mslite_run.c -o mslite_run \
 *      -L/system/lib64/ndk -lmindspore_lite_ndk
 *   LD_LIBRARY_PATH=/system/lib64/ndk ./mslite_run <model.ms> [nnrt|cpu]
 *
 * 默认走 NNRt 后端（OH_AI_DEVICETYPE_NNRT = 60）。传 cpu 可作对照：
 * 离线模型被包成 custom 算子、CPU 侧没有实现，输出通常全是 0；
 * 而 NNRt 后端能拿到正确数值，才说明整条链真的通了。
 */
#include <stdio.h>
#include <string.h>
#include "mindspore/context.h"
#include "mindspore/model.h"
#include "mindspore/tensor.h"
#include "mindspore/types.h"

static void dump_shape(OH_AI_TensorHandle t) {
    size_t nd = 0;
    const int64_t *sh = OH_AI_TensorGetShape(t, &nd);
    printf("[");
    for (size_t i = 0; i < nd; i++) printf("%lld%s", (long long) sh[i], i + 1 < nd ? "," : "");
    printf("]");
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("用法: %s <model.ms> [nnrt|cpu]\n", argv[0]); return 2; }
    int use_nnrt = !(argc > 2 && strcmp(argv[2], "cpu") == 0);

    OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
    OH_AI_DeviceInfoHandle dev = OH_AI_DeviceInfoCreate(
        use_nnrt ? OH_AI_DEVICETYPE_NNRT : OH_AI_DEVICETYPE_CPU);
    OH_AI_ContextAddDeviceInfo(ctx, dev);
    printf("后端: %s (devicetype=%d)\n", use_nnrt ? "NNRT" : "CPU",
           (int) (use_nnrt ? OH_AI_DEVICETYPE_NNRT : OH_AI_DEVICETYPE_CPU));

    OH_AI_ModelHandle m = OH_AI_ModelCreate();
    OH_AI_Status st = OH_AI_ModelBuildFromFile(m, argv[1], OH_AI_MODELTYPE_MINDIR, ctx);
    printf("OH_AI_ModelBuildFromFile -> %d %s\n", (int) st,
           st == OH_AI_STATUS_SUCCESS ? "(SUCCESS)" : "(FAILED)");
    if (st != OH_AI_STATUS_SUCCESS) return 1;

    OH_AI_TensorHandleArray ins = OH_AI_ModelGetInputs(m);
    OH_AI_TensorHandleArray outs = OH_AI_ModelGetOutputs(m);
    printf("输入 %zu 个 / 输出 %zu 个\n", ins.handle_num, outs.handle_num);

    for (size_t i = 0; i < ins.handle_num; i++) {
        OH_AI_TensorHandle t = ins.handle_list[i];
        printf("  in[%zu] %-8s dtype=%d shape=", i, OH_AI_TensorGetName(t),
               (int) OH_AI_TensorGetDataType(t));
        dump_shape(t);
        int64_t n = OH_AI_TensorGetElementNum(t);
        float *p = (float *) OH_AI_TensorGetMutableData(t);
        printf(" 元素=%lld 数据指针=%p\n", (long long) n, (void *) p);
        if (p) for (int64_t k = 0; k < n; k++) p[k] = 1.0f;   /* x1 = x2 = 1 ⇒ y = 2 */
    }

    st = OH_AI_ModelPredict(m, ins, &outs, NULL, NULL);
    printf("OH_AI_ModelPredict -> %d %s\n", (int) st,
           st == OH_AI_STATUS_SUCCESS ? "(SUCCESS)" : "(FAILED)");

    for (size_t i = 0; i < outs.handle_num; i++) {
        OH_AI_TensorHandle t = outs.handle_list[i];
        printf("  out[%zu] %-8s shape=", i, OH_AI_TensorGetName(t));
        dump_shape(t);
        int64_t n = OH_AI_TensorGetElementNum(t);
        const float *p = (const float *) OH_AI_TensorGetData(t);
        printf(" →");
        for (int64_t k = 0; k < n; k++) printf(" %.3f", p ? (double) p[k] : -1.0);
        printf("   （期望 %.1f）\n", 2.0);
    }
    OH_AI_ModelDestroy(&m);
    OH_AI_ContextDestroy(&ctx);
    return 0;
}
