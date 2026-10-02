/* mlp_run.c —— 用 NNRt 后端跑带权重的离线模型 mlp_w.ms，并与参考输出比对 */
#include <stdio.h>
#include <math.h>
#include "mindspore/context.h"
#include "mindspore/model.h"
#include "mindspore/tensor.h"
#include "mindspore/types.h"

static const float X[16] = { -1.16512012f,-0.90826511f,0.44977710f,-3.19734550f,-1.09267867f,0.79548407f,-0.58673990f,-1.62648308f,1.92556667f,-1.41053522f,-0.52336556f,-0.37273151f,0.08314440f,-0.36950520f,-0.08096508f,0.05749458f };
static const float REF[16] = { -0.07954413f,0.02162828f,0.08233339f,-0.13554707f,-0.46802299f,-0.11318561f,-0.12096452f,-0.19175090f,-0.12703167f,-0.12170612f,-0.08568357f,0.14427299f,-0.13003812f,-0.13615016f,-0.28488797f,0.31247185f };

int main(int argc, char **argv) {
    const char *mp = argc > 1 ? argv[1] : "mlp_w.ms";
    int nnrt = !(argc > 2 && argv[2][0]=='c');
    OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
    OH_AI_ContextAddDeviceInfo(ctx, OH_AI_DeviceInfoCreate(
        nnrt ? OH_AI_DEVICETYPE_NNRT : OH_AI_DEVICETYPE_CPU));
    OH_AI_ModelHandle m = OH_AI_ModelCreate();
    OH_AI_Status st = OH_AI_ModelBuildFromFile(m, mp, OH_AI_MODELTYPE_MINDIR, ctx);
    printf("后端=%s  Build -> %d %s\n", nnrt?"NNRT":"CPU", (int)st,
           st==OH_AI_STATUS_SUCCESS?"(SUCCESS)":"(FAILED)");
    if (st != OH_AI_STATUS_SUCCESS) return 1;
    OH_AI_TensorHandleArray in = OH_AI_ModelGetInputs(m), out = OH_AI_ModelGetOutputs(m);
    float *p = (float *) OH_AI_TensorGetMutableData(in.handle_list[0]);
    for (int i = 0; i < 16; i++) p[i] = X[i];
    OH_AI_ModelPredict(m, in, &out, NULL, NULL);
    const float *q = (const float *) OH_AI_TensorGetData(out.handle_list[0]);
    double maxd = 0; int bad = 0;
    printf("  NPU :"); for (int i=0;i<4;i++) printf(" %8.4f", (double)q[i]); printf(" …\n");
    printf("  参考:"); for (int i=0;i<4;i++) printf(" %8.4f", (double)REF[i]); printf(" …\n");
    for (int i = 0; i < 16; i++) { double d = fabs((double)q[i]-REF[i]); if (d>maxd) maxd=d; if (d>1e-2) bad++; }
    printf("  最大偏差 %.6f · 超差(>1e-2) %d/16 %s\n", maxd, bad, bad==0?"★ 与参考一致 ✓":"✗ 不一致");
    return 0;
}
