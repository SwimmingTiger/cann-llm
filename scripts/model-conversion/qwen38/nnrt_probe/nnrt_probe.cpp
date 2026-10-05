// 最小 NNRt 探针：① 枚举设备（看 NPU 是否暴露 ✓）② 试加载离线模型 ✓
#include <cstdio>
#include <cstdlib>
#include "neural_network_core.h"

int main(int argc, char **argv) {
    uint32_t count = 0;
    const size_t *ids = nullptr;
    OH_NN_ReturnCode rc = OH_NNDevice_GetAllDevicesID(&ids, &count);
    printf("① OH_NNDevice_GetAllDevicesID rc=%d count=%u\n", (int)rc, count);
    for (size_t i = 0; i < count; ++i) {
        const char *name = nullptr;
        OH_NN_ReturnCode r2 = OH_NNDevice_GetName(ids[i], &name);
        OH_NN_DeviceType type = (OH_NN_DeviceType)0;
        OH_NN_ReturnCode r3 = OH_NNDevice_GetType(ids[i], &type);
        printf("   device[%zu] id=%zu name=%s  type=%d  (rc name=%d type=%d)\n",
               i, ids[i], name ? name : "(null)", (int)type, (int)r2, (int)r3);
    }
    if (argc > 1) {
        printf("② ConstructWithOfflineModelFile(%s)\n", argv[1]);
        OH_NNCompilation *c = OH_NNCompilation_ConstructWithOfflineModelFile(argv[1]);
        printf("   => %p\n", (void *)c);
        if (c != nullptr) {
            if (count > 0) {
                OH_NN_ReturnCode r = OH_NNCompilation_SetDevice(c, ids[0]);
                printf("   SetDevice(ids[0]) rc=%d\n", (int)r);
            }
            // ★加缓存路径★（有些实现要求它 ✓）
            const char *cache = getenv("NNRT_CACHE");
            if (cache && cache[0]) {
                OH_NN_ReturnCode rc2 = OH_NNCompilation_SetCache(c, cache, 1);
                printf("   SetCache(%s) rc=%d\n", cache, (int)rc2);
            }
            // ★试编译选项（性能模式/优先级 ✓）
            OH_NN_ReturnCode rp = OH_NNCompilation_SetPerformanceMode(c, OH_NN_PERFORMANCE_HIGH);
            OH_NN_ReturnCode rr = OH_NNCompilation_SetPriority(c, OH_NN_PRIORITY_HIGH);
            printf("   SetPerformanceMode(HIGH)=%d SetPriority(HIGH)=%d\n", (int)rp, (int)rr);
            OH_NN_ReturnCode r = OH_NNCompilation_Build(c);
            printf("   Build rc=%d %s\n", (int)r, r == OH_NN_SUCCESS ? "★BUILD OK★" : "✗");
            OH_NNCompilation_Destroy(&c);
        }
    }
    return 0;
}
