/* nnrt_offline_probe.c - 能用 Neural Network Runtime（NNRt）直接加载我们 OMG 产出的
 * 离线模型（.omc）吗？
 *
 * 为什么值得试：LLM 引擎（libhiai/cann_llm_engine.so）对模型结构有硬约束
 * （它要求"逐层 1:1 K/V + 每层 past_key_value"），新架构（Gemma 4 的 per-layer /
 * K=V / KV 跨层共享）表达不了。而 NNRt 的【离线模型】路径里，输入/输出张量是
 * 由模型（或转换配置）自己声明的 —— 若它能吃 .omc，就等于拿到一条【绕开 LLM 引擎】的
 * 通用 NPU 执行通路。
 *
 * 本程序只做只读探测：建编译实例 → 编译 → 建执行器 → 读输入输出张量描述。
 * 不喂数据、不改任何文件。
 *
 * Build: cc -O1 nnrt_offline_probe.c -o nnrt_offline_probe -ldl
 * Run:   LD_LIBRARY_PATH=/system/lib64/ndk ./nnrt_offline_probe <模型.omc>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#define NDK "/system/lib64/ndk/"

/* OH_NN_ReturnCode */
static const char *rc_name(int rc) {
    switch (rc) {
        case 0: return "OH_NN_SUCCESS";
        case 1: return "OH_NN_FAILED";
        case 2: return "OH_NN_INVALID_PARAMETER";
        case 3: return "OH_NN_MEMORY_ERROR";
        case 4: return "OH_NN_OPERATION_FORBIDDEN";
        case 5: return "OH_NN_NULL_PTR";
        case 6: return "OH_NN_INVALID_FILE";
        case 7: return "OH_NN_UNAVALIDABLE_DEVICE";
        case 8: return "OH_NN_INVALID_PATH";
        default: return "?";
    }
}

#define GET(h, name) dlsym(h, name)
#define NEED(h, name)                                                          \
    do {                                                                       \
        name = GET(h, #name);                                                  \
        if (!name) { printf("    !! 缺符号 %s\n", #name); return 2; }          \
    } while (0)

/* 函数指针 */
static void *(*OH_NNCompilation_ConstructWithOfflineModelFile)(const char *);
static void *(*OH_NNCompilation_ConstructWithOfflineModelBuffer)(const void *, size_t);
static int   (*OH_NNCompilation_SetDevice)(void *, size_t);
static int   (*OH_NNCompilation_SetPerformanceMode)(void *, int);
static int   (*OH_NNCompilation_SetPriority)(void *, int);
static int   (*OH_NNCompilation_EnableFloat16)(void *, int);
static int   (*OH_NNCompilation_Build)(void *);
static void  (*OH_NNCompilation_Destroy)(void **);
static void *(*OH_NNExecutor_Construct)(void *);
static int   (*OH_NNExecutor_GetInputCount)(const void *, size_t *);
static int   (*OH_NNExecutor_GetOutputCount)(const void *, size_t *);
static void *(*OH_NNExecutor_CreateInputTensorDesc)(const void *, size_t);
static void *(*OH_NNExecutor_CreateOutputTensorDesc)(const void *, size_t);
static int   (*OH_NNTensorDesc_GetName)(const void *, const char **);
static int   (*OH_NNTensorDesc_GetDataType)(const void *, int *);
static int   (*OH_NNTensorDesc_GetShape)(const void *, int32_t **, size_t *);
static void  (*OH_NNTensorDesc_Destroy)(void **);
static void  (*OH_NNExecutor_Destroy)(void **);
static int   (*OH_NNDevice_GetAllDevicesID)(const size_t **, unsigned *);
static int   (*OH_NNDevice_GetName)(size_t, const char **);
static int   (*OH_NNDevice_GetType)(size_t, int *);

static void dump_desc(const char *tag, void *d) {
    if (!d) { printf("      %s: (null desc)\n", tag); return; }
    const char *name = NULL;
    int dt = -1, rc;
    int32_t *shape = NULL;
    size_t dim = 0;
    rc = OH_NNTensorDesc_GetName(d, &name);
    if (rc != 0) name = NULL;
    rc = OH_NNTensorDesc_GetDataType(d, &dt);
    if (rc != 0) dt = -1;
    rc = OH_NNTensorDesc_GetShape(d, &shape, &dim);
    printf("      %s name=%s dtype=%d shape=[", tag, name ? name : "?", dt);
    if (rc == 0 && shape) {
        for (size_t i = 0; i < dim; i++) printf("%d%s", shape[i], i + 1 < dim ? "," : "");
    } else {
        printf("?");
    }
    printf("] (%zu dims)\n", dim);
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model.omc>\n", argv[0]); return 1; }
    const char *model = argv[1];

    printf("=== 0) 加载 NNRt ===\n");
    void *h = dlopen(NDK "libneural_network_runtime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("    dlopen 失败: %s\n", dlerror()); return 1; }
    printf("    libneural_network_runtime.so OK\n");

    NEED(h, OH_NNCompilation_ConstructWithOfflineModelFile);
    NEED(h, OH_NNCompilation_ConstructWithOfflineModelBuffer);
    NEED(h, OH_NNCompilation_SetDevice);
    NEED(h, OH_NNCompilation_SetPerformanceMode);
    NEED(h, OH_NNCompilation_SetPriority);
    NEED(h, OH_NNCompilation_EnableFloat16);
    NEED(h, OH_NNCompilation_Build);
    NEED(h, OH_NNCompilation_Destroy);
    NEED(h, OH_NNExecutor_Construct);
    NEED(h, OH_NNExecutor_GetInputCount);
    NEED(h, OH_NNExecutor_GetOutputCount);
    NEED(h, OH_NNExecutor_CreateInputTensorDesc);
    NEED(h, OH_NNExecutor_CreateOutputTensorDesc);
    NEED(h, OH_NNTensorDesc_GetName);
    NEED(h, OH_NNTensorDesc_GetDataType);
    NEED(h, OH_NNTensorDesc_GetShape);
    NEED(h, OH_NNExecutor_Destroy);
    NEED(h, OH_NNDevice_GetAllDevicesID);
    NEED(h, OH_NNDevice_GetName);
    NEED(h, OH_NNDevice_GetType);
    printf("    全部符号解析成功\n");

    printf("\n=== 1) 枚举设备 ===\n");
    const size_t *ids = NULL;
    unsigned n = 0;
    int rc = OH_NNDevice_GetAllDevicesID(&ids, &n);
    printf("    GetAllDevicesID rc=%d(%s) count=%u\n", rc, rc_name(rc), n);
    const char *tn[] = { "OTHERS", "CPU", "GPU", "ACCELERATOR" };
    size_t dev = 0;
    for (unsigned i = 0; i < n; i++) {
        const char *nm = NULL; int t = -1;
        OH_NNDevice_GetName(ids[i], &nm);
        OH_NNDevice_GetType(ids[i], &t);
        printf("      device[%u] id=%zu type=%s name=%s\n", i, ids[i],
               (t >= 0 && t <= 3) ? tn[t] : "?", nm ? nm : "(null)");
        if (t == 3) dev = ids[i];
    }
    if (!n) { printf("    (没有设备)\n"); return 3; }
    if (!dev) dev = ids[0];
    printf("    使用 device id=%zu\n", dev);

    printf("\n=== 2) 用离线模型文件建编译实例 ===\n");
    printf("    modelPath = %s\n", model);
    void *c = OH_NNCompilation_ConstructWithOfflineModelFile(model);
    printf("    -> %p   %s\n", c, c ? "非空（说明 NNRt 接受了这个文件）" : "NULL（拒绝）");
    if (!c) {
        printf("\n=== 2b) 退一步：自己读文件、用 Buffer 接口 ===\n");
        FILE *f = fopen(model, "rb");
        if (!f) { printf("    读不到文件: %s\n", model); return 4; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        void *buf = malloc((size_t) sz);
        size_t rd = fread(buf, 1, (size_t) sz, f);
        fclose(f);
        printf("    文件 %ld 字节，实读 %zu\n", sz, rd);
        c = OH_NNCompilation_ConstructWithOfflineModelBuffer(buf, rd);
        printf("    Buffer 接口 -> %p\n", c);
        if (!c) { printf("    结论：NNRt 不接受这个 .omc（两种接口都拒绝）\n"); return 5; }
    }

    printf("\n=== 3) SetDevice / 性能模式 / 编译 ===\n");
    rc = OH_NNCompilation_SetDevice(c, dev);
    printf("    SetDevice       rc=%d(%s)\n", rc, rc_name(rc));
    rc = OH_NNCompilation_SetPerformanceMode(c, 3 /*EXTREME*/);
    printf("    SetPerformanceMode rc=%d(%s)\n", rc, rc_name(rc));
    rc = OH_NNCompilation_SetPriority(c, 2 /*HIGH*/);
    printf("    SetPriority     rc=%d(%s)\n", rc, rc_name(rc));
    rc = OH_NNCompilation_EnableFloat16(c, 0);
    printf("    EnableFloat16   rc=%d(%s)\n", rc, rc_name(rc));
    rc = OH_NNCompilation_Build(c);
    printf("    Build           rc=%d(%s)   ← 关键：编译是否成功\n", rc, rc_name(rc));
    if (rc != 0) {
        printf("    结论：NNRt 接受了文件但【编译失败】—— 多半不是它期望的格式\n");
        return 6;
    }

    printf("\n=== 4) 建执行器 + 读输入输出张量 ===\n");
    void *e = OH_NNExecutor_Construct(c);
    printf("    Executor -> %p\n", e);
    if (!e) return 7;
    size_t in = 0, out = 0;
    OH_NNExecutor_GetInputCount(e, &in);
    OH_NNExecutor_GetOutputCount(e, &out);
    printf("    输入 %zu 个 / 输出 %zu 个\n", in, out);
    for (size_t i = 0; i < in && i < 12; i++) {
        void *d = OH_NNExecutor_CreateInputTensorDesc(e, i);
        char tag[32];
        snprintf(tag, sizeof tag, "in[%zu]", i);
        dump_desc(tag, d);
        if (d && OH_NNTensorDesc_Destroy) OH_NNTensorDesc_Destroy(&d);
    }
    for (size_t i = 0; i < out && i < 12; i++) {
        void *d = OH_NNExecutor_CreateOutputTensorDesc(e, i);
        char tag[32];
        snprintf(tag, sizeof tag, "out[%zu]", i);
        dump_desc(tag, d);
        if (d && OH_NNTensorDesc_Destroy) OH_NNTensorDesc_Destroy(&d);
    }
    printf("\n★★ 结论：NNRt 能直接加载并编译我们的 .omc —— 拿到了一条绕开 LLM 引擎的 NPU 通路\n");
    return 0;
}
