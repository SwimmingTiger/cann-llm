/* npu_infer_cann.c - 用 **cann 后端**（NDK 的 libcann_llm_engine.so）跑一轮推理。
 *
 * 与 npu_infer_hiai.c 的区别（两套完全不同的 API）：
 *   hiai: /system/lib64/libhiai_llm_engine.so + HIAI_LLMEngine_*
 *         —— 逐个 SetOption 组装配置（InitOption / ModelInfo）
 *   cann: /system/lib64/ndk/libcann_llm_engine.so + HMS_LLMEngine*
 *         —— ★ 吃 JSON：executor 传【文件名】，context 传【JSON 文本】
 *
 * 调用序列照 src/cann_llm/backends/cann.py：
 *     executor = HMS_LLMEngineExecutor_CreateFromExecutorJson("executor.json")   ← 文件名（相对 cwd）
 *     ctx      = HMS_LLMEngineContext_CreateFromContextJson(<文件名>)   ← ★ 也是文件名！
 *     rc       = HMS_LLMEngineExecutor_Generate(executor, ctx, prompt【文本】)
 *     输出      = HMS_LLMEngineContext_GetAllGenerationLen / GetAllGeneration（明文）
 *     计时      = GetPrefillTimeMs / GetDecodeTimeMs / GetTotalTimeMs
 *   ⚠ 不要调 Destroy —— cann.py 的注释写明它们的 Destroy 会崩。
 *
 * Build: cc -O1 npu_infer_cann.c -o npu_infer_cann -ldl
 * Run:   LD_LIBRARY_PATH=/system/lib64/ndk ./npu_infer_cann <model_dir> [prompt]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define LIB_PATH "/system/lib64/ndk/libcann_llm_engine.so"

static void *(*Executor_CreateFromExecutorJson)(const char *);
static void *(*Context_CreateFromContextJson)(const char *);
static int   (*Executor_Generate)(void *, void *, const char *);
static int   (*Context_GetAllGenerationLen)(void *, unsigned *);
static int   (*Context_GetAllGeneration)(void *, char *, unsigned);
static int   (*Context_GetInputTokenCount)(void *, unsigned long *);
static int   (*Context_GetOutputTokenCount)(void *, unsigned long *);
static int   (*Context_GetPrefillTimeMs)(void *, double *);
static int   (*Context_GetDecodeTimeMs)(void *, double *);
static int   (*Context_GetTotalTimeMs)(void *, double *);
static int   (*Context_SetOnOneTokenGenerateDoneFunc)(void *, void *);

#define LOAD(v, s) do { *(void **) (&v) = dlsym(h, s); \
    if (!v) { printf("missing symbol %s\n", s); return 1; } } while (0)

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t) n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model_dir> [prompt]\n", argv[0]); return 1; }
    const char *dir = argv[1];
    const char *prompt = (argc > 2) ? argv[2]
        : "<|im_start|>user\n\u4f60\u597d<|im_end|>\n<|im_start|>assistant\n";

    /* ★ cann 的 executor 参数是【文件名】，相对于当前目录 —— 必须 chdir 到模型目录 */
    if (chdir(dir) != 0) { printf("chdir %s failed\n", dir); return 1; }
    printf("cwd     : %s\n", dir);
    printf("executor: executor.json   (cann 传的是文件名)\n");

    char *ctx_json = slurp("context.json");
    if (!ctx_json) { printf("读不到 context.json\n"); return 1; }
    printf("context : context.json（%zu 字节；★ 传给引擎的是【文件名】不是内容）\n",
           strlen(ctx_json));
    printf("prompt  : %.60s%s\n\n", prompt, strlen(prompt) > 60 ? "..." : "");

    void *h = dlopen(LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen %s failed: %s\n", LIB_PATH, dlerror()); return 1; }

    LOAD(Executor_CreateFromExecutorJson, "HMS_LLMEngineExecutor_CreateFromExecutorJson");
    LOAD(Context_CreateFromContextJson,   "HMS_LLMEngineContext_CreateFromContextJson");
    LOAD(Executor_Generate,               "HMS_LLMEngineExecutor_Generate");
    LOAD(Context_GetAllGenerationLen,     "HMS_LLMEngineContext_GetAllGenerationLen");
    LOAD(Context_GetAllGeneration,        "HMS_LLMEngineContext_GetAllGeneration");
    LOAD(Context_GetInputTokenCount,      "HMS_LLMEngineContext_GetInputTokenCount");
    LOAD(Context_GetOutputTokenCount,     "HMS_LLMEngineContext_GetOutputTokenCount");
    LOAD(Context_GetPrefillTimeMs,        "HMS_LLMEngineContext_GetPrefillTimeMs");
    LOAD(Context_GetDecodeTimeMs,         "HMS_LLMEngineContext_GetDecodeTimeMs");
    LOAD(Context_GetTotalTimeMs,          "HMS_LLMEngineContext_GetTotalTimeMs");
    LOAD(Context_SetOnOneTokenGenerateDoneFunc,
         "HMS_LLMEngineContext_SetOnOneTokenGenerateDoneFunc");

    /* ---- 1) executor：传文件名，引擎自己读 --------------- */
    double t0 = (double) clock() / CLOCKS_PER_SEC * 1000.0;
    void *ex = Executor_CreateFromExecutorJson("executor.json");
    printf("[1] Executor_CreateFromExecutorJson -> %p%s\n", ex,
           ex ? "  ← 引擎加载成功" : "  ← 失败");
    if (!ex) return 2;
    printf("    (%.0f ms)\n", (double) clock() / CLOCKS_PER_SEC * 1000.0 - t0);

    /* ---- 2) context：传 JSON 文本 ----------------------- */
    /* ★ 这里也是【文件名】：cann.py 里传的是它自己写的
       os.path.join(model_dir, ".context.live.json")，不是 JSON 文本 */
    void *ctx = Context_CreateFromContextJson("context.json");
    printf("[2] Context_CreateFromContextJson -> %p\n", ctx);
    if (!ctx) { printf("FAILED: 检查 context.json\n"); return 3; }

    /* ---- 3) 一轮推理 ------------------------------------ */
    t0 = (double) clock() / CLOCKS_PER_SEC * 1000.0;
    int rc = Executor_Generate(ex, ctx, prompt);
    double dt = (double) clock() / CLOCKS_PER_SEC * 1000.0 - t0;
    printf("[3] Generate rc=%d  (%.0f ms)\n", rc, dt);
    if (rc != 0) { printf("FAILED: 生成返回 %d\n", rc); return 4; }

    /* ---- 4) 取输出与计时 -------------------------------- */
    unsigned n = 0;
    unsigned long nin = 0, nout = 0;
    double prefill = 0, decode = 0, total = 0;
    Context_GetAllGenerationLen(ctx, &n);
    Context_GetInputTokenCount(ctx, &nin);
    Context_GetOutputTokenCount(ctx, &nout);
    Context_GetPrefillTimeMs(ctx, &prefill);
    Context_GetDecodeTimeMs(ctx, &decode);
    Context_GetTotalTimeMs(ctx, &total);
    printf("[4] gen_len=%u · in=%lu tok · out=%lu tok · prefill %.0f ms · "
           "decode %.0f ms · total %.0f ms\n", n, nin, nout, prefill, decode, total);

    if (n) {
        char *buf = calloc(1, (size_t) n + 256);
        if (buf) {
            Context_GetAllGeneration(ctx, buf, n);
            printf("\n===== 引擎输出（明文）=====\n%s\n===========================\n", buf);
            free(buf);
        }
    }
    free(ctx_json);
    printf("\nOK: cann 后端（libcann_llm_engine.so）完成了一轮推理\n");
    return 0;
}
