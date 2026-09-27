/* npu_infer.c - 纯 C（原生 ELF）直接驱动 cann LLM 引擎跑一轮推理。
 *
 * 与 python 后端（src/cann_llm/backends/hiai.py）走同一套 API，
 * 只是为了回答"绕过 python、原生 ELF 能不能打开 NPU 并完成推理"。
 *
 * 要点（都是从 hiai.py 的实测注释里抄的，别自己改）：
 *   · GenerateAsync 的第 3 个参数是 prompt【文本】(std::string::c_str())，
 *     引擎自己分词（tokenizer 由 InitOption_SetTokenizer 给它）→ 不用自己 encode。
 *   · 取输出用 GetAllGenerationLen + GetAllGeneration，拿到的是【明文】。
 *   · 这一族 getter 的返回值【不是成败标志】，长度以出参为准。
 *   · opt / mi 的生命周期必须覆盖全程（引擎把它们存进 executor），别提前 Destroy。
 *   · SetStopSeq 的签名是 (ctx, char**, int)。
 *
 * Build: cc -O1 npu_infer.c -o npu_infer -ldl
 * Run:   LD_LIBRARY_PATH=/system/lib64/ndk ./npu_infer <model_dir> [prompt]
 *        model_dir 里要有 <name>.omc / <name>.json / tokenizer.json
 */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define LIB_PATH "/system/lib64/libhiai_llm_engine.so"

static volatile int g_done = 0;
static volatile int g_failed = 0;

static void on_done(void *p)  { (void) p; g_done = 1; }
static void on_fail(void *p)  { (void) p; g_failed = 1; }

/* ---- 引擎 API（签名照 src/cann_llm/backends/hiai.py 的 ctypes 声明） ---- */
static void *(*InitOption_Create)(void);
static int   (*InitOption_SetInferType)(void *, int);
static int   (*InitOption_SetTokenizer)(void *, int, const char *);
static int   (*InitOption_SetModel)(void *, int, void *);
static void *(*ModelInfo_Create)(void);
static int   (*ModelInfo_SetWeightDir)(void *, const char *);
static int   (*ModelInfo_SetModelPath)(void *, const char *);
static int   (*ModelInfo_SetModelType)(void *, int);
static void *(*Executor_Create)(void);
static int   (*Executor_Init_Use_Option)(void *, void *);
static int   (*Executor_GenerateAsync)(void *, void *, const char *);
static void *(*Context_Create)(void);
static int   (*Context_SetInitTokenLen)(void *, int);
static int   (*Context_SetMaxGenTokens)(void *, int);
static int   (*Context_SetSampleGreedy)(void *, unsigned char);
static int   (*Context_SetTopK)(void *, int);
static int   (*Context_SetStopSeq)(void *, const char **, int);
static int   (*Context_SetOnAllTokensGenerateDoneFunc)(void *, void *);
static int   (*Context_SetOnGenerateAsyncFailed)(void *, void *);
static int   (*Context_GetAllGenerationLen)(void *, int *);
static int   (*Context_GetAllGeneration)(void *, char *, int);
static int   (*Context_GetOutputTokenCount)(void *, int *);
static int   (*Context_GetInputTokenCount)(void *, int *);

/* dlsym 要的是全名：InitOption_* / Executor_* / Context_* 都带 HIAI_LLMEngine_ 前缀，
   ModelInfo_* 带 HIAI_LMEngine_ 前缀（见 src/cann_llm/backends/hiai.py 的符号表）。 */
#define LOAD(v, s) do { *(void **) (&v) = dlsym(h, s); \
    if (!v) { printf("missing symbol %s\n", s); return 1; } } while (0)

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model_dir> [prompt]\n", argv[0]);
        return 1;
    }
    const char *dir = argv[1];
    const char *prompt = (argc > 2) ? argv[2]
        : "<|im_start|>user\n\u4f60\u597d<|im_end|>\n<|im_start|>assistant\n";

    /* 模型目录里取 .omc / .json 的文件名（约定：<name>.omc + <name>.json） */
    char model_path[1024], tok_path[1024];
    snprintf(tok_path, sizeof tok_path, "%s", "tokenizer.json");

    void *h = dlopen(LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("dlopen %s failed: %s\n", LIB_PATH, dlerror()); return 1; }

    LOAD(InitOption_Create, "HIAI_LLMEngine_InitOption_Create"); LOAD(InitOption_SetInferType, "HIAI_LLMEngine_InitOption_SetInferType");
    LOAD(InitOption_SetTokenizer, "HIAI_LLMEngine_InitOption_SetTokenizer"); LOAD(InitOption_SetModel, "HIAI_LLMEngine_InitOption_SetModel");
    LOAD(ModelInfo_Create, "HIAI_LMEngine_ModelInfo_Create"); LOAD(ModelInfo_SetWeightDir, "HIAI_LMEngine_ModelInfo_SetWeightDir");
    LOAD(ModelInfo_SetModelPath, "HIAI_LMEngine_ModelInfo_SetModelPath"); LOAD(ModelInfo_SetModelType, "HIAI_LMEngine_ModelInfo_SetModelType");
    LOAD(Executor_Create, "HIAI_LLMEngine_Executor_Create"); LOAD(Executor_Init_Use_Option, "HIAI_LLMEngine_Executor_Init_Use_Option");
    LOAD(Executor_GenerateAsync, "HIAI_LLMEngine_Executor_GenerateAsync"); LOAD(Context_Create, "HIAI_LLMEngine_Context_Create");
    LOAD(Context_SetInitTokenLen, "HIAI_LLMEngine_Context_SetInitTokenLen"); LOAD(Context_SetMaxGenTokens, "HIAI_LLMEngine_Context_SetMaxGenTokens");
    LOAD(Context_SetSampleGreedy, "HIAI_LLMEngine_Context_SetSampleGreedy"); LOAD(Context_SetTopK, "HIAI_LLMEngine_Context_SetTopK");
    LOAD(Context_SetStopSeq, "HIAI_LLMEngine_Context_SetStopSeq");
    LOAD(Context_SetOnAllTokensGenerateDoneFunc, "HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc");
    LOAD(Context_SetOnGenerateAsyncFailed, "HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed");
    LOAD(Context_GetAllGenerationLen, "HIAI_LLMEngine_Context_GetAllGenerationLen"); LOAD(Context_GetAllGeneration, "HIAI_LLMEngine_Context_GetAllGeneration");
    LOAD(Context_GetOutputTokenCount, "HIAI_LLMEngine_Context_GetOutputTokenCount"); LOAD(Context_GetInputTokenCount, "HIAI_LLMEngine_Context_GetInputTokenCount");

    /* 从目录里挑一个 .omc（用 opendir 太重，这里按约定拼接；模型名固定为 qwen15b_repo） */
    const char *name = "qwen15b_repo";
    snprintf(model_path, sizeof model_path, "%s.omc", name);
    if (chdir(dir) != 0) { printf("chdir %s failed\n", dir); return 1; }
    printf("cwd     : %s (已切到模型目录)\n", dir);

    printf("model   : %s\n", model_path);
    printf("tokenizer: %s\n", tok_path);
    printf("prompt  : %.60s%s\n\n", prompt, strlen(prompt) > 60 ? "..." : "");

    /* ---- 1) 组装 InitOption（顺序照 hiai.py：weightDir 在 modelPath 之前） ---- */
    void *opt = InitOption_Create();
    void *mi  = ModelInfo_Create();
    InitOption_SetInferType(opt, 0);
    InitOption_SetTokenizer(opt, 4, tok_path);      /* 4 = qwen */
    /* ★ 照 api_config.json 的原样传：modelPath 只用【文件名】，weightDir 用 "./"。
       引擎按 modelPath 去掉扩展名 + ".json" 找模型配置，所以必须在【模型目录里】运行
       （main 里已 chdir 过去）。传绝对路径反而找不到。 */
    ModelInfo_SetWeightDir(mi, "./");
    ModelInfo_SetModelPath(mi, model_path);
    ModelInfo_SetModelType(mi, 0);
    InitOption_SetModel(opt, 0, mi);
    printf("[1] InitOption 组装完成\n");

    /* ---- 2) 建 executor 并初始化（★ 这一步就是 python 侧报过 FAIL 的地方） ---- */
    void *ex = Executor_Create();
    double t0 = now_ms();
    int rc = Executor_Init_Use_Option(ex, opt);
    printf("[2] Executor_Init_Use_Option = %d  (%.0f ms)%s\n", rc, now_ms() - t0,
           rc == 0 ? "  ← 引擎加载成功" : "  ← 加载失败");
    if (rc != 0) { printf("FAILED to load model\n"); return 2; }

    /* ---- 3) 每请求一个 Context ---- */
    void *ctx = Context_Create();
    Context_SetInitTokenLen(ctx, 1024);              /* 必须 < kv_cache_max_len */
    Context_SetMaxGenTokens(ctx, 32);
    Context_SetSampleGreedy(ctx, 1);
    Context_SetTopK(ctx, 1);
    const char *stops[] = { "<|im_end|>" };
    Context_SetStopSeq(ctx, stops, 1);
    Context_SetOnAllTokensGenerateDoneFunc(ctx, (void *) on_done);
    Context_SetOnGenerateAsyncFailed(ctx, (void *) on_fail);
    printf("[3] Context 配好（initTokenLen=1024, maxGen=32, greedy）\n");

    /* ---- 4) 一轮推理：文本进，等回调，明文出 ---- */
    t0 = now_ms();
    rc = Executor_GenerateAsync(ex, ctx, prompt);
    if (rc != 0) { printf("[4] GenerateAsync rc=%d\n", rc); return 3; }
    while (!g_done && !g_failed && (now_ms() - t0) < 120000) {
        struct timespec ts = { 0, 20 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    double dt = now_ms() - t0;
    if (g_failed) { printf("[4] 引擎触发了失败回调\n"); return 4; }
    if (!g_done)  { printf("[4] 超时（120s）\n"); return 5; }

    int nin = 0, nout = 0, nlen = 0;
    Context_GetInputTokenCount(ctx, &nin);
    Context_GetOutputTokenCount(ctx, &nout);
    Context_GetAllGenerationLen(ctx, &nlen);
    printf("[4] 推理完成 %.0f ms · in=%d tok out=%d tok gen_len=%d\n", dt, nin, nout, nlen);

    char *buf = calloc(1, (size_t) (nlen > 0 ? nlen : 1) + 256);
    if (buf && nlen > 0) {
        Context_GetAllGeneration(ctx, buf, nlen + 63);
        printf("\n===== 引擎输出（明文）=====\n%s\n===========================\n", buf);
    }
    free(buf);
    printf("\nOK: 原生 ELF 用 cann LLM 引擎完成了一轮推理\n");
    return 0;
}
