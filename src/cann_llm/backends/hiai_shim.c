// hiai_shim.c —— 极小的 C++ 辅助库，绕过 Python 侧的 C++ ABI 猜测。
//
// 为什么需要它：
//   反编译确认 HIAI_LLMEngine_InitOption_SetModel 的第 3 个参数不是字符串，
//   而是 **modelInfo* 结构体**（其 +40 处是 std::string weightDir）：
//       SetModel(opt, modelType, void* a3):
//           *(int*)(opt + 32)  = modelType;
//           *(void**)(opt + 40) = a3;        // 原样存指针
//   而 Init_Use_Option 会把 opt+40 当结构体读，取 +40 的 std::string：
//       "initOptionImpl->modelInfo->weightDir.size() > 0"  "false, return FAIL."
//   从 Python/ctypes 手工拼 libc++ std::string 极易出错（且错误取值会让引擎段错误），
//   所以这里用**真正的 C++** 组装，交给编译器处理 ABI。
//
// 编译（设备上）：
//   CC=/data/service/hnp/bin/aarch64-unknown-linux-ohos-clang
//   $CC -O0 -g -shared -fPIC -o libhiai_shim.so hiai_shim.c \
//       -I/usr/include/c++/v1 -lc++ \        # 按设备实际头文件位置调整
//       -L/system/lib64 -lhiai_llm_engine
//
// 说明：modelInfo 的布局来自反编译推断，只有 +0（int modelType）
//       与 +40（std::string weightDir）是确定的；中间留空。

#include <string>
#include <cstdint>

// 引擎导出的接口（签名取自反编译）
extern "C" {
void *HIAI_LLMEngine_InitOption_Create(void);
int HIAI_LLMEngine_InitOption_SetModel(void *initOption, int modelType, void *modelInfo);
int HIAI_LLMEngine_InitOption_SetTokenizer(void *initOption, int tokenizerType, const char *path);
int HIAI_LLMEngine_InitOption_SetInferType(void *initOption, int inferType);
void *HIAI_LLMEngine_Executor_Create(void);
int HIAI_LLMEngine_Executor_Init_Use_Option(void *executor, void *initOption);
}

namespace {

//: 反编译推断的 modelInfo 布局：+0 = int modelType，+40 = std::string weightDir
struct ModelInfo {
    int32_t model_type;          // +0
    char    reserved[36];        // +4 .. +40（未使用，全 0）
    std::string weight_dir;      // +40  ★ Init_Use_Option 真正读的字段
};

//: 引擎会长期持有 modelInfo 指针，所以不能是栈上临时对象。
ModelInfo g_model_info;

}  // namespace

extern "C" {

//: 建 InitOption，并把模型信息按引擎期望的结构喂进去。
//: 成功返回 initOption 指针，失败返回 nullptr。
//: weight_dir 一般传模型目录（官方 api_config.json 的 weightDir，如 "./" 的绝对形式）。
void *hiai_make_opt(const char *model_file_path,
                    int model_type,
                    int tokenizer_type,
                    const char *tokenizer_path,
                    const char *weight_dir,
                    int infer_type)
{
    if (!model_file_path) {
        return nullptr;
    }
    void *opt = HIAI_LLMEngine_InitOption_Create();
    if (!opt) {
        return nullptr;
    }

    // ★ 关键：第 3 个参数必须是 modelInfo 结构体，不是路径字符串
    g_model_info.model_type = model_type;
    g_model_info.weight_dir = weight_dir ? weight_dir : "";
    if (HIAI_LLMEngine_InitOption_SetModel(opt, model_type, &g_model_info) != 0) {
        return nullptr;
    }

    HIAI_LLMEngine_InitOption_SetTokenizer(opt, tokenizer_type, tokenizer_path);
    HIAI_LLMEngine_InitOption_SetInferType(opt, infer_type);
    return opt;
}

//: 建 Executor 并用上面的 opt 初始化。成功返回 executor，失败返回 nullptr。
void *hiai_make_executor(void *opt)
{
    if (!opt) {
        return nullptr;
    }
    void *ex = HIAI_LLMEngine_Executor_Create();
    if (!ex) {
        return nullptr;
    }
    if (HIAI_LLMEngine_Executor_Init_Use_Option(ex, opt) != 0) {
        return nullptr;
    }
    return ex;
}

//: 供 Python 侧观察：确认 weight_dir 到底被写进了什么（调试用）
const char *hiai_debug_weight_dir(void)
{
    return g_model_info.weight_dir.c_str();
}

}  // extern "C"
