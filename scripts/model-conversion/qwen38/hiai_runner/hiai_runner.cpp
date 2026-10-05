// hiai_runner —— 用 DDK 的 hiai C++ API 直接加载并运行我们的编译产物（§61/§62）
//
// 背景：设备的 hiai 通路有两族 API
//   · LLM 引擎（libhiai_llm_engine.so）—— 固定 KV 接口 ✗ 不适合 qwen3_5 这种
//     "18 层线性注意力 + 6 层全注意力" 的混合架构（线性层要传卷积/递归状态 ✗）
//   · ★DDK hiai API（本文件用的）★ —— IO 由我们自己定 ✓（CreateBuiltModel +
//     CreateModelManager，头文件来自官方 HiAIDemo / DDK_Demo/V2/include ✓）
//
// 用法：
//   hiai_runner <model.omc|model.om>            只加载 + 打印真实 IO（不推理）
//   hiai_runner <model.omc|model.om> --run      再分配缓冲跑一次（输入置零）
//
// 编译（设备上用 OHOS clang，musl）：
//   aarch64-unknown-linux-ohos-clang++ -std=c++11 -O2 -fPIC -Iinclude hiai_runner.cpp \
//       -o hiai_runner lib64/libhiai.so -ldl
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "model/built_model.h"
#include "model_builder/model_builder.h"
#include "model_manager/model_manager.h"
#include "tensor/nd_tensor_buffer.h"

using namespace hiai;

namespace {

std::string EndsWith(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0 ? "yes" : "no";
}

void Dump(const char *tag, const std::vector<NDTensorDesc> &descs) {
    // ★防御式遍历★：NDTensorDesc 内含 std::vector ✓，直接按下标读容易越界崩溃 ✗（§83 踩过 ✓）
    size_t n = 0;
    try {
        n = descs.size();
    } catch (...) {
        std::cout << "  " << tag << " size() 抛异常" << std::endl;
        return;
    }
    std::cout << "  " << tag << " 共 " << n << " 个：" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        try {
            size_t r = descs[i].dims.size();
            std::cout << "    [" << i << "] rank=" << r << " dims=[";
            for (size_t j = 0; j < r; ++j) {
                std::cout << descs[i].dims[j] << (j + 1 < r ? "," : "");
            }
            std::cout << "] dtype=" << static_cast<int>(descs[i].dataType)
                      << " format=" << static_cast<int>(descs[i].format) << std::endl;
        } catch (...) {
            std::cout << "    [" << i << "] 读取抛异常（结构不完整 ✓）" << std::endl;
            break;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "用法: hiai_runner <model.omc|model.om> [--run]" << std::endl;
        return 2;
    }
    const std::string path = argv[1];
    const bool doRun = (argc > 2 && std::string(argv[2]) == "--run");

    std::shared_ptr<IBuiltModel> built = nullptr;
    if (path.size() > 3 && path.compare(path.size() - 3, 3, ".om") == 0) {
        // ①a .om（离线模型）⇒ 走 ModelBuilder 编译 ✓
        std::cout << "① 用 ModelBuilder 编译 " << path << std::endl;
        auto builder = CreateModelBuilder();
        if (builder == nullptr) {
            std::cout << "  ✗ CreateModelBuilder 返回空" << std::endl;
            return 3;
        }
        ModelBuildOptions buildOptions;
        Status ret = builder->Build(buildOptions, "qwen38", path.c_str(), built);
        std::cout << "  Build rc=" << static_cast<int>(ret) << std::endl;
    } else {
        // ①b .omc（已编译）⇒ 直接 RestoreFromFile ✓
        std::cout << "① 用 CreateBuiltModel + RestoreFromFile 加载 " << path << std::endl;
        built = CreateBuiltModel();
        if (built == nullptr) {
            std::cout << "  ✗ CreateBuiltModel 返回空" << std::endl;
            return 3;
        }
        Status ret = built->RestoreFromFile(path.c_str());
        std::cout << "  RestoreFromFile rc=" << static_cast<int>(ret) << std::endl;
    }
    if (built == nullptr) {
        std::cout << "✗ 没有拿到 IBuiltModel" << std::endl;
        return 4;
    }
    std::cout << "  模型名: " << built->GetName() << std::endl;
    bool compatible = false;
    built->CheckCompatibility(compatible);
    std::cout << "  CheckCompatibility: " << (compatible ? "兼容 ✓" : "不兼容 ✗") << std::endl;

    // ② 只取输入/输出的【个数】✓（★不做下标遍历✗★：demo 头文件与设备库的 ABI 可能不一致，
    //    读 descs[i] 会段错误 ✓，而且 try/catch 抓不住段错误 ✓ —— §84 实测踩到 ✓）
    std::vector<NDTensorDesc> inDesc, outDesc;
    try {
        inDesc = built->GetInputTensorDescs();
        outDesc = built->GetOutputTensorDescs();
        std::cout << "  输入个数 = " << inDesc.size() << " · 输出个数 = " << outDesc.size() << std::endl;
    } catch (...) {
        std::cout << "  取 descs 抛异常 ✓" << std::endl;
    }
    if (getenv("DUMP_IO")) { Dump("输入", inDesc); Dump("输出", outDesc); }

    // ③ 初始化执行器
    std::cout << "③ CreateModelManager + Init" << std::endl;
    std::shared_ptr<IModelManager> manager = CreateModelManager();
    if (manager == nullptr) {
        std::cout << "  ✗ CreateModelManager 返回空" << std::endl;
        return 5;
    }
    // ★关键★：默认 options 里 buildOptions.formatMode = USE_NCHW ✗
    //   而我们的图是【3 维】张量 ✗ ⇒ 被当 NCHW 处理 ⇒ DDK 报 "check dimCnt 2 != 3" ✗
    //   ⇒ 必须显式设成 ★USE_ORIGIN★ ✓（§105 从 model_builder_types.h 查到 ✓）
    //   精度按 omc 的权重类型给 FP16 ✓；fallback 显式 ENABLE ✓
    ModelInitOptions options;
    const char *fn = getenv("INIT_NCHW");
    if (fn && fn[0] == '1') {
        std::cout << "  [Init] formatMode=USE_NCHW（对照 ✗）" << std::endl;
    } else {
        options.buildOptions.formatMode = FormatMode::USE_ORIGIN;          // ★★
        options.buildOptions.precisionMode = PRECISION_MODE_FP16;          // ★
        options.buildOptions.modelDeviceConfig.fallBackMode = FallBackMode::ENABLE;
        // ★实验开关★：INIT_CPU=1 ⇒ 把执行设备强制成 CPU ✓
        //   ⇒ 若 Init 就过了 ⇒ 说明问题在【NPU 侧的算子/kernel 组合】✗
        const char *cpu = getenv("INIT_CPU");
        if (cpu && cpu[0] == '1') {
            options.buildOptions.modelDeviceConfig.modelDeviceOrder = {ExecuteDevice::CPU};
            std::cout << "  [Init] ★强制 modelDeviceOrder = {CPU}★" << std::endl;
        }
        std::cout << "  [Init] formatMode=USE_ORIGIN · precision=FP16 ✓" << std::endl;
    }
    // ★★★ 实验：显式填 inputTensorDescs ★★★
    //   为什么：DDK 默认自己推断 IO ✓；我们的图是【3 维】张量 ✗
    //   ⇒ 显式用 ★Format::ND★（Nd Tensor ✓）而不是默认的 NCHW ✗
    //   输入规格文件格式（每行一个）：name dim0,dim1,... DTYPE
    const char *isf = getenv("HW_INPUTS_FILE");
    if (isf && isf[0]) {
        std::ifstream fin(isf);
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string nm, dimstr, dt;
            iss >> nm >> dimstr >> dt;
            NDTensorDesc d;
            std::stringstream ds(dimstr);
            std::string tok;
            while (std::getline(ds, tok, ',')) d.dims.push_back(atoi(tok.c_str()));
            d.format = Format::ND;                       // ★关键：ND 而不是 NCHW ✓★
            if (dt == "INT32") d.dataType = DataType::INT32;
            else if (dt == "INT64") d.dataType = DataType::INT64;
            else if (dt == "FLOAT16") d.dataType = DataType::FLOAT16;
            else d.dataType = DataType::FLOAT32;
            options.buildOptions.inputTensorDescs.push_back(d);
            std::cout << "  [Init] 显式输入 " << nm << " dims=" << dimstr
                      << " dtype=" << dt << " format=ND ✓" << std::endl;
        }
    }
    Status ret = manager->Init(options, built, nullptr);
    std::cout << "  Init rc=" << static_cast<int>(ret) << std::endl;
    if (ret != SUCCESS) {
        return 6;
    }

    if (!doRun) {
        manager->DeInit();
        std::cout << "★ 加载成功（未推理）✓" << std::endl;
        return 0;
    }

    // ④ 跑一次（输入置零 ✓）
    std::cout << "④ 分配缓冲并 Run（输入置零）" << std::endl;
    std::vector<std::unique_ptr<int8_t[]>> inStore;
    std::vector<std::shared_ptr<INDTensorBuffer>> inputs, outputs;
    for (size_t i = 0; i < inDesc.size(); ++i) {
        size_t size = 1;
        for (int32_t d : inDesc[i].dims) {
            size *= static_cast<size_t>(d > 0 ? d : 1);
        }
        size_t bytes = size * (1u << static_cast<int>(inDesc[i].dataType));  // 粗略：按 2^dtype
        if (bytes == 0 || bytes > (size_t)8 << 30) {
            bytes = 4;
        }
        inStore.emplace_back(new int8_t[bytes]);
        std::memset(inStore.back().get(), 0, bytes);
        inputs.push_back(CreateNDTensorBuffer(inDesc[i], inStore.back().get(), bytes));
        std::cout << "    输入[" << i << "] bytes=" << bytes << std::endl;
    }
    for (size_t i = 0; i < outDesc.size(); ++i) {
        outputs.push_back(CreateNDTensorBuffer(outDesc[i]));
    }
    std::cout << "  开始 Run …" << std::endl;
    ret = manager->Run(inputs, outputs);
    std::cout << "  ★Run rc=" << static_cast<int>(ret) << std::endl;
    manager->DeInit();
    std::cout << (ret == SUCCESS ? "★★ 推理成功 ✓★" : "✗ 推理失败") << std::endl;
    return ret == SUCCESS ? 0 : 7;
}
