// ★通用 NNRt 在线建图运行时★：读 graph.txt（onnx_to_nnrt.py 生成 ✓）⇒ 在线建图 ⇒ NPU Build ✓
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "neural_network_runtime.h"
#include "neural_network_core.h"

struct Tensor { std::string name; std::string dt; int rank = 0; std::vector<int32_t> dims;
                int ptype = 0; bool isInit = false; std::string file; long off = 0, bytes = 0; };
static std::vector<Tensor> T;
static uint8_t *BLOB = nullptr; static long BLOB_SZ = 0;

static OH_NN_DataType DT(const std::string &s) {
    if (s == "F32") return OH_NN_FLOAT32; if (s == "F16") return OH_NN_FLOAT16;
    if (s == "I32") return OH_NN_INT32;   if (s == "I64") return OH_NN_INT64;
    if (s == "B")   return OH_NN_BOOL;    if (s == "I8")  return OH_NN_INT8;
    if (s == "U8")  return OH_NN_UINT8;   return OH_NN_FLOAT32;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("用法: nnrt_build <graph.txt> <blob.bin>\n"); return 1; }
    std::ifstream fin(argv[1]);
    if (!fin) { printf("打不开 %s\n", argv[1]); return 1; }
    std::ifstream bf(argv[2], std::ios::binary | std::ios::ate);
    if (bf) { BLOB_SZ = (long)bf.tellg(); bf.seekg(0); BLOB = new uint8_t[BLOB_SZ];
              bf.read((char *)BLOB, BLOB_SZ); }
    printf("  权重 blob %ld B\n", BLOB_SZ);

    std::string line;
    struct Node { int op; std::vector<uint32_t> prm, ins, outs; };
    std::vector<Node> N; std::vector<uint32_t> mins, mouts;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line); std::string k; is >> k;
        if (k == "T") { Tensor t; int idx; is >> idx >> t.name >> t.dt >> t.rank;
            for (int i = 0; i < t.rank; ++i) { int32_t d; is >> d; t.dims.push_back(d); }
            std::string tk; if (is >> tk && tk == "PARAM") { is >> t.ptype; }   // ★参数枚举值✓★
            if ((int)T.size() <= idx) T.resize(idx + 1);
            T[idx] = t;
        } else if (k == "I") { int idx; Tensor tmp; is >> idx >> tmp.file >> tmp.off >> tmp.bytes;
            if ((int)T.size() > idx) { T[idx].isInit = true; T[idx].file = tmp.file;
                T[idx].off = tmp.off; T[idx].bytes = tmp.bytes; }
        } else if (k == "N") { Node n; int np, ni, no; is >> n.op >> np;
            for (int i = 0; i < np; ++i) { uint32_t v; is >> v; n.prm.push_back(v); }
            is >> ni; for (int i = 0; i < ni; ++i) { uint32_t v; is >> v; n.ins.push_back(v); }
            is >> no; for (int i = 0; i < no; ++i) { uint32_t v; is >> v; n.outs.push_back(v); }
            N.push_back(n);
        } else if (k == "IN") { uint32_t v; while (is >> v) mins.push_back(v);
        } else if (k == "OUT") { uint32_t v; while (is >> v) mouts.push_back(v); }
    }
    printf("  张量 %zu · 节点 %zu · 模型输入 %zu · 输出 %zu\n", T.size(), N.size(), mins.size(), mouts.size());

    // 设备 ✓
    size_t npu = 0; const size_t *ids = nullptr; uint32_t cnt = 0;
    OH_NNDevice_GetAllDevicesID(&ids, &cnt);
    for (uint32_t i = 0; i < cnt; ++i) { const char *nm = nullptr; OH_NNDevice_GetName(ids[i], &nm);
        if (nm && strstr(nm, "NPU")) npu = ids[i]; }
    printf("  NPU=%zu\n", npu);

    // ★持久化描述符 ✓（NNRt 可能不拷贝 ✓ §132 ✓）
    // ★改用纯静态数组保存描述符与维度✓★（怀疑 NNRt 存了指针 ✗ §135 ✓）
    static OH_NN_Tensor DESC[4096];
    static int32_t DIMS[4096][8];
    static const char *NAMES[4096];
    OH_NNModel *m = OH_NNModel_Construct();
    int bad = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        OH_NN_Tensor &d = DESC[i];
        NAMES[i] = T[i].name.c_str();
        for (int k = 0; k < T[i].rank && k < 8; ++k) DIMS[i][k] = T[i].dims[k];
        d.dataType = DT(T[i].dt);
        d.dimensionCount = (uint32_t)T[i].rank;
        d.dimensions = T[i].rank ? DIMS[i] : nullptr;             // ★秩 0 传 nullptr ✓★
        d.quantParam = nullptr;
        // ★参数张量必须设成对应的 OH_NN_TensorType✓★（这是 §135 找到的根因 ✓）
        d.type = T[i].ptype ? (OH_NN_TensorType)T[i].ptype : OH_NN_TENSOR;
        int r = OH_NNModel_AddTensor(m, &d);
        printf("  AT[%zu] %-32s dt=%d rank=%d rc=%d\n", i, T[i].name.c_str(), (int)d.dataType, (int)d.dimensionCount, r);
        if (r) ++bad;
        // ★紧跟其常量数据✓★（能跑通的那份就是"加完常量张量立刻 SetTensorData"✓）
        if (T[i].isInit && BLOB && T[i].off + T[i].bytes <= BLOB_SZ) {
            int rs = OH_NNModel_SetTensorData(m, (uint32_t)i, BLOB + T[i].off, (size_t)T[i].bytes);
            printf("  SD[%zu] %-32s off=%ld bytes=%ld rc=%d\n", i, T[i].name.c_str(), T[i].off, T[i].bytes, rs);
        }
    }
    printf("  AddTensor 失败 %d 个\n", bad);
    int badop = 0;
    for (size_t i = 0; i < N.size(); ++i) {
        // ★诊断模式：用硬编码局部数组复刻能跑通那份的写法✓★
        static uint32_t prmD[8], insD[8], outsD[8];
        for (size_t k = 0; k < N[i].prm.size() && k < 8; ++k) prmD[k] = N[i].prm[k];
        for (size_t k = 0; k < N[i].ins.size() && k < 8; ++k) insD[k] = N[i].ins[k];
        for (size_t k = 0; k < N[i].outs.size() && k < 8; ++k) outsD[k] = N[i].outs[k];
        OH_NN_UInt32Array P, I, O;
        P.data = prmD; P.size = (uint32_t)N[i].prm.size();
        I.data = insD; I.size = (uint32_t)N[i].ins.size();
        O.data = outsD; O.size = (uint32_t)N[i].outs.size();
        int r = OH_NNModel_AddOperation(m, (OH_NN_OperationType)N[i].op, &P, &I, &O);
        std::string ps, is_, os_;
        for (auto v : N[i].prm) ps += std::to_string(v) + ",";
        for (auto v : N[i].ins) is_ += std::to_string(v) + ",";
        for (auto v : N[i].outs) os_ += std::to_string(v) + ",";
        printf("  OP[%zu] op=%d params=[%s] ins=[%s] outs=[%s] rc=%d\n", i, N[i].op,
               ps.c_str(), is_.c_str(), os_.c_str(), r);
        if (r) ++badop;
    }
    printf("  AddOperation 失败 %d 个\n", badop);
    OH_NN_UInt32Array MI, MO;
    MI.data = mins.data(); MI.size = (uint32_t)mins.size();
    MO.data = mouts.data(); MO.size = (uint32_t)mouts.size();
    printf("  Specify=%d\n", (int)OH_NNModel_SpecifyInputsAndOutputs(m, &MI, &MO));
    printf("  Finish=%d\n", (int)OH_NNModel_Finish(m));
    // ★★直接问设备：每个算子支不支持（权威 ✓ §140 下一步 ✓）★★
    {
        const bool *sup = nullptr; uint32_t opCnt = 0;
        OH_NN_ReturnCode rg = OH_NNModel_GetAvailableOperations(m, npu, &sup, &opCnt);
        printf("  ★GetAvailableOperations rc=%d opCount=%u★\n", (int)rg, opCnt);
        if (rg == 0 && sup) {
            // 按算子类型统计 ✓
            std::vector<uint32_t> seen(300, 0), okc(300, 0);
            for (size_t i = 0; i < N.size() && i < opCnt; ++i) {
                int op = N[i].op;
                if (op < 0 || op >= 300) continue;
                seen[op]++; if (sup[i]) okc[op]++;
            }
            for (int op = 0; op < 300; ++op)
                if (seen[op]) printf("     op=%-3d 支持 %u / %u\n", op, okc[op], seen[op]);
        }
    }
    OH_NNCompilation *c = OH_NNCompilation_Construct(m);
    printf("  SetDevice=%d\n", (int)OH_NNCompilation_SetDevice(c, npu));
    int rb = OH_NNCompilation_Build(c);
    printf("  ★★Build=%d★★ %s\n", rb, rb == 0 ? "★NPU 编译成功★" : "✗");
    return 0;
}
