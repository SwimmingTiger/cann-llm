/* w8a16.c —— ★按 hiai 白名单构造：FP16 激活 × INT8 权重（W8A16）★
 * 依据：IsCompatibleQuantType 白名单第 1 条 = (DT_FLOAT16, DT_INT8) ✓
 * 结构：y[FP16] = MATMUL(x[FP16], w[INT8])，w 挂量化参数
 */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "neural_network_runtime/neural_network_runtime.h"
#define RCN(r) ((r)==0?"SUCCESS":(r)==2?"INVALID_PARAM":(r)==1?"FAILED":"?")
int main(void) {
    const size_t *ids=NULL; uint32_t n=0; OH_NNDevice_GetAllDevicesID(&ids,&n);
    size_t dev=ids[0];
    for (uint32_t i=0;i<n;i++){OH_NN_DeviceType t;OH_NNDevice_GetType(ids[i],&t);if((int)t==3)dev=ids[i];}
    printf("  设备 id=%zu\n", dev);

    /* 变体：{名称, 激活dtype, 权重dtype, 输出dtype, 参数张量dtype} */
    struct V { const char*nm; OH_NN_DataType act, wgt, out, par; } V[] = {
        {"★合法形态：FP32 激活 × INT8 权重（W8A32）★", OH_NN_FLOAT32, OH_NN_INT8, OH_NN_FLOAT32, OH_NN_INT8},
    };
    for (unsigned k=0;k<sizeof(V)/sizeof(V[0]);k++) {
        printf("\n  ======== 变体 %u: %s ========\n", k, V[k].nm);
        OH_NNModel *m = OH_NNModel_Construct();
        int32_t s_x[2]={1,4}, s_w[2]={4,4}, s_y[2]={1,4}, s_p[1]={1};
        OH_NN_DataType dt[6] = {V[k].act, V[k].wgt, OH_NN_BOOL, OH_NN_BOOL, OH_NN_INT8, V[k].out};
        const int32_t *sh[6] = {s_x, s_w, s_p, s_p, s_p, s_y};
        OH_NN_TensorType tt[6] = {OH_NN_TENSOR, OH_NN_TENSOR,
                                  OH_NN_MATMUL_TRANSPOSE_A, OH_NN_MATMUL_TRANSPOSE_B,
                                  OH_NN_MATMUL_ACTIVATION_TYPE, OH_NN_TENSOR};   /* ★第3个参数★ */
        int ok=1;
        for (int i=0;i<6;i++){
            NN_TensorDesc *d=OH_NNTensorDesc_Create();
            OH_NNTensorDesc_SetName(d,"t"); OH_NNTensorDesc_SetDataType(d,dt[i]);
            OH_NNTensorDesc_SetShape(d,sh[i],(i==0||i==1||i==5)?2:1);
            OH_NNTensorDesc_SetFormat(d,OH_NN_FORMAT_NONE);
            int r=(int)OH_NNModel_AddTensorToModel(m,d); OH_NNTensorDesc_Destroy(&d);
            printf("    AddTensor t%d(%s) rc=%d\n", i,
                   dt[i]==OH_NN_FLOAT16?"FP16":dt[i]==OH_NN_FLOAT32?"FP32":dt[i]==OH_NN_INT8?"INT8":dt[i]==OH_NN_INT32?"INT32":"?", r);
            if(r){ok=0;break;}
            int rt=(int)OH_NNModel_SetTensorType(m,(uint32_t)i,tt[i]);
            if(rt){ printf("    SetTensorType t%d rc=%d\n", i, rt); ok=0; break; }
        }
        if(!ok){OH_NNModel_Destroy(&m);continue;}
        {   /* ★给 INT8 权重挂量化参数★ */
            NN_QuantParam *qp=OH_NNQuantParam_Create();
            double sc=0.05; int32_t zp=0; uint32_t nb=8;
            OH_NNQuantParam_SetNumBits(qp,&nb,1); OH_NNQuantParam_SetScales(qp,&sc,1);
            OH_NNQuantParam_SetZeroPoints(qp,&zp,1);
            int r=(int)OH_NNModel_SetTensorQuantParams(m,1,qp);
            printf("    ★SetTensorQuantParams(w) rc=%d★\n", r);
            OH_NNQuantParam_Destroy(&qp);
        }
        {   int8_t wv[16]; for(int i=0;i<16;i++) wv[i]=(int8_t)(i-8);
            int r=(int)OH_NNModel_SetTensorData(m,1,wv,16);
            printf("    SetTensorData(w) rc=%d\n", r);
            {   bool f = false;   /* ★TransposeA/B = OH_NN_BOOL 标量（源码要求 ✓）★ */
                printf("    SetTensorData(TransA/B) rc=%d\n",
                       (int)OH_NNModel_SetTensorData(m,2,&f,1));
                OH_NNModel_SetTensorData(m,3,&f,1); }
            int8_t act = 0;   /* ★ActivationType = OH_NN_FUSED_NONE★ */
            printf("    SetTensorData(ActType) rc=%d\n", (int)OH_NNModel_SetTensorData(m,4,&act,1)); }
        {   uint32_t pi[3]={2,3,4}, ii[2]={0,1}, oi[1]={5};
            OH_NN_UInt32Array par={pi,3}, inp={ii,2}, outp={oi,1};   /* ★3 个参数★ */
            int r=(int)OH_NNModel_AddOperation(m,OH_NN_OPS_MATMUL,&par,&inp,&outp);
            printf("    AddOperation(MATMUL) rc=%d(%s)\n",r,RCN(r));
            if(r){OH_NNModel_Destroy(&m);continue;}
            OH_NN_UInt32Array mi={ii,1}, mo={oi,1};
            printf("    SpecifyInputsAndOutputs rc=%d\n",(int)OH_NNModel_SpecifyInputsAndOutputs(m,&mi,&mo)); }
        {   int rf=(int)OH_NNModel_Finish(m);
            printf("    Finish rc=%d\n",rf);
            if(rf){OH_NNModel_Destroy(&m);continue;} }
        {   OH_NNCompilation *c=OH_NNCompilation_Construct(m);
            OH_NNCompilation_SetDevice(c,dev);
            OH_NNCompilation_SetPerformanceMode(c,OH_NN_PERFORMANCE_EXTREME);
            int rb=(int)OH_NNCompilation_Build(c);
            printf("    ★★Build rc=%d(%s)★★  %s\n",rb,RCN(rb),
                   rb==0?"⇒ ★★★ 白名单形态编译通过！！！★★★":"⇒ 仍失败"); }
        OH_NNModel_Destroy(&m);
    }
    return 0;
}
