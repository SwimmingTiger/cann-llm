/* qcast.c —— 穷举 QUANT_DTYPE_CAST 的合法形态（找 GE/设备接受的那一种 ✓）*/
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "neural_network_runtime/neural_network_runtime.h"
#define RCN(r) ((r)==0?"SUCCESS":(r)==2?"INVALID_PARAM":"?")
int main(void) {
    const size_t *ids=NULL; uint32_t n=0; OH_NNDevice_GetAllDevicesID(&ids,&n);
    size_t dev=ids[0];
    for (uint32_t i=0;i<n;i++){OH_NN_DeviceType t;OH_NNDevice_GetType(ids[i],&t);if((int)t==3)dev=ids[i];}
    printf("  设备 id=%zu\n", dev);
    /* 变体：{名称, 输入dtype, 输出dtype, srcT值, dstT值, 是否带axis} */
    struct V { const char*nm; OH_NN_DataType in,out; int32_t src,dst; int withAxis; } V[] = {
        {"U8->F32 (反量化)",  OH_NN_UINT8,   OH_NN_FLOAT32, 6, 11, 1},
        {"I8->F32 (反量化)",  OH_NN_INT8,    OH_NN_FLOAT32, 2, 11, 1},
        {"F32->U8 (量化)",    OH_NN_FLOAT32, OH_NN_UINT8,  11, 6, 1},
        {"F32->I8 (量化)",    OH_NN_FLOAT32, OH_NN_INT8,   11, 2, 1},
        {"U8->F32 无axis",   OH_NN_UINT8,   OH_NN_FLOAT32, 6, 11, 0},
        {"F32->F32 (同类型)", OH_NN_FLOAT32, OH_NN_FLOAT32,11, 11, 1},
    };
    for (unsigned k=0;k<sizeof(V)/sizeof(V[0]);k++) {
        printf("\n  ---- 变体 %u: %s ----\n", k, V[k].nm);
        OH_NNModel *m = OH_NNModel_Construct();
        int32_t sh4[4]={1,2,2,3}, sh1[1]={1};
        OH_NN_DataType dts[5]={V[k].in, OH_NN_INT64, OH_NN_INT64, OH_NN_INT64, V[k].out};
        const int32_t *shp[5]={sh4,sh1,sh1,sh1,sh4};
        int ok=1;
        for (int i=0;i<5;i++){
            NN_TensorDesc *d=OH_NNTensorDesc_Create();
            OH_NNTensorDesc_SetName(d,"t"); OH_NNTensorDesc_SetDataType(d,dts[i]);
            OH_NNTensorDesc_SetShape(d,shp[i],i==0||i==4?4:1);
            OH_NNTensorDesc_SetFormat(d,OH_NN_FORMAT_NONE);
            int r=(int)OH_NNModel_AddTensorToModel(m,d); OH_NNTensorDesc_Destroy(&d);
            if(r){printf("    AddTensor t%d rc=%d\n",i,r); ok=0; break;}
        }
        if(!ok){OH_NNModel_Destroy(&m);continue;}
        OH_NN_TensorType tt[5]={OH_NN_TENSOR,OH_NN_QUANT_DTYPE_CAST_SRC_T,
                                OH_NN_QUANT_DTYPE_CAST_DST_T,OH_NN_QUANT_DTYPE_CAST_AXIS,OH_NN_TENSOR};
        for(int i=0;i<5;i++) OH_NNModel_SetTensorType(m,i,tt[i]);
        {   NN_QuantParam *qp=OH_NNQuantParam_Create();
            double sc=0.05; int32_t zp=0; uint32_t nb=8;
            OH_NNQuantParam_SetNumBits(qp,&nb,1); OH_NNQuantParam_SetScales(qp,&sc,1);
            OH_NNQuantParam_SetZeroPoints(qp,&zp,1);
            OH_NNModel_SetTensorQuantParams(m,0,qp);      /* 输入挂量化参数 ✓ */
            OH_NNQuantParam_Destroy(&qp); }
        {   int64_t v[1];
            v[0]=V[k].src; OH_NNModel_SetTensorData(m,1,v,8);
            v[0]=V[k].dst; OH_NNModel_SetTensorData(m,2,v,8);
            v[0]=0;        OH_NNModel_SetTensorData(m,3,v,8); }
        uint32_t pi[3]={1,2,3}, ii[1]={0}, oi[1]={4};
        if (V[k].withAxis==0) { pi[0]=1; pi[1]=2; }
        OH_NN_UInt32Array par={pi,V[k].withAxis?3u:2u}, inp={ii,1}, outp={oi,1};
        int ra=(int)OH_NNModel_AddOperation(m,OH_NN_OPS_QUANT_DTYPE_CAST,&par,&inp,&outp);
        printf("    AddOperation rc=%d(%s)\n",ra,RCN(ra));
        if(ra){OH_NNModel_Destroy(&m);continue;}
        OH_NN_UInt32Array mi={ii,1}, mo={oi,1};
        OH_NNModel_SpecifyInputsAndOutputs(m,&mi,&mo);
        int rf=(int)OH_NNModel_Finish(m);
        printf("    Finish rc=%d\n",rf);
        if(rf){OH_NNModel_Destroy(&m);continue;}
        OH_NNCompilation *c=OH_NNCompilation_Construct(m);
        OH_NNCompilation_SetDevice(c,dev);
        int rb=(int)OH_NNCompilation_Build(c);
        printf("    ★★Build rc=%d(%s)★★ %s\n",rb,RCN(rb),
               rb==0?"⇒ ★★★ 这个形态可行！！！★★★":"⇒ 仍失败");
        OH_NNModel_Destroy(&m);
    }
    return 0;
}
