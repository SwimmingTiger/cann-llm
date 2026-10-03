/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NEURAL_NETWORK_RUNTIME_INNER_H
#define NEURAL_NETWORK_RUNTIME_INNER_H

#include "neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 定义Tensor信息结构体。包含名字，数据类型，维度信息，格式信息。
 *
 * @since 10
 * @version 1.1
 */
typedef struct OH_NN_TensorInfo {
    char name[128];
    OH_NN_DataType dataType;
    uint32_t dimensionCount;
    const int32_t *dimensions;
    OH_NN_Format format;
} OH_NN_TensorInfo;

/**
 * @brief 定义扩展字段结构体。
 *
 * @since 10
 * @version 1.1
 */
typedef struct OH_NN_Extension {
    char name[128];
    char *value;
    size_t valueSize;
} OH_NN_Extension;

/**
 * @brief 直接加载LiteGraph，完成模型搭建。
 *
 * 调用{@link OH_NNModel_Construct}创建模型实例后，直接调用本方法加载LiteGraph。加载LiteGraph后，只能调用
 * {@link OH_NNCompilation_Construct}创建模型编译器，或者调用{@link OH_NNModel_Destroy}销毁模型实例。\n
 *
 * 不允许本方法与{@link OH_NNModel_AddTensor}、{@link OH_NNModel_AddOperation}、
 * {@link OH_NNModel_SetTensorData}和{@link OH_NNModel_SpecifyInputsAndOutputs}
 * 等构图接口混用，否则返回{@link OH_NN_OPERATION_FORBIDDEN}错误。\n
 *
 * 如果本方法调用成功，返回{@link OH_NN_SUCCESS}，liteGraph将由NNRt管理，调用者无需释放，避免造成二次释放;
 * 如果方法返回其他错误码，则NNRt不会持有liteGraph，此时需要调用者主动释放内存。\n
 *
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param liteGraph 指向LiteGraph的指针。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @throw std::bad_alloc 本方法可能在转换原始指针到智能指针的过程中，抛出std::bad_alloc异常，此时liteGraph将被
 *        主动释放。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph,
    const OH_NN_Extension *extensions, size_t extensionSize);

/**
 * @brief 设置MetaGraph的输入输出信息。
 *
 * 调用{@link OH_NNModel_Construct}创建模型实例后，直接调用本方法设置MetaGraph的输入输出信息。然后调用{@link OH_NNModel_BuildFromMetaGraph}
 * 加载MetaGraph，完成模型搭建。\n
 *
 * 不允许本方法与{@link OH_NNModel_AddTensor}、和{@link OH_NNModel_SpecifyInputsAndOutputs}
 * 等构图接口混用，否则返回{@link OH_NN_OPERATION_FORBIDDEN}错误。\n
 *
 * 如果本方法调用成功，返回{@link OH_NN_SUCCESS}。\n
 *
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param inputsInfo 指向{@link OH_NN_TensorInfo}数组的指针，代表传入的输入Tensor信息。
 * @param inputSize 代表inputsInfo数组大小。
 * @param outputsInfo 指向{@link OH_NN_TensorInfo}数组的指针，代表传入的输出Tensor信息。
 * @param outputSize 代表outputsInfo数组大小。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @since 10
 * @version 1.1
 */
OH_NN_ReturnCode OH_NNModel_SetInputsAndOutputsInfo(OH_NNModel *model, const OH_NN_TensorInfo *inputsInfo,
    size_t inputSize, const OH_NN_TensorInfo *outputsInfo, size_t outputSize);

/**
 * @brief 直接加载MetaGraph，完成模型搭建。
 *
 * 调用{@link OH_NNModel_SetInputsAndOutputsInfo}设置好MetaGraph输入输出信息后，直接调用本方法加载MetaGraph。加载MetaGraph后，只能调用
 * {@link OH_NNCompilation_Construct}创建模型编译器，或者调用{@link OH_NNModel_Destroy}销毁模型实例。\n
 *
 * 不允许本方法与{@link OH_NNModel_AddTensor}、{@link OH_NNModel_AddOperation}、
 * {@link OH_NNModel_SetTensorData}、{@link OH_NNModel_SpecifyInputsAndOutputs}和{@link OH_NNModel_Finish}
 * 等构图接口混用，否则返回{@link OH_NN_OPERATION_FORBIDDEN}错误。\n
 *
 * 如果本方法调用成功，返回{@link OH_NN_SUCCESS}。\n
 *
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param metaGraph 指向MetaGraph的指针。
 * @param extensions 指向{@ OH_NN_Extension}数组的指针，代表传入的扩展字段。
 *                   例如，传递量化信息时，指定Extension.name为"QuantBuffer"，将量化Buffer赋给value和valueSize。
 * @param extensionSize 代表extensions数组大小。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @since 10
 * @version 1.1
 */
OH_NN_ReturnCode OH_NNModel_BuildFromMetaGraph(OH_NNModel *model, const void *metaGraph,
    const OH_NN_Extension *extensions, size_t extensionSize);

/**
 * @brief 判断cache文件是否存在。
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param cacheDir cache文件所在文件夹。
 * @param modelName 模型名字，与构图时传入的一致。
 * @return bool值，true表示cache文件存在，false表示cache文件不存在。
 * @since 11
 * @version 1.0
 */
bool OH_NNModel_HasCache(const char *cacheDir, const char *modelName, uint32_t version);

/**
 * @brief 获取NNRt device信息。
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param nnrtDevice 传出device参数。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NN_GetDeviceID(char *nnrtDevice, size_t len);

/**
 * @brief 获取NNRt isSupport AIPP信息
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param support 传出support参数。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NN_IsSupportAIPP(bool& support);

/**
 * @brief Synchronous execution of the model inference with aipp.
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param inputTensor An array of input tensors {@link NN_Tensor}.
 * @param inputCount Number of input tensors.
 * @param outputTensor An array of output tensors {@link NN_Tensor}.
 * @param outputCount Number of output tensors.
 * @param aippString The factor of aipp.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned.
 *         If the operation fails, an error code is returned.
 *         For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_RunSyncWithAipp(OH_NNExecutor *executor,
                                               NN_Tensor *inputTensor[],
                                               size_t inputCount,
                                               NN_Tensor *outputTensor[],
                                               size_t outputCount,
                                               const char* aippString);

/**
 * @brief 对cache进行crc校验和检验
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param buffer 传入buffer参数。
 * @param length 传入length参数。
 * @return unsigned short值，函数的crc校验和。
 * @since 11
 * @version 1.0
 */
unsigned short CacheInfoGetCrc16(char* buffer, size_t length);
#ifdef __cplusplus
}
#endif // __cpluscplus
#endif // NEURAL_NETWORK_RUNTIME_INNER_H