"""统一异常层次。

上层（CLI / HTTP 服务）只依赖这里，不感知具体后端；HTTP 层会把它们映射成
合适的 status code，见 ``api.openai.error_response``。
"""

from __future__ import annotations


class CannLlmError(Exception):
    """本项目的异常基类。"""

    #: 供 HTTP 层映射用的建议状态码
    http_status = 500
    #: OpenAI 风格的错误类型字符串
    error_type = "internal_error"


class BackendUnavailableError(CannLlmError):
    """后端不可用：动态库缺失、平台不支持、设备无 NPU 等。"""

    http_status = 503
    error_type = "backend_unavailable"


class ModelLoadError(CannLlmError):
    """模型目录不完整或加载失败。"""

    http_status = 500
    error_type = "model_load_failed"


class InvalidRequestError(CannLlmError):
    """请求参数非法（采样参数越界、消息格式错误等）。"""

    http_status = 400
    error_type = "invalid_request_error"


class GenerationError(CannLlmError):
    """推理过程中失败。"""

    http_status = 500
    error_type = "generation_failed"


class BusyError(CannLlmError):
    """引擎一次只能跑一路推理，排队超时。"""

    http_status = 503
    error_type = "server_busy"
