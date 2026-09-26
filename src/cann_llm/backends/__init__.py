"""后端实现包。

import 本包会触发内置后端的注册（见文件末尾）。
"""

from .base import (  # noqa: F401
    EngineBackend,
    SerializedBackend,
    available_backends,
    backend_class,
    create_backend,
    register_backend,
)

# --- 内置后端：import 即注册 ---
# 注意：CANN 后端只在鸿蒙设备上可用（依赖 /system/lib64/ndk/libcann_llm_engine.so），
# 但 import 本身是安全的（不会加载 .so，加载发生在 load()）。
try:  # pragma: no cover - 平台相关
    from . import cann  # noqa: F401
except Exception as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"CANN 后端不可用: {_e}", RuntimeWarning, stacklevel=2)
