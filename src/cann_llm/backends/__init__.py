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

# --- nnrt 后端：用 MindSpore Lite NDK + NNRt 跑【第三方离线模型】（.ms）---
# 与前两个后端的根本区别：它不经过华为 LLM 引擎，所以模型结构不受
# 「逐层一对 K/V」那类约束。见 docs/offline-model-nnrt.md。
try:  # pragma: no cover - 平台相关
    from . import nnrt  # noqa: F401
except Exception as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"nnrt 后端不可用: {_e}", RuntimeWarning, stacklevel=2)

# --- hiai 后端：驱动系统内部引擎（libhiai_llm_engine.so），认官方模型目录结构 ---
try:  # pragma: no cover - 平台相关
    from . import hiai  # noqa: F401
except Exception as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"hiai 后端不可用: {_e}", RuntimeWarning, stacklevel=2)
