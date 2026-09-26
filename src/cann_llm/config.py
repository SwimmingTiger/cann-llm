"""配置加载：TOML 文件 < 环境变量 < 命令行，后者覆盖前者。

之所以自己写而不依赖 pydantic-settings，是为了保持核心零依赖。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

ENV_PREFIX = "CANN_LLM_"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    #: 引擎一次只跑一路推理，这里限制排队长度，避免挂死
    max_queue: int = 8
    queue_timeout_s: float = 120.0
    request_timeout_s: float = 600.0
    api_key: Optional[str] = None          # 非空则要求 Authorization: Bearer
    cors_allow_origin: str = "*"
    #: True = 本服务实现不了的字段一律 400；
    #: False（默认）= 只忽略并在响应头 X-Cann-Llm-Ignored-Fields 回报。
    #: 默认宽容是有意的：多数现代客户端即使普通聊天也会带 tools，
    #: 一律报错会让它们完全用不了。想「宁可报错也别给我假象」就设 True。
    reject_unsupported: bool = False
    #: 工具执行模式：
    #:   "auto"（默认）—— 客户端声明的工具若本服务已注册，就由服务端执行并
    #:                    返回最终答案（服务端即 agent）；否则按 OpenAI 标准
    #:                    把 tool_calls 返回给客户端自行执行
    #:   "off"         —— 永远按 OpenAI 标准返回 tool_calls
    #:   "on"          —— 永远由服务端执行（未注册的工具会作为错误回给模型）



@dataclass(frozen=True)
class ModelConfig:
    #: 模型目录：需包含 omc / SubGraph_0.weight / embedding / tokenizer / json
    model_dir: Optional[str] = None
    #: 对外暴露的模型 id（OpenAI 的 model 字段）
    model_id: str = "qwen2.5-1.5b"
    backend: str = "cann"
    chat_template: str = "chatml"
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    #: 上下文长度（用于 /v1/models 展示）
    context_length: int = 2048


@dataclass(frozen=True)
class AgentSettings:
    """agent 循环的配置。

    注意：HTTP 层**不**执行工具，只按 OpenAI 标准把 tool_calls 返回给客户端；
    这个配置只作用于 CLI 交互式对话（以及直接调用 AgentLoop 的场合）。
    """

    #: 最多迭代几轮（含最后一轮强制收尾）
    max_steps: int = 4
    #: 工具结果回填时的截断长度
    max_result_chars: int = 4000
    #: 在工具说明后追加「该用就用」的强指令。默认关闭 —— 它会往 prompt 里
    #: 塞调用方没写的指令，属于改变模型行为；实测对小模型很有效，想要就显式打开。
    force_tool_use: bool = False


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    server: ServerConfig = ServerConfig()
    agent: AgentSettings = AgentSettings()

    def merged(self, **sections: Any) -> "AppConfig":
        """返回覆盖了若干字段的新配置，例如 ``merged(model={"port": 1})`` 非法，
        只接受与 dataclass 字段同名的字典。"""
        out = self
        if "model" in sections:
            out = replace(out, model=replace(out.model, **sections["model"]))
        if "server" in sections:
            out = replace(out, server=replace(out.server, **sections["server"]))
        if "agent" in sections:
            out = replace(out, agent=replace(out.agent, **sections["agent"]))
        return out


_SECTION_TYPES = {"model": ModelConfig, "server": ServerConfig,
                  "agent": AgentSettings}


def _coerce(cls, data: dict) -> Any:
    """只保留 dataclass 认识的字段，并做基本类型转换。"""
    import dataclasses

    known = {f.name: f.type for f in dataclasses.fields(cls)}
    kw = {}
    for k, v in data.items():
        if k not in known:
            continue
        kw[k] = v
    return cls(**kw)


def load_config(path: Optional[str | os.PathLike] = None,
                env: Optional[dict] = None) -> AppConfig:
    """从 TOML 文件 + 环境变量加载配置。

    环境变量形式：``CANN_LLM_MODEL__MODEL_DIR`` / ``CANN_LLM_SERVER__PORT``
    （双下划线分隔 section 与字段）。
    """
    env = dict(os.environ if env is None else env)
    raw: dict[str, dict] = {}

    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with p.open("rb") as f:
            raw = tomllib.load(f)

    for key, value in env.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section, _, field_name = key[len(ENV_PREFIX):].partition("__")
        section, field_name = section.lower(), field_name.lower()
        if section not in _SECTION_TYPES:
            continue
        raw.setdefault(section, {})[field_name] = _parse_scalar(value)

    cfg = AppConfig()
    for section, cls in _SECTION_TYPES.items():
        if section in raw and isinstance(raw[section], dict):
            cfg = replace(cfg, **{section: _coerce(cls, raw[section])})
    return cfg


def _parse_scalar(v: str) -> Any:
    low = v.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v
