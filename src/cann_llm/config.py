"""配置加载：TOML 文件 < 环境变量 < 命令行，后者覆盖前者。

之所以自己写而不依赖 pydantic-settings，是为了保持核心零依赖。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    #: 对外暴露的模型 id（OpenAI 的 model 字段）。
    #: 留空则**自动取模型目录名**（见 :attr:`resolved_id`）—— 换模型时不用手工改。
    model_id: Optional[str] = None
    backend: str = "cann"
    chat_template: str = "chatml"
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    max_tokens: int = 256
    # ---- 采样参数：None = 【未指定 → 跟随模型自带的配置】----
    # ★ 这里绝不写死"推荐值"。模型目录里的 api_config.json 才是权威来源
    #   （temperature / topK / topP / repetitionPenalty / sampleFlag）。
    #   实测踩过的坑：本文件曾写死 top_p=0.95，而官方模型配置是 0.8 ——
    #   每请求下发 setter 时把模型的真值盖掉了，而且从输出上完全看不出来。
    #   命令行 / 配置文件里**显式**给出的值仍然最高优先。
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    #: 采样随机种子。None = **每次请求换一个随机种子**（模型配置里的 seed 也【不】沿用）
    #: —— 这样同一提示每次的回答才会不一样。设成固定值则输出可复现。
    #  （引擎自己的默认是写死 seed=99，沿用会让每次结果完全相同。）
    seed: Optional[int] = None
    #: 上下文长度（用于 /v1/models 展示）
    # 0 = 未指定，交给后端按【模型自己的配置】探测（kv_cache_max_len）。
    # ★ 这里绝不能写死一个数字：它会覆盖模型真值，还会让"输入超出 KV 缓存（本模型 N token）"
    #   这类提示写出属于别的模型的上限（实测踩过：默认 2048 覆盖了实际 4096）。
    context_length: int = 0

    @property
    def resolved_id(self) -> str:
        """实际对外暴露的模型 id。

        优先用显式设置的 ``model_id``；没设就取 ``model_dir`` 的目录名
        （例如 ``/path/to/models/qwen3_4b`` -> ``qwen3_4b``）。
        这样 ``-d`` 换模型时名字会跟着变，不需要额外参数。
        """
        if self.model_id:
            return self.model_id
        if self.model_dir:
            # ★ 用 derive_model_name 而不是 basename(model_dir)：
            #   basename(".") 会返回 "."，于是 `-d .` 时模型名变成一个点。
            #   derive_model_name 会先 abspath（`-d .` → 真实目录名），
            #   必要时再从 <model>.json / api_config.json 的 modelPath 推。
            from .modelpkg import derive_model_name
            name = derive_model_name(self.model_dir)
            if name and name != "unknown":
                return name
        return "cann-llm"


#: 采样参数的内置兜底值 —— **只在「配置与模型都没给」时生效**，绝不拿来覆盖模型真值
_SAMPLER_FALLBACK: "Dict[str, Any]" = {
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}


def resolve_sampler(cfg: "ModelConfig", model_sampler: "Mapping[str, Any]") -> "Dict[str, Any]":
    """合并出采样参数，供构造 :class:`GenerationParams` 使用。

    优先级：**配置 / 命令行显式给的 > 模型自带（api_config.json）的 > 内置兜底**。

    ★ 兜底只在两者都没有时生效 —— 绝不用它去覆盖模型的真值。
    ★ ``seed`` **不**从模型配置取：模型里写的是引擎默认的 99，沿用会让每次输出
      完全相同；这里保持 ``None``（= 每次请求换一个随机种子）。
    """
    out: "Dict[str, Any]" = {}
    for key, fallback in _SAMPLER_FALLBACK.items():
        val = getattr(cfg, key, None)
        out[key] = model_sampler.get(key, fallback) if val is None else val
    out["seed"] = cfg.seed
    return out


@dataclass(frozen=True)
class AgentSettings:
    """agent 循环的配置。

    注意：HTTP 层**不**执行工具，只按 OpenAI 标准把 tool_calls 返回给客户端；
    这个配置只作用于 CLI 交互式对话（以及直接调用 AgentLoop 的场合）。
    """

    #: 最多迭代几轮（含最后一轮强制收尾）
    max_steps: int = 4


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
