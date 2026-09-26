"""交互式 CLI 的斜杠命令。

单独成模块，让 ``chat.py`` 只保留会话主循环；也方便单测。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..chat.session import ChatSession
from ..config import AppConfig
from ..types import GenerationParams

#: /temp /topk ... 到 GenerationParams 字段的映射
_PARAM_FIELDS = {
    "temp": ("temperature", float),
    "topk": ("top_k", int),
    "topp": ("top_p", float),
    "rep": ("repetition_penalty", float),
    "maxtok": ("max_tokens", int),
}


def _show_params(session: ChatSession, stream: bool) -> None:
    p = session.params
    mode = "贪心" if p.greedy else "采样"
    print(f"  temperature={p.temperature}  top-k={p.top_k}  top-p={p.top_p}  "
          f"repetition_penalty={p.repetition_penalty}  max_tokens={p.max_tokens}  "
          f"({mode})  流式={'on' if stream else 'off'}")


def _save(path: str, session: ChatSession, cfg: AppConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 对话记录\n\n")
        f.write(f"- 模型：`{cfg.model.model_id}`\n")
        f.write(f"- 后端：`{cfg.model.backend}`\n")
        f.write(f"- 模板：`{cfg.model.chat_template}`\n\n")
        if session.system_prompt:
            f.write(f"**system**: {session.system_prompt}\n\n")
        for m in session.messages:
            f.write(f"**{m.role}**: {m.content}\n\n")
    print(f"  [已保存到 {path}]")


def handle_command(line: str, session: ChatSession, cfg: AppConfig,
                   stream: bool) -> Tuple[bool, bool]:
    """处理一条 ``/xxx`` 命令。

    :return: ``(stream, quit_now)``
    """
    cmd, _, rest = line[1:].partition(" ")
    cmd, rest = cmd.lower(), rest.strip()

    if cmd in ("quit", "exit", "q"):
        return stream, True

    if cmd in ("help", "h", "?"):
        from .chat import HELP
        print(HELP)
        return stream, False

    if cmd == "reset":
        session.reset()
        print("  [已清空对话历史]")
        return stream, False

    if cmd == "system":
        if not rest:
            print(f"  当前 system prompt: {session.system_prompt!r}")
        else:
            session.set_system(rest)
            print("  [已设置 system prompt 并清空历史]")
        return stream, False

    if cmd == "stream":
        if rest in ("on", "1", "true"):
            stream = True
        elif rest in ("off", "0", "false"):
            stream = False
        print(f"  [逐字输出 {'开启' if stream else '关闭'}]")
        return stream, False

    if cmd in _PARAM_FIELDS:
        field, caster = _PARAM_FIELDS[cmd]
        if not rest:
            _show_params(session, stream)
            return stream, False
        try:
            val = caster(rest)
            # 复用 GenerationParams 的校验
            session.params = replace(session.params, **{field: val})
        except (ValueError, TypeError) as e:
            print(f"  /{cmd} 取值非法: {e}")
            return stream, False
        print("  [已更新] ", end="")
        _show_params(session, stream)
        return stream, False

    if cmd == "params":
        _show_params(session, stream)
        return stream, False

    if cmd == "stats":
        st = session.last_stats
        if st is None:
            print("  还没有生成记录")
        else:
            print(f"  in={st.prompt_tokens} out={st.completion_tokens} "
                  f"prefill={st.prefill_ms:.0f}ms decode={st.decode_ms:.0f}ms "
                  f"total={st.total_ms:.0f}ms wall={st.wall_s:.2f}s "
                  f"({st.tokens_per_second:.1f} tok/s)  "
                  f"finish={session.last_finish_reason}")
        return stream, False

    if cmd == "save":
        if not rest:
            print("  用法: /save <文件>")
        else:
            try:
                _save(rest, session, cfg)
            except OSError as e:
                print(f"  保存失败: {e}")
        return stream, False

    print(f"  未知命令 /{cmd}（/help 看帮助）")
    return stream, False


def apply_param_constraints(session: ChatSession) -> None:
    """占位：将来在这里统一处理参数之间的约束（如贪心时忽略 top_k）。"""
    return None


__all__ = ["handle_command", "_PARAM_FIELDS", "GenerationParams"]
