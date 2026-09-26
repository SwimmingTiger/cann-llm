"""交互式 CLI 的斜杠命令。"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ..config import AppConfig
from ..tools import default_registry
from .state import CliState

#: /temp /topk ... 到 GenerationParams 字段的映射
_PARAM_FIELDS = {
    "temp": ("temperature", float),
    "topk": ("top_k", int),
    "topp": ("top_p", float),
    "rep": ("repetition_penalty", float),
    "maxtok": ("max_tokens", int),
}


def _show_params(state: CliState) -> None:
    p = state.params
    mode = "贪心" if p.greedy else "采样"
    tools = ", ".join(state.tool_names) if state.tool_names else "关闭"
    print(f"  temperature={p.temperature}  top-k={p.top_k}  top-p={p.top_p}  "
          f"repetition_penalty={p.repetition_penalty}  max_tokens={p.max_tokens}  "
          f"({mode})\n  流式={'on' if state.stream else 'off'}  工具={tools}")


def _list_tools(state: CliState) -> None:
    reg = default_registry()
    active = set(state.tool_names)
    print(f"  可用工具（当前启用 {len(active)} 个）:")
    for name in reg.names():
        tool = reg.get(name)
        mark = "✓" if name in active else " "
        risk = "   [dangerous：需显式启用]" if tool.dangerous else ""
        first = (tool.description or "").splitlines()[0][:50]
        print(f"    [{mark}] {name:18} {first}{risk}")
    if not reg.names():
        print("    （无）")
    print("  用法: /tools <名字>…  切换启用；/tools all 全部安全工具；/tools none 全关")


def _toggle_tools(arg: str, state: CliState) -> None:
    reg = default_registry()
    if arg in ("all", "safe"):
        state.tool_names = list(reg.select().names())
    elif arg == "none":
        state.tool_names = []
    else:
        names = [x for x in arg.replace(",", " ").split() if x]
        unknown = [n for n in names if n not in reg]
        if unknown:
            print(f"  未注册的工具: {', '.join(unknown)}（用 /tools 看可用列表）")
            return
        cur = set(state.tool_names)
        for n in names:
            tool = reg.get(n)
            if tool.dangerous and n not in cur:
                print(f"  工具 {n!r} 标记为 dangerous，CLI 不自动开启；"
                      f"请在配置里显式选择并自行评估风险")
                continue
            cur.symmetric_difference_update({n})
        state.tool_names = [n for n in reg.names() if n in cur]
    print(f"  已启用工具: {', '.join(state.tool_names) or '（无，退化为普通对话）'}")


def _save(path: str, state: CliState, cfg: AppConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 对话记录\n\n")
        f.write(f"- 模型：`{cfg.model.model_id}`\n")
        f.write(f"- 后端：`{cfg.model.backend}`\n")
        f.write(f"- 模板：`{cfg.model.chat_template}`\n")
        if state.tool_names:
            f.write(f"- 工具：{', '.join(state.tool_names)}\n")
        f.write("\n")
        if state.system_prompt:
            f.write(f"**system**: {state.system_prompt}\n\n")
        for m in state.messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    f.write(f"**assistant→tool**: `{tc.name}` {tc.arguments}\n\n")
                if m.content:
                    f.write(f"**assistant**: {m.content}\n\n")
            elif m.role == "tool":
                f.write(f"**tool→{m.name}**: {m.content[:500]}\n\n")
            else:
                f.write(f"**{m.role}**: {m.content}\n\n")
    print(f"  [已保存到 {path}]")


def handle_command(line: str, state: CliState, cfg: AppConfig) -> bool:
    """处理一条 ``/xxx`` 命令。

    :return: 是否退出
    """
    cmd, _, rest = line[1:].partition(" ")
    cmd, rest = cmd.lower(), rest.strip()

    if cmd in ("quit", "exit", "q"):
        return True

    if cmd in ("help", "h", "?"):
        from .chat import HELP
        print(HELP)
        return False

    if cmd == "reset":
        state.reset()
        print("  [已清空对话历史]")
        return False

    if cmd == "system":
        if not rest:
            print(f"  当前 system prompt: {state.system_prompt!r}")
        else:
            state.set_system(rest)
            print("  [已设置 system prompt 并清空历史]")
        return False

    if cmd == "stream":
        if rest in ("on", "1", "true"):
            state.stream = True
        elif rest in ("off", "0", "false"):
            state.stream = False
        print(f"  [逐字输出 {'开启' if state.stream else '关闭'}]")
        return False

    if cmd == "tools":
        if not rest:
            _list_tools(state)
        else:
            _toggle_tools(rest, state)
        return False

    if cmd in _PARAM_FIELDS:
        field, caster = _PARAM_FIELDS[cmd]
        if not rest:
            _show_params(state)
            return False
        try:
            val = caster(rest)
            state.params = replace(state.params, **{field: val})   # 复用构造校验
        except (ValueError, TypeError) as e:
            print(f"  /{cmd} 取值非法: {e}")
            return False
        print("  [已更新] ", end="")
        _show_params(state)
        return False

    if cmd == "params":
        _show_params(state)
        return False

    if cmd == "stats":
        st = state.last_stats
        if st is None:
            print("  还没有生成记录")
        else:
            print(f"  in={st.prompt_tokens} out={st.completion_tokens} "
                  f"prefill={st.prefill_ms:.0f}ms decode={st.decode_ms:.0f}ms "
                  f"total={st.total_ms:.0f}ms wall={st.wall_s:.2f}s "
                  f"({st.tokens_per_second:.1f} tok/s)  "
                  f"finish={state.last_finish_reason}  agent步数={state.last_steps}")
        return False

    if cmd == "save":
        if not rest:
            print("  用法: /save <文件>")
        else:
            try:
                _save(rest, state, cfg)
            except OSError as e:
                print(f"  保存失败: {e}")
        return False

    print(f"  未知命令 /{cmd}（/help 看帮助）")
    return False


__all__ = ["handle_command", "_PARAM_FIELDS"]
