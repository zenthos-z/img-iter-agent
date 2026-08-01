"""Agent 系统提示词外部化加载器。

把写死在 agents/*.py 里的系统提示词移到 data/agents_config/<agent>.md。
读不到文件时回退代码默认（向后兼容，旧测试不破）。

约定：每个 agent 一个 .md 文件，正文就是系统提示词。
- generator.md → Generator 的「润色」提示词
- critic.md    → Critic 的「二分判定」提示词
- summarizer.md → Summarizer 的「归纳」提示词

只做这一件事。工具/skill 不在此列（后续单独接入）。
"""

from __future__ import annotations

from ..config import get_settings


def load_system_prompt(agent: str, default: str) -> str:
    """读 data/agents_config/<agent>.md；读不到回退 default。"""
    settings = get_settings()
    p = settings.data_root / "agents_config" / f"{agent}.md"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            return default
    return default


__all__ = ["load_system_prompt"]
