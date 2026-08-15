"""Agent 系统提示词 + 模型 id 外部化加载器。

把写死在 agents/*.py 里的系统提示词移到 data/agents_config/<agent>.md；
模型 id 移到 data/agents_config/<agent>.json（``{"model": "..."}``）。
读不到文件时回退代码默认（向后兼容，旧测试不破）。

约定：每个 agent 一个 .md（正文=系统提示词）+ 一个 .json（model 覆盖，可选）。
- generator.md / .json → Generator
- critic.md    / .json → Critic
- distiller.md / .json → Distiller（skill-author authoring 阶段；model 复用 summarizer_model 默认）

load_agent_model 返回空串时，调用方应回退到 settings 里的 role 默认 model（.env）。
只做这两件事。工具/skill 不在此列。
"""

from __future__ import annotations

import json

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


def load_agent_model(agent: str, default: str = "") -> str:
    """读 data/agents_config/<agent>.json 的 ``model`` 字段；读不到/为空回退 default。

    与 ``load_system_prompt`` 对称：提示词走 .md，model 走 .json。
    返回 default 的情形：文件不存在 / JSON 损坏 / model 字段空——调用方据此回退 settings 默认。
    """
    settings = get_settings()
    p = settings.data_root / "agents_config" / f"{agent}.json"
    if not p.exists():
        return default
    try:
        model = json.loads(p.read_text(encoding="utf-8")).get("model", "")
    except (json.JSONDecodeError, OSError):
        return default
    return model or default


__all__ = ["load_agent_model", "load_system_prompt"]
