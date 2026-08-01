"""Agent 配置服务：把系统提示词 + 模型 id 外部化到 data/agents_config/。

只 2 个可改项：system_prompt + model_id。工具/skill 不在此列（后续单独接入）。
文件布局：
  data/agents_config/<agent>.md   — 系统提示词正文
  data/agents_config/<agent>.json — {"model": "..."}（可选）

读不到时回退代码内默认（向后兼容，见 agents/*.py 的 _load_config）。

支持三个 agent：generator / critic / summarizer。
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import Settings, get_settings

AGENTS = ("generator", "critic", "summarizer")

# 代码内默认值（与 agents/*.py 里硬编码一致）。首次写盘时用它们初始化。
_DEFAULT_PROMPTS: dict[str, str] = {
    "generator": (
        "你是生图提示词工程师。把下面的生图指令精炼成一段清晰的生图 prompt"
        "（英文优先），保留所有关键约束，不要多余解释。"
    ),
    "critic": (
        "你是严格的产品图评判员。对下列 checklist 项逐项判定 通过/不通过，"
    ),
    "summarizer": (
        "你是生图经验归纳员。根据本轮评分，提炼 1-3 条可复用的生图经验"
        "（针对该模型/模式），只输出要点。"
    ),
}


def config_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    d = settings.data_root / "agents_config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prompt_path(agent: str, settings: Settings) -> Path:
    return config_dir(settings) / f"{agent}.md"


def _meta_path(agent: str, settings: Settings) -> Path:
    return config_dir(settings) / f"{agent}.json"


def get_agent_config(agent: str, *, settings: Settings | None = None) -> dict:
    """读一个 agent 的配置。读不到文件则返回代码默认。"""
    settings = settings or get_settings()
    prompt_path = _prompt_path(agent, settings)
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = _DEFAULT_PROMPTS.get(agent, "")

    model = ""
    meta_path = _meta_path(agent, settings)
    if meta_path.exists():
        try:
            model = json.loads(meta_path.read_text(encoding="utf-8")).get("model", "")
        except (json.JSONDecodeError, OSError):
            model = ""

    # 若配置里没填 model，回退 settings 默认
    if not model:
        model = _settings_default_model(agent, settings)

    return {"agent": agent, "system_prompt": system_prompt, "model": model}


def save_agent_config(
    agent: str, *, system_prompt: str | None = None, model: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """写一个 agent 的配置（只写非 None 的字段）。返回写后的完整配置。"""
    settings = settings or get_settings()
    cur = get_agent_config(agent, settings=settings)

    if system_prompt is not None:
        _prompt_path(agent, settings).write_text(system_prompt, encoding="utf-8")
        cur["system_prompt"] = system_prompt
    if model is not None:
        _meta_path(agent, settings).write_text(
            json.dumps({"model": model}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cur["model"] = model
    return cur


def reset_agent_config(agent: str, *, settings: Settings | None = None) -> dict:
    """恢复代码默认（删配置文件）。"""
    settings = settings or get_settings()
    for p in (_prompt_path(agent, settings), _meta_path(agent, settings)):
        if p.exists():
            p.unlink()
    return get_agent_config(agent, settings=settings)


def _settings_default_model(agent: str, settings: Settings) -> str:
    """settings 里各 agent 的默认 model 字段。"""
    mapping = {
        "generator": settings.generator_model,
        "critic": settings.critic_model,
        "summarizer": settings.summarizer_model,
    }
    return mapping.get(agent, "")
