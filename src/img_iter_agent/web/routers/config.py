"""Agent 配置路由（屏⑤）：读写系统提示词 + 模型 id。仅 2 个可改项。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...config import get_settings
from ..services.agent_config import (
    AGENTS,
    get_agent_config,
    list_bench_ids,
    reset_agent_config,
    save_agent_config,
)

router = APIRouter()

# 生图模型 key→（settings 字段名, 展示标签）。前端启动弹窗从这里拉可选模型。
_IMAGE_MODEL_DEFS = [
    ("seedream_pro", "model_seedream_pro", "Seedream Pro"),
    ("gpt_image", "model_gpt_image", "GPT Image"),
    ("gemini_image", "model_gemini_image", "Gemini Image"),
    ("qwen_image", "model_qwen_image", "Qwen Image"),
]
# agent LLM key→（settings 字段名, 展示标签）。distiller 复用 summarizer_model（蒸馏要审图，
# 需多模态 LLM）。与 agent_config.AGENTS / _settings_default_model 保持一致。
_AGENT_MODEL_DEFS = [
    ("generator", "generator_model", "Generator"),
    ("critic", "critic_model", "Critic"),
    ("distiller", "summarizer_model", "Distiller"),
]


@router.get("/models")
def list_models() -> dict:
    """列出 .env 中已配置（非空）的模型，供启动弹窗下拉选择。

    image_models 是生图模型（决定 loop 的 model 字段）；
    agent_models 是 generator/critic/distiller 的 LLM（仅展示，由全局 .env 控制）。
    """
    s = get_settings()
    image_models = [
        {"key": k, "label": label, "model_id": mid}
        for k, fld, label in _IMAGE_MODEL_DEFS
        if (mid := getattr(s, fld, ""))
    ]
    agent_models = [
        {"key": k, "label": label, "model_id": mid}
        for k, fld, label in _AGENT_MODEL_DEFS
        if (mid := getattr(s, fld, ""))
    ]
    return {"image_models": image_models, "agent_models": agent_models}


class AgentConfigUpdate(BaseModel):
    system_prompt: str | None = None
    model: str | None = None


@router.get("/config")
def list_agents() -> dict:
    """列出所有 agent 及当前配置 + 可选 benchmark 列表（供配置页下拉切换 per-bench 技能）。"""
    return {
        "agents": [get_agent_config(a) for a in AGENTS],
        "benches": list_bench_ids(),
    }


@router.get("/config/{agent}")
def get_config(agent: str, bench: str | None = None) -> dict:
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return get_agent_config(agent, bench_id=bench)


@router.post("/config/{agent}")
def update_config(agent: str, update: AgentConfigUpdate, bench: str | None = None) -> dict:
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return save_agent_config(agent, system_prompt=update.system_prompt, model=update.model, bench_id=bench)


@router.post("/config/{agent}/reset")
def reset_config(agent: str, bench: str | None = None) -> dict:
    """恢复代码默认。"""
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return reset_agent_config(agent, bench_id=bench)
