"""Agent 配置路由（屏⑤）：读写系统提示词 + 模型 id。仅 2 个可改项。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.agent_config import AGENTS, get_agent_config, reset_agent_config, save_agent_config

router = APIRouter()


class AgentConfigUpdate(BaseModel):
    system_prompt: str | None = None
    model: str | None = None


@router.get("/config")
def list_agents() -> dict:
    """列出所有 agent 及当前配置。"""
    return {"agents": [get_agent_config(a) for a in AGENTS]}


@router.get("/config/{agent}")
def get_config(agent: str) -> dict:
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return get_agent_config(agent)


@router.post("/config/{agent}")
def update_config(agent: str, update: AgentConfigUpdate) -> dict:
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return save_agent_config(agent, system_prompt=update.system_prompt, model=update.model)


@router.post("/config/{agent}/reset")
def reset_config(agent: str) -> dict:
    """恢复代码默认。"""
    if agent not in AGENTS:
        raise HTTPException(status_code=404, detail=f"未知 agent: {agent}")
    return reset_agent_config(agent)
