"""总览路由（屏①）：聚合所有 bench/sample/loop 的状态 + 待打分数。只读。"""

from __future__ import annotations

from fastapi import APIRouter

from ..services.data_access import build_overview

router = APIRouter()


@router.get("/overview")
def get_overview() -> dict:
    """全量总览：每个 bench 下每个 sample 的 loop 列表 + trace 计数 + 待打分数。"""
    return build_overview().model_dump()
