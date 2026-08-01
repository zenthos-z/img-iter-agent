"""loop 路由（屏②③）：loop 详情、trace 详情、loop 远程控制。

远程控制复用 loop_runner：start → 跑到 interrupt → resume/stop。
状态监测：status 返回 phase/round/interrupt_payload。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import LoopControlRequest, LoopStartRequest
from ..services.data_access import build_loop_detail
from ..services.loop_runner import get_runner

router = APIRouter()


@router.get("/loops/{loop_id}")
def get_loop(loop_id: str) -> dict:
    """loop 详情：trace 列表 + 经验 + target 图 + 运行态。"""
    runner = get_runner()
    handle = runner.get(loop_id)
    status_extra = None
    if handle is not None:
        status_extra = {
            "status": handle.phase,
            "round": handle.round,
            "interrupt_payload": handle.interrupt_payload,
        }
    detail = build_loop_detail(loop_id, status_extra=status_extra)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    return detail.model_dump()


@router.get("/loops/{loop_id}/status")
def get_loop_status(loop_id: str) -> dict:
    """轻量状态：只返回 phase/round/interrupt_payload（前端轮询用）。"""
    handle = get_runner().get(loop_id)
    if handle is None:
        # 不是 web runner 启动的 loop：从 meta 推断
        detail = build_loop_detail(loop_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
        return {
            "loop_id": loop_id,
            "phase": detail.status,
            "round": detail.round,
            "interrupt_payload": None,
            "controlled": False,
        }
    return {
        "loop_id": loop_id,
        "phase": handle.phase,
        "round": handle.round,
        "interrupt_payload": handle.interrupt_payload,
        "last_error": handle.last_error,
        "controlled": True,
    }


# ---- 远程控制 ----


@router.post("/loops")
def start_loop(req: LoopStartRequest) -> dict:
    """启动一个新 loop，跑到第一个 interrupt。返回 loop_id。"""
    loop_id = get_runner().start(
        bench_id=req.bench_id, sample_id=req.sample_id, model=req.model, note=req.note
    )
    return {"loop_id": loop_id, "phase": "running"}


@router.post("/loops/{loop_id}/resume")
def resume_loop(loop_id: str, req: LoopControlRequest) -> dict:
    """用一个 decision 续跑一轮。decision: continue / stop / 任意调整方向文本。"""
    ok = get_runner().resume(loop_id, req.decision)
    if not ok:
        handle = get_runner().get(loop_id)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"loop {loop_id} 不是 web 启动的")
        raise HTTPException(
            status_code=409,
            detail=f"loop 当前 phase={handle.phase}，不能 resume（只在 awaiting_review 时可继续）",
        )
    return {"loop_id": loop_id, "phase": "running", "decision": req.decision}
