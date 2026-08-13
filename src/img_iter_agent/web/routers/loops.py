"""loop 路由（屏②③）：loop 详情、trace 详情、loop 远程控制。

远程控制复用 loop_runner：start → 跑到 interrupt → resume/stop。
状态监测：status 返回 phase/round/interrupt_payload。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...config import get_settings
from ..models import HintCreateRequest, HintOut, LoopControlRequest, LoopStartRequest, MemoryWriteRequest
from ..services.data_access import build_loop_detail, read_events_since
from ..services.loop_runner import LoopBusyError, get_runner

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
            "last_error": handle.last_error,
        }
    detail = build_loop_detail(loop_id, status_extra=status_extra)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    # 补当前生效的人工提示词（loop+sample 合并视图）
    detail.hints = [HintOut(**h) for h in (get_runner().get_hints(loop_id) or [])]
    return detail.model_dump()


@router.get("/loops/{loop_id}/status")
def get_loop_status(loop_id: str) -> dict:
    """轻量状态：phase/round/interrupt_payload + 当前 agent 节点（前端轮询用）。"""
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
            "current_node": None,
            "rounds_remaining": 0,
            "controlled": False,
        }
    # 实时查 graph 的「即将执行节点」，让前端看到 generator→critic→summarizer 推进
    current_node = None
    if handle.app is not None and handle.cfg is not None:
        try:
            snap = handle.app.get_state(handle.cfg)
            nxt = getattr(snap, "next", ()) or ()
            # next 是「下一步要跑的节点名」；interrupt 时为空
            current_node = nxt[0] if nxt else None
        except Exception:  # noqa: BLE001
            current_node = None
    return {
        "loop_id": loop_id,
        "phase": handle.phase,
        "round": handle.round,
        "interrupt_payload": handle.interrupt_payload,
        "current_node": current_node,
        "rounds_remaining": handle.rounds_remaining,
        "last_error": handle.last_error,
        "controlled": True,
    }


@router.get("/loops/{loop_id}/events")
def get_loop_events(loop_id: str, since: int = 0) -> dict:
    """活动流事件（前端轮询用）：读 events.jsonl，返回 since 之后的增量 + 总行数（下次游标）。

    直接读文件、不依赖内存 LoopHandle——CLI/脚本起的 loop（web 内存无 handle）也能拿到事件。
    """
    if build_loop_detail(loop_id) is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    run_dir = get_settings().run_dir(loop_id)
    events, total = read_events_since(run_dir, since)
    return {
        "loop_id": loop_id,
        "events": events,
        "total": total,
        "controlled": get_runner().get(loop_id) is not None,
    }


# ---- 远程控制 ----


@router.post("/loops")
def start_loop(req: LoopStartRequest) -> dict:
    """启动一个新 loop，跑到第一个 interrupt。返回 loop_id。"""
    hints = [h.model_dump() for h in req.hints] if req.hints else None
    loop_id = get_runner().start(
        bench_id=req.bench_id, sample_id=req.sample_id,
        model=req.model, note=req.note, rounds=req.rounds, hints=hints,
    )
    return {"loop_id": loop_id, "phase": "running"}


@router.post("/loops/{loop_id}/resume")
def resume_loop(loop_id: str, req: LoopControlRequest) -> dict:
    """用一个 decision 续跑一轮。decision: continue / stop / 任意调整方向文本。"""
    ok = get_runner().resume(loop_id, req.decision)
    if not ok:
        handle = get_runner().get(loop_id)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在或不可续跑")
        raise HTTPException(
            status_code=409,
            detail=f"loop 当前 phase={handle.phase}，不能 resume（只在 awaiting_review 时可继续）",
        )
    return {"loop_id": loop_id, "phase": "running", "decision": req.decision}


# ---- 人工提示词（hints）----


@router.get("/loops/{loop_id}/hints")
def get_loop_hints(loop_id: str) -> dict:
    """当前生效的人工提示词（loop+sample 合并视图）。"""
    return {"hints": get_runner().get_hints(loop_id) or []}


@router.post("/loops/{loop_id}/hints", status_code=201)
def add_loop_hint(loop_id: str, req: HintCreateRequest) -> dict:
    """运行中新增一条人工提示词（按 scope 落盘，下一轮 invoke 生效）。"""
    hint = get_runner().add_hint(loop_id, req.agent, req.text, req.scope)
    if hint is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在或不可写")
    return hint


@router.delete("/loops/{loop_id}/hints/{hint_id}", status_code=204)
def delete_loop_hint(loop_id: str, hint_id: str) -> None:
    ok = get_runner().remove_hint(loop_id, hint_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"hint {hint_id} 不存在")


# ---- Generator 本 loop 记忆（系统托管，按 loop 隔离；前端查看/编辑/清空）----


@router.get("/loops/{loop_id}/memory")
def get_loop_memory(loop_id: str) -> dict:
    """读 generator 本 loop 记忆原文（markdown，含头部）。loop 不存在 → 404。"""
    if build_loop_detail(loop_id) is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    from ...memory.loop_memory import generator_memory_path, read_memory_raw

    run_dir = get_settings().run_dir(loop_id)
    content = read_memory_raw(run_dir)
    return {
        "loop_id": loop_id,
        "content": content,
        "exists": generator_memory_path(run_dir).exists(),
    }


@router.put("/loops/{loop_id}/memory")
def put_loop_memory(loop_id: str, req: MemoryWriteRequest) -> dict:
    """编辑（覆盖写）generator 本 loop 记忆。下一轮 generator 注入即生效。"""
    if build_loop_detail(loop_id) is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    from ...memory.loop_memory import write_memory_raw

    write_memory_raw(get_settings().run_dir(loop_id), req.content)
    return {"loop_id": loop_id, "saved": True}


@router.delete("/loops/{loop_id}/memory", status_code=204)
def delete_loop_memory(loop_id: str) -> None:
    """清空 generator 本 loop 记忆（重建空文件，保留头部）。"""
    if build_loop_detail(loop_id) is None:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")
    from ...memory.loop_memory import reset_memory

    reset_memory(get_settings().run_dir(loop_id))


# ---- 删除（loop / 单轮 attempt）----


@router.delete("/loops/{loop_id}", status_code=204)
def delete_loop(loop_id: str) -> None:
    """删除整个 loop（run 目录 + loop 内经验 conclusions.json 等），不动跨 loop 蒸馏 skill 包。"""
    try:
        ok = get_runner().delete_loop(loop_id)
    except LoopBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"loop {loop_id} 不存在")


@router.delete("/loops/{loop_id}/attempts/{attempt_id}", status_code=204)
def delete_attempt(loop_id: str, attempt_id: str) -> None:
    """删除 loop 内一轮：trajectory 移除该行 + 删 out/<id>/ + 更新 index.json + 移除人工排序。"""
    try:
        ok = get_runner().delete_attempt(loop_id, attempt_id)
    except LoopBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"loop {loop_id} 或轮次 {attempt_id} 不存在"
        )
