"""排序路由（屏④）：提交人工排序 → 触发自动权重校准 + 查校准状态。

人工只排序（不打分）：ranks 列表，trace_id=attempt_id，rank 越大越好。
提交即触发：写入 human_scores → 立即异步 fit_weights → sample 级落盘。
"""

from __future__ import annotations

from fastapi import APIRouter

from ...config import get_settings
from ..models import CalibrationStatusOut, RankSubmission
from ..services.calibrator import get_calibrator
from ..services.data_access import _human_scores_path, load_human_ranks

router = APIRouter()


@router.get("/scoring/{bench_id}/{sample_id}/ranks")
def get_ranks(bench_id: str, sample_id: str) -> dict:
    """读该 sample 已提交的人工排序（前端回显用）。"""
    settings = get_settings()
    ranks = load_human_ranks(settings, sample_id)
    return {
        "bench_id": bench_id,
        "sample_id": sample_id,
        "ranks": [{"trace_id": tid, "rank": r} for tid, r in ranks.items()],
    }


@router.post("/scoring/{bench_id}/{sample_id}/ranks")
def submit_ranks(bench_id: str, sample_id: str, sub: RankSubmission) -> dict:
    """提交人工排序 → 写入 human_scores → 立即触发自动校准。

    ranks: [{trace_id, rank}]，rank 越大越好。trace_id = attempt_id。
    """
    import json
    import time

    settings = get_settings()
    p = _human_scores_path(settings, sample_id)
    payload = {
        "bench_id": bench_id,
        "sample_id": sample_id,
        "ranks": [{"trace_id": r.trace_id, "rank": r.rank} for r in sub.ranks],
        "note": sub.note,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 提交即触发校准（异步）
    calibrator = get_calibrator()
    calibrator.trigger(bench_id, sample_id)

    return {"ok": True, "submitted": len(sub.ranks), "calibration_triggered": True}


@router.get("/scoring/{bench_id}/{sample_id}/calibration")
def get_calibration_status(bench_id: str, sample_id: str) -> dict:
    """查该 sample 的校准状态（前端轮询）。"""
    calibrator = get_calibrator()
    st = calibrator.status(bench_id, sample_id)
    out = CalibrationStatusOut(
        bench_id=bench_id,
        sample_id=sample_id,
        state=st.state,
        message=st.message,
        weights=st.result.weights if st.result else None,
        prior_weights=st.result.prior_weights if st.result else None,
        pairwise_accuracy=st.result.pairwise_accuracy if st.result else None,
        n_traces=st.result.n_traces if st.result else None,
        n_pairs=st.result.n_pairs if st.result else None,
        loss=st.result.loss if st.result else None,
        converged=st.result.converged if st.result else None,
        updated_at=st.updated_at,
    )
    return out.model_dump()


@router.post("/scoring/{bench_id}/{sample_id}/calibrate")
def recalibrate(bench_id: str, sample_id: str) -> dict:
    """手动重新触发校准（攒了更多排序后重跑）。"""
    get_calibrator().trigger(bench_id, sample_id)
    return {"ok": True, "calibration_triggered": True}
