"""自动权重校准服务：人工提交排序 → 触发 fit_weights → sample 级落盘。

校准范围：按 sample 跨 loop（同题所有 trace 一起拟合）。
权重落盘：data/calibration/<bench_id>/<sample_id>_weights.json（sample 级）。
回灌路径：data/weights.load_weights 会优先读 sample 级文件（见 weights.py 改动）。

提交排序即触发：ranks 写入 → 立即异步跑 calibrate。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ...calibration.fit_weights import CalibrationResult, RankedTrace, fit_weights
from ...config import Settings, get_settings
from ...data.benchmark import load_benchmark
from ...data.trajectory import TrajectoryReader
from .data_access import load_human_ranks


# sample 级权重落盘目录
def sample_weights_path(settings: Settings, bench_id: str, sample_id: str) -> Path:
    """data/calibration/<bench_id>/<sample_id>_weights.json。"""
    d = settings.data_root / "calibration" / bench_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample_id}_weights.json"


@dataclass
class _CalibState:
    """每个 sample 的校准状态（内存）。"""

    state: str = "idle"  # idle / running / done / error / insufficient
    message: str | None = None
    result: CalibrationResult | None = None
    updated_at: str | None = None
    error: str | None = None


class WeightCalibrator:
    """异步权重校准器。提交排序即触发。"""

    def __init__(self) -> None:
        self._states: dict[str, _CalibState] = {}  # key = f"{bench}:{sample}"
        self._lock = threading.Lock()

    # --- 状态查询 ---
    def status(self, bench_id: str, sample_id: str) -> _CalibState:
        key = f"{bench_id}:{sample_id}"
        with self._lock:
            st = self._states.get(key)
        if st:
            return st
        # 没跑过校准，但可能磁盘上已有历史结果（之前跑过）
        return self._load_disk_state(bench_id, sample_id)

    def _load_disk_state(self, bench_id: str, sample_id: str) -> _CalibState:
        """从磁盘的 sample 级权重文件恢复一个 done 状态。"""
        settings = get_settings()
        p = sample_weights_path(settings, bench_id, sample_id)
        if not p.exists():
            return _CalibState(state="idle")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            res = CalibrationResult(
                weights=data["weights"],
                prior_weights=data["prior_weights"],
                pairwise_accuracy=data["pairwise_accuracy"],
                margin=data["margin"],
                n_traces=data["n_traces"],
                n_pairs=data["n_pairs"],
                converged=data["converged"],
                loss=data["loss"],
            )
            return _CalibState(
                state="done",
                result=res,
                updated_at=data.get("updated_at"),
                message="已从磁盘加载历史校准结果",
            )
        except Exception:  # noqa: BLE001
            return _CalibState(state="idle")

    # --- 触发（提交排序后调用）---
    def trigger(self, bench_id: str, sample_id: str) -> None:
        """异步触发校准。不阻塞调用方。"""
        key = f"{bench_id}:{sample_id}"
        with self._lock:
            self._states[key] = _CalibState(state="running", message="校准中…")
        t = threading.Thread(target=self._run, args=(bench_id, sample_id), daemon=True)
        t.start()

    def _run(self, bench_id: str, sample_id: str) -> None:
        key = f"{bench_id}:{sample_id}"
        settings = get_settings()
        try:
            result = self._calibrate(settings, bench_id, sample_id)
            self._save(settings, bench_id, sample_id, result)
            with self._lock:
                if result.n_pairs == 0:
                    self._states[key] = _CalibState(
                        state="insufficient",
                        message="数据不足（trace <2 或排序无差异），未更新权重",
                        result=result,
                        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    )
                else:
                    self._states[key] = _CalibState(
                        state="done",
                        result=result,
                        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        message=None,
                    )
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._states[key] = _CalibState(state="error", error=str(e))

    # --- 核心：收集 trace + 人工 rank → fit_weights ---
    def _calibrate(self, settings: Settings, bench_id: str, sample_id: str) -> CalibrationResult:
        lb = load_benchmark(bench_id, settings=settings)
        bench = lb.bench

        # 跨 loop 收集该 sample 的所有 trace
        runs_dir = settings.runs_dir
        loop_dirs = sorted(
            d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
        )
        ranks = load_human_ranks(settings, sample_id)  # {attempt_id: rank}

        ranked: list[RankedTrace] = []
        for ld in loop_dirs:
            traj_path = ld / "trajectory.jsonl"
            if not traj_path.exists():
                continue
            for rec in TrajectoryReader(traj_path).iter_records():
                if rec.sample_id != sample_id or rec.bench_id != bench_id:
                    continue
                if rec.verdict is None:
                    continue
                if rec.attempt_id not in ranks:
                    # 未被人工排序的 trace 不纳入校准
                    continue
                ranked.append(
                    RankedTrace(
                        trace_id=rec.attempt_id,
                        features=rec.verdict.features,
                        human_rank=ranks[rec.attempt_id],
                    )
                )

        return fit_weights(ranked, bench)

    def _save(
        self, settings: Settings, bench_id: str, sample_id: str, result: CalibrationResult
    ) -> None:
        p = sample_weights_path(settings, bench_id, sample_id)
        payload = {
            "weights": result.weights,
            "prior_weights": result.prior_weights,
            "pairwise_accuracy": result.pairwise_accuracy,
            "margin": result.margin,
            "n_traces": result.n_traces,
            "n_pairs": result.n_pairs,
            "converged": result.converged,
            "loss": result.loss,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "bench_id": bench_id,
            "sample_id": sample_id,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 单例
_calibrator: WeightCalibrator | None = None


def get_calibrator() -> WeightCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = WeightCalibrator()
    return _calibrator
