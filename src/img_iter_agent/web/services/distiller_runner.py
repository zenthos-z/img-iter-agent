"""跨 loop 经验蒸馏的异步执行服务（web 触发）。

镜像 ``calibrator.py`` 的范式：``threading.Thread`` + 内存 ``_states`` + ``Lock`` + 单例。
``trigger`` 起 daemon 线程跑 ``ExperienceDistiller``，``save_general_experience`` 双写
``general.json`` + ``SKILL.md``。

状态机：``idle`` / ``running`` / ``done`` / ``error`` / ``no_runs``
（该 bench 无含 trajectory 的 run → ``no_runs``，对应 calibrator 的 ``insufficient``）。
"""

from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ...agents.experience_distiller import ExperienceDistiller
from ...config import Settings, get_settings
from ...data.benchmark import load_benchmark
from ...llm.chat_model import build_chat_model
from ...memory.experience import load_general_experience, save_general_experience


@dataclass
class _DistillState:
    """每个 bench 的蒸馏状态（内存）。"""

    state: str = "idle"  # idle / running / done / error / no_runs
    message: str | None = None
    n_lessons: int | None = None
    updated_at: str | None = None
    error: str | None = None


class DistillerRunner:
    """异步经验蒸馏器。web「重新蒸馏」按钮触发，后台跑蒸馏不阻塞调用方。"""

    def __init__(self) -> None:
        self._states: dict[str, _DistillState] = {}  # key = bench_id
        self._lock = threading.Lock()

    # --- 状态查询 ---
    def status(self, bench_id: str) -> _DistillState:
        with self._lock:
            st = self._states.get(bench_id)
        if st:
            return st
        # 没蒸馏过，但盘上可能有历史 general.json → 回退 done 态
        return self._load_disk_state(bench_id)

    def _load_disk_state(self, bench_id: str) -> _DistillState:
        settings = get_settings()
        exp = load_general_experience(settings.data_root, bench_id)
        if exp.lessons:
            return _DistillState(
                state="done",
                n_lessons=len(exp.lessons),
                updated_at=exp.updated_at or None,
                message="已从磁盘加载历史经验",
            )
        return _DistillState(state="idle")

    # --- 触发 ---
    def trigger(self, bench_id: str) -> None:
        """异步触发蒸馏。不阻塞调用方。"""
        with self._lock:
            self._states[bench_id] = _DistillState(state="running", message="蒸馏中…")
        t = threading.Thread(target=self._run, args=(bench_id,), daemon=True)
        t.start()

    def _run(self, bench_id: str) -> None:
        settings = get_settings()
        try:
            run_dirs = self._collect_runs(settings, bench_id)
            if not run_dirs:
                with self._lock:
                    self._states[bench_id] = _DistillState(
                        state="no_runs",
                        message="该 benchmark 没有含 trajectory 的 run，无法蒸馏",
                        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    )
                return
            exp = self._distill(settings, bench_id, run_dirs)
            save_general_experience(settings.data_root, bench_id, exp)  # 双写 json + SKILL.md
            with self._lock:
                self._states[bench_id] = _DistillState(
                    state="done",
                    n_lessons=len(exp.lessons),
                    updated_at=exp.updated_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    message=f"蒸馏完成，{len(exp.lessons)} 条经验",
                )
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._states[bench_id] = _DistillState(state="error", error=str(e))

    # --- 核心：发现 run → 蒸馏（复刻 cli.py cmd_distill）---
    def _collect_runs(self, settings: Settings, bench_id: str) -> list[Path]:
        run_dirs = [Path(p) for p in glob.glob(str(settings.runs_dir / f"{bench_id}-*"))]
        return [rd for rd in run_dirs if (rd / "trajectory.jsonl").exists()]

    def _distill(self, settings: Settings, bench_id: str, run_dirs: list[Path]):
        lb = load_benchmark(bench_id, settings=settings)
        chat = build_chat_model(settings, role="summarizer")
        distiller = ExperienceDistiller(
            chat,
            run_dirs=run_dirs,
            lb=lb,
            data_root=settings.data_root,
            previous=load_general_experience(settings.data_root, bench_id),
        )
        return distiller.distill()


# 单例
_distiller_runner: DistillerRunner | None = None


def get_distiller_runner() -> DistillerRunner:
    global _distiller_runner
    if _distiller_runner is None:
        _distiller_runner = DistillerRunner()
    return _distiller_runner
