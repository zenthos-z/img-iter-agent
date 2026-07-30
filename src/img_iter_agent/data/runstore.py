"""run 目录管理：`data/runs/<run_id>/`。

run 目录布局（ARCH §3.5）：
  runs/<run_id>/
    meta.json            # 本次 run 的固定参数（bench/model/起止/说明）
    index.json           # 记忆索引：attempt 列表（model/mode/scores/tags/链接）
    lessons/             # 经验 MD（Summarizer 产出，文件链接）
    out/                 # 生成图产物（三视图等，按 attempt 子目录）
    human_scores/        # 人工复评（异步补）
    checkpoints.sqlite   # LangGraph checkpoint（Step 4）
    calibrated_weights.json  # 排序校准产物（Step 5）
    trajectory.jsonl     # 头等公民：完整轨迹

git 忽略 runs/* 内容（只留 .gitkeep）—— run 是系统产出，不进版本库。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings


@dataclass
class RunMeta:
    """meta.json 的内容：本次 run 的固定参数。"""

    run_id: str
    bench_id: str
    model: str  # 本闭环固定模型（ADR-007：分开独立评测）
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    finished_at: str | None = None
    note: str | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "bench_id": self.bench_id,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "note": self.note,
            **self.extras,
        }


class RunStore:
    """某次 run 目录的创建与访问。"""

    def __init__(self, run_dir: Path, meta: RunMeta | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.meta = meta

    # --- 工厂方法 ---
    @classmethod
    def create(
        cls,
        run_id: str,
        bench_id: str,
        model: str,
        *,
        note: str | None = None,
        settings: Settings | None = None,
        extras: dict | None = None,
    ) -> RunStore:
        s = settings or get_settings()
        run_dir = s.run_dir(run_id)
        store = cls(run_dir)
        store._init_dirs()
        meta = RunMeta(
            run_id=run_id, bench_id=bench_id, model=model, note=note, extras=extras or {}
        )
        store.meta = meta
        store._write_meta(meta)
        return store

    @classmethod
    def open(cls, run_id: str, *, settings: Settings | None = None) -> RunStore:
        s = settings or get_settings()
        run_dir = s.run_dir(run_id)
        store = cls(run_dir)
        store._load_meta()
        return store

    # --- 目录与路径 ---
    def _init_dirs(self) -> None:
        for sub in ("lessons", "out", "human_scores"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def trajectory_path(self) -> Path:
        return self.run_dir / "trajectory.jsonl"

    @property
    def index_path(self) -> Path:
        return self.run_dir / "index.json"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    def out_dir(self, attempt_id: str) -> Path:
        d = self.run_dir / "out" / attempt_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def lessons_dir(self) -> Path:
        return self.run_dir / "lessons"

    def human_scores_dir(self) -> Path:
        return self.run_dir / "human_scores"

    # --- meta ---
    def _write_meta(self, meta: RunMeta) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_meta(self) -> None:
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.meta = RunMeta(
                run_id=data.get("run_id", self.run_dir.name),
                bench_id=data.get("bench_id", ""),
                model=data.get("model", ""),
                started_at=data.get("started_at", ""),
                finished_at=data.get("finished_at"),
                note=data.get("note"),
                extras={k: v for k, v in data.items()
                        if k not in {"run_id", "bench_id", "model", "started_at",
                                     "finished_at", "note"}},
            )

    def finish(self, note: str | None = None) -> None:
        if self.meta is None:
            self._load_meta()
        assert self.meta is not None
        self.meta.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if note:
            self.meta.note = note
        self._write_meta(self.meta)

    # --- index.json（记忆索引骨架；完整查询见 memory/index.py，Step 4）---
    def init_index(self) -> None:
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps({"attempts": []}, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def append_index_entry(self, entry: dict) -> None:
        """追加一条 attempt 摘要到 index.json（model/mode/scores/links）。"""
        self.init_index()
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        data.setdefault("attempts", []).append(entry)
        self.index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = ["RunMeta", "RunStore"]
