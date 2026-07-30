"""trajectory.jsonl 的读写。

trajectory 是头等公民（ADR-009）：每轮一行、自包含、可脱离运行时独立重放，
支撑策略对比 / 权重校准 / 泛化分析。一行 = 一个 `AttemptRecord`。

写入要求**原子追加**（避免中途崩溃产生半行）。这里用「写临时行 + 换行」+ flush 的稳妥方式。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..memory.schema import AttemptRecord


class TrajectoryWriter:
    """向某次 run 的 trajectory.jsonl 追加 attempt 记录。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AttemptRecord) -> None:
        """原子追加一条记录（单行 JSON + 换行）。"""
        line = record.model_dump_json()
        # 确保行内无换行（model_dump_json 默认无，但防御一下）
        line = line.replace("\n", " ")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
            f.flush()

    def extend(self, records: list[AttemptRecord]) -> None:
        for r in records:
            self.append(r)


class TrajectoryReader:
    """读取 trajectory.jsonl，逐行反序列化成 AttemptRecord。

    容错：跳过空行；坏行不致命（记录行号后跳过），避免某行损坏导致整条轨迹不可用。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[AttemptRecord]:
        return self.iter_records()

    def iter_records(self) -> Iterator[AttemptRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield AttemptRecord.model_validate_json(raw)
                except Exception:  # noqa: BLE001 - 坏行跳过，保留可读性
                    # 不抛错，但留下 stderr 提示便于排查
                    import sys

                    print(
                        f"[trajectory] 跳过损坏的第 {lineno} 行: {self.path}",
                        file=sys.stderr,
                    )
                    continue

    def read_all(self) -> list[AttemptRecord]:
        return list(self.iter_records())


def trajectory_path(run_dir: Path) -> Path:
    """某次 run 的 trajectory.jsonl 标准路径。"""
    return Path(run_dir) / "trajectory.jsonl"


def write_jsonl_atomically(path: Path, records: list[dict]) -> None:
    """一次性写多条 dict 到 jsonl（用于校准/分析中间产物，非 trajectory 主线）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False).replace("\n", " "))
            f.write("\n")
    tmp.replace(path)


__all__ = [
    "TrajectoryReader",
    "TrajectoryWriter",
    "trajectory_path",
    "write_jsonl_atomically",
]
