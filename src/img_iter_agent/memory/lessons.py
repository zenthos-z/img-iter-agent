"""经验 MD 读写：Summarizer 写经验、Generator 读经验注入。

经验是双层记忆的「人类可读层」（ADR-004）：经验正文用 MD 写（便于人读/改），
JSON 索引只记参数与文件链接（见 index.py）。这里只管 MD 文件的落地与读取。

经验 MD 命名：runs/<run_id>/lessons/lesson_<round>_<short_id>.md
"""

from __future__ import annotations

import time
from pathlib import Path


def write_lesson(run_dir: Path, *, round: int, title: str, body: str,
                 short_id: str = "") -> Path:
    """把一条经验写成 MD，返回相对 run 目录的路径。"""
    lessons_dir = run_dir / "lessons"
    lessons_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    sid = short_id or ts
    fname = f"lesson_{round:03d}_{sid}.md"
    path = lessons_dir / fname
    content = f"# {title}\n\n> 轮次 {round} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return Path("lessons") / fname


def read_lesson(run_dir: Path, rel_path: str) -> str:
    """读一条经验 MD（rel_path 相对 run 目录）。"""
    return (run_dir / rel_path).read_text(encoding="utf-8")


def list_lessons(run_dir: Path) -> list[Path]:
    """列出某 run 的所有经验 MD（按文件名排序）。"""
    lessons_dir = run_dir / "lessons"
    if not lessons_dir.exists():
        return []
    return sorted(lessons_dir.glob("lesson_*.md"))


__all__ = ["list_lessons", "read_lesson", "write_lesson"]
