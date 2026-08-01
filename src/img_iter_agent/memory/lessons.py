"""经验层入口：统一走结构化知识库 conclusions.json（替代原单轮 MD）。

历史：原为「每轮一个 lesson_<round>_<id>.md」的散落碎片，无法跨轮沉淀。
现改为：所有经验沉淀进 `lessons/conclusions.json`（结构化、Critic 驱动验证，
见 knowledge.py）。本模块对外保留兼容函数名，但语义指向 conclusions.json。

经验归属 sample（一题一 loop）：每个 run 目录一份 conclusions.json。
读写/判定逻辑见 knowledge.py；本模块仅作薄封装与目录约定。
"""

from __future__ import annotations

from pathlib import Path


def conclusions_path(run_dir: Path) -> Path:
    """conclusions.json 的路径（相对 run 目录）。"""
    return run_dir / "lessons" / "conclusions.json"


def list_lessons(run_dir: Path) -> list[Path]:
    """列出经验文件。新模型下只有 conclusions.json 一个文件。

    保留原函数名供 web 层兼容；返回单元素列表（或空）。
    """
    p = conclusions_path(run_dir)
    return [p] if p.exists() else []


def read_lesson(run_dir: Path, rel_path: str) -> str:
    """读经验内容。新模型下 rel_path 指向 conclusions.json，返回其 JSON 文本。"""
    return (run_dir / rel_path).read_text(encoding="utf-8")


__all__ = ["conclusions_path", "list_lessons", "read_lesson"]
