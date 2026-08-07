"""跨 loop 通用经验：独立「经验蒸馏」deepagent 的产物 schema + 读写。

与 `memory/knowledge.py` 的 `conclusions.json`（每 run、规则驱动、按 (dim,change) 的
effective/ineffective）区分：
  - conclusions.json = 单 loop 内、Critic 前后对比**机器验证**的结论（in-loop Summarizer 写）。
  - general.json     = 跨 loop、LLM **综合**的通用经验（独立蒸馏器写），是 conclusions 的上层归纳。

载体：`<data_root>/experience/<bench_id>/general.json`，按 bench 分库（不同 bench 维度不同）。
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field


class DistilledLesson(BaseModel):
    """一条跨 loop 蒸馏出的通用经验。"""

    dim: str = Field(description="关联的评分维度（跨维度共性可填 'general'）")
    insight: str = Field(description="一句话通用经验/规律")
    dos: list[str] = Field(default_factory=list, description="应该这样做")
    donts: list[str] = Field(default_factory=list, description="不要这样做（已验证无效）")
    evidence: list[str] = Field(
        default_factory=list,
        description="支撑来源，如 ['<run_id>/round3', ...']",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度 0-1")


class DistilledExperience(BaseModel):
    """经验蒸馏 deepagent 的结构化输出（response_format）。"""

    summary: str = Field(default="", description="跨 loop 的总体总结")
    lessons: list[DistilledLesson] = Field(default_factory=list, description="蒸馏出的通用经验")


class GeneralExperience(BaseModel):
    """落盘的跨 loop 通用经验库（general.json 内容）。"""

    bench_id: str
    updated_at: str = ""
    source_runs: list[str] = Field(default_factory=list, description="参与蒸馏的 run_id 列表")
    summary: str = ""
    lessons: list[DistilledLesson] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def general_experience_path(data_root: Path, bench_id: str) -> Path:
    """general.json 标准路径：<data_root>/experience/<bench_id>/general.json。"""
    d = Path(data_root) / "experience" / bench_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "general.json"


def load_general_experience(data_root: Path, bench_id: str) -> GeneralExperience:
    """读 general.json；不存在则返回空（仅带 bench_id）。"""
    import json

    p = general_experience_path(data_root, bench_id)
    if not p.exists():
        return GeneralExperience(bench_id=bench_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return GeneralExperience.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return GeneralExperience(bench_id=bench_id)


def save_general_experience(data_root: Path, bench_id: str, exp: GeneralExperience) -> Path:
    """写 general.json，返回路径。"""
    import json

    exp.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    p = general_experience_path(data_root, bench_id)
    p.write_text(json.dumps(json.loads(exp.model_dump_json()), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


__all__ = [
    "DistilledExperience",
    "DistilledLesson",
    "GeneralExperience",
    "general_experience_path",
    "load_general_experience",
    "save_general_experience",
]
