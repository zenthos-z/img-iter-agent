"""经验知识库：结构化、Critic 驱动验证的沉淀结论。

替代原散落的单轮 lesson MD（ADR-004 双层记忆的「经验层」演进）。
载体：`data/runs/<loop_id>/lessons/conclusions.json`，归属 sample（一题一 loop）。

闭环语义（Critic 是客观裁判）：
  轮次 N: Generator 改 prompt（change=delta_note）→ Critic verdict_N
  轮次 N+1: 对比 verdict_{N-1} vs verdict_N → 判 status → 沉淀结论
  Generator 下轮读 conclusions：effective 保留约束 / ineffective 换思路

判定规则（update_status_on_evidence）：
  - 上轮该维度失败项在本轮全部消除，或分数上升 → verified_effective
  - 仍失败或分数下降 → ineffective（lesson 记 Critic reason，解释"为什么没用"）
  - 首次提出（无上轮对比）→ pending，等下轮 Critic 验证
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ..memory.schema import (
    ConclusionStatus,
    CriticEvidence,
    CriticVerdict,
    KnowledgeConclusion,
)


class KnowledgeBase(BaseModel):
    """一个 sample 的经验知识库（conclusions.json 的内容）。"""

    sample_id: str
    loop_id: str = ""
    updated_at: str = ""
    conclusions: list[KnowledgeConclusion] = Field(default_factory=list)

    def by_id(self, cid: str) -> KnowledgeConclusion | None:
        return next((c for c in self.conclusions if c.id == cid), None)

    def pending(self) -> list[KnowledgeConclusion]:
        """待验证的结论（供下一轮 Critic 验证）。"""
        return [c for c in self.conclusions if c.status == "pending"]

    def verified_for_generator(self) -> dict[str, list[KnowledgeConclusion]]:
        """供 Generator 读：effective（应保持的约束）/ ineffective（应换思路）。"""
        return {
            "effective": [c for c in self.conclusions if c.status == "verified_effective"],
            "ineffective": [c for c in self.conclusions if c.status == "ineffective"],
        }


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def _conclusions_path(run_dir: Path) -> Path:
    """conclusions.json 放在 lessons/ 子目录下（沿用原目录约定）。"""
    d = run_dir / "lessons"
    d.mkdir(parents=True, exist_ok=True)
    return d / "conclusions.json"


def load_conclusions(run_dir: Path, *, sample_id: str = "", loop_id: str = "") -> KnowledgeBase:
    """读 conclusions.json；不存在则返回空 KnowledgeBase。"""
    p = _conclusions_path(run_dir)
    if not p.exists():
        return KnowledgeBase(sample_id=sample_id, loop_id=loop_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return KnowledgeBase.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return KnowledgeBase(sample_id=sample_id, loop_id=loop_id)


def save_conclusions(run_dir: Path, kb: KnowledgeBase) -> Path:
    """写 conclusions.json，返回路径。"""
    kb.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    p = _conclusions_path(run_dir)
    p.write_text(kb.model_dump_json(indent=2), encoding="utf-8")
    return p


def _next_id(kb: KnowledgeBase) -> str:
    """自增 id：exp_001, exp_002, ..."""
    nums = [int(c.id.split("_")[1]) for c in kb.conclusions if c.id.startswith("exp_")]
    n = max(nums, default=0) + 1
    return f"exp_{n:03d}"


def upsert_conclusion(
    kb: KnowledgeBase,
    *,
    dim: str,
    finding: str,
    change: str,
    tags: list[str] | None = None,
    created_round: int,
    status: ConclusionStatus = "pending",
    critic_evidence: CriticEvidence | None = None,
    lesson: str | None = None,
    verified_round: int | None = None,
) -> KnowledgeConclusion:
    """新增或更新一条结论。

    去重键：dim + change（同一维度+同一改动视为同一条，更新而非重复新增）。
    """
    existing = next(
        (c for c in kb.conclusions if c.dim == dim and c.change == change), None
    )
    if existing:
        # 更新已存在的（如从 pending → verified）
        existing.status = status
        if critic_evidence is not None:
            existing.critic_evidence = critic_evidence
        if lesson is not None:
            existing.lesson = lesson
        if verified_round is not None:
            existing.verified_round = verified_round
        return existing
    c = KnowledgeConclusion(
        id=_next_id(kb),
        dim=dim,
        finding=finding,
        change=change,
        status=status,
        critic_evidence=critic_evidence,
        lesson=lesson,
        tags=tags or [],
        created_round=created_round,
        verified_round=verified_round,
    )
    kb.conclusions.append(c)
    return c


# ---------------------------------------------------------------------------
# Critic 驱动的 status 判定（闭环核心）
# ---------------------------------------------------------------------------


def _dim_snapshot(verdict: CriticVerdict, dim: str) -> dict:
    """取某维度在 verdict 里的快照：{value, failed:[id...], reason}。"""
    d = next((x for x in verdict.dimensions if x.dim == dim), None)
    if d is None:
        return {"value": 0.0, "failed": [], "reason": ""}
    failed = [it.id for it in (d.items or []) if not it.passed] if d.scoring_type == "binary" else []
    reason = d.raw if d.scoring_type == "continuous" else (
        "; ".join(it.reason for it in (d.items or []) if not it.passed) or ""
    )
    return {"value": float(d.value), "failed": failed, "reason": reason}


def judge_status(
    prev_verdict: CriticVerdict, cur_verdict: CriticVerdict, dim: str
) -> tuple[ConclusionStatus, CriticEvidence, str]:
    """对比 Critic 前后 verdict，判定该维度改动是否有效。

    返回 (status, evidence, lesson_text)。
    - 失败项全部消除（二分）或分数上升（连续）→ verified_effective
    - 否则 → ineffective，lesson 记 Critic reason 解释"为什么没用"
    """
    before = _dim_snapshot(prev_verdict, dim)
    after = _dim_snapshot(cur_verdict, dim)
    tested_round = 0  # 由调用方填

    failed_before = set(before["failed"])
    failed_after = set(after["failed"])
    value_delta = after["value"] - before["value"]

    # 判定：失败项清空 或 分数上升 → 有效
    resolved = bool(failed_before) and failed_before.isdisjoint(failed_after) and not (failed_before & failed_after)
    improved = value_delta > 0.01  # 容忍微小噪声
    if resolved or improved:
        status: ConclusionStatus = "verified_effective"
    else:
        status = "ineffective"

    delta_desc = f"分 {before['value']:.2f}→{after['value']:.2f}"
    if failed_before:
        delta_desc += f"; 失败项 {sorted(failed_before)}→{sorted(failed_after)}"
    evidence = CriticEvidence(
        tested_round=tested_round,
        before=before,
        after=after,
        verdict_delta=delta_desc,
    )

    # lesson：有效则肯定+建议保持；无效则记 Critic reason 解释原因
    if status == "verified_effective":
        lesson = f"[{dim}] 改动有效（{delta_desc}），建议保持该方向"
    else:
        why = after["reason"] or before["reason"] or "无明显改善"
        lesson = f"[{dim}] 改动无效（{delta_desc}）：{why}；需换思路"

    return status, evidence, lesson


__all__ = [
    "KnowledgeBase",
    "judge_status",
    "load_conclusions",
    "save_conclusions",
    "upsert_conclusion",
]
