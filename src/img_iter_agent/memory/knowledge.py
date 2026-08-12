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
    # per-dim 连续失败轮数（B 复发检测，落盘）。键=dim，值=截至最近一轮该 dim 连续失败的轮数。
    # 放 KnowledgeBase 而非每条 conclusion：复发是 per-dim 跨「不同 change」的累计——每轮 change=delta_note
    # 文本不同 → upsert 新建 conclusion；若存 conclusion 上，同 dim 多条各自 streak=1 看不出连续失败。
    fail_streaks: dict[str, int] = Field(default_factory=dict)

    def by_id(self, cid: str) -> KnowledgeConclusion | None:
        return next((c for c in self.conclusions if c.id == cid), None)

    def pending(self) -> list[KnowledgeConclusion]:
        """待验证的结论（供下一轮 Critic 验证）。"""
        return [c for c in self.conclusions if c.status == "pending"]

    def verified_for_generator(self) -> dict[str, list[KnowledgeConclusion]]:
        """供 Generator 读：effective（应保持的约束）/ ineffective（应换思路）/ escalated（已撞模型上限，勿微调）。"""
        return {
            "effective": [c for c in self.conclusions if c.status == "verified_effective"],
            "ineffective": [c for c in self.conclusions if c.status == "ineffective"],
            "escalated": [c for c in self.conclusions if c.escalated],
        }

    def escalated_dims(self) -> set[str]:
        """当前已达升级阈值的 dim 集合（连续失败 ≥ ESCALATION_THRESHOLD）。"""
        return {d for d, s in self.fail_streaks.items() if s >= ESCALATION_THRESHOLD}


# 升级阈值：同 dim 连续失败 ≥2 轮 → escalated（提示换根本思路，勿再 prompt 微调）。
# 依据：trajectory 显示崩后第 2 轮「原样复发」（如 s003 r5→r6）即应升级，让 r7 换思路。
# 阈值=1 太激进（偶发失败误伤），=3 慢一轮。
ESCALATION_THRESHOLD = 2


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


# ---------------------------------------------------------------------------
# 复发检测 + 升级（B）：per-dim 连续失败计数驱动，与 judge_status 职责分离
# ---------------------------------------------------------------------------


def _dim_failed(verdict: CriticVerdict, dim: str) -> bool:
    """该 dim 在本轮 verdict 是否算失败（binary 有未过项 OR 连续维度 <0.7）。"""
    d = next((x for x in verdict.dimensions if x.dim == dim), None)
    if d is None:
        return False
    if d.scoring_type == "binary":
        return any(not it.passed for it in (d.items or []))
    return d.value < 0.7


def update_fail_streaks(
    kb: KnowledgeBase, *, cur_verdict: CriticVerdict
) -> dict[str, str]:
    """按本轮 verdict 更新 per-dim 连续失败计数（B 核心）。

    语义：dim 本轮失败 → streak += 1；通过 → streak = 0（复位）。
    返回 {dim: "incremented" | "reset" | "escalated"}：
      "escalated" = 本轮累加后恰好跨过 ESCALATION_THRESHOLD（升级瞬间，供 summarizer 写 trace）。
    """
    changes: dict[str, str] = {}
    for d in cur_verdict.dimensions:
        dim = d.dim
        prev = kb.fail_streaks.get(dim, 0)
        if _dim_failed(cur_verdict, dim):
            new_streak = prev + 1
            kb.fail_streaks[dim] = new_streak
            changes[dim] = (
                "escalated" if prev < ESCALATION_THRESHOLD <= new_streak else "incremented"
            )
        elif prev > 0:
            kb.fail_streaks[dim] = 0
            changes[dim] = "reset"
    return changes


def apply_escalation(kb: KnowledgeBase, *, cur_round: int) -> list[str]:
    """把升级阈值命中的 dim 标记到对应 conclusion（置 escalated=True）。

    在 update_fail_streaks 之后调用。对每个 escalated dim 的最近一条 conclusion 打标记
    （status 不动——escalated 是与 status 正交的叠加标注）。返回本轮新标记升级的 dim 列表。
    """
    newly: list[str] = []
    for dim in kb.escalated_dims():
        cands = [c for c in kb.conclusions if c.dim == dim]
        if not cands:
            continue
        c = cands[-1]
        if not c.escalated:
            c.escalated = True
            newly.append(dim)
    return newly


# ---------------------------------------------------------------------------
# 精简渲染：供 Generator user message 自动注入（与 query_experience 工具的全文 _format_experience 区分）
# ---------------------------------------------------------------------------


def render_conclusions_brief(
    kb: KnowledgeBase,
    *,
    failed_dims: list[str] | None = None,
    k_per_group: int = 4,
) -> str:
    """精简渲染本题已验证经验，供 Generator 每轮 user message 自动注入。

    区别于 ``query_experience`` 工具的 ``_format_experience``（全文 + escalated 分组 + 连续失败轮数）：
    这里只给「一行一条」的精简摘要（用户要求「已提取过一次的」形式，非前端大段展示）——分
    ineffective（勿重复）/ effective（保持）两组，``failed_dims`` 命中优先，每组 cap ``k_per_group``。
    无 verified 经验 → ``""``（调用方据此跳过注入）。
    """
    groups = kb.verified_for_generator()
    ineffective = groups["ineffective"]
    effective = groups["effective"]
    if not ineffective and not effective:
        return ""
    failed = set(failed_dims or [])

    def _rank(cs: list[KnowledgeConclusion]) -> list[KnowledgeConclusion]:
        # failed_dims 命中优先；再按 created_round 近期优先（降序）
        return sorted(cs, key=lambda c: (c.dim not in failed, -((c.created_round or 0))))

    def _line(c: KnowledgeConclusion) -> str:
        change = c.change or "(无改动说明)"
        lesson = (c.lesson or "").strip()
        # lesson 形如 "[dim] 改动无效（…）：why" —— 去掉 "[dim] " 前缀避免与行首重复
        prefix = f"[{c.dim}] "
        if lesson.startswith(prefix):
            lesson = lesson[len(prefix):]
        tail = f"：{lesson}" if lesson else ""
        return f"- [{c.dim}] 「{change}」{tail}"

    lines = ["【本题已验证经验（机器验证，直接遵循）】"]
    in_top = _rank(ineffective)[:k_per_group]
    if in_top:
        lines.append("勿重复（已判无效，换思路）：")
        lines.extend(_line(c) for c in in_top)
    ef_top = _rank(effective)[:k_per_group]
    if ef_top:
        lines.append("保持（已验证有效，勿丢）：")
        lines.extend(_line(c) for c in ef_top)
    return "\n".join(lines)


__all__ = [
    "ESCALATION_THRESHOLD",
    "KnowledgeBase",
    "apply_escalation",
    "judge_status",
    "load_conclusions",
    "render_conclusions_brief",
    "save_conclusions",
    "update_fail_streaks",
    "upsert_conclusion",
]
