"""Summarizer：Critic 驱动的经验闭环验证（替代原单轮事实记录）。

闭环核心（ARCH §1.3 演进）：
  不再只记录"本轮参数+分数"的快照，而是做跨轮因果验证——
  上轮改了什么（prev_delta_note）→ Critic 前后 verdict 对比 → 判定有效/无效 → 沉淀。

Critic 是客观裁判：其 verdict（分 + 失败项 + reason）是"改动有效性"的唯一证据。
reason 文本本身是可复用知识（"为什么有效/为什么没用"）。

产出：
  - 更新 lessons/conclusions.json（结构化经验知识库，见 memory/knowledge.py）
  - 返回 conclusions.json 的相对路径（供 AttemptRecord.lesson_ref 指向）
  - 追加 index.json 一条 entry
"""

from __future__ import annotations

from pathlib import Path

from ..llm import LlmClient
from ..memory import index, knowledge
from ..memory.schema import CriticVerdict
from .agent_config_loader import load_system_prompt
from .generator import GenOutcome

# 代码默认（agents_config/summarizer.md 缺失时回退）
_DEFAULT_SUMMARIZE_PROMPT = (
    "你是生图经验归纳员。根据 Critic 的前后评分对比，提炼本轮迭代的可复用结论："
    "改动是否有效、为什么、后续建议。只输出要点。"
)


class Summarizer:
    """经验闭环验证器。LLM 注入（可空）。"""

    def __init__(self, llm: LlmClient | None = None) -> None:
        self.llm = llm

    def summarize(
        self,
        *,
        run_dir: Path,
        round: int,
        outcome: GenOutcome,
        verdict: CriticVerdict,
        sample_id: str,
        prev_verdict: CriticVerdict | None = None,
        prev_delta_note: str | None = None,
    ) -> str:
        """做 Critic 驱动的经验闭环验证，更新 conclusions.json，返回其相对 run 目录路径。

        Args:
            outcome/verdict: 本轮产出与 Critic 判定。
            prev_verdict: 上一轮的 Critic 判定（验证上轮改动的"后"证据来源）。
            prev_delta_note: 上一轮 Generator 的改动说明（验证对象）。
                两者需同时提供才能做跨轮验证；首轮无上轮则只登记本轮。
        """
        kb = knowledge.load_conclusions(run_dir, sample_id=sample_id, loop_id=outcome.model)

        # 1) 验证上轮的 pending 结论：用本轮 verdict 作为"后"证据判定 status
        if prev_verdict is not None and prev_delta_note:
            self._verify_pending(kb, prev_verdict=prev_verdict, cur_verdict=verdict,
                                 cur_round=round, prev_delta_note=prev_delta_note)

        # 2) 登记本轮的改动为新 pending 结论（待下轮 Critic 验证）
        if outcome.delta_note:
            self._register_round_changes(kb, outcome=outcome, verdict=verdict, round=round)

        # 3) 可选 LLM 归纳：用 effective/ineffective 上下文提炼可复用经验
        if self.llm is not None and kb.conclusions:
            self._llm_refine(kb)

        lesson_ref = str(knowledge.save_conclusions(run_dir, kb).relative_to(run_dir))

        # 追加 index 条目
        entry = index.make_entry(
            attempt_id=outcome.attempt_id,
            round=round,
            model=outcome.model,
            gen_mode=outcome.gen_mode,
            test_variable=outcome.test_variable,
            baseline_ref=outcome.baseline_ref,
            size=outcome.size,
            restoration=verdict.restoration,
            output_image_refs=outcome.output_image_refs,
            lesson_ref=lesson_ref,
            delta_note=outcome.delta_note,
            prompt=outcome.prompt,
        )
        index.append_entry(run_dir, entry)
        return lesson_ref

    def _verify_pending(
        self, kb: knowledge.KnowledgeBase, *, prev_verdict: CriticVerdict,
        cur_verdict: CriticVerdict, cur_round: int, prev_delta_note: str,
    ) -> None:
        """验证上轮 pending 结论：对比 prev→cur Critic 判定，更新 status + critic_evidence。"""
        for c in kb.pending():
            status, evidence, lesson = knowledge.judge_status(prev_verdict, cur_verdict, c.dim)
            evidence.tested_round = cur_round
            c.status = status
            c.critic_evidence = evidence
            c.lesson = lesson
            c.verified_round = cur_round

    def _register_round_changes(
        self, kb: knowledge.KnowledgeBase, *, outcome: GenOutcome,
        verdict: CriticVerdict, round: int,
    ) -> None:
        """把本轮 delta_note 登记为新 pending 结论（按失败维度拆分）。"""
        # 本轮针对的失败维度（来自 Critic 判定）
        failed_dims = {
            d.dim for d in verdict.dimensions
            if d.scoring_type == "binary" and any(not it.passed for it in (d.items or []))
        }
        low_continuous = {d.dim for d in verdict.dimensions if d.scoring_type == "continuous" and d.value < 0.7}
        target_dims = failed_dims or low_continuous or {"general"}
        for dim in target_dims:
            finding = self._finding_for_dim(verdict, dim)
            knowledge.upsert_conclusion(
                kb, dim=dim, finding=finding, change=outcome.delta_note,
                tags=[outcome.test_variable or "prompt"], created_round=round,
            )

    def _finding_for_dim(self, verdict: CriticVerdict, dim: str) -> str:
        """从 verdict 提取该维度的问题描述（失败项理由 / 连续维度低分理由）。"""
        d = next((x for x in verdict.dimensions if x.dim == dim), None)
        if d is None:
            return ""
        if d.scoring_type == "binary":
            reasons = [f"{it.id}: {it.reason}" for it in (d.items or []) if not it.passed]
            return "; ".join(reasons) or f"{dim} 未通过"
        return d.raw or f"{dim} 低分({d.value:.2f})"

    def _llm_refine(self, kb: knowledge.KnowledgeBase) -> None:
        """用 LLM 对刚验证的结论做一句可复用归纳（写到 lesson 字段）。"""
        verified = [c for c in kb.conclusions if c.status in ("verified_effective", "ineffective")]
        if not verified:
            return
        facts = "\n".join(
            f"- [{c.dim}] 改动「{c.change}」→ {c.status}（{c.critic_evidence.verdict_delta if c.critic_evidence else ''}）"
            for c in verified
        )
        msgs = [
            {"role": "system", "content": load_system_prompt("summarizer", _DEFAULT_SUMMARIZE_PROMPT)},
            {"role": "user", "content": facts},
        ]
        try:
            summary = self.llm.complete(msgs).strip()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return
        if summary:
            # 追加到最近一条 verified 结论的 lesson（不覆盖 Critic 驱动的判定文本）
            for c in reversed(verified):
                if c.lesson:
                    c.lesson = c.lesson + "\n归纳: " + summary
                else:
                    c.lesson = summary
                break


__all__ = ["Summarizer"]
