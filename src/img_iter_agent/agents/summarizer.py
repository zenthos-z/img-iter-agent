"""Summarizer：把本轮 Critic 结果归纳成经验 MD + 更新 index.json。

跨轮归纳（ARCH §1.3）：把零散的 Critic 评分理由抽象成可复用的经验。
默认走**确定性归纳**（直接从 verdict 结构化出经验文本，无 LLM），便于离线测试；
注入 LlmClient 时用它做自然语言总结。

产出：
  - 经验 MD（lessons/lesson_<round>_<id>.md），返回相对 run 目录路径
  - index.json 追加一条 entry（参数 + 链接 + 还原度）
"""

from __future__ import annotations

from pathlib import Path

from ..llm import LlmClient
from ..memory import index, lessons
from ..memory.schema import CriticVerdict
from .generator import GenOutcome


class Summarizer:
    """经验归纳器。LLM 注入（可空）。"""

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
    ) -> str:
        """写经验 MD + 追加 index 条目，返回经验 MD 相对 run 目录的路径。"""
        title = f"轮次{round} 还原度={verdict.restoration:.3f} ({outcome.model_family}族 {outcome.model})"
        body = self._build_body(outcome, verdict)
        lesson_ref = lessons.write_lesson(
            run_dir, round=round, title=title, body=body,
            short_id=outcome.attempt_id[:8],
        )

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
            lesson_ref=str(lesson_ref),
            prompt=outcome.prompt,
        )
        index.append_entry(run_dir, entry)
        return str(lesson_ref)

    def _build_body(self, outcome: GenOutcome, verdict: CriticVerdict) -> str:
        """确定性归纳：把 verdict 的维度分 + 失败项写成人可读经验。"""
        lines = ["## 生成参数", f"- 模型: {outcome.model} ({outcome.model_family}族)",
                 f"- 模式: {outcome.gen_mode}", f"- 变量: {outcome.test_variable}",
                 f"- 对照: {outcome.baseline_ref}", f"- size: {outcome.size}",
                 f"- 参考图: {outcome.reference_image_refs}",
                 f"- 产出图: {outcome.output_image_refs}", "", "## 评分"]

        for d in verdict.dimensions:
            tag = "✓" if d.value >= 0.5 else "✗"
            if d.scoring_type == "binary":
                passed = sum(1 for it in (d.items or []) if it.passed)
                total = len(d.items or [])
                lines.append(f"- {tag} **{d.dim}** = {d.value:.2f} ({passed}/{total} 通过)")
                for it in (d.items or []):
                    if not it.passed:
                        lines.append(f"  - 失败项 {it.id}: {it.reason}")
            else:
                lines.append(f"- {tag} **{d.dim}** = {d.value:.2f} (连续) {d.raw or ''}")

        lines.append(f"\n**还原度总分 = {verdict.restoration:.3f}**")

        if self.llm is not None:
            # 用 LLM 做一句可复用的经验提炼
            facts = "\n".join(lines)
            msgs = [
                {"role": "system", "content": "你是生图经验归纳员。根据本轮评分，提炼 1-3 条可复用的生图经验（针对该模型/模式），只输出要点。"},
                {"role": "user", "content": facts},
            ]
            summary = self.llm.complete(msgs).strip()
            if summary:
                lines += ["", "## 可复用经验", summary]

        return "\n".join(lines)


__all__ = ["Summarizer"]
