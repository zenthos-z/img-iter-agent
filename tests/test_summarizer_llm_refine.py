"""Summarizer LLM 富化（A）单元测试。

chat_model 注入后，ineffective 结论的 lesson 含 LLM 产出的「建议: 具体替代思路」；
chat_model.invoke 抛异常时 _llm_refine 退化（return None），lesson 保留规则套话，闭环不炸。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from img_iter_agent.agents.generator import GenOutcome
from img_iter_agent.agents.summarizer import Summarizer
from img_iter_agent.memory.knowledge import load_conclusions
from img_iter_agent.memory.schema import CriticItemJudgment, CriticVerdict, DimensionScore
from tests._fakes import FakeToolCallingChatModel


def _verdict(value: float, failed_ids: set[str]) -> CriticVerdict:
    items = [CriticItemJudgment(id=i, passed=(i not in failed_ids), reason=f"r{i}")
             for i in ["A1", "A4"]]
    return CriticVerdict(
        sample_id="s1",
        dimensions=[DimensionScore(dim="artifact_defect", scoring_type="binary",
                                   value=value, items=items)],
        weights_used={"artifact_defect": 1.0}, restoration=value,
    )


def _outcome(round_n: int, delta: str) -> GenOutcome:
    return GenOutcome(
        attempt_id=f"a00{round_n}", test_variable="prompt", baseline_ref=None,
        gen_mode="image_edit", prompt="p", delta_note=delta, size="2K",
        reference_image_refs=[], output_image_refs=[], model="m", model_family="B",
    )


def test_llm_refine_enriches_ineffective_lesson(tmp_path):
    """ineffective 结论的 lesson 被 LLM 富化（含「建议:」+ 具体替代思路）。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    fake = FakeToolCallingChatModel(responses=[AIMessage(
        content="[artifact_defect] 改从底部光源方向描述接地阴影；或用 target 作 reference_image 风格锚"
    )])
    summ = Summarizer(chat_model=fake)
    # round2 登记 pending（change="加阴影"）
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.75, {"A4"}), sample_id="s1",
                   prev_verdict=None, prev_delta_note=None)
    # round3 仍失败 A4 → ineffective → LLM 富化
    summ.summarize(run_dir=run_dir, round=3, outcome=_outcome(3, "再调阴影"),
                   verdict=_verdict(0.75, {"A4"}), sample_id="s1",
                   prev_verdict=_verdict(0.75, {"A4"}), prev_delta_note="加阴影")

    kb = load_conclusions(run_dir, sample_id="s1")
    ineffective = [c for c in kb.conclusions if c.status == "ineffective"]
    assert ineffective, "应有 ineffective 结论"
    lesson = ineffective[0].lesson or ""
    assert "建议:" in lesson
    assert "阴影" in lesson or "reference" in lesson.lower()


def test_llm_failure_degrades_gracefully(tmp_path):
    """chat_model.invoke 抛异常 → _llm_refine return None，lesson 保留规则套话，闭环不炸。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)

    class _RaisingModel:
        def invoke(self, msgs):
            raise RuntimeError("boom")

    summ = Summarizer(chat_model=_RaisingModel())  # type: ignore[arg-type]
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.75, {"A4"}), sample_id="s1",
                   prev_verdict=None, prev_delta_note=None)
    # round3 让结论变 ineffective，触发 _llm_refine（会抛异常）
    summ.summarize(run_dir=run_dir, round=3, outcome=_outcome(3, "再调阴影"),
                   verdict=_verdict(0.75, {"A4"}), sample_id="s1",
                   prev_verdict=_verdict(0.75, {"A4"}), prev_delta_note="加阴影")

    kb = load_conclusions(run_dir, sample_id="s1")
    assert kb.conclusions  # 闭环没炸，结论照常登记
    ineffective = [c for c in kb.conclusions if c.status == "ineffective"]
    assert ineffective
    lesson = ineffective[0].lesson or ""
    assert "建议:" not in lesson  # LLM 失败 → 未富化
    assert "无效" in lesson  # 保留 judge_status 规则套话
