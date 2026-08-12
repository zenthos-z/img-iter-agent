"""critic 兼任 in-loop 经验总结（note_experience 工具 + Summarizer agent_lessons 分支）测试。

覆盖新逻辑的核心切面，端到端（critic agent 真调 note_experience）依赖真实 LLM 循环，
留给集成验证；这里聚焦可离线稳定测试的单元：
  - note_experience 工具回写 sink
  - make_critic_tools 装配 note_experience（+ run_dir 时挂 query_experience）
  - Summarizer.summarize(agent_lessons=...) 用 agent 第一手判断替代 judge_status 规则 + _llm_refine：
    ineffective / effective / escalated 三类 judgment → status + lesson，critic_evidence 客观，
    _register_round_changes 把 agent lesson 填进新 pending。
"""

from __future__ import annotations

import pytest

from img_iter_agent.agents.generator import GenOutcome
from img_iter_agent.agents.summarizer import Summarizer
from img_iter_agent.agents.tools.critic_tools import (
    make_critic_tools,
    make_note_experience_tool,
)
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.memory.knowledge import load_conclusions
from img_iter_agent.memory.schema import CriticItemJudgment, CriticVerdict, DimensionScore

_BENCH_ID = "furniture_product_whitebg"


# ---- note_experience 工具回写 sink ----


def test_note_experience_writes_sink():
    sink = {}
    tool = make_note_experience_tool(sink=sink)
    tool.invoke({"notes": [
        {"dim": "consistency", "judgment": "ineffective", "lesson": "换光源方向"},
    ]})
    assert sink["lessons"] == [
        {"dim": "consistency", "judgment": "ineffective", "lesson": "换光源方向"},
    ]


def test_note_experience_empty_notes():
    sink = {}
    tool = make_note_experience_tool(sink=sink)
    tool.invoke({"notes": []})
    assert sink["lessons"] == []


# ---- make_critic_tools 装配 ----


@pytest.fixture(scope="module")
def bench_spec():
    lb = load_benchmark(_BENCH_ID)
    return lb.bench, lb.sample("s001").spec


def test_make_critic_tools_includes_note_experience(bench_spec):
    bench, spec = bench_spec
    tools = make_critic_tools(bench=bench, spec=spec, sink={})
    names = {t.name for t in tools}
    assert "query_rubric" in names
    assert "note_experience" in names
    assert "query_experience" not in names  # 未提供 run_dir → 不挂


def test_make_critic_tools_adds_query_experience_with_run_dir(bench_spec, tmp_path):
    bench, spec = bench_spec
    tools = make_critic_tools(bench=bench, spec=spec, sink={}, run_dir=tmp_path)
    assert "query_experience" in {t.name for t in tools}


# ---- Summarizer.summarize(agent_lessons=...) ----


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


def test_agent_lessons_ineffective(tmp_path):
    """agent 判 ineffective → status=ineffective，lesson 来自 agent（非规则套话），evidence 客观。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    summ = Summarizer(chat_model=None)  # 跳过 _llm_refine/_discover_standards，专注 agent 分支
    # round2 登记 pending（artifact_defect 失败 A4）
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.5, {"A4"}), sample_id="s1")
    # round3 仍失败，agent 写第一手判断
    agent_lesson = "阴影方向偏离 target 接地，换 bottom-up 光源或加 reference_image"
    summ.summarize(
        run_dir=run_dir, round=3, outcome=_outcome(3, "再调阴影"),
        verdict=_verdict(0.5, {"A4"}), sample_id="s1",
        prev_verdict=_verdict(0.5, {"A4"}), prev_delta_note="加阴影",
        agent_lessons=[{"dim": "artifact_defect", "judgment": "ineffective", "lesson": agent_lesson}],
    )
    kb = load_conclusions(run_dir, sample_id="s1")
    ineffective = [c for c in kb.conclusions if c.status == "ineffective"]
    assert ineffective, "agent 判 ineffective → 应有 ineffective 结论"
    c = ineffective[0]
    assert c.lesson == agent_lesson            # lesson 来自 agent，非 judge_status 套话
    assert c.verified_round == 3
    assert c.critic_evidence is not None        # 客观证据（前后快照 + verdict_delta）
    assert c.critic_evidence.tested_round == 3


def test_agent_lessons_escalated_marks_conclusion(tmp_path):
    """judgment=escalated → status=ineffective + escalated=True。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    summ = Summarizer(chat_model=None)
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.5, {"A4"}), sample_id="s1")
    summ.summarize(
        run_dir=run_dir, round=3, outcome=_outcome(3, "再调阴影"),
        verdict=_verdict(0.5, {"A4"}), sample_id="s1",
        prev_verdict=_verdict(0.5, {"A4"}), prev_delta_note="加阴影",
        agent_lessons=[{"dim": "artifact_defect", "judgment": "escalated",
                        "lesson": "连续失败，换 test_variable=reference_images"}],
    )
    kb = load_conclusions(run_dir, sample_id="s1")
    c = next(c for c in kb.conclusions if c.dim == "artifact_defect" and c.status == "ineffective")
    assert c.escalated is True


def test_agent_lessons_effective(tmp_path):
    """judgment=effective → status=verified_effective，lesson 来自 agent。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    summ = Summarizer(chat_model=None)
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.5, {"A4"}), sample_id="s1")
    agent_lesson = "阴影接地正确，保持光源描述方向"
    # round3 A4 通过（value=1.0, failed=∅）
    summ.summarize(
        run_dir=run_dir, round=3, outcome=_outcome(3, "保持"),
        verdict=_verdict(1.0, set()), sample_id="s1",
        prev_verdict=_verdict(0.5, {"A4"}), prev_delta_note="加阴影",
        agent_lessons=[{"dim": "artifact_defect", "judgment": "effective", "lesson": agent_lesson}],
    )
    kb = load_conclusions(run_dir, sample_id="s1")
    effective = [c for c in kb.conclusions if c.status == "verified_effective"]
    assert effective
    assert effective[0].lesson == agent_lesson


def test_agent_lessons_register_fills_pending_lesson(tmp_path):
    """agent 模式下 _register_round_changes 把 agent 对新失败 dim 的 lesson 填进 pending。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    summ = Summarizer(chat_model=None)
    agent_lesson = "首轮发现的接地阴影问题"
    # round2 首轮：artifact_defect 失败，agent 写 lesson（无 prev → 不验证，只登记）
    summ.summarize(
        run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
        verdict=_verdict(0.5, {"A4"}), sample_id="s1",
        agent_lessons=[{"dim": "artifact_defect", "judgment": "ineffective", "lesson": agent_lesson}],
    )
    kb = load_conclusions(run_dir, sample_id="s1")
    pending = [c for c in kb.conclusions if c.status == "pending"]
    assert pending, "首轮登记 pending"
    assert any(c.lesson == agent_lesson for c in pending), "agent lesson 应填进新 pending"


def test_no_agent_lessons_falls_back_to_rules(tmp_path):
    """agent_lessons=None → 走旧逻辑（向后兼容）：judge_status 规则判 status。"""
    run_dir = tmp_path / "loop"
    (run_dir / "lessons").mkdir(parents=True)
    summ = Summarizer(chat_model=None)
    summ.summarize(run_dir=run_dir, round=2, outcome=_outcome(2, "加阴影"),
                   verdict=_verdict(0.5, {"A4"}), sample_id="s1")
    # round3 仍失败，不传 agent_lessons → 走 _verify_pending 规则（无 chat_model → 跳过 _llm_refine）
    summ.summarize(
        run_dir=run_dir, round=3, outcome=_outcome(3, "再调阴影"),
        verdict=_verdict(0.5, {"A4"}), sample_id="s1",
        prev_verdict=_verdict(0.5, {"A4"}), prev_delta_note="加阴影",
    )
    kb = load_conclusions(run_dir, sample_id="s1")
    ineffective = [c for c in kb.conclusions if c.status == "ineffective"]
    assert ineffective
    # 规则套话（judge_status）含"无效"，且无 agent lesson
    assert ineffective[0].lesson is not None
    assert "无效" in ineffective[0].lesson
