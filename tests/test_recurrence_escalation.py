"""复发检测 + 升级（B）单元测试。

不依赖 LLM / graph / 网络：直接驱动 knowledge.update_fail_streaks / apply_escalation /
escalated_dims，验证 per-dim 连续失败计数、升级阈值、复位语义；以及 _format_experience
的 escalated 分组透出。
"""

from __future__ import annotations

from img_iter_agent.memory.knowledge import (
    ESCALATION_THRESHOLD,
    KnowledgeBase,
    apply_escalation,
    save_conclusions,
    update_fail_streaks,
    upsert_conclusion,
)
from img_iter_agent.memory.schema import (
    CriticItemJudgment,
    CriticVerdict,
    DimensionScore,
)


def _bin_verdict(failed_ids: set[str], dim: str = "consistency", value: float = 0.5) -> CriticVerdict:
    items = [CriticItemJudgment(id=i, passed=(i not in failed_ids), reason="r")
             for i in ["C1", "C3"]]
    return CriticVerdict(
        sample_id="s", dimensions=[DimensionScore(
            dim=dim, scoring_type="binary", value=value, items=items)],
        weights_used={dim: 1.0}, restoration=value,
    )


def test_streak_increments_then_escalates_then_resets():
    """round1 失败→streak1（不升）；round2 再失败→streak2 升级；round3 通过→归零。"""
    kb = KnowledgeBase(sample_id="s")
    ch = update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))
    assert kb.fail_streaks["consistency"] == 1
    assert ch["consistency"] == "incremented"
    assert "consistency" not in kb.escalated_dims()

    ch = update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))
    assert kb.fail_streaks["consistency"] == 2
    assert ch["consistency"] == "escalated"  # 跨阈值瞬间
    assert "consistency" in kb.escalated_dims()

    ch = update_fail_streaks(kb, cur_verdict=_bin_verdict(set()))
    assert kb.fail_streaks["consistency"] == 0
    assert ch["consistency"] == "reset"
    assert "consistency" not in kb.escalated_dims()  # 复位后不再算升级


def test_threshold_boundary():
    assert ESCALATION_THRESHOLD == 2
    kb = KnowledgeBase(sample_id="s")
    update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))
    assert kb.escalated_dims() == set()  # streak=1 不升级
    update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))
    assert kb.escalated_dims() == {"consistency"}  # streak=2 升级


def test_apply_escalation_marks_latest_conclusion():
    kb = KnowledgeBase(sample_id="s")
    upsert_conclusion(kb, dim="consistency", finding="f", change="c1", created_round=1)
    update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))
    update_fail_streaks(kb, cur_verdict=_bin_verdict({"C1"}))  # streak=2
    assert kb.conclusions[-1].escalated is False
    apply_escalation(kb, cur_round=2)
    assert kb.conclusions[-1].escalated is True


def test_continuous_low_score_counts_as_failure():
    """连续维度 <0.7 也算失败，参与 streak。"""
    kb = KnowledgeBase(sample_id="s")
    v = CriticVerdict(
        sample_id="s",
        dimensions=[DimensionScore(dim="material_texture", scoring_type="continuous",
                                   value=0.4, raw="材质失真")],
        weights_used={"material_texture": 1.0}, restoration=0.4,
    )
    update_fail_streaks(kb, cur_verdict=v)
    assert kb.fail_streaks["material_texture"] == 1


def test_format_experience_has_escalated_group(tmp_path):
    """_format_experience：escalated 结论单独分组，标注连续失败轮数。"""
    from img_iter_agent.agents.tools.generator_tools import _format_experience

    run_dir = tmp_path / "loop"
    kb = KnowledgeBase(sample_id="s")
    upsert_conclusion(kb, dim="consistency", finding="f", change="c",
                      tags=["prompt"], created_round=1)
    kb.conclusions[0].escalated = True
    kb.fail_streaks["consistency"] = 3
    save_conclusions(run_dir, kb)

    out = _format_experience(run_dir)
    assert "已升级" in out
    assert "连续失败 3 轮" in out
    assert "consistency" in out
