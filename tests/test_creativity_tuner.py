"""Unit tests for creativity_tuner: signal extraction + merge clamps (pure-Python, no LLM)."""

import pytest

from img_iter_agent.calibration import creativity_tuner as ct
from img_iter_agent.calibration.creativity_tuner import (
    CreativityRenoItem,
    CreativityRenovation,
    extract_signals,
    merge_criteria,
    merge_weights,
)
from img_iter_agent.memory.schema import (
    AttemptRecord,
    CriticVerdict,
    DimensionScore,
)


def _rec(round_n: int, ref_ids: list[str], cd: float, od=0.5, ce=0.5, ri=0.5, cd_raw: str = "") -> AttemptRecord:
    """Build an AttemptRecord with a verdict carrying creativity dims."""
    dims = [
        DimensionScore(dim="creative_departure", scoring_type="continuous", value=cd, raw=cd_raw),
        DimensionScore(dim="originality_degree", scoring_type="continuous", value=od),
        DimensionScore(dim="concept_expression", scoring_type="binary", value=ce),
        DimensionScore(dim="reference_independence", scoring_type="binary", value=ri),
    ]
    verdict = CriticVerdict(sample_id="s001", dimensions=dims, weights_used={}, restoration=cd)
    return AttemptRecord(
        attempt_id=f"a{round_n:03d}", run_id="r", round=round_n, sample_id="s001",
        bench_id="anthropic_og_style", model="gemini-3.1-flash-image",
        reference_ids=ref_ids, verdict=verdict,
    )


# ---------------- extract_signals ----------------

def test_extract_signals_copy_reward_positive_correlation():
    # more refs -> higher creativity (gaming): expect positive corr
    recs = [
        _rec(1, [], 0.2),
        _rec(2, [], 0.3),
        _rec(3, ["hand-abacus"], 0.6),
        _rec(4, ["hand-abacus", "object-laptop"], 0.8),
    ]
    sig = extract_signals(recs)
    assert sig["n_records"] == 4
    assert sig["copy_reward_corr"] is not None
    assert sig["copy_reward_corr"] > 0.0  # positive = rewarding copying
    assert sig["discriminates"] is True  # variance > 0.01


def test_extract_signals_noise_as_creative_and_over_strict():
    recs = [
        _rec(1, [], 0.8, od=0.2, ce=0.2),  # high creativity but low originality/concept = noise (false positive)
        _rec(2, [], 0.2, od=0.9, ce=0.9),  # low creativity but high originality = over-strict (false negative)
    ]
    sig = extract_signals(recs)
    assert sig["noise_as_creative_count"] == 1
    assert sig["over_strict_count"] == 1


def test_extract_signals_no_creativity_dims():
    # records whose verdict lacks creative_departure -> n_records 0
    rec = AttemptRecord(attempt_id="a001", run_id="r", round=1, sample_id="s001",
                        bench_id="anthropic_og_style", model="m", reference_ids=[],
                        verdict=None)
    sig = extract_signals([rec])
    assert sig["n_records"] == 0
    assert sig["discriminates"] is False


def test_extract_signals_low_discrimination():
    # all creativity scores nearly identical -> var < 0.01 -> discriminates False
    recs = [_rec(i, [], 0.50 + 0.001 * i) for i in range(1, 6)]
    sig = extract_signals(recs)
    assert sig["discriminates"] is False


# ---------------- merge_criteria ----------------

def _seed_criteria():
    return {
        "creative_departure": {"scoring_type": "continuous",
                               "points": ["p1 隐喻新颖", "p2 服务概念", "p3 神韵内创新"]},
        "reference_independence": {"scoring_type": "binary",
                                   "items": [{"id": "reference_independence-1", "check": "c1", "anchor": None},
                                             {"id": "reference_independence-2", "check": "c2", "anchor": None}]},
    }


def test_merge_criteria_new_point_appended_and_clamped():
    seed = _seed_criteria()
    plan = CreativityRenovation(items=[
        CreativityRenoItem(dim="creative_departure", action="new", text="p4 新要点"),
        CreativityRenoItem(dim="creative_departure", action="new", text="p5"),
        CreativityRenoItem(dim="creative_departure", action="new", text="p6"),
        CreativityRenoItem(dim="creative_departure", action="new", text="p7 超出上限应被砍"),
    ])
    out = merge_criteria(seed, plan)
    assert len(out["creative_departure"]["points"]) == ct.SUBCRITERIA_MAX  # clamped to 6
    assert "p4 新要点" in out["creative_departure"]["points"]


def test_merge_criteria_revise_and_retire():
    seed = _seed_criteria()
    plan = CreativityRenovation(items=[
        CreativityRenoItem(dim="creative_departure", action="revise",
                           existing_id="p1 隐喻新颖", text="p1 改写后的隐喻新颖度"),
        CreativityRenoItem(dim="reference_independence", action="retire",
                           existing_id="reference_independence-2"),
    ])
    out = merge_criteria(seed, plan)
    assert "p1 改写后的隐喻新颖度" in out["creative_departure"]["points"]
    assert "p1 隐喻新颖" not in out["creative_departure"]["points"]
    # retire must not drop below SUBCRITERIA_MIN (2): reference_independence had 2 -> stays 2
    assert len(out["reference_independence"]["items"]) >= ct.SUBCRITERIA_MIN


def test_merge_criteria_new_binary_item():
    seed = _seed_criteria()
    plan = CreativityRenovation(items=[
        CreativityRenoItem(dim="reference_independence", action="new", item_id="reference_independence-3",
                           text="不得复刻传入参考的手+物组合", anchor="对照 reference_ids"),
    ])
    out = merge_criteria(seed, plan)
    ids = [it["id"] for it in out["reference_independence"]["items"]]
    assert "reference_independence-3" in ids


# ---------------- merge_weights ----------------

def test_merge_weights_delta_clamped_and_within_bounds():
    cur = {"creative_departure": 0.20, "reference_independence": 0.10}
    plan = CreativityRenovation(weight_delta={
        "creative_departure": 0.5,    # over cap -> clamped to +0.05
        "reference_independence": -0.9,  # over cap -> clamped to -0.05
    })
    signals = {"discriminates": True}
    out = merge_weights(cur, plan, signals)
    assert out["creative_departure"] == 0.25  # 0.20 + 0.05
    assert out["reference_independence"] == 0.05  # 0.10 - 0.05
    assert all(ct.WEIGHT_FLOOR <= v <= ct.WEIGHT_CEIL for v in out.values())


def test_merge_weights_floor_enforced():
    cur = {"creative_departure": 0.03, "reference_independence": 0.10}
    plan = CreativityRenovation(weight_delta={"creative_departure": -0.05})
    out = merge_weights(cur, plan, {"discriminates": True})
    assert out["creative_departure"] == ct.WEIGHT_FLOOR  # 0.03-0.05 -> clamped to 0.02


def test_merge_weights_skips_creative_departure_when_not_discriminating():
    cur = {"creative_departure": 0.20, "reference_independence": 0.10}
    plan = CreativityRenovation(weight_delta={
        "creative_departure": 0.05,        # should be SKIPPED (no discrimination)
        "reference_independence": 0.05,    # allowed
    })
    out = merge_weights(cur, plan, {"discriminates": False})
    assert out["creative_departure"] == 0.20  # unchanged
    assert out["reference_independence"] == pytest.approx(0.15)  # applied


def test_merge_weights_ignores_non_creativity_dims():
    cur = {"creative_departure": 0.20, "reference_independence": 0.10}
    plan = CreativityRenovation(weight_delta={"spirit_hand_form": 0.05})  # not a creativity dim
    out = merge_weights(cur, plan, {"discriminates": True})
    assert "spirit_hand_form" not in out
