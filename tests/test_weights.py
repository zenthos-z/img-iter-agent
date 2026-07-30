"""测试权重数学：features 向量 + 加权还原度 + 校准权重覆盖。

纯函数，无网络、无 LLM。覆盖混合评分的两种维度类型。
"""

from __future__ import annotations

import json

import pytest

from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.weights import (
    apply_weights,
    compute_features,
    init_weights,
    load_weights,
    recompute_restoration,
    weighted_restoration,
)
from img_iter_agent.memory.schema import (
    CriticItemJudgment,
    CriticVerdict,
    DimensionScore,
)


def _make_verdict(values: dict[str, float], weights: dict[str, float],
                  binary_items: dict[str, list[tuple[str, bool]]] | None = None) -> CriticVerdict:
    """构造一个 verdict，二分维度的 items 可选。"""
    dims = []
    for dim, val in values.items():
        items = None
        if binary_items and dim in binary_items:
            items = [CriticItemJudgment(id=i, passed=p, reason="") for i, p in binary_items[dim]]
        st = "binary" if (binary_items and dim in binary_items) else "continuous"
        dims.append(DimensionScore(dim=dim, scoring_type=st, value=val, items=items))
    return CriticVerdict(
        sample_id="s001", dimensions=dims, weights_used=dict(weights),
        restoration=weighted_restoration(values, weights),
    )


def test_weighted_restoration_basic():
    feats = {"a": 1.0, "b": 0.0}
    w = {"a": 0.5, "b": 0.5}
    assert weighted_restoration(feats, w) == pytest.approx(0.5)


def test_weighted_restoration_empty_and_zero():
    assert weighted_restoration({}, {"a": 1.0}) == 0.0
    assert weighted_restoration({"a": 1.0}, {}) == 0.0
    assert weighted_restoration({"a": 1.0}, {"a": 0.0}) == 0.0


def test_weighted_restoration_handles_missing_dim():
    # features 缺 b：按"缺失=0 贡献"语义，b 计 0
    # 归一化后 = (0.5/1.0)*0.8 + (0.5/1.0)*0 = 0.4
    feats = {"a": 0.8}
    w = {"a": 0.5, "b": 0.5}
    assert weighted_restoration(feats, w) == pytest.approx(0.4)


def test_compute_features_extracts_values():
    v = _make_verdict({"a": 0.7, "b": 0.3}, {"a": 0.5, "b": 0.5})
    assert compute_features(v) == {"a": 0.7, "b": 0.3}


def test_apply_weights_recomputes_restoration():
    feats = {"a": 0.6, "b": 0.4}
    w0 = {"a": 0.5, "b": 0.5}
    v = _make_verdict(feats, w0)
    # 用新权重重算
    w1 = {"a": 0.9, "b": 0.1}
    v2 = apply_weights(v, w1)
    assert v2.restoration == pytest.approx(0.9 * 0.6 + 0.1 * 0.4)
    assert v2.weights_used == w1
    # 原对象不变
    assert v.restoration != v2.restoration


def test_init_weights_from_real_benchmark(bench_id: str):
    lb = load_benchmark(bench_id)
    w = init_weights(lb.bench)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["consistency"] == pytest.approx(0.25)
    assert w["material_texture"] == pytest.approx(0.18)


def test_load_weights_falls_back_to_prior(bench_id: str, tmp_path):
    lb = load_benchmark(bench_id)
    # 无 calibrated_weights.json → 用先验
    w = load_weights(lb.bench, run_dir=tmp_path)
    assert w["consistency"] == pytest.approx(0.25)


def test_load_weights_uses_calibration_when_present(bench_id: str, tmp_path):
    lb = load_benchmark(bench_id)
    prior = init_weights(lb.bench)
    # 写一份校准权重：一致性权重提高（人为压低连续维度偏差）
    calibrated = {"consistency": 0.50, "material_texture": 0.10}
    (tmp_path / "calibrated_weights.json").write_text(
        json.dumps({"weights": calibrated}), encoding="utf-8"
    )
    w = load_weights(lb.bench, run_dir=tmp_path)
    assert w["consistency"] == pytest.approx(0.50)
    assert w["material_texture"] == pytest.approx(0.10)
    # 未出现在校准文件里的维度保留先验
    assert w["artifact_defect"] == pytest.approx(prior["artifact_defect"])
    # 校准权重之和可能 != 1，但 weighted_restoration 会归一化，故总分仍 ∈[0,1]


def test_load_weights_ignores_foreign_dims_and_negatives(bench_id: str, tmp_path):
    lb = load_benchmark(bench_id)
    prior = init_weights(lb.bench)
    # 含非法维度名/负值 → 被忽略
    bad = {"consistency": 0.3, "nonexistent_dim": 0.9, "color_accuracy": -0.5}
    (tmp_path / "calibrated_weights.json").write_text(
        json.dumps({"weights": bad}), encoding="utf-8"
    )
    w = load_weights(lb.bench, run_dir=tmp_path)
    assert w["consistency"] == pytest.approx(0.3)
    assert "nonexistent_dim" not in w
    # color_accuracy 负值被忽略 → 回落先验
    assert w["color_accuracy"] == pytest.approx(prior["color_accuracy"])


def test_recompute_restoration_from_dimension_scores():
    dims = [
        DimensionScore(dim="a", scoring_type="binary", value=0.5),
        DimensionScore(dim="b", scoring_type="continuous", value=0.8),
    ]
    w = {"a": 0.4, "b": 0.6}
    assert recompute_restoration(dims, w) == pytest.approx(0.4 * 0.5 + 0.6 * 0.8)


def test_full_hybrid_scenario_on_real_bench(bench_id: str):
    """端到端：真实 6 维 benchmark + 一份混合 verdict → 还原度。"""
    lb = load_benchmark(bench_id)
    w = init_weights(lb.bench)
    # 二分维度：各项通过率；连续维度：LLM 0-1 分
    feats = {
        "consistency": 3 / 4,        # C1-C4 通过 3 项
        "product_structure": 3 / 4,
        "material_texture": 0.7,     # LLM 连续分（带偏差）
        "color_accuracy": 0.85,
        "artifact_defect": 1.0,
        "commercial_focus": 2 / 3,
    }
    binary_items = {
        "consistency": [("C1", True), ("C2", True), ("C3", False), ("C4", True)],
        "product_structure": [("S1", True), ("S2", False), ("S3", True), ("S4", True)],
        "artifact_defect": [("A1", True), ("A2", True), ("A3", True), ("A4", True)],
        "commercial_focus": [("B1", True), ("B2", True), ("B3", False)],
    }
    v = _make_verdict(feats, w, binary_items=binary_items)
    expected = sum(w[d] * f for d, f in feats.items())
    assert v.restoration == pytest.approx(expected, abs=1e-9)
    assert 0.0 <= v.restoration <= 1.0
    # features 可取回
    assert compute_features(v) == pytest.approx(feats)
    # 二分维度的逐项判定可取回
    assert len(v.item_judgments("consistency")) == 4
    assert v.item_judgments("material_texture") == []  # 连续维度无逐项
