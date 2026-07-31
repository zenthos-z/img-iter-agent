"""排序校准测试：验证 fit_weights 能拟合出吻合人工排序的权重。

纯离线，用合成 features。核心断言：
  - 给定明确的人工排序 + 对应的 features，拟合后 w·features 的排序应高度吻合人工排序
  - 权重满足约束（Σ=1, w≥0）
  - 若某维度与人工排序正相关、某维度反相关，权重应反映这个差异
"""

from __future__ import annotations

import json

import pytest

from img_iter_agent.calibration.fit_weights import (
    RankedTrace,
    fit_weights,
    save_calibrated_weights,
    traces_from_verdicts,
)
from img_iter_agent.calibration.report import render_report_md, write_report
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.memory.schema import (
    CriticVerdict,
    DimensionScore,
)

# ---- 用真实 benchmark 拿维度名 ----

@pytest.fixture(scope="module")
def bench(bench_id):
    return load_benchmark(bench_id).bench


DIMS = ["consistency", "product_structure", "material_texture",
        "color_accuracy", "artifact_defect", "commercial_focus"]


def _make_traces(features_list, ranks, ids=None):
    """构造 RankedTrace 列表。"""
    ids = ids or [f"t{i}" for i in range(len(features_list))]
    return [
        RankedTrace(trace_id=ids[i], features=features_list[i], human_rank=ranks[i])
        for i in range(len(features_list))
    ]


# ---- 核心拟合 ----

def test_fit_recovers_consistent_ranking(bench):
    """当某维度与人工排序完美正相关时，拟合后排序应高度吻合。"""
    # 让 consistency 完美决定排序：trace0 最好(consistency=1)，trace3 最差(0.25)
    feats = [
        {d: 1.0 if d == "consistency" else 0.5 for d in DIMS},
        {d: 0.75 if d == "consistency" else 0.5 for d in DIMS},
        {d: 0.5 for d in DIMS},
        {d: 0.25 if d == "consistency" else 0.5 for d in DIMS},
    ]
    ranks = [4.0, 3.0, 2.0, 1.0]  # 与 consistency 正相关
    traces = _make_traces(feats, ranks)

    result = fit_weights(traces, bench)
    # 排序应完全吻合
    assert result.pairwise_accuracy == pytest.approx(1.0, abs=1e-6)
    # consistency 权重应被显著抬升（它才是决定排序的维度）
    assert result.weights["consistency"] > result.weights["material_texture"]


def test_fit_lowers_irrelevant_dim(bench):
    """某维度与人工排序反相关时，其权重应被压低。"""
    # material_texture 与排序反相关（越高反而越差）
    feats = [
        {d: 0.3 if d == "material_texture" else 0.8 for d in DIMS},  # 最好
        {d: 0.9 if d == "material_texture" else 0.8 for d in DIMS},  # 最差
    ]
    ranks = [2.0, 1.0]  # trace0 好
    traces = _make_traces(feats, ranks)

    result = fit_weights(traces, bench)
    # material_texture 反相关 → 权重应被压低（甚至接近 0）
    assert result.weights["material_texture"] < result.prior_weights["material_texture"]


def test_constraints_satisfied(bench):
    """权重满足 Σ=1, w≥0。"""
    feats = [
        {d: (i % 3) / 2 + 0.3 for d in DIMS}
        for i in range(5)
    ]
    ranks = [5.0, 4.0, 3.0, 2.0, 1.0]
    traces = _make_traces(feats, ranks)

    result = fit_weights(traces, bench)
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-9 for v in result.weights.values())


def test_fit_too_few_traces_returns_prior(bench):
    """trace 不足 2 条时，返回先验权重。"""
    traces = _make_traces([{d: 0.8 for d in DIMS}], [1.0])
    result = fit_weights(traces, bench)
    assert result.n_traces == 1
    assert result.weights == result.prior_weights


def test_fit_no_rank_difference_returns_prior(bench):
    """所有 trace 人工排序相同时（无 pair），返回先验。"""
    feats = [{d: 0.8 for d in DIMS}, {d: 0.6 for d in DIMS}]
    traces = _make_traces(feats, [1.0, 1.0])  # 同分
    result = fit_weights(traces, bench)
    assert result.n_pairs == 0
    assert result.weights == result.prior_weights


# ---- 保存/加载 ----

def test_save_calibrated_weights(bench, tmp_path):
    traces = _make_traces(
        [{d: 1.0 if d == "consistency" else 0.5 for d in DIMS},
         {d: 0.3 if d == "consistency" else 0.5 for d in DIMS}],
        [2.0, 1.0],
    )
    result = fit_weights(traces, bench)
    path = save_calibrated_weights(result, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "weights" in data
    assert data["pairwise_accuracy"] == result.pairwise_accuracy


def test_saved_weights_loadable_by_weights_module(bench, tmp_path):
    """校准存的 weights.json 能被 data.weights.load_weights 读回。"""
    from img_iter_agent.data.weights import load_weights

    # 用足够差异化的 features 让拟合出明确信号：consistency 与排序强正相关，
    # 其他维度有噪音波动（不与排序一致）
    traces = _make_traces(
        [{d: 0.95 if d == "consistency" else 0.5 + 0.1 * (i % 3) for d in DIMS} for i in range(1)],
        [3.0],
    )
    traces += _make_traces(
        [{d: 0.70 if d == "consistency" else 0.4 + 0.15 * (i % 2) for d in DIMS} for i in range(1)],
        [2.0],
    )
    traces += _make_traces(
        [{d: 0.35 if d == "consistency" else 0.6 - 0.1 * (i % 2) for d in DIMS} for i in range(1)],
        [1.0],
    )
    result = fit_weights(traces, bench)
    save_calibrated_weights(result, tmp_path)
    loaded = load_weights(bench, run_dir=tmp_path)
    # 拟合达成高排序吻合度
    assert result.pairwise_accuracy >= 0.8
    # load_weights 读回后权重仍是 6 维、和为 1
    assert set(loaded) == set(DIMS)
    assert sum(loaded.values()) == pytest.approx(1.0, abs=1e-6)


# ---- 报告 ----

def test_render_report_md(bench):
    traces = _make_traces(
        [{d: 1.0 if d == "consistency" else 0.5 for d in DIMS},
         {d: 0.3 if d == "consistency" else 0.5 for d in DIMS}],
        [2.0, 1.0],
    )
    result = fit_weights(traces, bench)
    md = render_report_md(result, bench_label="furniture")
    assert "排序吻合度" in md
    assert "consistency" in md
    assert "权重变化" in md


def test_write_report(bench, tmp_path):
    traces = _make_traces(
        [{d: 0.9 for d in DIMS}, {d: 0.4 for d in DIMS}], [2.0, 1.0],
    )
    result = fit_weights(traces, bench)
    path = write_report(result, tmp_path)
    assert path.exists()
    assert path.name == "weight_calibration_report.md"


# ---- 从 verdict 构造 trace ----

def test_traces_from_verdicts(bench):
    v1 = CriticVerdict(sample_id="s001", dimensions=[
        DimensionScore(dim=d, scoring_type="binary", value=0.9) for d in DIMS],
        weights_used={}, restoration=0.9)
    v2 = CriticVerdict(sample_id="s001", dimensions=[
        DimensionScore(dim=d, scoring_type="binary", value=0.4) for d in DIMS],
        weights_used={}, restoration=0.4)
    traces = traces_from_verdicts([v1, v2], [2.0, 1.0])
    assert len(traces) == 2
    assert traces[0].features["consistency"] == 0.9
    assert traces[0].human_rank == 2.0
