"""测试 benchmark 加载：跑在真实家具考题（furniture_product_whitebg）上。"""

from __future__ import annotations

from pathlib import Path

import pytest

from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark, load_sample
from img_iter_agent.memory.schema import CheckItem, ContinuousRubric


def test_manifest_loads_with_hybrid_scoring(bench_id: str):
    lb = load_benchmark(bench_id)
    b = lb.bench
    assert b.bench_id == bench_id
    assert b.scoring_method == "hybrid_with_rank_calibration"
    # 六个维度，且二分/连续分布与 manifest 一致
    by = {d.dim: d.scoring_type for d in b.score_dimensions}
    assert by == {
        "consistency": "binary",
        "product_structure": "binary",
        "material_texture": "continuous",
        "color_accuracy": "continuous",
        "artifact_defect": "binary",
        "commercial_focus": "binary",
    }
    assert {s.sample_id for s in b.samples} == {"s001", "s002", "s003"}


def test_init_weights_normalized_to_one(bench_id: str):
    lb = load_benchmark(bench_id)
    w = lb.bench.init_weights()
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    # 一致性是最重维度（三视图核心）
    assert max(w, key=w.get) == "consistency"


def test_each_sample_has_target_image_and_dual_checklist(bench_id: str):
    lb = load_benchmark(bench_id)
    for sid in ("s001", "s002", "s003"):
        s = lb.sample(sid)
        # target 实物参考图真实存在
        assert s.target_path.exists(), f"{sid} target 缺失: {s.target_path}"
        # 二分维度是 CheckItem 列表
        binary = s.spec.binary_dims()
        assert set(binary) == {
            "consistency", "product_structure", "artifact_defect", "commercial_focus"
        }
        for dim in binary:
            items = s.spec.checklist[dim]
            assert items, f"{sid}.{dim} checklist 为空"
            assert all(isinstance(i, CheckItem) for i in items)
            assert all(i.id and i.check for i in items)
        # 连续维度是 ContinuousRubric
        cont = s.spec.continuous_dims()
        assert set(cont) == {"material_texture", "color_accuracy"}
        for dim in cont:
            rubric = s.spec.checklist[dim]
            assert isinstance(rubric, ContinuousRubric)
            assert rubric.points, f"{sid}.{dim} 没有 rubric points"
        # anchor_for 覆盖所有对比型维度
        assert set(s.spec.anchor_for) == {"consistency", "product_structure",
                                          "material_texture", "color_accuracy"}


def test_load_sample_convenience(bench_id: str):
    b, s = load_sample(bench_id, "s001")
    assert b.bench_id == bench_id
    assert s.spec.sample_id == "s001"


def test_missing_benchmark_raises(project_root: Path):
    s = Settings(data_root=project_root / "data")
    with pytest.raises(FileNotFoundError):
        load_benchmark("does_not_exist", settings=s)


def test_missing_sample_raises(bench_id: str):
    lb = load_benchmark(bench_id)
    with pytest.raises(KeyError):
        lb.sample("s999")
