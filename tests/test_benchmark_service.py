"""benchmark 管理写操作 + loop/attempt 删除 的测试。

覆盖：
  - create_benchmark：写 manifest / rubric / content_spec 脚手架 / target.jpg，且能被 load_benchmark 读回
  - create_benchmark 守卫：非法名 / 已存在 → 报错
  - delete_sample：删 sample 目录 + 其所有 loop + human_hints；manifest samples 减少；
    **不动** data/experience/<bench>/（跨 loop 蒸馏 skill 保留）
  - LoopRunner.delete_loop：删整个 run 目录；在跑抛 LoopBusyError
  - LoopRunner.delete_attempt：trajectory 移除该行 + 删 out/<id>/ + index.json 少一条
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.data.trajectory import TrajectoryWriter
from img_iter_agent.memory.schema import AttemptRecord
from img_iter_agent.web.models import DimensionIn, SampleIn
from img_iter_agent.web.services.benchmark_service import (
    BenchmarkNotFound,
    create_benchmark,
    delete_sample,
    get_benchmark_detail,
    list_benchmarks,
)
from img_iter_agent.web.services.loop_runner import LoopBusyError, LoopHandle, LoopRunner

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path, monkeypatch):
    """把全局 get_settings() 指向临时 data_root（所有 service 的 get_settings 都读这个全局）。"""
    import img_iter_agent.config as cfg

    s = Settings(data_root=tmp_path, dmxapi_key="")
    monkeypatch.setattr(cfg, "_settings", s)
    return s


def _furniture_dims() -> list[DimensionIn]:
    return [
        DimensionIn(dim="consistency", desc="三视图一致性", weight_init=0.25,
                    ref_needed=True, scoring_type="binary",
                    check_items=["C1 三视图同产品", "C2 颜色一致"]),
        DimensionIn(dim="material_texture", desc="材质还原", weight_init=0.2,
                    ref_needed=True, scoring_type="continuous", rubric_ref="rubric.md#材质"),
        DimensionIn(dim="artifact_defect", desc="无瑕疵", weight_init=0.1,
                    ref_needed=False, scoring_type="binary",
                    check_items=["A1 直线不弯", "A2 不悬浮"]),
    ]


def _make_bench(settings: Settings, bench_id: str = "test_whitebg",
                samples: list[SampleIn] | None = None,
                targets: dict[str, bytes] | None = None) -> str:
    samples = samples or [
        SampleIn(sample_id="s001", product="椅子", category="座椅"),
        SampleIn(sample_id="s002", product="桌子", category="桌几"),
    ]
    return create_benchmark(
        bench_id=bench_id, scene="白底三视图", description="测试用",
        scoring_method="hybrid_with_rank_calibration",
        task_type="three_view_whitebg_single_image", views="front,side,perspective",
        dimensions=_furniture_dims(), samples=samples, target_files=targets or {},
        settings=settings,
    )


def _make_loop(data_root: Path, loop_id: str, sample_id: str, bench_id: str,
               n_rounds: int, *, settings: Settings) -> str:
    """在 data_root/runs/<loop_id> 下造一个带 n 轮 trace 的 loop（含 out/<id>/ + index.json）。"""
    store = RunStore.create(loop_id, bench_id, "test-model", settings=settings, note="synthetic")
    tw = TrajectoryWriter(store.trajectory_path)
    for r in range(1, n_rounds + 1):
        aid = f"a{r:03d}_{loop_id[:6]}"
        (store.run_dir / "out" / aid).mkdir(parents=True, exist_ok=True)
        (store.run_dir / "out" / aid / "three_view.png").write_bytes(b"\x89PNG fake")
        rec = AttemptRecord(
            attempt_id=aid, run_id=loop_id, round=r, sample_id=sample_id,
            bench_id=bench_id, model="test-model",
            test_variable="prompt" if r > 1 else None,
            prompt=f"prompt round {r}", size="2K",
            output_image_refs=[f"out/{aid}/three_view.png"],
            verdict=None,
        )
        tw.append(rec)
    # index.json 补齐 attempts 摘要（delete_attempt 依赖它）
    idx_path = store.run_dir / "index.json"
    idx_path.write_text(json.dumps({"attempts": [
        {"attempt_id": f"a{r:03d}_{loop_id[:6]}", "round": r} for r in range(1, n_rounds + 1)
    ]}, ensure_ascii=False), encoding="utf-8")
    return loop_id


# ---------------------------------------------------------------------------
# create_benchmark
# ---------------------------------------------------------------------------

def test_create_benchmark_writes_manifest_and_scaffold(settings: Settings):
    targets = {"s001": b"\x89PNG target1", "s002": b"\x89PNG target2"}
    bid = _make_bench(settings, targets=targets)

    assert bid == "test_whitebg"
    bench_dir = settings.benchmark_dir(bid)
    assert (bench_dir / "manifest.json").exists()
    assert (bench_dir / "rubric.md").exists()
    # target 图落盘
    assert (bench_dir / "samples" / "s001" / "target.jpg").read_bytes() == b"\x89PNG target1"
    assert (bench_dir / "samples" / "s002" / "target.jpg").exists()
    # content_spec 脚手架：连续维度带 points（dict keyed by dim），二分维度不写 check（继承 manifest）
    cs = json.loads((bench_dir / "samples" / "s001" / "content_spec.json").read_text())
    assert cs["sample_id"] == "s001"
    cl = cs.get("checklist", {})
    cont = [v for v in cl.values() if isinstance(v, dict) and v.get("_scoring") == "continuous"]
    assert cont, "连续维度应有 points 脚手架"

    # 能被 load_benchmark 读回，维度/题目齐全
    lb = load_benchmark(bench_dir)
    assert lb.bench.bench_id == bid
    assert {d.dim for d in lb.bench.score_dimensions} == {"consistency", "material_texture", "artifact_defect"}
    assert {s.sample_id for s in lb.bench.samples} == {"s001", "s002"}
    # comparative_dims = 需参考图的维度
    assert set(lb.bench.comparative_dims) == {"consistency", "material_texture"}


def test_create_benchmark_rejects_invalid_name_and_existing(settings: Settings):
    # 非法名（含空格）
    with pytest.raises(ValueError):
        _make_bench(settings, bench_id="bad name!")
    # 已存在
    _make_bench(settings, bench_id="exists_ok")
    with pytest.raises(FileExistsError):
        _make_bench(settings, bench_id="exists_ok")


def test_list_and_detail(settings: Settings):
    _make_bench(settings)
    benches = list_benchmarks(settings)
    assert any(b["bench_id"] == "test_whitebg" for b in benches)

    detail = get_benchmark_detail("test_whitebg", settings)
    assert detail["bench_id"] == "test_whitebg"
    assert len(detail["dimensions"]) == 3
    assert {s["sample_id"] for s in detail["samples"]} == {"s001", "s002"}
    # 消费者面板
    assert any("Generator" in c["name"] or "pipeline" in c["name"].lower()
               for c in detail["consumers"])


# ---------------------------------------------------------------------------
# delete_sample
# ---------------------------------------------------------------------------

def test_delete_sample_removes_loops_and_manifest_entry(settings: Settings):
    _make_bench(settings)
    # 给 s001 造一个 loop，给 s002 造一个 loop
    _make_loop(settings.data_root, "test_whitebg-s001", "s001", "test_whitebg", n_rounds=2, settings=settings)
    _make_loop(settings.data_root, "test_whitebg-s002", "s002", "test_whitebg", n_rounds=1, settings=settings)
    # 跨 loop 蒸馏 skill 包（必须保留）
    exp_dir = settings.data_root / "experience" / "test_whitebg" / "test-whitebg"
    exp_dir.mkdir(parents=True)
    (exp_dir / "SKILL.md").write_text("# distill skill")
    (exp_dir / "general.json").write_text("{}")

    delete_sample("test_whitebg", "s001", settings)

    bench_dir = settings.benchmark_dir("test_whitebg")
    # sample 目录没了
    assert not (bench_dir / "samples" / "s001").exists()
    # s002 还在
    assert (bench_dir / "samples" / "s002").exists()
    # manifest samples 少一条，其它字段（scene/scoring）保留
    manifest = json.loads((bench_dir / "manifest.json").read_text())
    assert {s["sample_id"] for s in manifest["samples"]} == {"s002"}
    assert manifest.get("scene") == "白底三视图"
    # s001 的 loop 整个没了；s002 的 loop 保留
    assert not (settings.runs_dir / "test_whitebg-s001").exists()
    assert (settings.runs_dir / "test_whitebg-s002").exists()
    # 跨 loop 蒸馏 skill 包原封不动
    assert (exp_dir / "SKILL.md").exists()
    assert (exp_dir / "general.json").read_text() == "{}"


def test_delete_sample_running_guard(settings: Settings):
    _make_bench(settings)
    _make_loop(settings.data_root, "test_whitebg-s001", "s001", "test_whitebg", n_rounds=1, settings=settings)
    # 注册一个「运行中」的内存 handle 模拟在跑
    runner = LoopRunner()
    runner._handles["test_whitebg-s001"] = LoopHandle(loop_id="test_whitebg-s001", phase="running")
    # delete_sample 内部 lazy import get_runner()——这里用全局单例注册 running handle
    from img_iter_agent.web.services.loop_runner import get_runner
    singleton = get_runner()
    singleton._handles["test_whitebg-s001"] = LoopHandle(loop_id="test_whitebg-s001", phase="running")
    try:
        with pytest.raises(LoopBusyError):
            delete_sample("test_whitebg", "s001", settings)
    finally:
        singleton._handles.pop("test_whitebg-s001", None)
        runner._handles.pop("test_whitebg-s001", None)
    # 没删成：sample 仍在
    assert (settings.benchmark_dir("test_whitebg") / "samples" / "s001").exists()


def test_delete_sample_not_found(settings: Settings):
    _make_bench(settings)
    with pytest.raises(BenchmarkNotFound):
        delete_sample("does_not_exist", "s001", settings)
    # bench 存在但 sample 不存在 → SampleNotFound
    from img_iter_agent.web.services.benchmark_service import SampleNotFound
    with pytest.raises(SampleNotFound):
        delete_sample("test_whitebg", "nope", settings)


# ---------------------------------------------------------------------------
# LoopRunner.delete_loop / delete_attempt
# ---------------------------------------------------------------------------

def test_delete_loop_removes_run_dir(settings: Settings):
    _make_loop(settings.data_root, "test_whitebg-s001", "s001", "test_whitebg",
               n_rounds=2, settings=settings)
    run_dir = settings.run_dir("test_whitebg-s001")
    assert run_dir.exists()

    runner = LoopRunner()
    assert runner.delete_loop("test_whitebg-s001") is True
    assert not run_dir.exists()
    # 不存在的 loop → False
    assert runner.delete_loop("test_whitebg-s001") is False


def test_delete_loop_running_guard(settings: Settings):
    _make_loop(settings.data_root, "test_whitebg-s001", "s001", "test_whitebg",
               n_rounds=1, settings=settings)
    runner = LoopRunner()
    runner._handles["test_whitebg-s001"] = LoopHandle(loop_id="test_whitebg-s001", phase="running")
    with pytest.raises(LoopBusyError):
        runner.delete_loop("test_whitebg-s001")
    # 仍在
    assert settings.run_dir("test_whitebg-s001").exists()


def test_delete_attempt_removes_round(settings: Settings):
    _make_loop(settings.data_root, "test_whitebg-s001", "s001", "test_whitebg",
               n_rounds=3, settings=settings)
    run_dir = settings.run_dir("test_whitebg-s001")
    # 与 _make_loop 里 f"a{r:03d}_{loop_id[:6]}" 对齐；loop_id[:6] == "test_w"
    aid = "a002_test_w"
    runner = LoopRunner()
    assert runner.delete_attempt("test_whitebg-s001", aid) is True

    # trajectory 少一行
    lines = [l for l in run_dir.joinpath("trajectory.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert not any(f'"attempt_id": "{aid}"' in l for l in lines)
    # out/<id>/ 没了
    assert not (run_dir / "out" / aid).exists()
    # index.json attempts 少一条
    idx = json.loads((run_dir / "index.json").read_text())
    assert all(a["attempt_id"] != aid for a in idx["attempts"])
    assert len(idx["attempts"]) == 2
    # 其它轮的 out 仍在
    assert (run_dir / "out" / "a001_test_w").exists()
    # 不存在的 attempt → False
    assert runner.delete_attempt("test_whitebg-s001", "nope") is False
