"""测试 trajectory.jsonl 读写 + RunStore 目录管理。

写进临时目录，绝不触碰真实 data/runs。
"""

from __future__ import annotations

import json

from img_iter_agent.config import Settings
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.data.trajectory import (
    TrajectoryReader,
    TrajectoryWriter,
    trajectory_path,
    write_jsonl_atomically,
)
from img_iter_agent.memory.schema import (
    AttemptRecord,
    CriticVerdict,
    DimensionScore,
)


def _record(attempt_id: str = "a001", round_: int = 1) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        run_id="run-test",
        round=round_,
        sample_id="s001",
        bench_id="furniture_product_whitebg",
        model="seedream-5.0-pro",
        test_variable="prompt",
        baseline_ref=None,
        gen_mode="image_edit",
        prompt="生成三视图白底...",
        reference_image_refs=["out/a001/target.jpg"],
        size="2K",
        output_image_refs=["out/a001/front.jpg", "out/a001/side.jpg", "out/a001/perspective.jpg"],
        verdict=CriticVerdict(
            sample_id="s001",
            dimensions=[
                DimensionScore(dim="consistency", scoring_type="binary", value=0.75),
                DimensionScore(dim="material_texture", scoring_type="continuous", value=0.7),
            ],
            weights_used={"consistency": 0.25, "material_texture": 0.18},
            restoration=0.3135,
        ),
    )


def test_trajectory_roundtrip(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    w = TrajectoryWriter(path)
    w.append(_record("a001", 1))
    w.append(_record("a002", 2))

    recs = TrajectoryReader(path).read_all()
    assert len(recs) == 2
    assert recs[0].attempt_id == "a001"
    assert recs[1].attempt_id == "a002"
    # verdict 嵌套结构完整保留
    assert recs[0].verdict is not None
    assert recs[0].verdict.features == {"consistency": 0.75, "material_texture": 0.7}
    assert recs[0].output_image_refs[0] == "out/a001/front.jpg"


def test_trajectory_each_record_one_line(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    TrajectoryWriter(path).append(_record())
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1  # 单行 JSON，无内部换行


def test_trajectory_reader_skips_bad_lines(tmp_path, capsys):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        json.dumps({"attempt_id": "a001", "run_id": "r", "round": 1,
                    "sample_id": "s001", "bench_id": "b", "model": "m"}) + "\n"
        + "{not valid json\n"
        + "\n"
        + json.dumps({"attempt_id": "a002", "run_id": "r", "round": 2,
                      "sample_id": "s001", "bench_id": "b", "model": "m"}) + "\n",
        encoding="utf-8",
    )
    recs = TrajectoryReader(path).read_all()
    assert [r.attempt_id for r in recs] == ["a001", "a002"]  # 坏行被跳过，不致命


def test_trajectory_reader_missing_file(tmp_path):
    assert TrajectoryReader(tmp_path / "nope.jsonl").read_all() == []


def test_write_jsonl_atomically(tmp_path):
    p = tmp_path / "out" / "data.jsonl"
    write_jsonl_atomically(p, [{"a": 1}, {"b": 2}])
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}


def test_trajectory_path_helper(tmp_path):
    assert trajectory_path(tmp_path) == tmp_path / "trajectory.jsonl"


# ---------------- RunStore ----------------


def test_runstore_create_and_layout(tmp_path):
    s = Settings(data_root=tmp_path)
    store = RunStore.create("run-1", "furniture_product_whitebg", "seedream-5.0-pro",
                            note="smoke", settings=s)
    # 固定子目录都建好
    assert (store.run_dir / "lessons").is_dir()
    assert (store.run_dir / "out").is_dir()
    assert (store.run_dir / "human_scores").is_dir()
    # meta.json 写入
    meta = json.loads((store.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "run-1"
    assert meta["model"] == "seedream-5.0-pro"
    assert meta["note"] == "smoke"
    # trajectory 标准路径
    assert store.trajectory_path.name == "trajectory.jsonl"


def test_runstore_open_reloads_meta(tmp_path):
    s = Settings(data_root=tmp_path)
    RunStore.create("run-2", "furniture_product_whitebg", "gpt-image-2", settings=s)
    reopened = RunStore.open("run-2", settings=s)
    assert reopened.meta is not None
    assert reopened.meta.model == "gpt-image-2"


def test_runstore_out_dir_creates_attempt_subdir(tmp_path):
    s = Settings(data_root=tmp_path)
    store = RunStore.create("run-3", "furniture_product_whitebg", "m", settings=s)
    d = store.out_dir("a001")
    assert d.is_dir()
    assert d.name == "a001"


def test_runstore_index_append(tmp_path):
    s = Settings(data_root=tmp_path)
    store = RunStore.create("run-4", "furniture_product_whitebg", "m", settings=s)
    store.append_index_entry({"attempt_id": "a001", "restoration": 0.82})
    store.append_index_entry({"attempt_id": "a002", "restoration": 0.91})
    idx = json.loads(store.index_path.read_text(encoding="utf-8"))
    assert len(idx["attempts"]) == 2
    assert idx["attempts"][0]["attempt_id"] == "a001"


def test_runstore_finish_sets_finished_at(tmp_path):
    s = Settings(data_root=tmp_path)
    store = RunStore.create("run-5", "furniture_product_whitebg", "m", settings=s)
    store.finish(note="done")
    meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
    assert meta["finished_at"] is not None
    assert meta["note"] == "done"
