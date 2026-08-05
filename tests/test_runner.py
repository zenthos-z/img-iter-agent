"""pipeline.runner 收口层的单测：make_loop_config / open_checkpointer / close_checkpointer / build_loop_context。

build_loop_context 内部构造真 agent（OpenAiCompatLlm + Router），但这里只验「构造出的
LoopContext 结构 + 标准 config + checkpointer setup」，不 invoke（避免真实 API 调用）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.pipeline.runner import (
    LoopContext,
    build_loop_context,
    close_checkpointer,
    make_loop_config,
    open_checkpointer,
)


def test_make_loop_config_has_thread_metadata_tags() -> None:
    cfg = make_loop_config("bench-s001", "bench", "s001", "seedream-pro")
    assert cfg["configurable"]["thread_id"] == "bench-s001"
    md = cfg["metadata"]
    assert md["loop_id"] == "bench-s001"
    assert md["bench_id"] == "bench"
    assert md["sample_id"] == "s001"
    assert md["model"] == "seedream-pro"
    assert cfg["tags"] == ["loop:bench-s001"]


def test_open_checkpointer_sets_up_tables(tmp_path: Path) -> None:
    saver = open_checkpointer(tmp_path)
    # setup() 后 checkpoints / writes 表应存在
    conn = saver.conn  # type: ignore[attr-defined]
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "checkpoints" in tables
    assert "writes" in tables
    close_checkpointer(saver)


def test_close_checkpointer_idempotent_and_none_safe() -> None:
    close_checkpointer(None)  # 不抛
    with tempfile.TemporaryDirectory() as d:
        saver = open_checkpointer(Path(d))
        close_checkpointer(saver)
        close_checkpointer(saver)  # 已关，幂等不抛


def _setup_bench(tmp_path: Path, bench_id: str, project_root: Path) -> Settings:
    settings = Settings(_env_file=None, data_root=tmp_path,
                        dmxapi_host="http://o.test", dmxapi_key="k",
                        model_seedream_pro="seedream-test")
    (tmp_path / "benchmarks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "benchmarks" / bench_id).symlink_to(
        project_root / "data" / "benchmarks" / bench_id)
    return settings


def test_build_loop_context_structure(tmp_path: Path, bench_id: str, project_root: Path) -> None:
    """build_loop_context 构造 LoopContext：app/cfg/checkpointer 就位，cfg 标准。不 invoke。"""
    settings = _setup_bench(tmp_path, bench_id, project_root)
    lb = load_benchmark(bench_id, settings=settings)
    store = RunStore.create("bench-s001", bench_id, model="seedream-test",
                            settings=settings, note="t")

    ctx = build_loop_context(lb, store, "s001", loop_model="seedream-test", settings=settings)

    assert isinstance(ctx, LoopContext)
    assert ctx.app is not None
    assert ctx.checkpointer is not None  # persist=True 默认 → SqliteSaver
    assert ctx.cfg["configurable"]["thread_id"] == "bench-s001"
    assert ctx.cfg["metadata"]["loop_id"] == "bench-s001"
    assert ctx.cfg["metadata"]["sample_id"] == "s001"
    assert ctx.cfg["tags"] == ["loop:bench-s001"]
    close_checkpointer(ctx.checkpointer)


def test_build_loop_context_inmemory_when_no_persist(
    tmp_path: Path, bench_id: str, project_root: Path,
) -> None:
    """persist=False 用 InMemorySaver（checkpointer=None，不落盘）。"""
    settings = _setup_bench(tmp_path, bench_id, project_root)
    lb = load_benchmark(bench_id, settings=settings)
    store = RunStore.create("bench-s001", bench_id, model="seedream-test",
                            settings=settings, note="t")

    ctx = build_loop_context(lb, store, "s001", persist=False, settings=settings)

    assert ctx.checkpointer is None
    assert not (store.run_dir / "checkpoints.sqlite").exists()
