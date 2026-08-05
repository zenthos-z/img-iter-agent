"""LoopRunner 自动连跑（auto_mode）的离线验证：不依赖真实生图/LLM API。

用 FakeRouter + FakeLlmClient 构造 mock graph，覆盖 LoopRunner._build_app，
验证：
  1. rounds=2 时，round1 完成后自动 resume(continue) 进 round2（remaining 1→0）；
  2. round2 完成后 auto_mode 触发 resume(stop)，loop 走到 END，phase=finished；
  3. rounds=1（非自动模式）首轮停在 awaiting_review（不自动续跑）。

这绕开了网络/API 超时，100% 确定性地覆盖 auto_mode 收尾逻辑。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.agents.summarizer import Summarizer
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from img_iter_agent.llm import FakeLlmClient
from img_iter_agent.pipeline.graph import build_graph
from img_iter_agent.web.services.loop_runner import get_runner


class _FakeRouter(Router):
    """假 Router：不联网，直接写占位图返回。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, req, out_dir: Path, *, config=None) -> GeneratedImage:  # type: ignore[override]
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake_{self.calls}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return GeneratedImage(
            image_path=p, model="fake-model", endpoint="fake",
            meta={"family": "B", "size": "2K"},
        )


def _critic_responses(bench):
    import json as _j
    out = []
    for dim_def in bench.score_dimensions:
        if dim_def.scoring_type == "binary":
            out.append(_j.dumps({"judgments": [
                {"id": f"{dim_def.dim[0].upper()}{i+1}", "passed": i == 0, "reason": "mock"}
                for i in range(3)
            ]}))
        else:
            out.append(_j.dumps({"score": 0.7, "reason": "mock"}))
    return out


@pytest.fixture()
def setup(tmp_path, bench_id, monkeypatch):
    """建临时 data_root + 软链 benchmark；monkeypatch 全局 settings 指向 tmp。"""
    settings = Settings(data_root=tmp_path)
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    real_bench = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / bench_id
    (tmp_path / "benchmarks" / bench_id).symlink_to(real_bench)

    lb = load_benchmark(bench_id, settings=settings)

    # 全局 get_settings 指向临时目录（loop_runner 内部会调 get_settings）
    monkeypatch.setattr(
        "img_iter_agent.web.services.loop_runner.get_settings", lambda: settings
    )
    # 每个测试用独立的 runner 实例（避免单例残留）
    import img_iter_agent.web.services.loop_runner as mod
    monkeypatch.setattr(mod, "_runner", None)

    return settings, lb, bench_id


def _patch_build_app(monkeypatch, lb, settings):
    """让 LoopRunner._build_app 用 FakeRouter + FakeLlm（不联网）。"""
    bench = lb.bench

    def _fake_build_app(self, settings, lb, store, sample_id):
        router = _FakeRouter()
        gen = Generator(router)
        # 给足多轮的 critic 响应（每轮 6 维）
        critic = Critic(FakeLlmClient(responses=_critic_responses(bench) * 8), bench=bench)
        summ = Summarizer()
        conn = sqlite3.connect(store.run_dir / "checkpoints.sqlite", check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver(conn)
        app = build_graph(
            bench=lb, run_store=store, generator=gen, critic=critic,
            summarizer=summ, sample_id=sample_id, checkpointer=checkpointer,
        )
        cfg = {"configurable": {"thread_id": store.run_dir.name}}
        return app, cfg

    import img_iter_agent.web.services.loop_runner as mod
    monkeypatch.setattr(mod.LoopRunner, "_build_app", _fake_build_app)


def _wait_phase(runner, loop_id, target, timeout=20):
    """轮询直到 phase 命中 target 集合或超时。"""
    import time
    targets = set(target)
    for _ in range(timeout * 4):
        h = runner.get(loop_id)
        if h and h.phase in targets:
            return h
        # 任务在线程池里，等它推进
        time.sleep(0.25)
    h = runner.get(loop_id)
    assert h is not None, "handle 消失"
    return h


def test_auto_mode_2_rounds_finishes(setup, monkeypatch):
    """rounds=2：round1 自动续跑 round2，跑满后 auto_mode 触发 stop → finished。"""
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    loop_id = runner.start(
        bench_id=bench_id, sample_id="s001", model="fake-model", rounds=2,
    )
    # 自动连跑：最终应到 finished（而非停在 awaiting_review）
    h = _wait_phase(runner, loop_id, {"finished", "error"})
    assert h.phase == "finished", f"期望 finished，实际 {h.phase}: {h.last_error}"

    # 跑了 2 轮（trajectory 2 条）
    traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
    rounds = [json.loads(l) for l in traj.strip().splitlines() if l.strip()]
    assert len(rounds) == 2
    assert [r["round"] for r in rounds] == [1, 2]
    # meta 标记 finished
    meta = json.loads((settings.run_dir(loop_id) / "meta.json").read_text(encoding="utf-8"))
    assert meta["finished_at"] is not None


def test_single_round_stops_at_review(setup, monkeypatch):
    """rounds=1（非自动模式）：首轮跑完停在 awaiting_review，不自动续跑。"""
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    loop_id = runner.start(
        bench_id=bench_id, sample_id="s001", model="fake-model", rounds=1,
    )
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"期望 awaiting_review，实际 {h.phase}"
    assert h.auto_mode is False
    # 只跑了 1 轮
    traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
    rounds = [json.loads(l) for l in traj.strip().splitlines() if l.strip()]
    assert len(rounds) == 1
