"""LoopRunner 自动连跑（auto_mode）的离线验证：不依赖真实生图/LLM API。

用 FakeRouter + FakeToolCallingChatModel（驱动 deepagent）构造 mock graph，覆盖 LoopRunner._build_app，
验证：
  1. rounds=2 时，round1 完成后自动 resume(continue) 进 round2（remaining 1→0）；
  2. round2 跑满后停在 awaiting_review（不再自动 stop→END），保持 graph 可续；
  3. 跑满后 resume("continue") 能继续跑第 3 轮（核心：跑满不堵死）；
  4. rounds=1（非自动模式）首轮停在 awaiting_review（不自动续跑）。

这绕开了网络/API 超时，100% 确定性地覆盖 auto_mode 收尾逻辑。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from img_iter_agent.pipeline.graph import build_graph
from img_iter_agent.web.services.loop_runner import get_runner
from tests._fakes import FakeToolCallingChatModel


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


def _critic_chat(lb) -> FakeToolCallingChatModel:
    """驱动 Critic deepagent 的 canned CriticAgentOutput（具名字段形态，多轮复用）。"""
    spec = lb.sample("s001").spec
    bench = lb.bench
    args: dict = {}
    for ddef in bench.score_dimensions:
        if ddef.scoring_type == "binary":
            items = spec.checklist.get(ddef.dim, [])
            items = items if isinstance(items, list) else []
            args[ddef.dim] = [
                {"id": it.id, "passed": i == 0, "reason": "mock"}
                for i, it in enumerate(items)
            ]
        else:
            args[ddef.dim] = {"value": 0.7, "reason": "mock"}
    resp = AIMessage(content="", tool_calls=[{
        "name": "CriticAgentOutput", "type": "tool_call", "id": "c1",
        "args": args,
    }])
    return FakeToolCallingChatModel(responses=[resp])


def _gen_responses(num_rounds: int) -> list[AIMessage]:
    """Generator deepagent 的 canned 序列（每轮 generate_image + GeneratorOutput 两步）。"""
    resps: list[AIMessage] = []
    for r in range(1, num_rounds + 1):
        prompt = f"product 3-view round {r}"
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "generate_image", "type": "tool_call", "id": f"g{r}",
            "args": {"prompt": prompt, "size": "2K"},
        }]))
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "GeneratorOutput", "type": "tool_call", "id": f"s{r}",
            "args": {"prompt": prompt, "delta_note": f"round {r}"},
        }]))
    return resps


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
        gen = Generator(router, chat_model=FakeToolCallingChatModel(responses=_gen_responses(3)),
                        skills_dir=None)
        critic = Critic(_critic_chat(lb), bench=bench)
        conn = sqlite3.connect(store.run_dir / "checkpoints.sqlite", check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver(conn)
        app = build_graph(
            bench=lb, run_store=store, generator=gen, critic=critic,
            sample_id=sample_id, checkpointer=checkpointer,
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


def test_auto_mode_2_rounds_stops_at_review(setup, monkeypatch):
    """rounds=2：round1 自动续跑 round2，跑满后停在 awaiting_review（不再自动 stop→END），
    且可由 resume("continue") 继续跑第 3 轮（核心：跑满不堵死）。"""
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    loop_id = runner.start(
        bench_id=bench_id, sample_id="s001", model="fake-model", rounds=2,
    )
    # 跑满 2 轮后停在 awaiting_review（不再自动 finished）
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"期望 awaiting_review，实际 {h.phase}: {h.last_error}"

    def _rounds():
        traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
        return [json.loads(l) for l in traj.strip().splitlines() if l.strip()]

    # 跑了 2 轮（trajectory 2 条）
    rs = _rounds()
    assert [r["round"] for r in rs] == [1, 2]
    # 跑满后未结束：meta 不写 finished_at
    meta = json.loads((settings.run_dir(loop_id) / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("finished_at") is None

    # 关键：跑满后仍可继续 —— resume(continue) 应跑出第 3 轮
    assert runner.resume(loop_id, "continue") is True
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"resume 后期望 awaiting_review，实际 {h.phase}: {h.last_error}"
    rs = _rounds()
    assert [r["round"] for r in rs] == [1, 2, 3], f"期望跑到第 3 轮，实际 {[r['round'] for r in rs]}"


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


def test_continue_finished_loop_runs_next_round(setup, monkeypatch):
    """END/finished 态 loop 继续：resume('continue') 应跑出 N+1 轮，而非空操作 finished。

    覆盖 loop_runner._invoke_round 的 END 重入分支：旧实现对 finished 线程 Command(resume)
    是空操作（直接 finished、不出新轮），「继续跑下一轮」失效。修复后从 START 重入跑新轮。
    """
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    def _rounds():
        traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
        return [json.loads(l)["round"] for l in traj.strip().splitlines() if l.strip()]

    # round1 → awaiting_review
    loop_id = runner.start(bench_id=bench_id, sample_id="s001", model="fake-model", rounds=1)
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"{h.phase}: {h.last_error}"
    assert _rounds() == [1]

    # stop → finished（END 态）
    assert runner.resume(loop_id, "stop") is True
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "finished", f"stop 后期望 finished，实际 {h.phase}"

    # 关键：继续一个已结束的 loop → 必须跑出第 2 轮（而非立即 finished 空操作）
    assert runner.resume(loop_id, "continue") is True
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"继续 finished 后期望 awaiting_review，实际 {h.phase}: {h.last_error}"
    assert _rounds() == [1, 2], f"期望跑到第 2 轮，实际 {_rounds()}"


def test_start_finished_loop_runs_next_round(setup, monkeypatch):
    """END/finished 态 loop 用 start() 继续（模拟 server 重启、内存无 handle）：
    应跑出 N+1 轮。覆盖 _run_first_round(resume_existing=True) → _invoke_round 的 END 重入分支。"""
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    def _rounds():
        traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
        return [json.loads(l)["round"] for l in traj.strip().splitlines() if l.strip()]

    loop_id = runner.start(bench_id=bench_id, sample_id="s001", model="fake-model", rounds=1)
    _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    runner.resume(loop_id, "stop")
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "finished"
    assert _rounds() == [1]

    # 模拟 server 重启：清空内存 handle（run 目录仍在 → start() 走 _continue_existing → resume_existing）
    runner._handles.clear()

    # 再次 start：应续跑第 2 轮，而非空操作 finished
    loop_id2 = runner.start(
        bench_id=bench_id, sample_id="s001", model="fake-model", rounds=1,
    )
    assert loop_id2 == loop_id  # 一题一条：同 loop_id
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"start 续跑期望 awaiting_review，实际 {h.phase}: {h.last_error}"
    assert _rounds() == [1, 2], f"期望跑到第 2 轮，实际 {_rounds()}"


def test_resume_adopts_handleless_loop(setup, monkeypatch):
    """resume() 收养无内存 handle 的 loop（外部 run_loop_auto 起 / server 重启后）：
    清空 _handles 后 resume('continue') 应重建 graph 跑出 N+1 轮，而非返回 False。

    覆盖 resume() → _adopt_and_resume → _run_first_round(resume_existing=True) 路径——
    这是「卡片/详情页一键续跑外部 loop」的后端依赖。
    """
    settings, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, settings)
    runner = get_runner()

    def _rounds():
        traj = (settings.run_dir(loop_id) / "trajectory.jsonl").read_text(encoding="utf-8")
        return [json.loads(l)["round"] for l in traj.strip().splitlines() if l.strip()]

    loop_id = runner.start(bench_id=bench_id, sample_id="s001", model="fake-model", rounds=1)
    _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    runner.resume(loop_id, "stop")
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "finished"
    assert _rounds() == [1]

    # 模拟外部进程起的 loop / server 重启：清空内存 handle（run 目录仍在盘上）
    runner._handles.clear()
    assert runner.get(loop_id) is None  # 确实无 handle

    # resume 无 handle 但盘上存在 → 收养重建 + 续跑第 2 轮（旧实现这里返回 False）
    assert runner.resume(loop_id, "continue") is True
    h = _wait_phase(runner, loop_id, {"awaiting_review", "finished", "error"})
    assert h.phase == "awaiting_review", f"resume 收养后续跑期望 awaiting_review，实际 {h.phase}: {h.last_error}"
    assert _rounds() == [1, 2], f"期望跑到第 2 轮，实际 {_rounds()}"


def test_resume_unknown_loop_returns_false(setup, monkeypatch):
    """resume() 对盘上不存在的 loop_id 返回 False（→ 路由 404）。"""
    _, lb, bench_id = setup
    _patch_build_app(monkeypatch, lb, setup[0])
    runner = get_runner()
    assert runner.resume("nope-not-exist", "continue") is False
