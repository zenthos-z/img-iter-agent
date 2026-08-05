"""闭环 A 测试：用假 Router（不联网出图）+ FakeLlmClient 跑完整 LangGraph 循环。

验证：generator→critic→summarizer→human_review(interrupt)→条件边 全链路，
且 trajectory.jsonl / index.json / lessons MD 都正确产出。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.agents.summarizer import Summarizer
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from img_iter_agent.llm import FakeLlmClient
from img_iter_agent.pipeline.graph import run_loop


class _FakeRouter(Router):
    """假 Router：不联网，直接写占位图返回。"""

    def __init__(self) -> None:
        # 不调用父类 __init__（避免创建真 client）
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
    """构造 6 维 canned 判定（与 test_critic 同款）。"""
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
def setup(tmp_path, bench_id):
    """加载真实 benchmark + 建 run 目录（写 tmp）。"""
    settings = Settings(data_root=tmp_path)
    # 软链 benchmark 进临时 data_root
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    real_bench = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / bench_id
    (tmp_path / "benchmarks" / bench_id).symlink_to(real_bench)

    lb = load_benchmark(bench_id, settings=settings)
    store = RunStore.create("test-run", bench_id, "seedream-5.0-pro",
                            settings=settings, note="pipeline test")
    return lb, store


def test_loop_runs_one_round_and_stops(setup):
    """跑 1 轮：首轮自动到 interrupt，resume 'stop' → END。"""
    lb, store = setup
    bench = lb.bench
    router = _FakeRouter()
    gen = Generator(router)
    critic = Critic(FakeLlmClient(responses=_critic_responses(bench)), bench=bench)
    summ = Summarizer()

    state = run_loop(
        bench=lb, run_store=store, sample_id="s001",
        generator=gen, critic=critic, summarizer=summ,
        decisions=["stop"],
    )
    # 跑了 1 轮（三视图 = 1 张图，单次生成）
    assert state["round"] == 1
    assert state["decision"] == "stop"
    assert router.calls == 1
    # trajectory 写了 1 条
    traj = (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(traj) == 1
    rec = json.loads(traj[0])
    assert rec["round"] == 1
    assert rec["verdict"] is not None
    assert len(rec["output_image_refs"]) == 1  # 三视图=一张图


def test_loop_two_rounds_then_stop(setup):
    """跑 2 轮：continue → 第二轮 → stop。"""
    lb, store = setup
    bench = lb.bench
    router = _FakeRouter()
    gen = Generator(router)
    # critic 每轮要 6 次响应；两轮共 12 次
    resp = _critic_responses(bench) * 2
    critic = Critic(FakeLlmClient(responses=resp), bench=bench)
    summ = Summarizer()

    state = run_loop(
        bench=lb, run_store=store, sample_id="s001",
        generator=gen, critic=critic, summarizer=summ,
        decisions=["continue", "stop"],
    )
    assert state["round"] == 2
    assert router.calls == 2  # 两轮各 1 张（三视图单图）
    # trajectory 2 条
    traj = (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(traj) == 2


def test_loop_writes_lessons_and_index(setup):
    """经验知识库 conclusions.json + index.json 都正确产出。"""
    lb, store = setup
    bench = lb.bench
    gen = Generator(_FakeRouter())
    critic = Critic(FakeLlmClient(responses=_critic_responses(bench)), bench=bench)
    summ = Summarizer()

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic, summarizer=summ, decisions=["stop"])

    # 经验知识库 conclusions.json（替代原单轮 MD）
    cpath = store.run_dir / "lessons" / "conclusions.json"
    assert cpath.exists()
    kb = json.loads(cpath.read_text(encoding="utf-8"))
    assert kb["sample_id"] == "s001"

    # index.json
    idx = json.loads((store.run_dir / "index.json").read_text(encoding="utf-8"))
    assert len(idx["attempts"]) == 1
    e = idx["attempts"][0]
    assert e["round"] == 1
    assert e["model"] == "fake-model"
    assert len(e["output_image_refs"]) == 1  # 三视图=一张图
    assert e["lesson_ref"].startswith("lessons/")  # 指向 conclusions.json
    assert "delta_note" in e  # 改动说明字段存在


def test_control_variable_baseline_ref_set_on_round2(setup):
    """第 2 轮的 baseline_ref 应指向第 1 轮的 attempt（控制变量法）。"""
    lb, store = setup
    bench = lb.bench
    gen = Generator(_FakeRouter())
    critic = Critic(FakeLlmClient(responses=_critic_responses(bench) * 2), bench=bench)
    summ = Summarizer()

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic, summarizer=summ,
             decisions=["continue", "stop"])

    traj = [json.loads(l) for l in
            (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    # 第 2 轮的 baseline_ref 指向第 1 轮 attempt_id
    assert traj[1]["baseline_ref"] == traj[0]["attempt_id"]
    assert traj[1]["round"] == 2


def test_round2_prompt_improved_from_round1_feedback(setup):
    """第 2 轮 prompt 应基于第 1 轮 Critic 失败项改进（确定性补强，无 LLM）。"""
    lb, store = setup
    bench = lb.bench
    # 无 LLM 的 Generator：_improve_prompt 走确定性补强（追加失败项理由）
    gen = Generator(_FakeRouter())
    # 构造一些失败项：让二分维度有失败（前 N-1 通过 → 最后 1 项失败）
    critic = Critic(FakeLlmClient(responses=_critic_responses(bench) * 2), bench=bench)
    summ = Summarizer()

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic, summarizer=summ,
             decisions=["continue", "stop"])

    traj = [json.loads(l) for l in
            (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    p1, p2 = traj[0]["prompt"], traj[1]["prompt"]
    # 第 2 轮 prompt 应不同于第 1 轮（基于失败项补强）
    assert p2 != p1
    # 第 2 轮应含补强标记（确定性补强会加 "改进点"）
    assert "改进点" in p2 or "确保" in p2
    # test_variable 始终是 prompt（不再是 size 轮换）
    assert traj[1]["test_variable"] == "prompt"
    # size 两轮一致（固定不动）
    assert traj[0]["size"] == traj[1]["size"]
