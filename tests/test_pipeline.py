"""闭环 A 测试：用假 Router（不联网出图）+ FakeToolCallingChatModel 驱动 deepagent 跑完整 LangGraph 循环。

验证：generator→critic→summarizer→human_review(interrupt)→条件边 全链路，
且 trajectory.jsonl / index.json / lessons MD 都正确产出。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from img_iter_agent.pipeline.graph import run_loop
from tests._fakes import FakeToolCallingChatModel


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


def _critic_chat(lb) -> FakeToolCallingChatModel:
    """构造驱动 Critic deepagent 的 canned 响应：一条 CriticAgentOutput（具名字段形态，
    每维度一个字段）。单条响应被多轮复用（fake 耗尽后重复最后一条）。"""
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


def _gen_chat_model(num_rounds: int) -> FakeToolCallingChatModel:
    """构造驱动 Generator deepagent 的 canned AIMessage 序列（每轮 2 条：
    generate_image 调用 + GeneratorOutput 结构化输出）。跨轮共享、顺序出队。"""
    resps: list[AIMessage] = []
    for r in range(1, num_rounds + 1):
        prompt = f"product white-bg 3-view, round {r}"
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "generate_image", "type": "tool_call", "id": f"g{r}",
            "args": {"prompt": prompt, "size": "2K"},
        }]))
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "GeneratorOutput", "type": "tool_call", "id": f"s{r}",
            "args": {"prompt": prompt, "delta_note": f"round {r} change"},
        }]))
    return FakeToolCallingChatModel(responses=resps)


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
    gen = Generator(router, chat_model=_gen_chat_model(1))
    critic = Critic(_critic_chat(lb), bench=bench)

    state = run_loop(
        bench=lb, run_store=store, sample_id="s001",
        generator=gen, critic=critic,
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
    gen = Generator(router, chat_model=_gen_chat_model(2))
    critic = Critic(_critic_chat(lb), bench=bench)

    state = run_loop(
        bench=lb, run_store=store, sample_id="s001",
        generator=gen, critic=critic,
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
    gen = Generator(_FakeRouter(), chat_model=_gen_chat_model(1))
    critic = Critic(_critic_chat(lb), bench=bench)

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic, decisions=["stop"])

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
    gen = Generator(_FakeRouter(), chat_model=_gen_chat_model(2))
    critic = Critic(_critic_chat(lb), bench=bench)

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic,
             decisions=["continue", "stop"])

    traj = [json.loads(l) for l in
            (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    # 第 2 轮的 baseline_ref 指向第 1 轮 attempt_id
    assert traj[1]["baseline_ref"] == traj[0]["attempt_id"]
    assert traj[1]["round"] == 2


def test_round2_prompt_improved_from_round1_feedback(setup):
    """第 2 轮：Generator deepagent 据上轮 Critic 失败项改进 prompt（结构化断言）。"""
    lb, store = setup
    bench = lb.bench
    gen = Generator(_FakeRouter(), chat_model=_gen_chat_model(2))
    # 构造一些失败项：让二分维度有失败（前 N-1 通过 → 最后 1 项失败）
    critic = Critic(_critic_chat(lb), bench=bench)

    run_loop(bench=lb, run_store=store, sample_id="s001",
             generator=gen, critic=critic,
             decisions=["continue", "stop"])

    traj = [json.loads(l) for l in
            (store.run_dir / "trajectory.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    p1, p2 = traj[0]["prompt"], traj[1]["prompt"]
    # 第 2 轮 prompt 应不同于第 1 轮（canned 序列里两轮 prompt 不同）
    assert p2 != p1
    # 第 2 轮应携带改动说明（delta_note 由 GeneratorOutput 结构化输出给出）
    assert traj[1].get("delta_note")
    # test_variable 始终是 prompt（不再是 size 轮换）
    assert traj[1]["test_variable"] == "prompt"
    # size 两轮一致（固定不动）
    assert traj[0]["size"] == traj[1]["size"]
