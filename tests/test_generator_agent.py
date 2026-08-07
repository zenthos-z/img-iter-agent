"""Generator deepagent 单元测试：FakeToolCallingChatModel 驱动，离线、不联网出图。

验证 generate_round 在 agent 路径下：构造/改进 prompt、调 generate_image 出图、
结构化输出 prompt+delta_note、组装 GenOutcome（字段/文件/控制变量标记正确）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.generator import Generator, PriorFeedback
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from img_iter_agent.memory.schema import CriticItemJudgment
from tests._fakes import FakeToolCallingChatModel


class _FakeRouter(Router):
    """假 Router：不联网，写占位图。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, req, out_dir: Path, *, config=None) -> GeneratedImage:  # type: ignore[override]
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fake_{self.calls}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return GeneratedImage(image_path=p, model="fake-model", endpoint="fake",
                              meta={"family": "B", "size": "2K"})


def _gen_responses(pairs: list[tuple[str, str]]) -> list[AIMessage]:
    """每对 (prompt, delta_note) → [generate_image 调用, GeneratorOutput 调用]。"""
    resps: list[AIMessage] = []
    for i, (prompt, delta) in enumerate(pairs, 1):
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "generate_image", "type": "tool_call", "id": f"g{i}",
            "args": {"prompt": prompt, "size": "2K"},
        }]))
        resps.append(AIMessage(content="", tool_calls=[{
            "name": "GeneratorOutput", "type": "tool_call", "id": f"s{i}",
            "args": {"prompt": prompt, "delta_note": delta},
        }]))
    return resps


@pytest.fixture()
def setup(tmp_path, bench_id):
    settings = Settings(data_root=tmp_path)
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    real_bench = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / bench_id
    (tmp_path / "benchmarks" / bench_id).symlink_to(real_bench)
    lb = load_benchmark(bench_id, settings=settings)
    store = RunStore.create("gen-run", bench_id, "seedream-5.0-pro",
                            settings=settings, note="generator agent test")
    return lb, store


def test_generate_round1_produces_image_and_outcome(setup):
    """首轮：agent 调 generate_image 出图，结构化输出 prompt+delta_note。"""
    lb, store = setup
    router = _FakeRouter()
    gen = Generator(router, chat_model=FakeToolCallingChatModel(
        responses=_gen_responses([("product white-bg 3-view, round 1", "round 1 change")])),
        skills_dir=None)

    outcome = gen.generate_round(
        sample=lb.sample("s001"), out_dir=store.run_dir / "out", run_dir=store.run_dir,
        round=1,
    )
    assert outcome.prompt == "product white-bg 3-view, round 1"
    assert router.calls == 1  # generate_image 工具调了一次 router
    assert len(outcome.output_image_refs) == 1
    # 图片确实落盘
    img_path = store.run_dir / outcome.output_image_refs[0]
    assert img_path.exists()
    assert outcome.model == "fake-model"
    assert outcome.test_variable is None  # 首轮无 baseline
    assert outcome.improved_from_feedback is False
    # 结构化输出带 delta_note
    assert outcome.delta_note == "round 1 change"


def test_generate_round2_uses_feedback_and_marks_improved(setup):
    """第 2 轮：带上轮失败反馈 → test_variable=prompt、improved_from_feedback=True。"""
    lb, store = setup
    router = _FakeRouter()
    gen = Generator(router, chat_model=FakeToolCallingChatModel(
        responses=_gen_responses([("product white-bg 3-view, round 2", "round 2 change")])),
        skills_dir=None)

    prior = PriorFeedback(failed_items=[
        CriticItemJudgment(id="C3", passed=False, reason="悬浮无阴影"),
    ])
    outcome = gen.generate_round(
        sample=lb.sample("s001"), out_dir=store.run_dir / "out", run_dir=store.run_dir,
        round=2, baseline_ref="a001_xxx", prior_feedback=prior,
    )
    assert outcome.prompt == "product white-bg 3-view, round 2"
    assert outcome.test_variable == "prompt"
    assert outcome.improved_from_feedback is True
    assert outcome.baseline_ref == "a001_xxx"
    assert outcome.delta_note == "round 2 change"
    assert (store.run_dir / outcome.output_image_refs[0]).exists()


def test_generate_round_fallback_when_agent_fails(setup):
    """agent 跑飞（无结构化输出）时：兜底用指令出图，不崩。"""
    lb, store = setup
    router = _FakeRouter()
    # 只给 generate_image 调用，不给 GeneratorOutput → structured_response 缺失 → 走兜底
    fake = FakeToolCallingChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "generate_image", "type": "tool_call", "id": "g1",
            "args": {"prompt": "fallback prompt", "size": "2K"},
        }]),
        AIMessage(content="(no structured output)"),
    ])
    gen = Generator(router, chat_model=fake, skills_dir=None)

    outcome = gen.generate_round(
        sample=lb.sample("s001"), out_dir=store.run_dir / "out", run_dir=store.run_dir,
        round=1,
    )
    # 兜底：generate_image 已被调用（sink 有 ref），prompt 用结构化或缺省指令
    assert len(outcome.output_image_refs) == 1
    assert (store.run_dir / outcome.output_image_refs[0]).exists()
