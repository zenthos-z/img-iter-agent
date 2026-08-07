"""经验蒸馏器（独立 Summarizer）测试：FakeToolCallingChatModel + 聚合工具单测。

离线、无 key。验证：跨 run 聚合工具读对、distill 结构化输出落 GeneralExperience、agent 失败降级。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.experience_distiller import ExperienceDistiller
from img_iter_agent.agents.tools.experience_tools import (
    make_list_runs_tool,
    make_query_dim_history_tool,
)
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.trajectory import TrajectoryWriter
from img_iter_agent.memory.experience import (
    load_general_experience,
    save_general_experience,
)
from img_iter_agent.memory.knowledge import KnowledgeBase, save_conclusions, upsert_conclusion
from img_iter_agent.memory.schema import (
    AttemptRecord,
    CriticEvidence,
    CriticVerdict,
    DimensionScore,
)
from tests._fakes import FakeToolCallingChatModel

# ---- fixtures ----


@pytest.fixture(scope="module")
def loaded(bench_id: str):
    return load_benchmark(bench_id)


def _attempt(round_n: int, sample_id: str, bench_id: str, restoration: float, delta_note: str) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=f"a{round_n:03d}_x", run_id="x", round=round_n, sample_id=sample_id,
        bench_id=bench_id, model="m",
        verdict=CriticVerdict(
            sample_id=sample_id,
            dimensions=[
                DimensionScore(dim="artifact_defect", scoring_type="binary", value=0.5),
                DimensionScore(dim="material_texture", scoring_type="continuous", value=restoration),
            ],
            weights_used={}, restoration=restoration,
        ),
        delta_note=delta_note,
    )


@pytest.fixture
def two_runs(loaded, tmp_path) -> tuple[Path, Path]:
    """造 2 个 run：各 2 轮 trajectory + conclusions（同 dim 一个 effective、一个 ineffective）。"""
    bench = loaded.bench
    run1 = tmp_path / "run-a-s001"
    run2 = tmp_path / "run-b-s002"
    (run1 / "lessons").mkdir(parents=True)
    (run2 / "lessons").mkdir(parents=True)

    TrajectoryWriter(run1 / "trajectory.jsonl").extend([
        _attempt(1, "s001", bench.bench_id, 0.5, "base prompt"),
        _attempt(2, "s001", bench.bench_id, 0.8, "加接地阴影"),
    ])
    TrajectoryWriter(run2 / "trajectory.jsonl").extend([
        _attempt(1, "s002", bench.bench_id, 0.4, "base prompt"),
        _attempt(2, "s002", bench.bench_id, 0.35, "改色温描述"),
    ])

    kb1 = KnowledgeBase(sample_id="s001")
    upsert_conclusion(kb1, dim="artifact_defect", finding="缺阴影", change="加接地阴影",
                      created_round=2, status="verified_effective",
                      critic_evidence=CriticEvidence(tested_round=2, before={}, after={},
                                                     verdict_delta="分 0.50→1.00"),
                      lesson="加阴影有效，保持")
    save_conclusions(run1, kb1)

    kb2 = KnowledgeBase(sample_id="s002")
    upsert_conclusion(kb2, dim="artifact_defect", finding="偏色", change="改色温描述",
                      created_round=2, status="ineffective",
                      critic_evidence=CriticEvidence(tested_round=2, before={}, after={},
                                                     verdict_delta="分 0.40→0.35"),
                      lesson="色温描述无效，需换思路")
    save_conclusions(run2, kb2)
    return run1, run2


# ---- 聚合工具单测 ----


def test_list_runs_summarizes_each_run(two_runs):
    tool = make_list_runs_tool(list(two_runs))
    out = tool.invoke({})
    assert "run-a-s001" in out and "run-b-s002" in out
    assert "0.800" in out  # run1 最好还原度


def test_query_dim_history_aggregates_across_runs(two_runs):
    tool = make_query_dim_history_tool(list(two_runs))
    out = tool.invoke({"dim": "artifact_defect"})
    # 跨 run：run1 effective（加接地阴影）、run2 ineffective（改色温描述）都出现
    assert "加接地阴影" in out and "verified_effective" in out
    assert "改色温描述" in out and "ineffective" in out
    assert "run-a-s001" in out and "run-b-s002" in out


# ---- distill 端到端 ----


def _canned_experience() -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "DistilledExperience", "type": "tool_call", "id": "e1",
        "args": {
            "summary": "跨 run：阴影相关改动有效，色温描述无效。",
            "lessons": [
                {"dim": "artifact_defect", "insight": "加接地阴影普遍提升还原度",
                 "dos": ["明确描述接地阴影"], "donts": ["忽略阴影"],
                 "evidence": ["run-a-s001/round2"], "confidence": 0.8},
            ],
        },
    }])


def test_distill_produces_general_experience(loaded, two_runs):
    bench = loaded.bench
    chat = FakeToolCallingChatModel(responses=[_canned_experience()])
    distiller = ExperienceDistiller(chat, run_dirs=list(two_runs), bench=bench, skills_dir=None)

    exp = distiller.distill()

    assert exp.bench_id == bench.bench_id
    assert set(exp.source_runs) == {"run-a-s001", "run-b-s002"}
    assert exp.summary.startswith("跨 run")
    assert len(exp.lessons) == 1
    assert exp.lessons[0].dim == "artifact_defect"
    assert exp.lessons[0].confidence == pytest.approx(0.8)
    # 写盘 + 读回一致
    path = save_general_experience(two_runs[0].parent, bench.bench_id, exp)
    assert path.exists()
    back = load_general_experience(two_runs[0].parent, bench.bench_id)
    assert back.bench_id == bench.bench_id
    assert len(back.lessons) == 1
    assert set(back.source_runs) == {"run-a-s001", "run-b-s002"}


def test_distill_degrades_when_agent_fails(loaded, two_runs):
    """agent 不给结构化输出 → 安全降级（空 lessons + 占位 summary），不抛错。"""
    bench = loaded.bench
    chat = FakeToolCallingChatModel(responses=[AIMessage(content="(no structured output)")])
    distiller = ExperienceDistiller(chat, run_dirs=list(two_runs), bench=bench, skills_dir=None)

    exp = distiller.distill()
    assert exp.lessons == []
    assert "降级" in exp.summary or "失败" in exp.summary
    assert set(exp.source_runs) == {"run-a-s001", "run-b-s002"}
