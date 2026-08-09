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
    skill_package_dir,
    validate_skill_md,
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
    """蒸馏器现用 RenovationPlan 作 response_format（v2 翻新）；首次蒸馏 previous=None → 全 new。"""
    return AIMessage(content="", tool_calls=[{
        "name": "RenovationPlan", "type": "tool_call", "id": "e1",
        "args": {
            "summary": "跨 run：阴影相关改动有效，色温描述无效。",
            "renovation": [
                {"action": "new", "reason": "跨 run 归纳",
                 "lesson": {
                    "dim": "artifact_defect", "insight": "加接地阴影普遍提升还原度",
                    "dos": ["明确描述接地阴影"], "donts": ["忽略阴影"],
                    "evidence": ["run-a-s001/round2"], "confidence": 0.8,
                    "category": "瑕疵", "applies_when": "fix",
                 }},
            ],
        },
    }])


def _canned_authored_skill() -> AIMessage:
    """author_skill 两阶段（draft/review）的 AuthoredSkill 结构化输出。

    skill_name 会被 author_skill 强制覆盖为 slug(bench)，故此处随便填。
    description 不含尖括号（过 quick_validate）；skill_md 含 lessons 前景化 + 输出格式模板。
    """
    return AIMessage(content="", tool_calls=[{
        "name": "AuthoredSkill", "type": "tool_call", "id": "s1",
        "args": {
            "summary": "产出白底三视图编辑策略",
            "skill_name": "will-be-overwritten-by-slug",
            "description": "输入一张产品照，产出电商白底三视图（正/侧/立体）的生成编辑策略。"
            "触发于：需要生成产品白底素材图、做电商三视图排版、或要把单品展成多视角时。",
            "skill_md": (
                "# 白底三视图编辑策略技能\n\n"
                "> ⚠️ 核心精华：生成前务必读 references/lessons.md 并严格遵循其 dos/donts。\n\n"
                "## 输入契约\n一张产品照。\n\n## 工作流\n1. 识别产品主特征\n2. 排三视图\n\n"
                "## 输出格式\n```json\n{\"prompt\":\"...\",\"strategy\":[]}\n```"
            ),
            "references": [{"path": "eval_criteria.md", "content": "# 评分标准\n三视角同产品同比例。"}],
            "asset_paths": [],
        },
    }])


def test_distill_produces_general_experience(loaded, two_runs):
    bench = loaded.bench
    # 3 个响应：renovation（lessons 翻新）→ skill draft → skill review（两阶段 authoring）。
    chat = FakeToolCallingChatModel(
        responses=[_canned_experience(), _canned_authored_skill(), _canned_authored_skill()]
    )
    distiller = ExperienceDistiller(
        chat, run_dirs=list(two_runs), lb=loaded, data_root=two_runs[0].parent, skills_dir=None,
    )

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
    # 两阶段 authoring 写出规范技能包 + 过 quick_validate 硬规则
    pkg = skill_package_dir(two_runs[0].parent, bench.bench_id)
    assert (pkg / "SKILL.md").exists(), "技能包 SKILL.md 未写出"
    assert (pkg / "references" / "lessons.md").exists(), "references/lessons.md 未渲染"
    ok, msg = validate_skill_md((pkg / "SKILL.md").read_text(encoding="utf-8"))
    assert ok, f"装配的 SKILL.md 未过结构校验：{msg}"


def test_distill_skill_dossier_enriched_anthropic():
    """style_transfer 类 dossier 富化：含 style_brief 全文 + 视觉参考图 image_url part。

    furniture（默认 bench_id）是 image_edit，不注入参考图；用 anthropic（style_transfer）验视觉注入。
    """
    from img_iter_agent.data.benchmark import load_benchmark

    from img_iter_agent.memory.experience import DistilledLesson, GeneralExperience

    lb = load_benchmark("anthropic_og_style")
    d = ExperienceDistiller(
        FakeToolCallingChatModel(responses=[]), run_dirs=[], lb=lb,
        data_root=Path("/tmp"), skills_dir=None,
    )
    exp = GeneralExperience(
        bench_id="anthropic_og_style",
        lessons=[DistilledLesson(dim="spirit_hand_form", insight="手部几何化抽象", confidence=0.9, category="结构")],
    )
    parts = d._build_skill_dossier(exp)
    text = next((p["text"] for p in parts if p.get("type") == "text"), "")
    # 头号燃料：style_brief 全文进 dossier（旧版只标「存在」没读内容）
    assert "style_summary" in text or "极简" in text, "dossier 缺 style_brief 全文"
    # 全量 lessons（dos/donts）进 dossier
    assert "蒸馏经验" in text and "spirit_hand_form" in text
    # style_transfer 视觉参考图注入（参考图即风格 spec）
    assert any(p.get("type") == "image_url" for p in parts), "style_transfer dossier 应注入参考图"


def test_distill_degrades_when_agent_fails(loaded, two_runs):
    """agent 不给结构化输出 → 安全降级（空 lessons + 占位 summary），不抛错。"""
    bench = loaded.bench
    chat = FakeToolCallingChatModel(responses=[AIMessage(content="(no structured output)")])
    distiller = ExperienceDistiller(
        chat, run_dirs=list(two_runs), lb=loaded, data_root=two_runs[0].parent, skills_dir=None,
    )

    exp = distiller.distill()
    assert exp.lessons == []
    assert "降级" in exp.summary or "失败" in exp.summary
    assert set(exp.source_runs) == {"run-a-s001", "run-b-s002"}
