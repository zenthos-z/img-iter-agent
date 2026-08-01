"""web 台 + 自动校准闭环 的测试。

覆盖：
  - 总览聚合（build_overview）：bench/sample/loop 归类、待打分计数
  - loop 详情（build_loop_detail）：trace/verdict/lesson 转换
  - load_weights 优先级链：loop 级 > sample 级 > 先验
  - 自动校准：提交排序 → fit_weights → sample 级落盘 → load_weights 能读到
  - Agent 配置外部化：写提示词 → load_system_prompt 读到
"""

from __future__ import annotations

import json
from pathlib import Path

from img_iter_agent.agents.agent_config_loader import load_system_prompt
from img_iter_agent.calibration.fit_weights import fit_weights
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.weights import load_weights
from img_iter_agent.web.services.calibrator import sample_weights_path
from img_iter_agent.web.services.data_access import (
    build_loop_detail,
    build_overview,
)


def _settings(data_root: Path) -> Settings:
    return Settings(data_root=data_root, dmxapi_key="")


def _make_verdict(restoration=0.8):
    """造一个 6 维 CriticVerdict（用 furniture benchmark 的维度）。"""
    from img_iter_agent.memory.schema import (
        CriticItemJudgment,
        CriticVerdict,
        DimensionScore,
    )
    dims = [
        DimensionScore(dim="consistency", scoring_type="binary", value=restoration,
                       items=[CriticItemJudgment(id="C1", passed=True, reason="ok")]),
        DimensionScore(dim="product_structure", scoring_type="binary", value=restoration),
        DimensionScore(dim="material_texture", scoring_type="continuous", value=restoration,
                       raw="material ok"),
        DimensionScore(dim="color_accuracy", scoring_type="continuous", value=restoration,
                       raw="color ok"),
        DimensionScore(dim="artifact_defect", scoring_type="binary", value=restoration),
        DimensionScore(dim="commercial_focus", scoring_type="binary", value=restoration),
    ]
    return CriticVerdict(sample_id="s001", dimensions=dims,
                         weights_used={d.dim: 0.16 for d in dims}, restoration=restoration)


def _make_loop(data_root: Path, loop_id: str, sample_id: str, n_rounds: int):
    """在 data_root/runs/<loop_id> 下造一个带 n 轮 trace 的 loop。"""
    from img_iter_agent.data.runstore import RunStore
    from img_iter_agent.data.trajectory import TrajectoryWriter
    from img_iter_agent.memory.schema import AttemptRecord

    s = _settings(data_root)
    store = RunStore.create(loop_id, "furniture_product_whitebg", "test-model",
                            settings=s, note="synthetic")
    tw = TrajectoryWriter(store.trajectory_path)
    for r in range(1, n_rounds + 1):
        rec = AttemptRecord(
            attempt_id=f"a{r:03d}_{loop_id[:6]}", run_id=loop_id, round=r,
            sample_id=sample_id, bench_id="furniture_product_whitebg", model="test-model",
            test_variable="prompt" if r > 1 else None, baseline_ref=f"a{r-1:03d}" if r > 1 else None,
            gen_mode="image_edit", prompt=f"prompt round {r}", size="2K",
            output_image_refs=[f"out/a{r:03d}/three_view.png"], verdict=_make_verdict(0.6 + r * 0.1),
            delta_note=f"round {r} delta" if r > 1 else None,
        )
        tw.append(rec)
    return loop_id


def test_build_overview_aggregates_by_sample(tmp_path):
    """总览按 bench→sample→loop 聚类，待打分数 = 未提交排序的 trace 数。"""
    _make_loop(tmp_path, "furniture_product_whitebg-s001", "s001", n_rounds=3)
    s = _settings(tmp_path)
    ov = build_overview(s)
    assert len(ov.benches) >= 1
    bench = next(b for b in ov.benches if b.bench_id == "furniture_product_whitebg")
    sample = next(sm for sm in bench.samples if sm.sample_id == "s001")
    # 一题一 loop：只 1 个 loop，3 trace
    assert len(sample.loops) == 1
    assert sample.n_traces == 3
    assert sample.pending == 3  # 未提交排序


def test_loop_detail_has_traces_and_verdict(tmp_path):
    """loop 详情读出 trace 列表，verdict 含 6 维，conclusions 可读。"""
    _make_loop(tmp_path, "furniture_product_whitebg-s001", "s001", n_rounds=2)
    s = _settings(tmp_path)
    detail = build_loop_detail("furniture_product_whitebg-s001", settings=s)
    assert detail is not None
    assert len(detail.traces) == 2
    t0 = detail.traces[0]
    assert t0.verdict is not None
    assert len(t0.verdict.dimensions) == 6
    assert t0.delta_note is None  # 首轮无改动
    assert detail.traces[1].delta_note == "round 2 delta"  # 后续轮有改动
    assert detail.conclusions is not None  # 经验字段存在（可能空）


def test_load_weights_priority_sample_level(tmp_path):
    """无 loop 级文件时，load_weights 回退到 sample 级校准权重。"""
    _make_loop(tmp_path, "furniture_product_whitebg-s001", "s001", n_rounds=1)
    s = _settings(tmp_path)
    lb = load_benchmark("furniture_product_whitebg", settings=_settings(Path(__file__).resolve().parents[1] / "data"))
    wpath = sample_weights_path(s, "furniture_product_whitebg", "s001")
    wpath.write_text(
        json.dumps({
            "weights": {"consistency": 0.9, "product_structure": 0.02,
                        "material_texture": 0.02, "color_accuracy": 0.02,
                        "artifact_defect": 0.02, "commercial_focus": 0.02},
            "prior_weights": {"consistency": 0.25}, "pairwise_accuracy": 1.0,
            "margin": 0.05, "n_traces": 2, "n_pairs": 1,
            "converged": True, "loss": 0.0,
        }),
        encoding="utf-8",
    )
    try:
        run_dir = s.run_dir("furniture_product_whitebg-s001")
        w = load_weights(lb.bench, run_dir=run_dir, sample_id="s001")
        assert abs(w["consistency"] - 0.9) < 1e-6
    finally:
        wpath.unlink(missing_ok=True)


def test_calibration_fits_weights_from_ranks(tmp_path):
    """fit_weights：3 trace 给明确排序 → 拟合出权重，吻合度=1。"""
    from img_iter_agent.calibration.fit_weights import RankedTrace

    lb = load_benchmark("furniture_product_whitebg",
                        settings=_settings(Path(__file__).resolve().parents[1] / "data"))
    bench = lb.bench
    # 用合成 verdict 造 3 条不同 rank 的 RankedTrace
    verdicts = [_make_verdict(0.6), _make_verdict(0.9), _make_verdict(0.7)]
    ranked = [
        RankedTrace(trace_id=f"a00{i}", features=v.features, human_rank=rank)
        for i, (v, rank) in enumerate(zip(verdicts, [1, 3, 2]))  # 第二条最好
    ]
    res = fit_weights(ranked, bench)
    assert res.n_pairs > 0
    assert res.pairwise_accuracy == 1.0
    assert abs(sum(res.weights.values()) - 1.0) < 1e-4  # Σw=1


def test_agent_config_externalization(repo_data_root, tmp_path):
    """写 agents_config/<agent>.md → load_system_prompt 读到，否则回退默认。"""
    s = _settings(repo_data_root)
    default = "默认提示词"
    # 未写文件 → 回退默认
    assert load_system_prompt("test_agent_xyz", default) == default
    # 写文件 → 读到
    cfg_dir = s.data_root / "agents_config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "test_agent_xyz.md").write_text("【外部】自定义提示词", encoding="utf-8")
    try:
        assert load_system_prompt("test_agent_xyz", default) == "【外部】自定义提示词"
    finally:
        (cfg_dir / "test_agent_xyz.md").unlink(missing_ok=True)


# ============ 经验闭环验证测试（Critic 驱动）============


def test_knowledge_judge_status_critic_driven():
    """judge_status：Critic 前后 verdict 对比 → 判定 effective/ineffective。

    场景：上轮 artifact_defect 有失败项 A4（value=0.75），本轮消除（value=1.0）→ effective。
    """
    from img_iter_agent.memory.knowledge import judge_status
    from img_iter_agent.memory.schema import (
        CriticItemJudgment,
        CriticVerdict,
        DimensionScore,
    )

    def mk(value, failed_ids):
        items = [CriticItemJudgment(id=i, passed=(i not in failed_ids), reason="r") for i in ["A1", "A4"]]
        dims = [DimensionScore(dim="artifact_defect", scoring_type="binary", value=value, items=items)]
        return CriticVerdict(sample_id="s", dimensions=dims, weights_used={"artifact_defect": 1.0},
                             restoration=value)

    # 失败项 A4 消除 + 分数升 → effective
    prev = mk(0.75, failed_ids={"A4"})
    cur = mk(1.0, failed_ids=set())
    status, evidence, lesson = judge_status(prev, cur, "artifact_defect")
    assert status == "verified_effective"
    assert "A4" in evidence.before["failed"]
    assert evidence.after["failed"] == []
    assert "有效" in lesson

    # 失败项仍存在 + 分数不升 → ineffective
    cur2 = mk(0.75, failed_ids={"A4"})
    status2, _, lesson2 = judge_status(prev, cur2, "artifact_defect")
    assert status2 == "ineffective"
    assert "无效" in lesson2


def test_summarizer_writes_critic_driven_conclusions(tmp_path):
    """Summarizer 闭环：登记本轮 pending → 下轮用 Critic 前后 verdict 验证 status。"""
    from img_iter_agent.agents.generator import GenOutcome
    from img_iter_agent.agents.summarizer import Summarizer
    from img_iter_agent.memory.schema import (
        CriticItemJudgment,
        CriticVerdict,
        DimensionScore,
    )

    run_dir = tmp_path / "loop1"
    (run_dir / "lessons").mkdir(parents=True)

    def mk(value, failed_ids):
        items = [CriticItemJudgment(id=i, passed=(i not in failed_ids), reason=f"reason {i}")
                 for i in ["A1", "A4"]]
        dims = [DimensionScore(dim="artifact_defect", scoring_type="binary", value=value, items=items)]
        return CriticVerdict(sample_id="s1", dimensions=dims, weights_used={"artifact_defect": 1.0},
                             restoration=value)

    summ = Summarizer()
    # 第2轮：有 delta_note，登记为 pending
    out2 = GenOutcome(attempt_id="a002", test_variable="prompt", baseline_ref="a001",
                      gen_mode="image_edit", prompt="p2", delta_note="加接地阴影",
                      size="2K", reference_image_refs=[], output_image_refs=[],
                      model="m", model_family="B")
    summ.summarize(run_dir=run_dir, round=2, outcome=out2, verdict=mk(0.75, {"A4"}),
                   sample_id="s1", prev_verdict=None, prev_delta_note=None)

    from img_iter_agent.memory.knowledge import load_conclusions
    kb = load_conclusions(run_dir, sample_id="s1")
    assert len(kb.conclusions) == 1
    assert kb.conclusions[0].status == "pending"  # 登记时待验证
    assert kb.conclusions[0].change == "加接地阴影"

    # 第3轮：上轮改动 + 上轮 verdict + 本轮 verdict（A4 消除）→ 判 effective
    out3 = GenOutcome(attempt_id="a003", test_variable="prompt", baseline_ref="a002",
                      gen_mode="image_edit", prompt="p3", delta_note="再细化阴影",
                      size="2K", reference_image_refs=[], output_image_refs=[],
                      model="m", model_family="B")
    summ.summarize(run_dir=run_dir, round=3, outcome=out3, verdict=mk(1.0, set()),
                   sample_id="s1", prev_verdict=mk(0.75, {"A4"}), prev_delta_note="加接地阴影")
    kb = load_conclusions(run_dir, sample_id="s1")
    # 上轮的 pending 现在应是 verified_effective
    eff = [c for c in kb.conclusions if c.status == "verified_effective"]
    assert len(eff) >= 1
    assert eff[0].critic_evidence is not None
    assert "A4" in eff[0].critic_evidence.before["failed"]


def test_generator_reads_conclusions(tmp_path):
    """Generator 读 conclusions.json，effective/ineffective 上下文正确生成。"""
    from img_iter_agent.agents.generator import Generator
    from img_iter_agent.config import Settings
    from img_iter_agent.generation.router import Router
    from img_iter_agent.memory.knowledge import (
        KnowledgeBase,
        save_conclusions,
        upsert_conclusion,
    )

    run_dir = tmp_path / "loop1" / "runs" / "test-s1"
    (run_dir / "lessons").mkdir(parents=True)
    kb = KnowledgeBase(sample_id="s1")
    upsert_conclusion(kb, dim="artifact_defect", finding="缺阴影", change="加阴影",
                      tags=["prompt"], created_round=2, status="verified_effective",
                      lesson="加阴影有效，保持")
    upsert_conclusion(kb, dim="color_accuracy", finding="偏色", change="加色温描述",
                      tags=["prompt"], created_round=3, status="ineffective",
                      lesson="色温描述无效")
    save_conclusions(run_dir, kb)

    gen = Generator(Router(settings=Settings(data_root=tmp_path, dmxapi_key=""), client=None))  # type: ignore[arg-type]
    ctx = gen.knowledge_context(run_dir)
    assert "加阴影" in ctx and "有效" in ctx  # effective 出现
    assert "色温描述" in ctx and "无效" in ctx  # ineffective 出现

