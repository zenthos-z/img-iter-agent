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


def test_build_overview_reflects_live_runner_phase(tmp_path):
    """总览必须合并 loop_runner 内存态：后台正跑的 loop 在 overview 里 status=running。

    regression：曾只按盘上 meta.finished_at 推断（→ unknown），导致前端「运行中」页
    过滤不到任何 loop、整页空白。
    """
    from img_iter_agent.web.services.loop_runner import LoopHandle, get_runner

    loop_id = "furniture_product_whitebg-s001"
    _make_loop(tmp_path, loop_id, "s001", n_rounds=1)
    s = _settings(tmp_path)

    runner = get_runner()
    handle = LoopHandle(loop_id=loop_id, phase="running", round=2)
    runner._handles[loop_id] = handle
    try:
        def loop_status() -> str:
            ov = build_overview(s)
            bench = next(b for b in ov.benches if b.bench_id == "furniture_product_whitebg")
            sample = next(sm for sm in bench.samples if sm.sample_id == "s001")
            return sample.loops[0].status

        # 无内存态时盘上未结束 → unknown（baseline）
        runner._handles.pop(loop_id, None)
        assert loop_status() == "unknown"
        runner._handles[loop_id] = handle

        for phase in ("running", "awaiting_review", "error"):
            handle.phase = phase
            assert loop_status() == phase, f"phase={phase} 未反映到总览"
    finally:
        runner._handles.pop(loop_id, None)


def test_build_overview_detects_external_pid_running(tmp_path):
    """CLI/批量脚本起的 loop 走另一进程（web 内存 LoopRunner 不知道）。

    用 run_dir/running.pid 跨进程标记：写一个存活 pid → build_overview 判 running；
    进程死了（残留孤儿文件）→ 判 unknown。run_loop_session 的 try/finally 负责写/清。
    """
    import os

    from img_iter_agent.data.runstore import run_is_alive

    loop_id = "furniture_product_whitebg-s001"
    _make_loop(tmp_path, loop_id, "s001", n_rounds=1)
    s = _settings(tmp_path)
    run_dir = s.run_dir(loop_id)

    def loop_status() -> str:
        ov = build_overview(s)
        bench = next(b for b in ov.benches if b.bench_id == "furniture_product_whitebg")
        sample = next(sm for sm in bench.samples if sm.sample_id == "s001")
        return sample.loops[0].status

    # 无 pid 文件 → unknown（baseline）
    assert loop_status() == "unknown"
    assert run_is_alive(run_dir) is False

    # 写存活 pid（当前测试进程）→ running
    pid_file = run_dir / "running.pid"
    pid_file.write_text(json.dumps({"pid": os.getpid(), "ts": "now"}), encoding="utf-8")
    try:
        assert run_is_alive(run_dir) is True
        assert loop_status() == "running"
    finally:
        pid_file.unlink(missing_ok=True)

    # 死 pid → run_is_alive 兜底判 False（残留孤儿文件不卡「运行中」）
    pid_file.write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    try:
        assert run_is_alive(run_dir) is False
    finally:
        pid_file.unlink(missing_ok=True)


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
    # 二分维度的逐项判定（含通过项 + reason）必须全量透出，不能只留 failed_items。
    # 旧 bug：consistency 的 C1 passed=True 被 _verdict_to_out 过滤掉，前端「全过维度零说明」。
    consistency = next(d for d in t0.verdict.dimensions if d.dim == "consistency")
    assert consistency.items == [{"id": "C1", "passed": True, "reason": "ok"}]
    assert consistency.failed_items == []  # 全过 → 无失败项，但 items 仍带通过项理由
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


def test_rescore_history_with_current_weights(tmp_path):
    """手动排序产生 sample 级新权重后，API 层按当前权重重算历史 trace 分数：

    rescored=True + restoration_original 保留冻结分；总览 best/last 同步重算并带标记；
    trajectory.jsonl 落盘数据不动（纯展示层重算）。
    """
    import shutil

    from img_iter_agent.data.runstore import RunStore
    from img_iter_agent.data.trajectory import TrajectoryReader, TrajectoryWriter
    from img_iter_agent.data.weights import init_weights
    from img_iter_agent.memory.schema import AttemptRecord, CriticVerdict, DimensionScore

    # benchmark 拷进 tmp data_root（bench 加载走 settings.benchmarks_dir）
    repo_data = Path(__file__).resolve().parents[1] / "data"
    shutil.copytree(
        repo_data / "benchmarks" / "furniture_product_whitebg",
        tmp_path / "benchmarks" / "furniture_product_whitebg",
    )
    s = _settings(tmp_path)
    bench = load_benchmark("furniture_product_whitebg", settings=s).bench
    prior = init_weights(bench)

    def mk_verdict(consistency_value: float) -> CriticVerdict:
        """只有 consistency 有分、其余维度 0 分的 verdict（冻结 restoration=0.123）。"""
        dims = [
            DimensionScore(
                dim=d,
                scoring_type="binary" if d not in ("material_texture", "color_accuracy") else "continuous",
                value=consistency_value if d == "consistency" else 0.0,
            )
            for d in ["consistency", "product_structure", "material_texture",
                      "color_accuracy", "artifact_defect", "commercial_focus"]
        ]
        return CriticVerdict(sample_id="s001", dimensions=dims,
                             weights_used=dict(prior), restoration=0.123)

    loop_id = "furniture_product_whitebg-s001"
    store = RunStore.create(loop_id, "furniture_product_whitebg", "test-model",
                            settings=s, note="synthetic")
    tw = TrajectoryWriter(store.trajectory_path)
    for r, cv in [(1, mk_verdict(1.0)), (2, mk_verdict(0.5))]:
        tw.append(AttemptRecord(
            attempt_id=f"a{r:03d}_t", run_id=loop_id, round=r,
            sample_id="s001", bench_id="furniture_product_whitebg", model="test-model",
            gen_mode="image_edit", prompt=f"p{r}", size="2K",
            output_image_refs=[f"out/a{r:03d}.png"], verdict=cv,
        ))

    # 基线：无校准文件，当前权重 == 冻结权重 → 不重算
    detail = build_loop_detail(loop_id, settings=s)
    assert detail.traces[0].verdict.rescored is False
    assert abs(detail.traces[0].verdict.restoration - 0.123) < 1e-9

    # 写 sample 级人工校准权重：consistency 0.9（Σw=1），其余维度 0 分
    wpath = sample_weights_path(s, "furniture_product_whitebg", "s001")
    wpath.write_text(
        json.dumps({"weights": {
            "consistency": 0.9, "product_structure": 0.02, "material_texture": 0.02,
            "color_accuracy": 0.02, "artifact_defect": 0.02, "commercial_focus": 0.02,
        }}),
        encoding="utf-8",
    )
    try:
        detail = build_loop_detail(loop_id, settings=s)
        v1, v2 = detail.traces[0].verdict, detail.traces[1].verdict
        assert v1.rescored is True and v2.rescored is True
        assert abs(v1.restoration_original - 0.123) < 1e-9
        assert abs(v1.restoration - 0.9) < 1e-6   # consistency=1.0 × w=0.9
        assert abs(v2.restoration - 0.45) < 1e-6  # consistency=0.5 × w=0.9

        # 总览摘要同步：best/last 按新权重重算 + rescored 标记
        ov = build_overview(s)
        bench_o = next(b for b in ov.benches if b.bench_id == "furniture_product_whitebg")
        sample_o = next(sm for sm in bench_o.samples if sm.sample_id == "s001")
        loop_sum = sample_o.loops[0]
        assert loop_sum.rescored is True
        assert abs(loop_sum.best_restoration - 0.9) < 1e-6
        assert abs(loop_sum.last_restoration - 0.45) < 1e-6

        # 落盘数据不动：trajectory 里冻结分保持 0.123
        recs = TrajectoryReader(store.trajectory_path).read_all()
        assert len(recs) == 2
        assert all(abs(r.verdict.restoration - 0.123) < 1e-9 for r in recs)
    finally:
        wpath.unlink(missing_ok=True)


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
    """经验注入：query_experience 工具背后的 _format_experience 读 conclusions.json，
    effective/ineffective 正确格式化。"""
    from img_iter_agent.agents.tools.generator_tools import _format_experience
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

    ctx = _format_experience(run_dir)
    assert "加阴影" in ctx and "有效" in ctx  # effective 出现
    assert "色温描述" in ctx and "无效" in ctx  # ineffective 出现

