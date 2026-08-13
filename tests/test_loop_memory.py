"""Generator 本 loop 记忆子系统测试（纯单元，无 LLM/无联网）。

覆盖：append_round 逐轮追加、load_memory_brief 头部剥离（保留 ### Round）、
read/write_memory_raw 前端 CRUD、reset_memory 清空、vs 上轮增减计算、失败维度提取。
"""

from __future__ import annotations

from pathlib import Path

from img_iter_agent.agents.generator import GenOutcome
from img_iter_agent.memory import loop_memory as lm
from img_iter_agent.memory.schema import CriticItemJudgment, CriticVerdict, DimensionScore


def _verdict(resto: float, feat: dict[str, float]) -> CriticVerdict:
    return CriticVerdict(
        sample_id="s1", restoration=resto, weights_used={},
        dimensions=[DimensionScore(dim=k, scoring_type="continuous", value=v) for k, v in feat.items()],
    )


def _verdict_binary(resto: float, dim: str, items: list[tuple[str, bool]]) -> CriticVerdict:
    return CriticVerdict(
        sample_id="s1", restoration=resto, weights_used={},
        dimensions=[DimensionScore(dim=dim, scoring_type="binary", value=0.0,
                                   items=[CriticItemJudgment(id=i, passed=p) for i, p in items])],
    )


def _outcome(**kw) -> GenOutcome:
    base = dict(
        attempt_id="a001", test_variable="prompt", baseline_ref=None,
        gen_mode="text_to_image", prompt="clean three-view product shot", size="2K",
        reference_image_refs=[], reference_ids=[], output_image_refs=["out/a001/three_view.png"],
        model="gemini-3.1-flash-image", model_family="D",
    )
    base.update(kw)
    return GenOutcome(**base)


def test_append_first_round_no_trend(tmp_path: Path):
    lm.append_round(tmp_path, round_n=1, outcome=_outcome(), verdict=_verdict(0.6, {"consistency": 0.5}),
                    prev_verdict=None)
    raw = (tmp_path / "generator_memory.md").read_text(encoding="utf-8")
    assert "### Round 1" in raw
    assert "vs上轮" not in raw  # 首轮无趋势
    assert "consistency(0.50)" in raw  # 低分连续维度入失败维度


def test_append_second_round_with_delta_and_levers(tmp_path: Path):
    v1 = _verdict(0.60, {"consistency": 0.5, "color": 0.7})
    lm.append_round(tmp_path, round_n=1, outcome=_outcome(), verdict=v1, prev_verdict=None)
    v2 = _verdict(0.66, {"consistency": 0.8, "color": 0.6})
    o2 = _outcome(model="qwen-image-2.0-pro", model_family="C", edit_previous=True,
                  negative_prompt="blurry, extra legs", seed=42, strategy_note="换 Qwen 试 consistency",
                  reference_ids=["hand-abacus"])
    lm.append_round(tmp_path, round_n=2, outcome=o2, verdict=v2, prev_verdict=v1)
    raw = (tmp_path / "generator_memory.md").read_text(encoding="utf-8")
    assert "vs上轮 +0.0600" in raw          # restoration 涨
    assert "edit_previous=是" in raw         # 改图杠杆
    assert "negative=blurry, extra legs" in raw
    assert "seed=42" in raw
    assert "参考=hand-abacus" in raw
    assert "改善: consistency +0.30" in raw  # per-dim 涨
    assert "退步: color -0.10" in raw


def test_load_brief_strips_header_keeps_round_headers(tmp_path: Path):
    lm.append_round(tmp_path, round_n=1, outcome=_outcome(), verdict=_verdict(0.6, {"x": 0.5}),
                    prev_verdict=None)
    brief = lm.load_memory_brief(tmp_path)
    assert "### Round 1" in brief          # 三井号轮次标题保留
    assert "动作记忆" not in brief          # 单井号头部注释被剥
    assert "还原度" in brief


def test_load_brief_empty_when_no_file(tmp_path: Path):
    assert lm.load_memory_brief(tmp_path) == ""


def test_read_write_raw_roundtrip(tmp_path: Path):
    assert lm.read_memory_raw(tmp_path) == ""   # 不存在→空
    lm.write_memory_raw(tmp_path, "# my notes\nround 1 stuff")
    assert lm.read_memory_raw(tmp_path) == "# my notes\nround 1 stuff"


def test_write_raw_empty_falls_back_to_header(tmp_path: Path):
    lm.write_memory_raw(tmp_path, "   ")        # 空内容→写头部（不留空文件）
    raw = lm.read_memory_raw(tmp_path)
    assert "动作记忆" in raw                     # 头部在场
    assert lm.load_memory_brief(tmp_path) == ""  # 正文为空


def test_reset_clears_content(tmp_path: Path):
    lm.append_round(tmp_path, round_n=1, outcome=_outcome(), verdict=_verdict(0.6, {"x": 0.5}),
                    prev_verdict=None)
    assert lm.load_memory_brief(tmp_path) != ""
    lm.reset_memory(tmp_path)
    assert lm.load_memory_brief(tmp_path) == ""  # 清空
    assert (tmp_path / "generator_memory.md").exists()  # 文件仍在（重建为头部）


def test_failed_dims_binary_and_continuous(tmp_path: Path):
    # 二分维度有未通过项 + 连续维度低分 → 都入失败维度
    v = _verdict_binary(0.5, "checklist", [("C1", True), ("C2", False)])
    v.dimensions.append(DimensionScore(dim="texture", scoring_type="continuous", value=0.4))
    failed = lm._failed_dims(v)
    assert "checklist" in failed
    assert any("texture" in f for f in failed)


def test_dim_delta_below_epsilon_filtered(tmp_path: Path):
    # 涨跌 < eps(0.03) 视为噪声不记
    v1 = _verdict(0.5, {"a": 0.50, "b": 0.50})
    v2 = _verdict(0.5, {"a": 0.52, "b": 0.48})  # ±0.02 都 < 0.03
    improved, regressed = lm._dim_delta(v1, v2)
    assert improved == [] and regressed == []


def test_append_isolated_per_run_dir(tmp_path: Path):
    """按 loop=run 隔离：两个 run_dir 的记忆互不串扰。"""
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir(); run_b.mkdir()  # run_dir 生产里由 RunStore 建好；测试需先建
    lm.append_round(run_a, round_n=1, outcome=_outcome(model="modelA"),
                    verdict=_verdict(0.5, {"x": 0.5}), prev_verdict=None)
    lm.append_round(run_b, round_n=1, outcome=_outcome(model="modelB"),
                    verdict=_verdict(0.5, {"x": 0.5}), prev_verdict=None)
    assert "modelA" in lm.load_memory_brief(run_a) and "modelB" not in lm.load_memory_brief(run_a)
    assert "modelB" in lm.load_memory_brief(run_b) and "modelA" not in lm.load_memory_brief(run_b)
