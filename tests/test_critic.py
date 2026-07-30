"""Critic 测试：用 FakeLlmClient 回 canned 判定，验证分派/解析/加权。

完全离线、无 key、无网络。跑在真实家具 benchmark 上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_iter_agent.agents.critic import (
    Critic,
    CriticInput,
    _extract_json,
    _parse_binary_items,
    _parse_continuous_score,
)
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.weights import init_weights
from img_iter_agent.llm import FakeLlmClient
from img_iter_agent.memory.schema import CheckItem

# ---- fixtures ----

@pytest.fixture(scope="module")
def loaded(bench_id: str):
    lb = load_benchmark(bench_id)
    return lb


@pytest.fixture
def three_view_images(loaded, tmp_path) -> list[Path]:
    """造 3 张占位视图图（Critic 不真看图，FakeLlm 回 canned；这里只要路径存在）。"""
    imgs = []
    for view in ("front", "side", "perspective"):
        p = tmp_path / f"{view}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake")  # 占位 jpg 头
        imgs.append(p)
    return imgs


# ---- 单元：JSON 解析容错 ----

def test_extract_json_plain():
    assert _extract_json('{"score":0.7}') == {"score": 0.7}


def test_extract_json_codefence():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}


def test_extract_json_embedded_in_prose():
    assert _extract_json('结果如下:\n{"judgments":[]}\n完毕') == {"judgments": []}


def test_extract_json_bad_returns_none():
    assert _extract_json("not json at all") is None
    assert _extract_json("{broken") is None


def test_parse_binary_items_handles_missing_and_bad():
    items = [CheckItem(id="C1", check="x"), CheckItem(id="C2", check="y")]
    raw = {"judgments": [{"id": "C1", "passed": True, "reason": "ok"}]}
    out = _parse_binary_items(raw, items)
    assert out[0].passed is True and out[0].reason == "ok"
    # C2 缺失 → 默认 False
    assert out[1].passed is False


def test_parse_binary_items_none_raw():
    items = [CheckItem(id="C1", check="x")]
    out = _parse_binary_items(None, items)
    assert out[0].passed is False


def test_parse_continuous_score_clamps_and_defaults():
    assert _parse_continuous_score({"score": 0.8, "reason": "good"}) == (0.8, "good")
    assert _parse_continuous_score({"score": 1.5}) == (1.0, "")
    assert _parse_continuous_score({"score": -0.3}) == (0.0, "")
    assert _parse_continuous_score({"score": "abc"}) == (0.0, "")
    assert _parse_continuous_score(None) == (0.0, "(解析失败)")


# ---- 集成：混合评分端到端 ----

def _make_responses(loaded):
    """构造 6 个维度的 canned LLM 响应（顺序 = manifest 维度顺序）。
    4 个二分维度（consistency/product_structure/artifact_defect/commercial_focus）
    + 2 个连续维度（material_texture/color_accuracy）。"""
    s1 = loaded.sample("s001")
    spec = s1.spec
    responses = []
    bench = loaded.bench
    for dim_def in bench.score_dimensions:
        name = dim_def.dim
        val = spec.checklist.get(name)
        if dim_def.scoring_type == "binary":
            items = val if isinstance(val, list) else []
            # 前 N-1 项通过，最后一项不通过
            judgments = [
                {"id": it.id, "passed": i < len(items) - 1, "reason": f"mock {it.id}"}
                for i, it in enumerate(items)
            ]
            responses.append(json.dumps({"judgments": judgments}))
        else:
            responses.append(json.dumps({"score": 0.75, "reason": "mock continuous"}))
    return responses


def test_critic_evaluate_full_hybrid(loaded, three_view_images):
    bench = loaded.bench
    weights = init_weights(bench)
    client = FakeLlmClient(responses=_make_responses(loaded))
    critic = Critic(client, bench=bench)

    inp = CriticInput(sample=loaded.sample("s001"),
                      generated_images=three_view_images, weights=weights)
    verdict = critic.evaluate(inp)

    # 6 个维度都评了
    assert {d.dim for d in verdict.dimensions} == {
        "consistency", "product_structure", "material_texture",
        "color_accuracy", "artifact_defect", "commercial_focus",
    }
    # 二分维度：通过率正确（前 N-1 通过 / 共 N 项）
    s1 = loaded.sample("s001")
    for d in verdict.dimensions:
        if d.scoring_type == "binary":
            items = s1.spec.checklist[d.dim]
            n = len(items)
            expected = (n - 1) / n
            assert d.value == pytest.approx(expected), f"{d.dim} 通过率错"
            assert len(d.items) == n
            assert all(it.reason for it in d.items)  # 每项有理由
        else:
            assert d.value == pytest.approx(0.75)
            assert d.raw == "mock continuous"
            assert d.items is None  # 连续维度无逐项

    # 还原度 = Σ(w·features)
    feats = {d.dim: d.value for d in verdict.dimensions}
    expected_restoration = sum(weights[k] * v for k, v in feats.items()) / sum(weights.values())
    assert verdict.restoration == pytest.approx(expected_restoration)
    assert 0.0 <= verdict.restoration <= 1.0
    # features 可取回
    assert verdict.features == pytest.approx(feats)


def _flatten_text(content) -> str:
    """把 content(str 或多模态 list) 折成纯文本，便于断言。"""
    if isinstance(content, str):
        return content
    # 多模态 list：拼所有 text part
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def _has_image_part(content) -> bool:
    """多模态 content 里是否含 image_url part。"""
    if isinstance(content, str):
        return False
    return any(p.get("type") == "image_url" for p in content if isinstance(p, dict))


def test_critic_all_dims_compare_against_target(loaded, three_view_images):
    """系统目标=还原度，所有维度都对照 target 评：每个维度都注入 target+生成图。"""
    bench = loaded.bench
    client = FakeLlmClient(responses=_make_responses(loaded))
    critic = Critic(client, bench=bench)
    critic.evaluate(CriticInput(sample=loaded.sample("s001"),
                                generated_images=three_view_images, weights=init_weights(bench)))
    target_name = loaded.sample("s001").target_path.name
    for dim_def, call in zip(bench.score_dimensions, client.calls):
        content = call[-1]["content"]
        text = _flatten_text(content)
        # 每个维度（含 artifact_defect / commercial_focus）都应含 target 文字锚
        assert target_name in text, f"{dim_def.dim} 应对照 target"
        assert _has_image_part(content), f"{dim_def.dim} 应注入图片"


def test_critic_no_generated_images_still_injects_target(loaded):
    """无生成图时，所有维度仍注入 target（对比基准）→ 多模态。"""
    bench = loaded.bench
    client = FakeLlmClient(responses=_make_responses(loaded))
    critic = Critic(client, bench=bench)
    critic.evaluate(CriticInput(sample=loaded.sample("s001"),
                                generated_images=[], weights=init_weights(bench)))
    for dim_def, call in zip(bench.score_dimensions, client.calls):
        content = call[-1]["content"]
        # 即使无生成图，target 仍注入 → 多模态
        assert _has_image_part(content), f"{dim_def.dim} 应注入 target"


def test_critic_robust_to_malformed_llm_output(loaded, three_view_images):
    """LLM 回乱码时：二分项默认不通过、连续分默认 0，但不抛错。"""
    bench = loaded.bench
    weights = init_weights(bench)
    client = FakeLlmClient(responses=["garbage"] * 6)
    critic = Critic(client, bench=bench)
    verdict = critic.evaluate(CriticInput(sample=loaded.sample("s001"),
                                          generated_images=three_view_images, weights=weights))
    for d in verdict.dimensions:
        if d.scoring_type == "binary":
            assert d.value == 0.0  # 全不通过
        else:
            assert d.value == 0.0
    assert verdict.restoration == 0.0


def test_critic_missing_checklist_treated_neutrally(loaded, three_view_images):
    """manifest 声明某维度为二分但考题 checklist 缺该项 → 中性处理不崩。"""
    bench = loaded.bench
    weights = init_weights(bench)
    # 构造一个故意缺 consistency checklist 的 spec 副本
    s = loaded.sample("s001")
    broken_spec = s.spec.model_copy()
    broken_spec.checklist.pop("consistency", None)
    from img_iter_agent.data.benchmark import Sample
    broken_sample = Sample(spec=broken_spec, target_path=s.target_path,
                           target_md_path=s.target_md_path, sample_dir=s.sample_dir)

    client = FakeLlmClient(responses=_make_responses(loaded))
    critic = Critic(client, bench=bench)
    # consistency 缺 checklist → 跳过 LLM（少一次调用），其余正常
    responses = _make_responses(loaded)
    # 去掉第一个（consistency 的响应），因为缺 checklist 会直接给中性分
    client = FakeLlmClient(responses=responses[1:])
    verdict = critic.evaluate(CriticInput(sample=broken_sample,
                                          generated_images=three_view_images, weights=weights))
    consistency = next(d for d in verdict.dimensions if d.dim == "consistency")
    assert consistency.value == 0.0
    assert consistency.items == []
