"""Critic deepagent 测试：FakeToolCallingChatModel 回 canned CriticAgentOutput，验证映射/加权。

完全离线、无 key、无网络。跑在真实家具 benchmark 上。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.critic import Critic, CriticInput
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.weights import init_weights
from tests._fakes import FakeToolCallingChatModel

# ---- fixtures ----


@pytest.fixture(scope="module")
def loaded(bench_id: str):
    lb = load_benchmark(bench_id)
    return lb


@pytest.fixture
def three_view_images(loaded, tmp_path) -> list[Path]:
    """造 3 张占位视图图（agent 的多模态注入需要路径存在）。"""
    imgs = []
    for view in ("front", "side", "perspective"):
        p = tmp_path / f"{view}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake")  # 占位 jpg 头
        imgs.append(p)
    return imgs


# ---- canned 响应构造 ----


def _critic_agent_response(
    loaded, *, continuous: float = 0.75, empty_binary: tuple[str, ...] = (),
) -> AIMessage:
    """构造一条 AIMessage：tool_call=CriticAgentOutput，含全部维度评分。

    二分维度：前 N-1 项 passed、最后一项不通过（与原 test_critic 同款）。
    连续维度：value=continuous。empty_binary 里的二分维度强制 items=[]。
    """
    spec = loaded.sample("s001").spec
    bench = loaded.bench
    dims = []
    for ddef in bench.score_dimensions:
        if ddef.scoring_type == "binary":
            items = [] if ddef.dim in empty_binary else spec.checklist.get(ddef.dim, [])
            items = items if isinstance(items, list) else []
            judgments = [
                {"id": it.id, "passed": i < len(items) - 1, "reason": f"mock {it.id}"}
                for i, it in enumerate(items)
            ]
            dims.append({"dim": ddef.dim, "scoring_type": "binary", "items": judgments})
        else:
            dims.append({
                "dim": ddef.dim, "scoring_type": "continuous",
                "value": continuous, "reason": "mock continuous",
            })
    return AIMessage(content="", tool_calls=[{
        "name": "CriticAgentOutput", "type": "tool_call", "id": "c1",
        "args": {"dimensions": dims},
    }])


def _flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def _has_image_part(content) -> bool:
    if isinstance(content, str):
        return False
    return any(p.get("type") == "image_url" for p in content if isinstance(p, dict))


def _all_contents(chat_model):
    for msgs in chat_model.calls:
        for m in msgs:
            yield m.content


# ---- 集成：混合评分端到端 ----


def test_critic_evaluate_full_hybrid(loaded, three_view_images):
    """agent 回 CriticAgentOutput → 映射 DimensionScore + 算 restoration，数学一致。"""
    bench = loaded.bench
    weights = init_weights(bench)
    chat = FakeToolCallingChatModel(responses=[_critic_agent_response(loaded)])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))

    # 6 个维度都评了
    assert {d.dim for d in verdict.dimensions} == {
        "consistency", "product_structure", "material_texture",
        "color_accuracy", "artifact_defect", "commercial_focus",
    }
    s1 = loaded.sample("s001")
    for d in verdict.dimensions:
        if d.scoring_type == "binary":
            items = s1.spec.checklist[d.dim]
            n = len(items)
            assert d.value == pytest.approx((n - 1) / n), f"{d.dim} 通过率错"
            assert len(d.items) == n
            assert all(it.reason for it in d.items)
        else:
            assert d.value == pytest.approx(0.75)
            assert d.raw == "mock continuous"
            assert d.items is None

    # 还原度 = Σ(w·features) / Σw
    feats = {d.dim: d.value for d in verdict.dimensions}
    expected = sum(weights[k] * v for k, v in feats.items()) / sum(weights.values())
    assert verdict.restoration == pytest.approx(expected)
    assert verdict.features == pytest.approx(feats)


def test_critic_injects_target_and_generated_images(loaded, three_view_images):
    """生成图 + target 以 image_url 注入初始 HumanMessage（agent 循环每步都看得到）。"""
    bench = loaded.bench
    chat = FakeToolCallingChatModel(responses=[_critic_agent_response(loaded)])
    critic = Critic(chat, bench=bench)
    critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images,
        weights=init_weights(bench),
    ))

    target_name = loaded.sample("s001").target_path.name
    contents = list(_all_contents(chat))
    # 至少有一条消息含 target 文字锚 + 图片
    assert any(target_name in _flatten_text(c) for c in contents), "应注入 target 文字锚"
    assert any(_has_image_part(c) for c in contents), "应注入图片（生成图 + target）"


def test_critic_robust_to_empty_output(loaded, three_view_images):
    """agent 回空 dimensions（结构性坏输出）→ 所有维度安全默认 0，restoration=0，不抛错。"""
    bench = loaded.bench
    weights = init_weights(bench)
    bad = AIMessage(content="", tool_calls=[{
        "name": "CriticAgentOutput", "type": "tool_call", "id": "c1",
        "args": {"dimensions": []},
    }])
    chat = FakeToolCallingChatModel(responses=[bad])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))
    for d in verdict.dimensions:
        assert d.value == 0.0
    assert verdict.restoration == 0.0


def test_critic_missing_checklist_items_default_zero(loaded, three_view_images):
    """某二分维度 agent 回 items=[] → value=0.0、items=[]（不崩）。"""
    bench = loaded.bench
    weights = init_weights(bench)
    chat = FakeToolCallingChatModel(
        responses=[_critic_agent_response(loaded, empty_binary=("consistency",))],
    )
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))
    consistency = next(d for d in verdict.dimensions if d.dim == "consistency")
    assert consistency.value == 0.0
    assert consistency.items == []
