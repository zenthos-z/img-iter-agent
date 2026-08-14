"""Critic deepagent 测试：FakeToolCallingChatModel 回 canned CriticAgentOutput，验证映射/加权。

完全离线、无 key、无网络。跑在真实家具 benchmark 上。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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
    drop_dims: tuple[str, ...] = (), mangle_binary_ids: tuple[str, ...] = (),
) -> AIMessage:
    """构造一条 AIMessage：tool_call=CriticAgentOutput，含全部维度评分（具名字段形态）。

    二分维度：前 N-1 项 passed、最后一项不通过。连续维度：{value, reason}。
    empty_binary 里的二分维度强制 items=[]；drop_dims 整维度缺字段（模拟漏填）；
    mangle_binary_ids 里的二分维度 id 全部改成自造 id（模拟 id 不对应）。
    """
    spec = loaded.sample("s001").spec
    bench = loaded.bench
    args: dict = {}
    for ddef in bench.score_dimensions:
        if ddef.dim in drop_dims:
            continue
        if ddef.scoring_type == "binary":
            items = [] if ddef.dim in empty_binary else spec.checklist.get(ddef.dim, [])
            items = items if isinstance(items, list) else []
            judgments = [
                {"id": (f"X{i+1}" if ddef.dim in mangle_binary_ids else it.id),
                 "passed": i < len(items) - 1, "reason": f"mock {it.id}"}
                for i, it in enumerate(items)
            ]
            args[ddef.dim] = judgments
        else:
            args[ddef.dim] = {"value": continuous, "reason": "mock continuous"}
    return AIMessage(content="", tool_calls=[{
        "name": "CriticAgentOutput", "type": "tool_call", "id": "c1",
        "args": args,
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
    """agent 持续回缺全部维度字段的坏输出 → 修正 2 轮仍失败 → 全维度安全默认 0，不抛错。

    fake model 每次都回同一条坏消息（schema 校验失败）：反馈修正回路 2 轮无果后退兜底，
    宽松解析抠不出任何维度 → 全 0。断言修正确实发生过（模型被反馈催了 2 次）。
    """
    bench = loaded.bench
    weights = init_weights(bench)
    bad = AIMessage(content="", tool_calls=[{
        "name": "CriticAgentOutput", "type": "tool_call", "id": "c1",
        "args": {"consistency": []},  # 缺其余 5 个维度字段 → 校验失败
    }])
    chat = FakeToolCallingChatModel(responses=[bad])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))
    for d in verdict.dimensions:
        assert d.value == 0.0
    assert verdict.restoration == 0.0
    # 修正回路确实跑了：初始 + 2 轮修正 = 3 次 agent invoke
    assert len(chat.calls) == 1 + critic.REPAIR_ROUNDS


def test_critic_output_schema_named_fields(loaded):
    """结构化 schema 是「每维度一个具名 required 字段」，没有自由数组——重复维度结构上不可能。"""
    bench = loaded.bench
    critic = Critic(FakeToolCallingChatModel(), bench=bench)
    schema = critic._output_schema.model_json_schema()
    props = set(schema.get("properties", {}))
    assert props == {d.dim for d in bench.score_dimensions}
    assert set(schema.get("required", [])) == props  # 全 required：漏填即校验失败
    assert "dimensions" not in props  # 旧自由数组已不存在


def test_critic_repairs_missing_dim_via_feedback(loaded, three_view_images):
    """第一轮回丢了一个维度的输出 → 校验错误反馈给模型 → 第二轮补全 → 正常出分。"""
    bench = loaded.bench
    weights = init_weights(bench)
    chat = FakeToolCallingChatModel(responses=[
        _critic_agent_response(loaded, drop_dims=("color_accuracy",)),
        _critic_agent_response(loaded),
    ])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))

    # 第二轮修正成功：所有维度都有真实评分
    color = next(d for d in verdict.dimensions if d.dim == "color_accuracy")
    assert color.value == pytest.approx(0.75)
    assert color.raw == "mock continuous"
    # 反馈确实送达模型：第二次 invoke 的消息里含校验错误反馈
    second_msgs = chat.calls[1]
    fb = [m for m in second_msgs if isinstance(m, HumanMessage) and "未通过 schema 校验" in str(m.content)]
    assert fb, "校验失败应反馈给模型让其修正"


def test_critic_repairs_item_id_mismatch(loaded, three_view_images):
    """二分维度 id 与 checklist 不对应（语义问题，schema 拦不住）→ 反馈修正 → 第二轮对齐。"""
    bench = loaded.bench
    weights = init_weights(bench)
    chat = FakeToolCallingChatModel(responses=[
        _critic_agent_response(loaded, mangle_binary_ids=("consistency",)),
        _critic_agent_response(loaded),
    ])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))

    consistency = next(d for d in verdict.dimensions if d.dim == "consistency")
    s1 = loaded.sample("s001")
    n = len(s1.spec.checklist["consistency"])
    assert consistency.value == pytest.approx((n - 1) / n)  # 修正后的真实通过率
    assert {it.id for it in consistency.items} == {it.id for it in s1.spec.checklist["consistency"]}
    # 语义问题反馈里点出了 id 不对应
    second_msgs = chat.calls[1]
    fb = [m for m in second_msgs if isinstance(m, HumanMessage) and "不对应" in str(m.content)]
    assert fb, "id 不对应应作为语义问题反馈给模型"


def test_critic_salvages_valid_dims_on_persistent_semantic_error(loaded, three_view_images):
    """语义问题修不好（fake 每次回同样的坏输出）→ 兜底保留可用维度，坏维度退 0 不拖垮全局。"""
    bench = loaded.bench
    weights = init_weights(bench)
    chat = FakeToolCallingChatModel(responses=[
        _critic_agent_response(loaded, empty_binary=("consistency",)),
    ])
    critic = Critic(chat, bench=bench)

    verdict = critic.evaluate(CriticInput(
        sample=loaded.sample("s001"), generated_images=three_view_images, weights=weights,
    ))
    consistency = next(d for d in verdict.dimensions if d.dim == "consistency")
    assert consistency.value == 0.0
    assert consistency.items == []
    # 其他维度不受牵连（部分 salvage，而非整轮全 0）
    color = next(d for d in verdict.dimensions if d.dim == "color_accuracy")
    assert color.value == pytest.approx(0.75)
    assert verdict.restoration > 0.0


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
