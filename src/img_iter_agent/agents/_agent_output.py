"""deepagent 结构化输出 schema（agent 的「最终交付物」类型）。

Generator/Critic 各自的 deepagent 用 `response_format=<这里的数据类>` 约束最终输出，
结果从 `result["structured_response"]` 取（已 pydantic 校验）。

注意：与 `memory/schema.py` 的 `CriticVerdict` 分开——
  - agent 不知道当前权重 w，不能返回 `restoration`（= Σ wᵢ×features）；
  - 故 agent 只返回每维度的原始评分，`restoration` 由代码用 `recompute_restoration(...)` 算。
这样保住现有「权重变更不影响打分、只影响还原度」的数学契约。
"""

from __future__ import annotations

import re

from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field, create_model

from ..memory.schema import CriticItemJudgment


def provider_structured(schema: type | dict) -> ProviderStrategy:
    """把结构化输出 schema 包成 ``ProviderStrategy``，走 provider 原生 json_schema response_format。

    根因：把裸 pydantic schema 直接传给 ``create_deep_agent(response_format=...)``（底层 langchain
    ``create_agent``）时，会被包成 ``AutoStrategy``→``ToolStrategy``（function_calling 路径），后者
    写死 ``tool_choice=<tool_name>``（object 形）。但 critic/generator/distiller 用的是 thinking 模型
    (qwen3.7-flash)，供应商(dmxapi→DashScope)在 thinking 模式下拒绝 ``tool_choice`` 设为
    required/object → 400 BadRequestError → agent 整轮失败退兜底（critic→评分全 0）。

    ``ProviderStrategy`` 改走 ``response_format={"type": "json_schema", ...}``，**不设 tool_choice**，
    绕开此限制（已用探针验证 qwen3.7-flash 经 dmxapi 支持 json_schema response_format）。对非 thinking
    模型同样安全（json_schema 是标准 OpenAI response_format）。

    另：嵌套 pydantic 模型（如 CriticDimensionOutput.items 里嵌 CriticItemJudgment）的 json_schema
    会带 ``$defs``/``$ref``。dmxapi 部分 Gemini 后端把 response_format 翻译成 Google 原生
    ``generation_config.response_schema``，后者不认 ``$defs`` → 400 "Unknown name $defs"（后端路由
    相关故偶发，生产实测 critic 整轮 5/5 重试全灭退兜底全 0 分）。这里把 ``$defs`` 递归内联成平化
    schema 再发出——只改**发出**的 json_schema，解析侧（structured_response → pydantic 实例）不动。
    """
    strategy = ProviderStrategy(schema)
    if not isinstance(schema, dict):
        strategy.schema_spec.json_schema = _dereference_defs(strategy.schema_spec.json_schema)
    return strategy


def _dereference_defs(schema: dict) -> dict:
    """递归内联 JSON schema 里的 ``$defs``/``$ref``（平化，Gemini 原生 response_schema 兼容）。

    解析规则：``{"$ref": "#/$defs/X", ...兄弟键}`` → 被引定义展开后兄弟键覆盖之。循环引用时
    放弃内联原样保留（本项目 schema 无环，仅防御）。
    """
    defs = schema.pop("$defs", {}) or {}

    def _inline(node, seen: frozenset):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name not in defs or name in seen:
                    return {k: v for k, v in node.items() if k != "$ref"}
                target = _inline(defs[name], seen | {name})
                return {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return {k: _inline(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [_inline(x, seen) for x in node]
        return node

    return _inline(schema, frozenset())


class GeneratorOutput(BaseModel):
    """Generator agent 的最终交付：本轮用的 prompt + 改动说明。"""

    prompt: str = Field(description="本轮最终采用的生图 prompt（应与调用 generate_image 时一致）")
    meaning: str = Field(
        default="",
        description="一句话图片含义解释：这张图如何用视觉隐喻表达文章主题/概念（风格神韵迁移场景必填，其他场景可空）",
    )
    delta_note: str = Field(
        default="",
        description="本轮相对上轮改了什么（针对哪些失败项做了什么；引用经验库的结论）",
    )
    strategy_note: str = Field(
        default="",
        description=(
            "本轮【非 prompt 杠杆】的生图策略声明：选了哪个 model、edit_previous 是否在上一版上改图、"
            "用了 negative_prompt/seed/steps 哪些。供记忆追加与蒸馏结构化捕获。"
            "与 delta_note 互补——delta_note 讲 prompt 改动，strategy_note 讲 model/参数/改图模式。"
            "纯 prompt 微调轮可留空。"
        ),
    )


class CriticContinuousScore(BaseModel):
    """连续维度的评分：value 与 reason **同在一条**，全 required。

    历史教训（2026-08-14 s002 R3）：旧的自由数组 schema 里，模型会把同一连续维度拆成
    「有 value 没 reason」+「有 reason 没 value」两条重复输出，代码 last-wins 去重后好评
    被记成 0 分。具名字段 schema 从结构上消灭重复；required 让漏填在解码/校验层就失败，
    由 critic 的反馈修正回路处理，而不是悄悄落 0。
    """

    value: float = Field(ge=0.0, le=1.0, description="该连续维度的 0-1 分")
    reason: str = Field(description="一句评分理由（必须引用图上具体可见的点）")


# create_model 的字段名不能撞 BaseModel 保留属性；维度名不是合法标识符时也加前缀。
_RESERVED_FIELD_NAMES = frozenset({
    "model_config", "model_fields", "model_computed_fields", "schema", "fields",
    "json", "copy", "construct", "validate",
})


def _dim_field_name(dim: str) -> str:
    """维度名 → 合法且不与 BaseModel 保留属性冲突的 schema 字段名。"""
    f = re.sub(r"\W", "_", dim, flags=re.UNICODE)
    if not f or f[0].isdigit() or f in _RESERVED_FIELD_NAMES or not f.isidentifier():
        f = f"d_{f}"
    return f


def build_critic_output_schema(score_dimensions) -> tuple[type[BaseModel], dict[str, str]]:
    """按 bench 维度动态生成 Critic 的结构化输出 schema：**每个维度一个具名字段**。

    旧 schema 是 ``dimensions: list[CriticDimensionOutput]`` 自由数组——模型可塞任意条数、
    重复维度、瞎编维度名（生产实测 6 维被塞成 9 条，last-wins 去重把好评压成 0 分）。
    改成具名字段后：
      - 重复维度 / 多余维度 / 编造维度名 → 结构上不可能（对象键唯一且固定）；
      - 漏填（缺维度键、连续维度缺 value/reason）→ 全 required，校验直接失败，
        走 critic 的反馈修正回路而不是悄悄落默认值。

    字段类型按 scoring_type 定制：binary → ``list[CriticItemJudgment]``（逐项判定），
    continuous → ``CriticContinuousScore``（value + reason）。字段一律非 Optional + 显式
    类型：Gemini 原生 functionDeclaration 要求每个 property 都带顶层 `type`，`X | None`
    的 anyOf 会被严格后端 400 拒（dmxapi 路由到不同严格度的后端故偶发）。

    Returns:
        (schema_model, dim_to_field)：schema_model 名为 ``CriticAgentOutput``（与
        ProviderStrategy/response_format 及测试 fake 的 schema 名约定一致）；
        dim_to_field 是 维度名→schema 字段名 的映射（字段名做过净化时用它映射回）。
    """
    fields: dict[str, tuple] = {}
    dim_to_field: dict[str, str] = {}
    for d in score_dimensions:
        fname = _dim_field_name(d.dim)
        dim_to_field[d.dim] = fname
        if d.scoring_type == "binary":
            fields[fname] = (
                list[CriticItemJudgment],
                Field(
                    description=(
                        f"二分维度「{d.dim}」：逐项判定列表，每项 {{id, passed, reason}}；"
                        f"id 与题面列出的 checklist 项严格一一对应，项数相等，不得合并/省略/自造"
                    ),
                ),
            )
        else:
            fields[fname] = (
                CriticContinuousScore,
                Field(
                    description=(
                        f"连续维度「{d.dim}」：{{value: 0-1 分, reason: 一句理由}}，两个字段都必须填"
                    ),
                ),
            )
    model = create_model("CriticAgentOutput", **fields)
    return model, dim_to_field


__all__ = [
    "CriticContinuousScore",
    "GeneratorOutput",
    "build_critic_output_schema",
    "provider_structured",
]
