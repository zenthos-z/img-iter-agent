"""deepagent 结构化输出 schema（agent 的「最终交付物」类型）。

Generator/Critic 各自的 deepagent 用 `response_format=<这里的数据类>` 约束最终输出，
结果从 `result["structured_response"]` 取（已 pydantic 校验）。

注意：与 `memory/schema.py` 的 `CriticVerdict` 分开——
  - agent 不知道当前权重 w，不能返回 `restoration`（= Σ wᵢ×features）；
  - 故 agent 只返回每维度的原始评分，`restoration` 由代码用 `recompute_restoration(...)` 算。
这样保住现有「权重变更不影响打分、只影响还原度」的数学契约。
"""

from __future__ import annotations

from typing import Literal

from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field

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


class CriticDimensionOutput(BaseModel):
    """单个维度的评分（agent 原始产出，不含 restoration）。

    字段一律非 Optional + 显式类型：Gemini 的 functionDeclaration schema 校验要求每个
    property 都带 `type`，而 `X | None` 会生成无顶层 type 的 anyOf → 被严格后端 400 拒
    （"schema didn't specify the schema type field"，dmxapi 会路由到不同严格度的后端故偶发）。
    binary 维度用 items（value 留默认 0.0）；continuous 维度用 value + reason。
    """

    dim: str = Field(description="维度名")
    scoring_type: Literal["binary", "continuous"] = Field(description="该维度的评分类型")
    # binary：逐项判定（通过率 = passed 数 / 总数，由代码算 value）
    items: list[CriticItemJudgment] = Field(default_factory=list, description="二分维度的逐项判定")
    # continuous：0-1 分 + 理由
    value: float = Field(default=0.0, ge=0.0, le=1.0, description="连续维度的 0-1 分")
    reason: str = Field(default="", description="连续维度的评分理由")


class CriticAgentOutput(BaseModel):
    """Critic agent 的最终交付：每个维度的评分。"""

    dimensions: list[CriticDimensionOutput] = Field(description="按 bench.score_dimensions 顺序的每维度评分")


__all__ = ["CriticAgentOutput", "CriticDimensionOutput", "GeneratorOutput", "provider_structured"]
