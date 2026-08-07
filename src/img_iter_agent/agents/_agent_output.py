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

from pydantic import BaseModel, Field

from ..memory.schema import CriticItemJudgment


class GeneratorOutput(BaseModel):
    """Generator agent 的最终交付：本轮用的 prompt + 改动说明。"""

    prompt: str = Field(description="本轮最终采用的生图 prompt（应与调用 generate_image 时一致）")
    delta_note: str = Field(
        default="",
        description="本轮相对上轮改了什么（针对哪些失败项做了什么；引用经验库的结论）",
    )


class CriticDimensionOutput(BaseModel):
    """单个维度的评分（agent 原始产出，不含 restoration）。"""

    dim: str = Field(description="维度名")
    scoring_type: Literal["binary", "continuous"] = Field(description="该维度的评分类型")
    # binary：逐项判定（通过率 = passed 数 / 总数，由代码算 value）
    items: list[CriticItemJudgment] | None = Field(default=None, description="二分维度的逐项判定")
    # continuous：0-1 分 + 理由
    value: float | None = Field(default=None, ge=0.0, le=1.0, description="连续维度的 0-1 分")
    reason: str | None = Field(default=None, description="连续维度的评分理由")


class CriticAgentOutput(BaseModel):
    """Critic agent 的最终交付：每个维度的评分。"""

    dimensions: list[CriticDimensionOutput] = Field(description="按 bench.score_dimensions 顺序的每维度评分")


__all__ = ["CriticAgentOutput", "CriticDimensionOutput", "GeneratorOutput"]
