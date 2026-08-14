"""打分台的 API 数据模型（pydantic v2）。

这些是给前端的响应/请求模型，与内部 schema 解耦：内部用 AttemptRecord 等，
对外用这些扁平、前端友好的模型。转换在 services 层完成。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 人工提示词（generator / critic；临时 vs 持久化，见 plans/wise-dreaming-shell.md）
# ---------------------------------------------------------------------------


class HintOut(BaseModel):
    """一条人工提示词（对外展示）。"""

    id: str
    agent: Literal["generator", "critic"]
    text: str
    scope: Literal["loop", "sample"]


class HintIn(BaseModel):
    """启动 loop 时随请求带入的人工提示词。"""

    agent: Literal["generator", "critic"]
    text: str
    scope: Literal["loop", "sample"] = "loop"


class HintCreateRequest(BaseModel):
    """运行中新增一条人工提示词。"""

    agent: Literal["generator", "critic"]
    text: str
    scope: Literal["loop", "sample"] = "loop"


# ---------------------------------------------------------------------------
# 总览（屏①）
# ---------------------------------------------------------------------------


class LoopSummary(BaseModel):
    """总览里一个 loop 的摘要。"""

    loop_id: str
    bench_id: str
    sample_id: str
    model: str
    started_at: str | None = None
    finished_at: str | None = None
    n_traces: int = 0
    best_restoration: float | None = None
    last_restoration: float | None = None
    status: str = "unknown"  # running / awaiting_review / finished / error / unknown
    has_checkpoint: bool = False
    thumbnail: str | None = None  # 最新一张生成图的相对路径
    # best/last_restoration 是否按当前生效权重（含人工校准回灌）重算过：
    # 手动排序产生新权重后历史 loop 的分数随之更新（展示层重算，不动落盘数据）。
    rescored: bool = False


class SampleOverview(BaseModel):
    """总览里一道题（sample）的聚合。打分按 sample 组织，故待打分数在这里。"""

    sample_id: str
    product: str | None = None
    category: str | None = None
    loops: list[LoopSummary] = Field(default_factory=list)
    n_traces: int = 0  # 该 sample 下所有 trace 总数
    pending: int = 0  # 待打分数（未提交排序的 trace 数）


class BenchOverview(BaseModel):
    bench_id: str
    description: str | None = None
    samples: list[SampleOverview] = Field(default_factory=list)
    general_experience_count: int = 0  # 跨 loop 通用经验条数（总览入口 badge；0=无）


class OverviewResponse(BaseModel):
    benches: list[BenchOverview] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# loop / trace 详情（屏②③）
# ---------------------------------------------------------------------------


class DimensionScoreOut(BaseModel):
    dim: str
    scoring_type: str
    value: float
    raw: str | None = None
    # 二分维度逐项判定（全量，含通过项）：{id, passed, reason}。
    # Critic 对每个 checklist 项都给理由（哪怕 passed=True），前端逐项展示 ✓/✗ + reason，
    # 避免「全过的维度零说明」——旧版只透出 failed_items，通过项的理由被吞掉。
    items: list[dict] = Field(default_factory=list)
    # 兼容旧字段：仅失败项 {id, reason}（现为 items 的 passed=False 子集）。
    failed_items: list[dict] = Field(default_factory=list)


class VerdictOut(BaseModel):
    restoration: float
    # restoration 是否按当前生效权重重算（trajectory 冻结值 ≠ 当前权重时 True）。
    rescored: bool = False
    # 重算前的落盘冻结分（rescored=False 时为 None）
    restoration_original: float | None = None
    weights_used: dict[str, float]
    dimensions: list[DimensionScoreOut]


class TraceOut(BaseModel):
    """一个 trace（一轮）的完整展示信息。"""

    loop_id: str
    trace_id: str  # attempt_id
    round: int
    sample_id: str
    bench_id: str
    model: str
    ts: str | None = None
    test_variable: str | None = None
    baseline_ref: str | None = None
    gen_mode: str | None = None
    prompt: str | None = None
    size: str | None = None
    output_image_refs: list[str] = Field(default_factory=list)
    reference_image_refs: list[str] = Field(default_factory=list)
    verdict: VerdictOut | None = None
    lesson_ref: str | None = None  # 指向 conclusions.json
    delta_note: str | None = None  # 本轮相对上轮的改动说明
    human_rank: float | None = None  # 该 trace 的人工排序值（来自 human_scores）


class CriticEvidenceOut(BaseModel):
    """经验结论的 Critic 验证证据（前后对比）。"""

    tested_round: int
    before: dict
    after: dict
    verdict_delta: str


class ConclusionOut(BaseModel):
    """经验知识库里的一条沉淀结论（Critic 驱动验证）。"""

    id: str
    dim: str
    finding: str
    change: str
    status: str  # pending / verified_effective / ineffective
    critic_evidence: CriticEvidenceOut | None = None
    lesson: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_round: int
    verified_round: int | None = None


class LoopDetail(BaseModel):
    loop_id: str
    bench_id: str
    sample_id: str
    model: str
    started_at: str | None = None
    finished_at: str | None = None
    note: str | None = None
    status: str = "unknown"
    round: int | None = None  # 当前轮（运行中时）
    interrupt_payload: dict | None = None  # 等审批时给前端展示的 payload
    last_error: str | None = None  # error 时的错误信息（供前端展示）
    traces: list[TraceOut] = Field(default_factory=list)
    conclusions: list[ConclusionOut] = Field(default_factory=list)  # 经验知识库
    target_image: str | None = None  # benchmark target 图路径
    target_md: str | None = None  # target 说明
    hints: list[HintOut] = Field(default_factory=list)  # 当前生效的人工提示词（loop+sample 合并）


# ---------------------------------------------------------------------------
# 排序（屏④）
# ---------------------------------------------------------------------------


class RankItem(BaseModel):
    trace_id: str  # attempt_id
    rank: float  # 越大越好


class RankSubmission(BaseModel):
    """人工提交的排序结果。trace_id = attempt_id；rank 越大越好。"""

    ranks: list[RankItem]
    note: str | None = None


class CalibrationStatusOut(BaseModel):
    bench_id: str
    sample_id: str
    state: str  # idle / running / done / error / insufficient
    message: str | None = None
    weights: dict[str, float] | None = None
    prior_weights: dict[str, float] | None = None
    pairwise_accuracy: float | None = None
    n_traces: int | None = None
    n_pairs: int | None = None
    loss: float | None = None
    converged: bool | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# 通用经验（跨 loop 蒸馏）
# ---------------------------------------------------------------------------


class DistilledLessonOut(BaseModel):
    """一条跨 loop 蒸馏出的通用经验（前端展示用）。"""

    id: str = ""
    dim: str
    insight: str
    dos: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    category: str = ""
    status: str = "active"
    applies_when: str = "always"
    successor_id: str = ""
    retire_reason: str = ""


class GeneralExperienceOut(BaseModel):
    """某 bench 的跨 loop 通用经验（general.json 对外模型，自描述）。"""

    bench_id: str
    summary: str = ""
    lessons: list[DistilledLessonOut] = Field(default_factory=list)
    source_runs: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    scene: str = ""
    dimensions: list[str] = Field(default_factory=list)
    bench_description: str = ""
    categories: list[str] = Field(default_factory=list)


class LessonEdit(BaseModel):
    """人工编辑一条 lesson 的可改字段（PATCH）。None 字段保持不变。"""

    insight: str | None = None
    dos: list[str] | None = None
    donts: list[str] | None = None
    category: str | None = None
    applies_when: str | None = None
    confidence: float | None = None


class LessonRefute(BaseModel):
    """人工标无效（refute）一条 lesson 的理由。"""

    reason: str = ""


class DistillStatusOut(BaseModel):
    """蒸馏任务状态（前端轮询）。"""

    bench_id: str
    state: str  # idle / running / done / error / no_runs
    message: str | None = None
    n_lessons: int | None = None
    updated_at: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# loop 控制
# ---------------------------------------------------------------------------


class LoopStartRequest(BaseModel):
    """启动一个新 loop。"""

    bench_id: str
    sample_id: str
    model: str | None = None  # 为空用 settings 默认
    note: str | None = None
    rounds: int | None = None  # 自动连跑轮数；None/1=单轮跑首停审批，>1=后台连跑不等审批
    hints: list[HintIn] | None = None  # 启动时附加的人工提示词（按各自 scope 落盘）


class LoopControlRequest(BaseModel):
    """继续/停止一个 loop。"""

    decision: str = "continue"  # continue / stop / 任意调整方向文本


class MemoryWriteRequest(BaseModel):
    """编辑 generator 本 loop 记忆（系统托管，按 loop 隔离）的请求体。"""

    content: str = ""  # 记忆原文（markdown）；空串 → 清空


# ---------------------------------------------------------------------------
# benchmark 管理（新增 benchmark 表单）
# ---------------------------------------------------------------------------


class DimensionIn(BaseModel):
    """新增 benchmark 表单里的一条维度定义（对应 manifest 的 score_dimensions[]）。"""

    dim: str
    desc: str | None = None
    weight_init: float = Field(default=0.0, ge=0.0, le=1.0)
    ref_needed: bool = False
    scoring_type: str = "binary"  # binary / continuous
    check_items: list[str] | None = None  # 仅二分维度（manifest 的 check_items，二分 checklist 真源）
    rubric_ref: str | None = None  # 仅连续维度


class SampleIn(BaseModel):
    """新增 benchmark 表单里的一道考题（对应 manifest 的 samples[] 一条）。"""

    sample_id: str
    product: str | None = None
    category: str | None = None
    difficulty_note: str | None = None
