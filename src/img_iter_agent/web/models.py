"""打分台的 API 数据模型（pydantic v2）。

这些是给前端的响应/请求模型，与内部 schema 解耦：内部用 AttemptRecord 等，
对外用这些扁平、前端友好的模型。转换在 services 层完成。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
    failed_items: list[dict] = Field(default_factory=list)  # {id, reason}


class VerdictOut(BaseModel):
    restoration: float
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
    traces: list[TraceOut] = Field(default_factory=list)
    conclusions: list[ConclusionOut] = Field(default_factory=list)  # 经验知识库
    target_image: str | None = None  # benchmark target 图路径
    target_md: str | None = None  # target 说明


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
# loop 控制
# ---------------------------------------------------------------------------


class LoopStartRequest(BaseModel):
    """启动一个新 loop。"""

    bench_id: str
    sample_id: str
    model: str | None = None  # 为空用 settings 默认
    note: str | None = None


class LoopControlRequest(BaseModel):
    """继续/停止一个 loop。"""

    decision: str = "continue"  # continue / stop / 任意调整方向文本
