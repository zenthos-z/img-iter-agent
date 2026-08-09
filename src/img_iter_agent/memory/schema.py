"""数据 schema（Pydantic v2）。

这是整个系统的数据契约。关键设计点见 `docs/ARCHITECTURE.md §2.6` 与
`docs/EVALUATION.md §4`：

- **混合评分**：维度分两类——
    * `binary`：按 checklist 逐项 ✓/✗（`CriticItemJudgment`），维度特征 = 通过率
    * `continuous`：LLM 给 0-1 分（承认有偏差，由排序校准的权重吸收）
  两类都产出 `∈[0,1]` 的特征值，组成 `features` 向量。

- **还原度 = w · features**：权重 `w` 由 benchmark 的 `weight_init` 先验给出，
  后续被排序校准闭环（learning-to-rank）更新，存 `runs/<id>/calibrated_weights.json`。

- **content_spec 的双形态 checklist**：
    * 二分维度 → `list[CheckItem]`
    * 连续维度 → `ContinuousRubric{"_scoring", "points"}`
  这正对应磁盘上 `content_spec.json` 的真实结构。

- **AttemptRecord**：trajectory.jsonl 的一行。字段对齐 ARCH §3.3.2，
  含控制变量法的 `test_variable` / `baseline_ref`，以及各类文件链接（`*_ref`）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# benchmark / content_spec
# ---------------------------------------------------------------------------

ScoringType = Literal["binary", "continuous"]


class CheckItem(BaseModel):
    """二分维度里的一条 checklist 项。"""

    model_config = ConfigDict(extra="ignore")
    id: str
    check: str
    anchor: str | None = None  # 可选：如 "对照 target"


class ContinuousRubric(BaseModel):
    """连续维度里的评分点集（content_spec.json 中 material_texture / color_accuracy 的形态）。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    scoring: str | None = Field(default=None, alias="_scoring")
    points: list[str] = Field(default_factory=list)


# checklist 的两种合法形态：二分项列表 / 连续 rubric 对象。
ChecklistValue = list[CheckItem] | ContinuousRubric


class ScoreDimension(BaseModel):
    """benchmark manifest.json 里的一条维度定义。"""

    model_config = ConfigDict(extra="ignore")

    dim: str
    desc: str | None = None
    weight_init: float = Field(default=0.0, ge=0.0, le=1.0)
    ref_needed: bool = False
    scoring_type: ScoringType
    check_items: list[str] | None = None  # 仅二分维度在 manifest 里有
    rubric_ref: str | None = None
    note: str | None = None


class TaskSpec(BaseModel):
    """benchmark 的任务定义（如三视图白底）。"""

    model_config = ConfigDict(extra="ignore")
    type: str
    views: list[str] = Field(default_factory=list)


class SampleRef(BaseModel):
    """manifest.json samples[] 里的一条样例引用。"""

    model_config = ConfigDict(extra="ignore")
    sample_id: str
    product: str | None = None
    category: str | None = None
    target: str  # 相对 bench 目录的路径
    difficulty_note: str | None = None


class Benchmark(BaseModel):
    """整个 benchmark（manifest.json）。"""

    model_config = ConfigDict(extra="ignore")

    bench_id: str
    version: str | None = None
    scene: str | None = None
    description: str | None = None
    scoring_method: str | None = None
    scoring_note: str | None = None

    score_dimensions: list[ScoreDimension]
    weight_note: str | None = None

    comparative_dims: list[str] = Field(default_factory=list)
    task: TaskSpec | None = None
    samples: list[SampleRef] = Field(default_factory=list)

    @property
    def dim_by_name(self) -> dict[str, ScoreDimension]:
        return {d.dim: d for d in self.score_dimensions}

    def init_weights(self) -> dict[str, float]:
        """benchmark 的初始先验权重（归一化后）。"""
        raw = {d.dim: d.weight_init for d in self.score_dimensions}
        total = sum(raw.values())
        if total <= 0:
            n = len(raw) or 1
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}


class ContentSpecTask(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mode: str
    input_assets: list[str] = Field(default_factory=list)
    instruction: str | None = None
    output: dict = Field(default_factory=dict)  # views/background/size 等（自由格式，按需读）
    article_topic: str | None = None  # 风格迁移场景：文章核心概念（诊断/展示用；评判从 checklist 读）


class ContentSpec(BaseModel):
    """一道考题：target 图 + 任务 + 约束 + 各维度 checklist。

    `checklist` 的键是维度名，值有两种形态（见 `ChecklistValue`）：
    二分维度→`list[CheckItem]`，连续维度→`ContinuousRubric`。
    """

    model_config = ConfigDict(extra="ignore")

    sample_id: str
    product: str | None = None
    category: str | None = None
    task: ContentSpecTask | None = None
    constraints: dict = Field(default_factory=dict)  # must_keep/may_change/must_avoid/forbidden_motifs（generator 注入用）
    checklist: dict[str, ChecklistValue] = Field(default_factory=dict)
    anchor_for: list[str] = Field(default_factory=list)

    def binary_dims(self) -> list[str]:
        """该考题里以二分 checklist 形态出现的维度名。"""
        return [k for k, v in self.checklist.items() if isinstance(v, list)]

    def continuous_dims(self) -> list[str]:
        """该考题里以连续 rubric 形态出现的维度名。"""
        return [k for k, v in self.checklist.items() if isinstance(v, ContinuousRubric)]


# ---------------------------------------------------------------------------
# Critic 评分产物
# ---------------------------------------------------------------------------


class CriticItemJudgment(BaseModel):
    """对一条二分 checklist 项的判定：✓/✗ + 一句理由。"""

    id: str
    passed: bool
    reason: str = ""


class DimensionScore(BaseModel):
    """一个维度的评分产物。

    - 二分维度：`value` = 通过率；`items` = 逐项判定；`raw` 留空。
    - 连续维度：`value` = LLM 0-1 分；`raw` 可放原始文本/理由；`items` 留空。
    """

    dim: str
    scoring_type: ScoringType
    value: float = Field(ge=0.0, le=1.0)
    items: list[CriticItemJudgment] | None = None
    raw: str | None = None  # 连续维度的 LLM 原始输出/理由


class CriticVerdict(BaseModel):
    """对单个 trace（一组生成图）的完整评分。

    `features` 是逐维度的 `∈[0,1]` 特征向量（二分=通过率，连续=LLM 分）；
    `restoration` = `Σ(wᵢ × features[i])`，用 `weights_used` 的权重算出。
    """

    sample_id: str
    dimensions: list[DimensionScore]
    weights_used: dict[str, float]
    restoration: float

    @property
    def features(self) -> dict[str, float]:
        return {d.dim: d.value for d in self.dimensions}

    def item_judgments(self, dim: str) -> list[CriticItemJudgment]:
        """某二分维度的逐项判定（连续维度返回空列表）。"""
        for d in self.dimensions:
            if d.dim == dim:
                return d.items or []
        return []


# ---------------------------------------------------------------------------
# trajectory 一行：AttemptRecord
# ---------------------------------------------------------------------------

# 控制变量法的五个可变维度（模型本身固定，不在此列）——见 ARCH §2.5.2
TestVariable = Literal["prompt", "reference_images", "size", "generation_mode", "model_params"]


class AttemptRecord(BaseModel):
    """trajectory.jsonl 的一行：一次完整尝试的自包含记录。

    设计目标：可脱离运行时独立加载、重放、做策略对比/校准（ARCH §3.5.3）。
    所有图片用**文件路径**（相对 run 目录），绝不用 base64（ADR-005）。
    """

    # --- 标识 ---
    attempt_id: str
    run_id: str
    round: int
    sample_id: str
    bench_id: str
    model: str  # 本闭环固定的模型（ADR-007：分开独立评测）

    # --- 时间 ---
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # --- 控制变量法（ARCH §2.5.2）---
    test_variable: TestVariable | None = None  # 本轮变了哪个维度
    baseline_ref: str | None = None  # 对照组 attempt_id（其余维度固定）

    # --- 生成输入（全部用文件路径/值，不用 base64）---
    gen_mode: str | None = None  # image_edit / multi_image_fusion / multiturn_edit ...
    prompt: str | None = None
    reference_image_refs: list[str] = Field(default_factory=list)  # 相对 run 目录
    conversation_history_refs: list[str] = Field(default_factory=list)
    size: str | None = None
    model_params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    # --- 生成产物 ---
    output_image_refs: list[str] = Field(default_factory=list)  # 三视图等生成图路径

    # --- 评分 ---
    verdict: CriticVerdict | None = None  # Critic 完整判定（含 features + restoration）

    # --- 迭代改动说明（Generator 产出：本轮相对上轮改了什么）---
    # 多策略基建：配合 test_variable 一起记录"该策略本轮具体改了什么"。
    # 当前只有 prompt 策略会填；未来 reference_images/size 策略扩展时也走这里。
    delta_note: str | None = None
    meaning: str | None = None  # 一句话图片含义解释（风格神韵迁移；Generator 产出，供 Critic 判概念表达）

    # --- 经验链接 ---
    # 指向该 sample 的经验知识库（lessons/conclusions.json），统一一份而非单轮 MD。
    lesson_ref: str | None = None

    def to_jsonl(self) -> str:
        """序列化成 trajectory.jsonl 的一行（单行 JSON，无换行）。"""
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# 经验知识库（结构化、Critic 驱动验证的沉淀结论）
# ---------------------------------------------------------------------------

# 经验验证状态：由 Critic 前后 verdict 驱动判定（update_status_on_evidence 见 knowledge.py）
ConclusionStatus = Literal["pending", "verified_effective", "ineffective"]


class CriticEvidence(BaseModel):
    """一条经验结论的 Critic 验证证据（前后对比）。

    Critic 是"改动有效性"的客观裁判：before/after 各记录该维度的分数、
    失败项、以及 Critic 给出的理由文本（reason 是可复用知识的一等内容）。
    """

    tested_round: int
    before: dict  # {value, failed: [id...], reason}
    after: dict  # 同上
    verdict_delta: str  # 人类可读的变化摘要，如 "A4: ✗→✓, 分 0.75→1.00"


class KnowledgeConclusion(BaseModel):
    """经验知识库里的一条沉淀结论。

    闭环语义：Generator 改了 `change`（delta_note）→ Critic 前后 verdict 对比 →
    判定 `status`（有效/无效/待验）→ 沉淀为可复用知识。
    """

    id: str
    dim: str  # 关联的评分维度
    finding: str  # 发现的问题（来自上轮 Critic 失败项/低分）
    change: str  # 本轮改动说明（delta_note）
    status: ConclusionStatus = "pending"
    critic_evidence: CriticEvidence | None = None  # 待验证时为空，验证后填入
    lesson: str | None = None  # 归纳的可复用结论（含 Critic reason）
    tags: list[str] = Field(default_factory=list)
    created_round: int  # 首次提出（改动）的轮次
    verified_round: int | None = None  # 验证完成的轮次
    # 升级标记（B 复发检测）：该 dim 连续失败 ≥ ESCALATION_THRESHOLD 轮时，最近一条结论置 True。
    # 与 status 正交（status 仍是 effective/ineffective/pending）—— escalated 是叠加的「已撞模型能力上限」标注，
    # 提示 generator 换根本思路 / 上报人工，而非继续 prompt 微调。
    escalated: bool = False


__all__ = [
    "AttemptRecord",
    "Benchmark",
    "CheckItem",
    "ChecklistValue",
    "ConclusionStatus",
    "ContentSpec",
    "ContentSpecTask",
    "ContinuousRubric",
    "CriticEvidence",
    "CriticItemJudgment",
    "CriticVerdict",
    "DimensionScore",
    "KnowledgeConclusion",
    "SampleRef",
    "ScoreDimension",
    "ScoringType",
    "TaskSpec",
    "TestVariable",
]
