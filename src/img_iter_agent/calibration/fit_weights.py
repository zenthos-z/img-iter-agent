"""排序校准：用人工对 trace 的排序拟合维度权重 w。

核心机制（ARCH §2.6.3 / EVALUATION §4）：
  人猜不准"材质 72 分"，但能可靠判断"trace A 比 trace B 好"——排序是人的强项。
  校准即：找权重 w（约束 Σw=1, w≥0）使 `w·features` 给出的排序**尽量吻合人工排序**。

  为什么修正偏差：若某连续维度 LLM 系统性偏高，拟合会把该维度权重压低，
  使加权排序仍贴合人工判断 → 天然吸收 LLM 连续分偏差。

算法：pairwise hinge loss（rank SVM 思想）。
  对人工排序里每对 (i 比 j 好)：希望 w·features[i] - w·features[j] ≥ margin
  loss = Σ_pairs max(0, margin - (w·fi - w·fj))  （成对 hinge）
  约束：Σw=1, w≥0（用 scipy.optimize SLSQP + simplex 约束）。
  相比直接回归（量化×量化，误差叠加），排序拟合只需人擅长的排序信号。

输入：每个 trace 的 features 向量（从 trajectory verdict 取）+ 人工排序。
输出：calibrated_weights.json，回灌 Critic 算还原度总分的权重。

注：sklearn 无现成 rank 权重拟合器（label_ranking_loss 只是评估指标），
故用 scipy.optimize 直接优化 pairwise loss——规模小（6 维权重）、约束精确可控。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..data.weights import init_weights
from ..memory.schema import Benchmark, CriticVerdict


@dataclass
class RankedTrace:
    """一条带人工排序的 trace（排序值越大越好；同分可并列）。"""

    trace_id: str  # 标识（attempt_id 或自定义）
    features: dict[str, float]  # 逐维度 ∈[0,1]
    human_rank: float  # 人工给的排序值，越大越好（可并列）


@dataclass
class CalibrationResult:
    """校准结果。"""

    weights: dict[str, float]  # 拟合出的权重（Σ=1, ≥0）
    prior_weights: dict[str, float]  # 校准前的先验
    pairwise_accuracy: float  # 排序吻合度：人工好对的 pair 中，w·features 也排对的占比
    margin: float
    n_traces: int
    n_pairs: int
    converged: bool
    loss: float


def _features_matrix(traces: list[RankedTrace], dims: list[str]) -> np.ndarray:
    """(n_traces, n_dims) 的特征矩阵。缺失维度填 0。"""
    return np.array([[t.features.get(d, 0.0) for d in dims] for t in traces])


def _build_pairs(traces: list[RankedTrace]) -> list[tuple[int, int]]:
    """从人工排序构造"好 vs 差"的 pair 下标（i 比 j 好，即 rank[i] > rank[j]）。"""
    pairs = []
    for i in range(len(traces)):
        for j in range(len(traces)):
            if i != j and traces[i].human_rank > traces[j].human_rank:
                pairs.append((i, j))
    return pairs


def _pairwise_loss(w: np.ndarray, X: np.ndarray, pairs: list[tuple[int, int]],
                   margin: float, alpha_reg: float) -> float:
    """pairwise hinge loss + 轻微 L2 正则（防止权重塌缩到单一维度）。"""
    scores = X @ w  # (n,)
    total = 0.0
    for i, j in pairs:
        diff = scores[i] - scores[j]  # 希望 ≥ margin
        total += max(0.0, margin - diff)
    reg = alpha_reg * float(np.sum(w ** 2))
    return total + reg


def fit_weights(
    traces: list[RankedTrace],
    bench: Benchmark,
    *,
    margin: float = 0.05,
    alpha_reg: float = 0.01,
) -> CalibrationResult:
    """拟合维度权重：让 w·features 的排序吻合人工排序。

    Args:
        traces: 带 human_rank 的 trace 列表（至少 2 条且 rank 有差异才有意义）。
        bench: benchmark（提供维度名 + 先验权重 + 维度数）。
        margin: pairwise 的安全间隔。
        alpha_reg: L2 正则强度（小，防止单维度塌缩）。
    """
    dims = [d.dim for d in bench.score_dimensions]
    prior = init_weights(bench)
    n_dims = len(dims)

    if len(traces) < 2:
        # 数据不足，直接返回先验
        return CalibrationResult(
            weights=dict(prior), prior_weights=dict(prior),
            pairwise_accuracy=1.0, margin=margin, n_traces=len(traces),
            n_pairs=0, converged=True, loss=0.0,
        )

    X = _features_matrix(traces, dims)
    pairs = _build_pairs(traces)
    if not pairs:
        return CalibrationResult(
            weights=dict(prior), prior_weights=dict(prior),
            pairwise_accuracy=1.0, margin=margin, n_traces=len(traces),
            n_pairs=0, converged=True, loss=0.0,
        )

    # 约束：Σw=1, w≥0
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_dims
    x0 = np.array([prior[d] for d in dims])  # 从先验起步
    x0 = x0 / x0.sum() if x0.sum() > 0 else np.ones(n_dims) / n_dims

    res = minimize(
        _pairwise_loss, x0, args=(X, pairs, margin, alpha_reg),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    w = np.clip(res.x, 0.0, None)
    if w.sum() > 0:
        w = w / w.sum()  # 归一化兜底
    weights = {d: float(w[i]) for i, d in enumerate(dims)}

    # 排序吻合度
    acc = _pairwise_accuracy(w, X, pairs)

    return CalibrationResult(
        weights=weights, prior_weights=dict(prior),
        pairwise_accuracy=acc, margin=margin, n_traces=len(traces),
        n_pairs=len(pairs), converged=bool(res.success), loss=float(res.fun),
    )


def _pairwise_accuracy(w: np.ndarray, X: np.ndarray,
                       pairs: list[tuple[int, int]]) -> float:
    """排序吻合度：人工好对中，w·features 也排对的比例。"""
    if not pairs:
        return 1.0
    scores = X @ w
    correct = sum(1 for i, j in pairs if scores[i] > scores[j])
    return correct / len(pairs)


def save_calibrated_weights(result: CalibrationResult, run_dir: Path) -> Path:
    """把校准结果写成 calibrated_weights.json（load_weights 会优先读它）。"""
    path = run_dir / "calibrated_weights.json"
    payload = {
        "weights": result.weights,
        "prior_weights": result.prior_weights,
        "pairwise_accuracy": result.pairwise_accuracy,
        "margin": result.margin,
        "n_traces": result.n_traces,
        "n_pairs": result.n_pairs,
        "converged": result.converged,
        "loss": result.loss,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---- 从 trajectory 提取 RankedTrace（人工填 rank 后用）----

def traces_from_verdicts(
    verdicts: list[CriticVerdict], human_ranks: list[float],
    *, trace_ids: list[str] | None = None,
) -> list[RankedTrace]:
    """从 CriticVerdict 列表 + 人工排序构造 RankedTrace。

    human_ranks[i] 对应 verdicts[i] 的人工排序值（越大越好）。
    """
    if len(verdicts) != len(human_ranks):
        raise ValueError("verdicts 与 human_ranks 长度不一致")
    ids = trace_ids or [v.sample_id for v in verdicts]
    return [
        RankedTrace(trace_id=ids[i], features=v.features, human_rank=human_ranks[i])
        for i, v in enumerate(verdicts)
    ]


__all__ = [
    "CalibrationResult",
    "RankedTrace",
    "fit_weights",
    "save_calibrated_weights",
    "traces_from_verdicts",
]
