"""权重与还原度的数学：features 向量 + 加权还原度 + 校准权重加载。

核心公式（ARCH §2.6.1 / EVALUATION §4.3）：

    features[dim] ∈ [0,1]      # 二分维度=通过率, 连续维度=LLM 0-1 分
    restoration = Σ(wᵢ × features[i])

权重来源优先级（Step 6 web 台加入 sample 级校准）：
    1. `runs/<run_id>/calibrated_weights.json`（单 loop 校准产物，若存在）
    2. `calibration/<bench_id>/<sample_id>_weights.json`（sample 级跨 loop 校准产物）
    3. benchmark manifest 的 `weight_init`（先验）

本模块是**纯函数**，无网络、无 LLM 依赖，可完全单测。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..memory.schema import Benchmark, CriticVerdict, DimensionScore


def compute_features(verdict: CriticVerdict) -> dict[str, float]:
    """从 CriticVerdict 提取 features 向量（逐维度 ∈[0,1]）。

    二分维度的 features 已在 `DimensionScore.value` 里算成了通过率；
    连续维度的 features 就是 LLM 的 0-1 分。故这里只是取 value。
    """
    return {d.dim: float(d.value) for d in verdict.dimensions}


def weighted_restoration(features: dict[str, float], weights: dict[str, float]) -> float:
    """还原度 = Σ(wᵢ × features[i])。

    只对同时在 weights 与 features 里出现的维度求和（校准可能丢弃某些维度）。
    若 weights 之和偏离 1 较多，按 weights 自身归一化后再加权，避免量纲漂移。
    """
    if not weights:
        return 0.0
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    acc = 0.0
    for dim, w in weights.items():
        if dim in features:
            acc += (w / total_w) * features[dim]
    return acc


def init_weights(bench: Benchmark) -> dict[str, float]:
    """benchmark 的初始先验权重（归一化，和为 1）。"""
    return bench.init_weights()


def load_weights(
    bench: Benchmark, *, run_dir: Path | None = None, sample_id: str | None = None
) -> dict[str, float]:
    """加载生效权重：按优先级链找校准文件，否则用 benchmark 先验。

    优先级（高→低）：
        1. `run_dir/calibrated_weights.json`（单 loop 校准产物）
        2. `data/calibration/<bench_id>/<sample_id>_weights.json`（sample 级跨 loop 校准产物）
        3. benchmark manifest 的 weight_init 先验

    Args:
        bench: benchmark（提供 weight_init 先验 + 合法维度名）。
        run_dir: 某次 run 目录；若其下有 calibrated_weights.json 则最高优先。
        sample_id: 若给，且 run_dir 没有校准文件，则查 sample 级校准文件。
    """
    prior = init_weights(bench)
    if run_dir is not None:
        w = _load_calib_file(run_dir / "calibrated_weights.json", prior)
        if w is not None:
            return w
    if sample_id is not None:
        # sample 级文件：data_root/calibration/<bench_id>/<sample_id>_weights.json
        # run_dir 存在时从它推断 data_root；否则回退 _DEFAULT_DATA_ROOT。
        data_root = run_dir.parents[1] if run_dir is not None else None
        if data_root is None:
            from ...config import get_settings  # 延迟导入避免循环

            data_root = get_settings().data_root
        p = Path(data_root) / "calibration" / bench.bench_id / f"{sample_id}_weights.json"
        w = _load_calib_file(p, prior)
        if w is not None:
            return w
    return prior


def _load_calib_file(path: Path, prior: dict[str, float]) -> dict[str, float] | None:
    """从单个校准文件加载权重。无文件/无效则返回 None（让调用方继续找下一级）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("weights", data) if isinstance(data, dict) else {}
    valid_dims = set(prior)
    calibrated: dict[str, float] = {}
    for dim, w in raw.items():
        if dim in valid_dims and isinstance(w, int | float) and w >= 0:
            calibrated[dim] = float(w)
    if not calibrated:
        return None
    return {**prior, **calibrated}


def apply_weights(verdict: CriticVerdict, weights: dict[str, float]) -> CriticVerdict:
    """用给定权重重算 verdict 的 restoration（返回新对象）。

    用于：用校准后的权重对历史 trace 重打总分（策略对比/排序）。
    """
    features = compute_features(verdict)
    new_restoration = weighted_restoration(features, weights)
    return verdict.model_copy(update={"restoration": new_restoration, "weights_used": dict(weights)})


def recompute_restoration(
    dimensions: list[DimensionScore], weights: dict[str, float]
) -> float:
    """给定维度分列表 + 权重，直接算还原度（不依赖完整 verdict）。"""
    features = {d.dim: d.value for d in dimensions}
    return weighted_restoration(features, weights)


__all__ = [
    "apply_weights",
    "compute_features",
    "init_weights",
    "load_weights",
    "recompute_restoration",
    "weighted_restoration",
]
