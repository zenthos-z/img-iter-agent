"""校准报告：把 CalibrationResult 渲染成人可读 MD + 权重变化图。

报告内容：
  - 排序吻合度、trace 数、pair 数
  - 各维度权重变化（先验 → 校准后），标出哪些维度被压低/抬升
  - 解读：哪些维度 LLM 系统性偏高（被压低）、哪些更受人工重视（被抬升）
"""

from __future__ import annotations

from pathlib import Path

from .fit_weights import CalibrationResult


def render_report_md(result: CalibrationResult, *, bench_label: str = "") -> str:
    """渲染校准报告 MD。"""
    lines = [f"# 评分校准报告 {bench_label}".rstrip(), ""]

    # 概况
    lines += [
        "## 概况",
        f"- 排序吻合度: **{result.pairwise_accuracy:.1%}**（人工好对中 w·features 也排对的比例）",
        f"- trace 数: {result.n_traces} | pair 数: {result.n_pairs}",
        f"- margin: {result.margin} | 收敛: {'是' if result.converged else '否'} | loss: {result.loss:.4f}",
        "",
    ]

    # 权重变化
    lines += ["## 维度权重变化（先验 → 校准后）", ""]
    lines += ["| 维度 | 先验 | 校准后 | 变化 | 趋势 |", "|---|---|---|---|---|"]
    all_dims = sorted(set(result.prior_weights) | set(result.weights))
    for d in all_dims:
        p = result.prior_weights.get(d, 0.0)
        c = result.weights.get(d, 0.0)
        delta = c - p
        trend = "↑ 抬升" if delta > 0.01 else ("↓ 压低" if delta < -0.01 else "→ 持平")
        lines.append(f"| {d} | {p:.3f} | {c:.3f} | {delta:+.3f} | {trend} |")
    lines.append("")

    # 解读
    lines += ["## 解读", ""]
    raised = [(d, result.weights[d] - result.prior_weights.get(d, 0))
              for d in result.weights
              if result.weights[d] - result.prior_weights.get(d, 0) > 0.01]
    lowered = [(d, result.prior_weights.get(d, 0) - result.weights[d])
               for d in result.weights
               if result.prior_weights.get(d, 0) - result.weights[d] > 0.01]
    if lowered:
        names = "、".join(d for d, _ in sorted(lowered, key=lambda x: -x[1]))
        lines.append(f"- **被压低的维度**（{names}）：人工排序里这些维度的高分没带来更高整体评价，"
                     "可能 LLM 在这些维度系统性偏高 → 拟合压低其权重以吸收偏差。")
    if raised:
        names = "、".join(d for d, _ in sorted(raised, key=lambda x: -x[1]))
        lines.append(f"- **被抬升的维度**（{names}）：这些维度对人工整体评价影响更大，权重被提高。")
    if not raised and not lowered:
        lines.append("- 各维度权重基本未变（先验已较合理，或排序信号不足）。")
    lines.append("")
    lines.append("> 校准后的权重已存 `calibrated_weights.json`，下一轮 Critic 算还原度总分时用它。")

    return "\n".join(lines)


def write_report(result: CalibrationResult, run_dir: Path, *,
                 bench_label: str = "") -> Path:
    """写报告 MD 到 run 目录的 analyses 子目录。"""
    reports_dir = run_dir / "analyses"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "weight_calibration_report.md"
    path.write_text(render_report_md(result, bench_label=bench_label), encoding="utf-8")
    return path


__all__ = ["render_report_md", "write_report"]
