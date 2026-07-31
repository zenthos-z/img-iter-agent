"""策略对比：跨 run/样本/轮次读 trajectory，对比还原度与各维度分。

只读（不污染原始）。用 pandas 汇总，可选 matplotlib 出图。
支撑问题：不同样本/模型/迭代轮次的还原度如何？哪些维度稳定、哪些波动？
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.trajectory import TrajectoryReader


def load_trajectories(run_dirs: list[Path]) -> pd.DataFrame:
    """把多个 run 的 trajectory 读成一个 DataFrame。

    列: run_id, sample_id, round, model, test_variable, restoration,
        各维度分（consistency/product_structure/...）, prompt, baseline_ref。
    """
    rows = []
    for rd in run_dirs:
        tp = Path(rd) / "trajectory.jsonl"
        if not tp.exists():
            continue
        run_id = Path(rd).name
        for rec in TrajectoryReader(tp).iter_records():
            row = {
                "run_id": run_id,
                "sample_id": rec.sample_id,
                "round": rec.round,
                "model": rec.model,
                "test_variable": rec.test_variable,
                "baseline_ref": rec.baseline_ref,
                "restoration": rec.verdict.restoration if rec.verdict else None,
                "prompt": (rec.prompt or "")[:80],
            }
            if rec.verdict:
                for d in rec.verdict.dimensions:
                    row[d.dim] = d.value
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """按样本汇总还原度（均值/标准差/轮次数）。"""
    return (df.groupby("sample_id")["restoration"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4))


def summarize_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """按 (sample_id, round) 看迭代轨迹。"""
    return (df.groupby(["sample_id", "round"])["restoration"]
            .mean().round(4).unstack("round"))


def plot_restoration_by_round(df: pd.DataFrame, out_path: Path) -> Path:
    """画各样本随轮次的还原度折线图。"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="round", columns="sample_id", values="restoration", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(ax=ax, marker="o")
    ax.set_xlabel("轮次")
    ax.set_ylabel("还原度")
    ax.set_title("各样本随迭代轮次的还原度变化")
    ax.set_ylim(0, 1)
    ax.legend(title="样本")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


__all__ = [
    "load_trajectories",
    "plot_restoration_by_round",
    "summarize_by_round",
    "summarize_by_sample",
]
