"""经验蒸馏 deepagent 的跨 loop 聚合工具（闭包捕获 run_dirs + bench）。

这些工具让 agent 能跨多个 run 看到模式：哪些改动跨 run 有效、哪些无效。数据来自每 run 的
`trajectory.jsonl`（每轮分数 + delta_note）与 `lessons/conclusions.json`（in-loop Summarizer
机器验证的 effective/ineffective 结论）。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from ...data.trajectory import TrajectoryReader
from ...memory.knowledge import load_conclusions


def _find_run(run_dirs: list[Path], run_id: str) -> Path | None:
    return next((rd for rd in run_dirs if rd.name == run_id), None)


def _read_records(run_dir: Path):
    tp = run_dir / "trajectory.jsonl"
    if not tp.exists():
        return []
    return list(TrajectoryReader(tp).iter_records())


def make_list_runs_tool(run_dirs: list[Path]) -> BaseTool:
    @tool
    def list_runs() -> str:
        """列出本次分析的全部 run：run_id / sample_id / model / 轮数 / 最好还原度。"""
        lines = []
        for rd in run_dirs:
            recs = _read_records(rd)
            if not recs:
                lines.append(f"- {rd.name}: (无 trajectory)")
                continue
            res = [r.verdict.restoration for r in recs if r.verdict]
            best = f"{max(res):.3f}" if res else "?"
            lines.append(
                f"- {rd.name}: sample={recs[0].sample_id}, model={recs[0].model}, "
                f"轮数={len(recs)}, 最好还原度={best}"
            )
        return "\n".join(lines) or "(无 run)"

    return list_runs


def make_query_run_tool(run_dirs: list[Path]) -> BaseTool:
    @tool
    def query_run(run_id: str) -> str:
        """查某个 run 的逐轮轨迹：每轮 round / 还原度 / 各维度分 / delta_note（本轮改了什么）。

        Args:
            run_id: run 目录名。
        """
        rd = _find_run(run_dirs, run_id)
        if rd is None:
            return f"(未知 run_id: {run_id})"
        recs = _read_records(rd)
        if not recs:
            return f"({run_id} 无 trajectory)"
        lines = []
        for r in recs:
            dims = ", ".join(f"{d.dim}={d.value:.2f}" for d in r.verdict.dimensions) if r.verdict else "?"
            res = f"{r.verdict.restoration:.3f}" if r.verdict else "?"
            delta = r.delta_note or "(无改动说明)"
            lines.append(f"round {r.round}: 还原度={res}; [{dims}]; delta: {delta}")
        return "\n".join(lines)

    return query_run


def make_query_dim_history_tool(run_dirs: list[Path]) -> BaseTool:
    @tool
    def query_dim_history(dim: str) -> str:
        """跨 run 聚合某维度的改动史：列出每个 run 里该 dim 的 change + 验证状态(effective/ineffective)
        + 前后分差(verdict_delta) + lesson。这是归纳「该维度怎么做有效/无效」的关键视图。

        Args:
            dim: 维度名（如 'artifact_defect'）。
        """
        lines = [f"维度 {dim} 跨 run 改动史："]
        found = 0
        for rd in run_dirs:
            kb = load_conclusions(rd)
            for c in kb.conclusions:
                if c.dim != dim:
                    continue
                found += 1
                delta = c.critic_evidence.verdict_delta if c.critic_evidence else "?"
                lines.append(
                    f"- [{rd.name}] 「{c.change}」→ {c.status}（{delta}）"
                    f" lesson: {c.lesson or '(无)'}"
                )
        if not found:
            lines.append("(该维度暂无已验证结论)")
        return "\n".join(lines)

    return query_dim_history


def make_query_conclusions_tool(run_dirs: list[Path]) -> BaseTool:
    @tool
    def query_conclusions(run_id: str, status: str = "") -> str:
        """查某个 run 的已验证结论（in-loop Summarizer 产出的 effective/ineffective）。

        Args:
            run_id: run 目录名。
            status: 可选过滤：'verified_effective' / 'ineffective' / 'pending'；留空看全部。
        """
        rd = _find_run(run_dirs, run_id)
        if rd is None:
            return f"(未知 run_id: {run_id})"
        kb = load_conclusions(rd)
        cs = [c for c in kb.conclusions if (not status or c.status == status)]
        if not cs:
            return f"({run_id} 无{status or ''}结论)"
        lines = [f"{run_id} 的结论："]
        for c in cs:
            delta = c.critic_evidence.verdict_delta if c.critic_evidence else "?"
            lines.append(
                f"- [{c.dim}] 「{c.change}」→ {c.status}（{delta}） lesson: {c.lesson or '(无)'}"
            )
        return "\n".join(lines)

    return query_conclusions


def make_experience_tools(run_dirs: list[Path], bench) -> list[BaseTool]:
    """组装经验蒸馏工具集。bench 用于在 system_prompt 侧给维度清单（工具本身不直接需要）。"""
    return [
        make_list_runs_tool(run_dirs),
        make_query_run_tool(run_dirs),
        make_query_dim_history_tool(run_dirs),
        make_query_conclusions_tool(run_dirs),
    ]


__all__ = [
    "make_experience_tools",
    "make_list_runs_tool",
    "make_query_conclusions_tool",
    "make_query_dim_history_tool",
    "make_query_run_tool",
]
