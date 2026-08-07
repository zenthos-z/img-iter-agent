"""Critic deepagent 的工具集（每轮由 evaluate 构建，闭包捕获每轮上下文）。

多模态策略（见方案「多模态」）：生成图 + target 直接注入初始 HumanMessage，agent 循环每步
都看得到；query_rubric 是按需取判定标准的文本工具。新增能力（如 query_checklist）= 新工具。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ...memory.schema import Benchmark, ContinuousRubric


def _format_rubric(bench: Benchmark, spec, dim_name: str) -> str:
    """格式化某维度的判定标准（二分 checklist / 连续 rubric 要点）。"""
    d = bench.dim_by_name.get(dim_name)
    if d is None:
        return f"(未知维度: {dim_name})"
    val = spec.checklist.get(dim_name)
    if d.scoring_type == "binary":
        items = val if isinstance(val, list) else []
        lines = [f"- {it.id}: {it.check}" for it in items] or ["(无 checklist 项)"]
        return f"维度 {dim_name}（{d.desc or ''}）二分判定项：\n" + "\n".join(lines)
    rubric = val if isinstance(val, ContinuousRubric) else ContinuousRubric(points=[])
    pts = "\n".join(f"- {p}" for p in rubric.points) or "- (按维度描述整体评分)"
    return f"维度 {dim_name}（{d.desc or ''}）连续评分要点：\n{pts}"


def make_query_rubric_tool(*, bench: Benchmark, spec) -> BaseTool:
    @tool
    def query_rubric(dim_name: str) -> str:
        """查询某评分维度的判定标准。

        Args:
            dim_name: 维度名（如 'consistency' / 'material_texture'）。
        返回该维度的 checklist 项（二分）或 rubric 评分要点（连续）。
        """
        return _format_rubric(bench, spec, dim_name)

    return query_rubric


def make_critic_tools(*, bench: Benchmark, spec) -> list[BaseTool]:
    """组装本轮 Critic 工具集。"""
    return [make_query_rubric_tool(bench=bench, spec=spec)]


__all__ = ["make_critic_tools", "make_query_rubric_tool"]
