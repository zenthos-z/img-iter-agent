"""Critic deepagent 的工具集（每轮由 evaluate 构建，闭包捕获每轮上下文）。

多模态策略（见方案「多模态」）：生成图 + target 直接注入初始 HumanMessage，agent 循环每步
都看得到；query_rubric 是按需取判定标准的文本工具。新增能力（如 query_checklist）= 新工具。

创造力维度标准可被 overlay（``creativity_criteria.json``，creativity_tuner 产物）覆盖种子
content_spec——``_effective_checklist`` 是统一入口，critic 的 _build_user_content 与 query_rubric 都走它，
避免两处看到不一致的标准。
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from ...config import get_settings
from ...memory.schema import Benchmark, CheckItem, ContinuousRubric


def _load_creativity_overlay(bench_id: str) -> dict | None:
    """读 bench 级创造力标准 overlay（无文件/无效返回 None）。Critic 启动时调一次。"""
    p = get_settings().benchmark_dir(bench_id) / "creativity_criteria.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _effective_checklist(spec, dim_name: str, overlay: dict | None):
    """返回某维度的生效判定标准：overlay 的 criteria 有该 dim 就用 overlay，否则回落 spec.checklist。

    返回类型与 spec.checklist 原生一致：binary→list[CheckItem]，continuous→ContinuousRubric。
    """
    crit = (overlay.get("criteria") or {}) if overlay else {}
    if dim_name in crit:
        c = crit[dim_name]
        if c.get("scoring_type") == "binary":
            return [
                CheckItem(id=(it.get("id") or f"{dim_name}-{i + 1}"),
                          check=it.get("check", ""),
                          anchor=it.get("anchor"))
                for i, it in enumerate(c.get("items") or [])
            ]
        return ContinuousRubric(points=list(c.get("points") or []))
    return (spec.checklist or {}).get(dim_name)


def _format_rubric(bench: Benchmark, spec, dim_name: str, overlay: dict | None = None) -> str:
    """格式化某维度的判定标准（二分 checklist / 连续 rubric 要点）。"""
    d = bench.dim_by_name.get(dim_name)
    if d is None:
        return f"(未知维度: {dim_name})"
    val = _effective_checklist(spec, dim_name, overlay)
    if d.scoring_type == "binary":
        items = val if isinstance(val, list) else []
        lines = [f"- {it.id}: {it.check}" for it in items] or ["(无 checklist 项)"]
        return f"维度 {dim_name}（{d.desc or ''}）二分判定项：\n" + "\n".join(lines)
    rubric = val if isinstance(val, ContinuousRubric) else ContinuousRubric(points=[])
    pts = "\n".join(f"- {p}" for p in rubric.points) or "- (按维度描述整体评分)"
    return f"维度 {dim_name}（{d.desc or ''}）连续评分要点：\n{pts}"


def make_query_rubric_tool(*, bench: Benchmark, spec, overlay: dict | None = None) -> BaseTool:
    @tool
    def query_rubric(dim_name: str) -> str:
        """查询某评分维度的判定标准。

        Args:
            dim_name: 维度名（如 'consistency' / 'material_texture' / 'creative_departure'）。
        返回该维度的 checklist 项（二分）或 rubric 评分要点（连续）。
        """
        return _format_rubric(bench, spec, dim_name, overlay)

    return query_rubric


def make_critic_tools(*, bench: Benchmark, spec, overlay: dict | None = None) -> list[BaseTool]:
    """组装本轮 Critic 工具集。overlay=创造力标准 overlay（覆盖种子 content_spec）。"""
    return [make_query_rubric_tool(bench=bench, spec=spec, overlay=overlay)]


__all__ = [
    "make_critic_tools",
    "make_query_rubric_tool",
    "_effective_checklist",
    "_load_creativity_overlay",
]
