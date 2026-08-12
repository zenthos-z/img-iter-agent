"""Critic deepagent 的工具集（每轮由 evaluate 构建，闭包捕获每轮上下文）。

多模态策略（见方案「多模态」）：生成图 + target 直接注入初始 HumanMessage，agent 循环每步
都看得到；query_rubric 是按需取判定标准的文本工具。新增能力（如 query_checklist）= 新工具。

创造力维度标准可被 overlay（``creativity_criteria.json``，creativity_tuner 产物）覆盖种子
content_spec——``_effective_checklist`` 是统一入口，critic 的 _build_user_content 与 query_rubric 都走它，
避免两处看到不一致的标准。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ...config import get_settings
from ...memory.schema import Benchmark, CheckItem, ContinuousRubric
from .generator_tools import _format_experience


def _load_creativity_overlay(bench_id: str) -> dict | None:
    """读 bench 级创造力标准 overlay（无文件/无效返回 None）。Critic 启动时调一次。"""
    p = get_settings().benchmark_dir(bench_id) / "creativity_criteria.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _effective_checklist(spec, dim_name: str, overlay: dict | None, bench=None):
    """返回某维度的生效判定标准（**两层 merge**）。

    优先级：creativity overlay > 【通用(manifest check_items) + 特定(sample checklist) merge】。
    设计（benchmark 两层标准）：
    - **通用底线**写在 manifest 的 ``check_items``（所有 sample 自动继承，如三视图一致/完整、
      无瑕疵、结构忠实还原的开放式对比）；sample 不再重复写这些。
    - **sample 特定补充**写在 sample checklist（如该产品额外的二分判定点）；binary 维度追加在通用项之后。
    - **continuous 维度**（材质/颜色）的 ``points`` 是 per-sample，必须写在 sample checklist（manifest 无 points）。

    overlay（creativity_tuner 产物）覆盖一切：不和 manifest/sample 合并。

    返回类型：binary→list[CheckItem]，continuous→ContinuousRubric | dict | None。
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
    # 两层 merge：通用(manifest check_items) + 特定(sample checklist)
    dim = bench.dim_by_name.get(dim_name) if bench else None
    sample_val = (spec.checklist or {}).get(dim_name)
    if dim and dim.scoring_type == "binary":
        # 通用项：manifest 的 "CX 描述" 串 → CheckItem（id=首词，check=剩余）
        items: list[CheckItem] = []
        for i, s in enumerate(dim.check_items or []):
            parts = s.split(None, 1)
            cid = parts[0] if parts else f"{dim_name}-{i + 1}"
            chk = parts[-1] if len(parts) > 1 else s
            items.append(CheckItem(id=cid, check=chk))
        # sample 特定补充：追加（sample 的 binary 项只写"该 sample 额外要查的"，不重复通用）
        if isinstance(sample_val, list):
            for it in sample_val:
                if isinstance(it, CheckItem):
                    items.append(it)
                else:
                    items.append(CheckItem(id=it.get("id", ""),
                                           check=it.get("check", ""),
                                           anchor=it.get("anchor")))
        return items
    # continuous：sample points 为主（manifest 无 points）
    return sample_val


def _format_rubric(bench: Benchmark, spec, dim_name: str, overlay: dict | None = None) -> str:
    """格式化某维度的判定标准（二分 checklist / 连续 rubric 要点）。"""
    d = bench.dim_by_name.get(dim_name)
    if d is None:
        return f"(未知维度: {dim_name})"
    val = _effective_checklist(spec, dim_name, overlay, bench=bench)
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


def make_critic_tools(
    *,
    bench: Benchmark,
    spec,
    overlay: dict | None = None,
    sink: dict,
    run_dir: Path | None = None,
) -> list[BaseTool]:
    """组装本轮 Critic 工具集。

    - overlay=创造力标准 overlay（覆盖种子 content_spec）。
    - sink=note_experience 工具回写 agent 第一手经验判断的字典（由 evaluate 持有，落盘时取用）。
    - run_dir=提供则额外挂 query_experience 工具（让 critic 按需深查历史经验，辅助写 note）。
    """
    tools: list[BaseTool] = [
        make_query_rubric_tool(bench=bench, spec=spec, overlay=overlay),
        make_note_experience_tool(sink=sink),
    ]
    if run_dir is not None:
        tools.append(make_query_experience_tool(run_dir=run_dir))
    return tools


# ---------------------------------------------------------------------------
# 经验总结工具（critic 兼任 in-loop 经验总结：note_experience + query_experience）
# ---------------------------------------------------------------------------
# 设计：critic 是裁判，打分时对「为什么有效/无效」理解最深最准——让它顺手把第一手经验判断
# 写进 note_experience，替代事后 Summarizer._llm_refine 的压缩行富化。工具不落盘（此时本轮
# 结构化 CriticAgentOutput 尚未产出，落盘需本轮 verdict 作"后"证据）；只把判断回写 sink，
# evaluate() 拿到 verdict 后由 Summarizer 用它替代 judge_status 规则 + _llm_refine。


class ExperienceNote(BaseModel):
    """critic 对单个维度本轮经验的第一手判断（note_experience 工具的单条参数）。"""

    dim: str = Field(description="评分维度名（如 consistency / material_texture / creative_departure）")
    judgment: Literal["effective", "ineffective", "escalated"] = Field(
        description=(
            "该维度本轮改动的有效性判断："
            "effective=改动有效，建议保持该方向；"
            "ineffective=改动无效，需换思路（换描述角度/参考图策略/约束措辞）；"
            "escalated=已连续多轮失败、疑似模型能力上限，需根本性换方向（换 test_variable 如 "
            "reference_images/size）或上报人工"
        ),
    )
    lesson: str = Field(
        description=(
            "可复用、可执行的经验总结：为什么有效/无效（引用你判分时看到的具体偏差）+ 后续具体怎么做。"
            "effective→为什么有效+如何保持；ineffective→为什么失败+至少一条具体替代思路"
            "（禁止只写『需换思路』）；escalated→标注模型上限+换 test_variable 的具体建议"
        ),
    )


class NoteExperienceArgs(BaseModel):
    """note_experience 工具 args_schema。

    显式 ``list[ExperienceNote]`` → 干净 array-of-object schema + Literal enum，规避 gemini/deepseek
    严格后端的 anyOf 拒绝（参考 generator_tools.GenerateImageArgs 的同款教训）。
    """

    notes: list[ExperienceNote] = Field(
        default_factory=list,
        description="本轮对各维度的经验判断。打完分后在输出最终评分前调用一次，写下第一手经验。",
    )


def make_note_experience_tool(*, sink: dict) -> BaseTool:
    """critic 兼任经验总结的工具：把 agent 打分时的第一手判断写进 sink，供 evaluate() 落盘。

    工具本身不落盘（本轮结构化 CriticAgentOutput 尚未产出）；只把 notes 回写 sink["lessons"]。
    """
    @tool(args_schema=NoteExperienceArgs)
    def note_experience(notes: list[dict]) -> str:
        """打完分后总结本轮经验：对每个有判断的维度写下『改动是否有效 + 可执行 lesson』。

        这些第一手判断会沉淀到经验知识库(conclusions.json)，供后续轮次/loop 复用——
        你（裁判）对"为什么有效/无效"的理解最深最准，请在输出最终评分前调用本工具一次。
        每条 note：dim（维度名）+ judgment(effective/ineffective/escalated) + lesson（可执行总结）。
        无经验可记时传空 list（如首轮，或本轮无失败维度）。
        """
        # args_schema 把输入解析成 ExperienceNote 对象；统一转回 dict 存 sink，让下游
        # Summarizer._apply_agent_lessons 可按 dict 取字段（al.get("dim")）。
        sink["lessons"] = [
            n.model_dump() if hasattr(n, "model_dump") else n for n in (notes or [])
        ]
        return f"已记录 {len(sink['lessons'])} 条经验判断，将在评分输出后落盘到 conclusions.json。"

    return note_experience


def make_query_experience_tool(*, run_dir: Path) -> BaseTool:
    """让 critic 按需深查本 loop 已沉淀的经验（复用 generator 侧 _format_experience 全文渲染）。"""
    @tool
    def query_experience(dim: str = "") -> str:
        """查询经验知识库里已验证的有效/无效/已升级经验（按需深查，辅助你写 note_experience）。

        Args:
            dim: 可选，只看某个评分维度；留空看全部。
        无经验时返回提示文本。
        """
        return _format_experience(run_dir, dim or None)

    return query_experience


__all__ = [
    "make_critic_tools",
    "make_note_experience_tool",
    "make_query_experience_tool",
    "make_query_rubric_tool",
    "_effective_checklist",
    "_load_creativity_overlay",
]
