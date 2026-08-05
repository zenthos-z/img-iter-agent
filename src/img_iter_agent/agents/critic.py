"""Critic：看图按 benchmark 维度打分，产出 CriticVerdict。

混合评分（ARCH §2.6.1 / EVALUATION §4）：
  - 二分维度(scoring_type=binary)：按 content_spec 的 checklist 逐项 ✓/✗ + 理由
    → features[dim] = 通过项数 / 总项数
  - 连续维度(scoring_type=continuous)：按 rubric points 让 LLM 给 0-1 分（承认偏差）
    → features[dim] = LLM 归一化分

喂图策略（由 manifest.comparative_dims / content_spec.anchor_for 决定）：
  - 对比型维度：同时喂「参考图(target) + 生成图(三视图)」
  - 绝对型维度：只喂「生成图」
  三视图任务里 consistency 是跨张对照（三视图之间 + target）。

LLM 调用走依赖注入（`LlmClient`），故可用 `FakeLlmClient` 完全离线测试分派与加权。
LLM 输出约定为 JSON，本模块负责解析 + 容错（解析失败给安全默认值，绝不抛错中断闭环）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from ..data.benchmark import Sample
from ..data.weights import recompute_restoration
from ..generation.image_io import file_to_data_uri
from ..llm import LlmClient
from ..memory.schema import (
    Benchmark,
    CheckItem,
    ContinuousRubric,
    CriticItemJudgment,
    CriticVerdict,
    DimensionScore,
)
from .agent_config_loader import load_system_prompt

# 代码默认提示词（agents_config/critic.md / critic_continuous.md 缺失时回退）
_DEFAULT_BINARY_PROMPT = (
    "你是严格的产品图评判员。对下列 checklist 项逐项判定 通过/不通过，"
    "每项给一句简短理由。只输出 JSON，不要任何额外文字。\n"
    'JSON 格式: {"judgments":[{"id":"C1","passed":true,"reason":"..."}, ...]}'
)
_DEFAULT_CONTINUOUS_PROMPT = (
    "你是产品图材质/颜色评判员。对生成图的还原图整体给一个 0-1 分"
    "（0=完全没还原，1=完美还原），并给一句理由。只输出 JSON。\n"
    'JSON 格式: {"score":0.72,"reason":"..."}'
)


@dataclass
class CriticInput:
    """一次 Critic 评判的输入（一个 trace）。"""

    sample: Sample
    generated_images: list[Path] = field(default_factory=list)  # 三视图等生成图
    weights: dict[str, float] = field(default_factory=dict)  # 当前生效权重


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------

def _build_multimodal_content(text: str, images: list[Path]) -> list[dict] | str:
    """构造 OpenAI 兼容的多模态 content：text + 图片(data-URI)。

    无图时直接返回 text（保持纯文本模型兼容）。有图时返回 content 数组：
    [{"type":"text",...}, {"type":"image_url","image_url":{"url":"data:..."}}, ...]
    """
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text}]
    for p in images:
        if Path(p).exists():
            parts.append({
                "type": "image_url",
                "image_url": {"url": file_to_data_uri(Path(p))},
            })
    return parts


def _images_block(comparative: bool, target: Path, generated: list[Path]) -> str:
    """文字版喂图说明（多模态 content 里也会带这段文字锚，便于 LLM 区分图序）。"""
    parts = ["[生成图]"] + [f"  - view {i+1}: {p.name}" for i, p in enumerate(generated)]
    if comparative:
        parts += ["[参考图(target 产品实物)]", f"  - {target.name}"]
    return "\n".join(parts)


def _binary_prompt(dim_name: str, dim_desc: str, items: list[CheckItem],
                   comparative: bool, target: Path, generated: list[Path]) -> list[dict]:
    """二分维度的 prompt：要求逐项 ✓/✗ + 理由，返回 JSON。多模态（带图）。"""
    item_lines = "\n".join(f"  - {it.id}: {it.check}" for it in items)
    sys_msg = load_system_prompt("critic", _DEFAULT_BINARY_PROMPT)
    text = (
        f"维度: {dim_name}\n描述: {dim_desc}\n\n"
        f"判定项:\n{item_lines}\n\n"
        f"{_images_block(comparative, target, generated)}\n\n"
        "请逐项判定。passed=true 表示通过，false 表示不通过。"
    )
    # 喂的图：生成图（必给）；对比型再加 target
    images = list(generated) + ([target] if comparative else [])
    user_content = _build_multimodal_content(text, images)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}]


def _continuous_prompt(dim_name: str, dim_desc: str, rubric: ContinuousRubric,
                       comparative: bool, target: Path, generated: list[Path]) -> list[dict]:
    """连续维度的 prompt：按 rubric points 整体给 0-1 分，返回 JSON。多模态（带图）。"""
    points = "\n".join(f"  - {p}" for p in rubric.points) or "  - (按维度描述整体评分)"
    sys_msg = load_system_prompt("critic_continuous", _DEFAULT_CONTINUOUS_PROMPT)
    text = (
        f"维度: {dim_name}\n描述: {dim_desc}\n评分要点:\n{points}\n\n"
        f"{_images_block(comparative, target, generated)}\n\n"
        "请给 0-1 的 score 与一句 reason。"
    )
    images = list(generated) + ([target] if comparative else [])
    user_content = _build_multimodal_content(text, images)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# LLM 输出解析（容错：解析失败给安全默认，不抛错）
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """从 LLM 文本里抠出第一个 JSON 对象。失败返回 None。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _parse_binary_items(raw: dict | None, items: list[CheckItem]) -> list[CriticItemJudgment]:
    """解析二分维度的逐项判定。缺失/坏项默认 passed=False。"""
    judgments = raw.get("judgments", []) if raw else []
    by_id = {}
    for j in judgments:
        jid = j.get("id")
        if jid:
            by_id[jid] = j
    out: list[CriticItemJudgment] = []
    for it in items:
        j = by_id.get(it.id)
        if j is None:
            out.append(CriticItemJudgment(id=it.id, passed=False, reason="(未返回)"))
        else:
            passed = bool(j.get("passed", False))
            reason = str(j.get("reason", "")).strip()
            out.append(CriticItemJudgment(id=it.id, passed=passed, reason=reason))
    return out


def _parse_continuous_score(raw: dict | None) -> tuple[float, str]:
    """解析连续维度的 0-1 分。坏值默认 0.0。"""
    if not raw:
        return 0.0, "(解析失败)"
    score = raw.get("score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))  # clamp 到 [0,1]
    reason = str(raw.get("reason", "")).strip()
    return score, reason


# ---------------------------------------------------------------------------
# Critic 主体
# ---------------------------------------------------------------------------


class Critic:
    """混合评分 Critic。LLM client 注入。"""

    def __init__(self, client: LlmClient, *, bench: Benchmark) -> None:
        self.client = client
        self.bench = bench

    def evaluate(self, inp: CriticInput, *, config: RunnableConfig | None = None) -> CriticVerdict:
        """对一个 trace 打分，产出 CriticVerdict。

        系统目标=还原度，故**所有维度都对照参考图(target)与生成图对比评判**：
        每个 prompt 都同时注入 target + 生成图（无绝对型维度）。每个维度的评分
        经 `_score_dimension` 作为一条 LangSmith chain run 上报（含 LLM 原始输出 +
        解析后的结构化分），便于在 LangSmith 里按维度排查。config 由 LangGraph 节点
        注入，透传给 @traceable 子方法以保证 trace 嵌套在节点 run 之下。
        """
        spec = inp.sample.spec
        target = inp.sample.target_path
        generated = inp.generated_images
        ls_extra: Any = {"config": config} if config is not None else {}

        dimension_scores: list[DimensionScore] = []
        for dim_def in self.bench.score_dimensions:
            checklist_val = spec.checklist.get(dim_def.dim)
            result = self._score_dimension(
                dim_def, target=target, generated=generated,
                checklist_val=checklist_val, langsmith_extra=ls_extra,
            )
            dimension_scores.append(result["score"])

        restoration = recompute_restoration(dimension_scores, inp.weights)
        return CriticVerdict(
            sample_id=spec.sample_id,
            dimensions=dimension_scores,
            weights_used=dict(inp.weights),
            restoration=restoration,
        )

    @traceable(name="critic.score_dimension", run_type="chain")
    def _score_dimension(
        self, dim_def, *, target: Path, generated: list[Path], checklist_val,
    ) -> dict:
        """单个维度的评分（作为一条 LangSmith chain run）。

        返回结构化 dict：`score` 是要累加进 verdict 的 DimensionScore；其余字段
        （raw_llm_output / parsed）作为 trace output，让 LangSmith 看到喂了什么、
        LLM 原文返回什么、解析后得多少分。解析失败给安全默认值，绝不抛错中断闭环。
        所有维度都是对比型（对照 target 评）。
        """
        name = dim_def.dim
        if dim_def.scoring_type == "binary":
            items = checklist_val if isinstance(checklist_val, list) else []
            if not items:
                # manifest 声明二分但考题缺 checklist → 当作零项，给中性分
                return {
                    "dimension": name,
                    "score": DimensionScore(dim=name, scoring_type="binary", value=0.0,
                                            items=[], raw="(无 checklist 项)"),
                    "raw_llm_output": None,
                }
            msgs = _binary_prompt(name, dim_def.desc or name, items, True, target, generated)
            raw_text = self.client.complete(msgs)
            raw_json = _extract_json(raw_text)
            judgments = _parse_binary_items(raw_json, items)
            value = sum(1 for j in judgments if j.passed) / len(judgments) if judgments else 0.0
            return {
                "dimension": name,
                "score": DimensionScore(dim=name, scoring_type="binary", value=value, items=judgments),
                "raw_llm_output": raw_text,
                "parsed": [j.model_dump() for j in judgments],
            }
        # continuous
        rubric = checklist_val if isinstance(checklist_val, ContinuousRubric) else ContinuousRubric(points=[])
        msgs = _continuous_prompt(name, dim_def.desc or name, rubric, True, target, generated)
        raw_text = self.client.complete(msgs)
        raw_json = _extract_json(raw_text)
        score, reason = _parse_continuous_score(raw_json)
        return {
            "dimension": name,
            "score": DimensionScore(dim=name, scoring_type="continuous", value=score, raw=reason),
            "raw_llm_output": raw_text,
            "parsed_reason": reason,
        }


__all__ = ["Critic", "CriticInput"]
