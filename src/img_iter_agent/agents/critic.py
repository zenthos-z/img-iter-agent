"""Critic：对照参考图(target)对生成图打分的 agent（deepagent 版）。

改造前是「逐维度单次 LLM 调用 + 手抠 JSON」；现在是一个真正的 tool-using agent：
  - 生成图 + target 直接注入初始 HumanMessage（多模态），agent 循环每步都看得到；
  - 可用工具 `query_rubric(dim_name)` 按需查某维度的判定标准；
  - 用 `response_format=CriticAgentOutput` 约束最终输出（每维度原始评分）；
  - 代码侧把原始评分映射成 `DimensionScore`，再用权重 `recompute_restoration` 算还原度
    （agent 不知道权重，不返回 restoration）——保住「权重变更不影响打分」的契约。

deepagent 在 `evaluate` 内部构建并同步跑完一轮（checkpointer=None），作为引擎嵌入外层
LangGraph 的 critic 节点。agent 跑飞时退安全默认评分，绝不中断闭环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..data.benchmark import Sample
from ..data.weights import recompute_restoration
from ..generation.image_io import file_to_data_uri
from ..memory.schema import (
    Benchmark,
    CriticVerdict,
    DimensionScore,
)
from ._agent_output import CriticAgentOutput
from ._narrow_tools import AGENT_RECURSION_LIMIT, invoke_with_retry, narrow_tools_middleware
from .agent_config_loader import load_system_prompt
from .tools.critic_tools import _effective_checklist, _load_creativity_overlay, make_critic_tools

_DEFAULT_CRITIC_SYS = (
    "你是严格的产品图评判员。对照参考图(target)对生成图打分：二分维度逐项判通过/不通过 + 一句理由，"
    "连续维度给 0-1 分 + 一句理由。可用 query_rubric 查某维度的判定标准。所有维度都是"
    "「生成图 vs target」的还原对比，不是绝对评判。拿不准倾向判不通过/给低分。"
    "最后结构化输出每个维度的评分（不要自己算加权和/还原度，权重不在你手上）。"
)


@dataclass
class CriticInput:
    """一次 Critic 评判的输入（一个 trace）。"""

    sample: Sample
    generated_images: list[Path] = field(default_factory=list)  # 三视图等生成图
    weights: dict[str, float] = field(default_factory=dict)  # 当前生效权重
    meaning: str | None = None  # Generator 的一句话图片含义解释（风格迁移场景；判概念表达时参考）
    reference_ids: list[str] = field(default_factory=list)  # 本次实际传给生图 API 的参考标识符；判 reference_independence 用


# ---------------------------------------------------------------------------
# 多模态内容构造（生成图 + target 注入 HumanMessage）
# ---------------------------------------------------------------------------


def _build_multimodal_content(text: str, images: list[Path]) -> list[dict] | str:
    """构造 OpenAI 兼容的多模态 content：text + 图片(data-URI)。无图时返回纯文本。"""
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


def _images_block(target: Path, generated: list[Path]) -> str:
    """文字版喂图说明（多模态 content 里也带这段文字锚，便于 LLM 区分图序）。"""
    parts = ["[生成图]"] + [f"  - view {i+1}: {p.name}" for i, p in enumerate(generated)]
    parts += ["[参考图(target 产品实物)]", f"  - {target.name}"]
    return "\n".join(parts)


def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
    cfg: dict = dict(config) if config else {}
    cfg["recursion_limit"] = limit
    return cfg


# ---------------------------------------------------------------------------
# Critic 主体
# ---------------------------------------------------------------------------


class Critic:
    """混合评分 Critic（deepagent 引擎）。chat model + bench 注入。"""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        bench: Benchmark,
        system_prompt: str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.bench = bench
        self.system_prompt = system_prompt or load_system_prompt("critic", _DEFAULT_CRITIC_SYS)
        # 创造力标准 overlay（creativity_tuner 产物）：Critic 实例构建时读一次，覆盖种子 content_spec。
        # 每个 loop 子进程构建一次 → tuner 只在批与批之间生效（不污染在跑的 loop）。
        self._creativity_overlay = _load_creativity_overlay(bench.bench_id)
        # 经验总结（原 in-loop Summarizer 职责）：critic 打分后兼做跨轮因果验证 + lesson 富化。
        # Summarizer 不再作为 graph 独立节点——loop 只剩 generator/critic 两个 agent；其成熟逻辑
        # （distiller 跨 loop 蒸馏依赖的 conclusions.json 产物）作为工具被 critic 调用。
        from .summarizer import Summarizer
        self._summarizer = Summarizer(chat_model=chat_model)

    def evaluate(
        self, inp: CriticInput, *, config: RunnableConfig | None = None,
        extra_hints: list[str] | None = None,
    ) -> CriticVerdict:
        """对一个 trace 打分，产出 CriticVerdict。

        所有维度都对照 target 评判。生成图 + target 以 image_url 注入初始 HumanMessage。
        agent 用 response_format=CriticAgentOutput 约束最终输出；代码侧映射 + 算 restoration。
        agent 跑飞 → 安全默认评分（不中断闭环）。
        """
        spec = inp.sample.spec
        target = inp.sample.target_path
        generated = inp.generated_images

        user_content = self._build_user_content(
            target, generated, spec, meaning=inp.meaning, extra_hints=extra_hints,
            reference_ids=inp.reference_ids,
        )
        tools = make_critic_tools(bench=self.bench, spec=spec, overlay=self._creativity_overlay)
        agent = create_deep_agent(
            model=self.chat_model, tools=tools,
            system_prompt=self.system_prompt,
            response_format=CriticAgentOutput, checkpointer=None, name="critic",
            middleware=narrow_tools_middleware(),
        )

        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=user_content)]},
            config=_merge_recursion(config, AGENT_RECURSION_LIMIT), label="critic",
        )
        out: CriticAgentOutput | None = (result.get("structured_response") if result else None)

        dim_scores = self._to_dimension_scores(out)
        restoration = recompute_restoration(dim_scores, inp.weights)
        return CriticVerdict(
            sample_id=spec.sample_id,
            dimensions=dim_scores,
            weights_used=dict(inp.weights),
            restoration=restoration,
        )

    def summarize_round(self, **kwargs) -> str:
        """critic 兼任的经验总结（原 in-loop Summarizer 职责）。

        打分后做跨轮因果验证 + 复发检测 + lesson 富化，更新 conclusions.json，返回 lesson_ref。
        逻辑见 Summarizer.summarize（成熟；distiller 跨 loop 蒸馏依赖其 conclusions.json 产物）。
        """
        return self._summarizer.summarize(**kwargs)

    # --- 辅助 ---

    def _build_user_content(
        self, target: Path, generated: list[Path], spec, meaning: str | None = None,
        extra_hints: list[str] | None = None, reference_ids: list[str] | None = None,
    ) -> list[dict] | str:
        """初始 HumanMessage：喂图说明 + 每维度**逐项 checklist** + 评分指令 + 生成图与 target(image_url)。

        关键：把每个二分维度的 checklist 项（id+判定+anchor）直接列出来，强制 agent 逐项判，
        返回与项数相等、id 逐项对应的 items——否则 agent 会偷懒只给一个聚合判断（导致 passed/1=满分级虚高）。
        """
        dim_lines: list[str] = []
        for d in self.bench.score_dimensions:
            cl = _effective_checklist(spec, d.dim, self._creativity_overlay, bench=self.bench)
            if d.scoring_type == "binary":
                items = list(cl) if isinstance(cl, list) else []
                item_lines = "\n".join(
                    f"    - {getattr(it, 'id', '?')}: {getattr(it, 'check', '')}"
                    + (f"（{it.anchor}）" if getattr(it, 'anchor', None) else "")
                    for it in items
                ) or "    (未定义 checklist 项)"
                dim_lines.append(
                    f"- {d.dim}（二分，逐项 ✓/✗ + 理由）：{d.desc or ''}\n"
                    f"  必须为下面**每一项**返回一条 item（id 严格对应）：\n{item_lines}"
                )
            else:
                pts = getattr(cl, "points", None) or []
                ptstr = "; ".join(pts) if pts else "(无)"
                dim_lines.append(
                    f"- {d.dim}（连续，0-1 分 + 理由）：{d.desc or ''}。评分要点：{ptstr}"
                )
        task = (
            "对照参考图(target)，对生成图按下列维度逐一打分（所有维度都是 生成图 vs target 的还原对比）。\n"
            "二分维度：为列出的**每一项** checklist 返回一条 {id, passed, reason}——"
            "items 数量必须等于列出的项数、id 逐项对应，**不得合并、不得省略**（通过率=通过项/总项，少一项分数就错）。"
            "连续维度：给 0-1 的 value + 一句 reason。\n"
            "**严格**：拿不准、有瑕疵、与 target 不完全一致时，倾向判不通过/给低分；只有确无问题才判通过。\n"
            "维度清单：\n" + "\n".join(dim_lines) + "\n\n请结构化输出每个维度的评分。"
        )
        # 参考图使用情况：让 critic 能判 reference_independence（对照「实际传入」的参考，而非全 7 张）
        if reference_ids is not None:
            ids_str = ", ".join(reference_ids) if reference_ids else "（空 = 纯文生图，未用参考图）"
            task += (
                f"\n\n【参考图使用情况】本次 Generator 实际传给生图 API 的参考图(reference_ids)：{ids_str}。"
                f"reference_independence 维度据此判定：只判生成图是否复制了**这些实际传入**参考图的 motif；"
                f"reference_ids 为空（纯文生图）时 reference_independence 默认通过。"
            )
        # 人工补充评分准则（运行时人工介入；与 checklist 同等效力，必须执行）
        if extra_hints:
            task += (
                "\n\n【额外评分准则（人工补充，与 checklist 同等效力，必须执行）】\n"
                + "\n".join(f"- {h}" for h in extra_hints)
            )
        if meaning:
            task = f"【Generator 的图片含义解释（它声称这张图想表达的概念）】{meaning}\n\n" + task
        if spec.task and spec.task.mode == "style_transfer":
            task = (
                "【风格神韵迁移 · 自主对照参考集判一致性（最高优先级）】\n"
                "严格对照参考集(target)：生成图任何与参考集风格不一致的细节——"
                "写实的解剖/纹理(指甲/指纹/关节纹/皮肤)、参考集里没有的元素、写实阴影/高光——"
                "即使下列 checklist 未明确写出，也**必须在对应 spirit_* 维度判不通过 / 给低分**，并在 reason 写明你发现的偏差。\n"
                "你要主动发现 checklist 之外的偏差(参考集是纯线条抽象、零解剖细节；生成图出现任何写实细节就是不一致)，不要只盯列出的项。\n\n"
            ) + task
        text = _images_block(target, generated) + "\n\n" + task
        images = list(generated) + ([target] if target.exists() else [])
        return _build_multimodal_content(text, images)

    def _to_dimension_scores(self, out: CriticAgentOutput | None) -> list[DimensionScore]:
        """把 agent 输出映射成 DimensionScore 列表（按 bench 维度顺序，缺失→安全默认）。"""
        by_dim = {d.dim: d for d in (out.dimensions if out else [])}
        scores: list[DimensionScore] = []
        for ddef in self.bench.score_dimensions:
            o = by_dim.get(ddef.dim)
            if o is None:
                scores.append(self._safe_dim(ddef.dim, ddef.scoring_type))
                continue
            if ddef.scoring_type == "binary":
                items = o.items or []
                value = (sum(1 for it in items if it.passed) / len(items)) if items else 0.0
                scores.append(DimensionScore(
                    dim=ddef.dim, scoring_type="binary", value=value, items=items,
                ))
            else:
                v = o.value if o.value is not None else 0.0
                scores.append(DimensionScore(
                    dim=ddef.dim, scoring_type="continuous", value=v, raw=o.reason or "",
                ))
        return scores

    @staticmethod
    def _safe_dim(dim: str, scoring_type: str) -> DimensionScore:
        if scoring_type == "binary":
            return DimensionScore(dim=dim, scoring_type="binary", value=0.0, items=[])
        return DimensionScore(dim=dim, scoring_type="continuous", value=0.0, raw="(eval failed)")


__all__ = ["Critic", "CriticInput"]
