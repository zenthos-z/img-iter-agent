"""Generator：基于 Critic 反馈自动迭代优化 prompt 的生图 agent（deepagent 版）。

迭代核心（用户明确）：**自动优化生图提示词**，不是改尺寸。size 由考题固定，不动。
每轮由一个真正的 tool-using agent（`create_deep_agent`）完成：
  - 读考题指令+约束（round>1 还看上轮 Critic 失败项）→ 用工具 `query_experience` 查已验证经验
    → 构造/改进英文 prompt → 调工具 `generate_image` 出图 → 结构化输出 `GeneratorOutput`。
  - baseline_ref 指向上轮 attempt（对比迭代效果），test_variable 始终记为 "prompt"。

deepagent 在 `generate_round` 内部构建并同步跑完一轮（checkpointer=None，无断点），
作为「引擎」嵌入外层 LangGraph 的 generator 节点——`RunState`/checkpointer/trace 编排不变。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..data.benchmark import Sample
from ..generation.base import GenRequest, ModelFamily
from ..generation.image_io import file_to_data_uri
from ..generation.router import Router
from ..memory.experience import generator_agent_fs
from ..memory.schema import CriticItemJudgment, TestVariable
from ._agent_output import GeneratorOutput, provider_structured
from ._narrow_tools import (
    AGENT_RECURSION_LIMIT,
    _GENERATOR_NARROW_EXCLUDED,
    invoke_with_retry,
    narrow_tools_middleware,
)
from .agent_config_loader import load_system_prompt
from .tools.generator_tools import _size_from_str, make_generator_tools

# 代码默认 system prompt（data/agents_config/generator.md 缺失时回退）。
# 短的「始终生效」角色放这里；更长的诀窍/流程在本 benchmark 的蒸馏技能包里（agent 按需读）。
# 关键：明确工具清单 + 「参考图已在消息里/不要找文件/每个工具至多一次/出图一次即输出」，
# 配合 NarrowToolsMiddleware 剥掉 fs 工具，杜绝在空 sandbox 里乱逛导致 GraphRecursionError。
_DEFAULT_GENERATOR_SYS = (
    "你是生图提示词工程师。每轮按以下精简流程，不要发散：\n"
    "1. 读用户消息里的考题指令与约束（round>1 时还含上轮 Critic 失败项 + 【本题已验证经验】摘要）。"
    "参考图（若有）已直接附在消息里，**不要去文件系统找图**。\n"
    "1b. 若用户消息给出【文章原文】路径：先 read_file 读完文章（长文用 offset/limit 翻页），"
    "吃透核心概念再构思视觉隐喻，**不要凭训练记忆臆测文章内容**。\n"
    "2. 取经验（每个工具至多一次）：若挂载了本 benchmark 的经验技能包，按提示 read_file 它的 SKILL.md"
    "（必要时再读 references/lessons.md）取跨 loop 蒸馏的生成要点；round>1 时调一次 query_experience 取本题"
    " in-loop 经验全文（每轮 user message 也有结论摘要）。\n"
    "3. 构造/改进**英文优先**的生图 prompt：保留用户消息里给出的所有考题约束；"
    "把每个失败项转成具体的、可执行的正面描述（不要只写『不要 X』）；保留原有正确部分；"
    "【本题已验证经验】里「勿重复」的改动绝对不要再试、「保持」的要延续。\n"
    "4. **只调一次** generate_image(prompt=..., size=..., reference_images=...) 出图。\n"
    "5. 【收尾·强制，违反会导致本轮作废】generate_image 只返回文件路径、**没有任何评分或质量反馈**，"
    "因此**绝不能**为『改善结果』再出图。出图成功后，**下一步必须且只能**调用结构化输出工具 GeneratorOutput"
    "（填 prompt + delta_note + meaning）结束本轮。**不要**在出图后停顿、**不要**输出纯文本就结束——"
    "那样本轮会被判无结构化输出、触发重试、放大成一堆废图。\n"
    "6. 【风格迁移·参考图用法】generate_image 的 reference_images 参数可选，传参考图标识符子集（如 "
    "['hand-abacus']）。Gemini 把它们作 inline_data 风格条件。**这是创意权衡**：0-2 张帮你锚定风格神韵；"
    ">2 张会过度锚定 motif、压制原创（creativity 的 reference_independence 维度会扣分）。多数情况建议 0-1 张，"
    "纯文生图(reference_images=[])是合法且常更原创的选择。可用标识符见用户消息。\n"
    "你的核心工具：generate_image / query_experience。若挂载了经验技能包或文章素材，还会自动出现 read_file"
    "（仅限读用户消息指出的文章路径，及技能包的 SKILL.md / references，不可读其它路径）。"
)


class PriorFeedback(BaseModel):
    """上轮 Critic 反馈的摘要，供 Generator 改进 prompt。"""

    failed_items: list[CriticItemJudgment] = Field(default_factory=list)
    continuous_notes: list[str] = Field(default_factory=list)  # 连续维度的低分理由
    failed_dims: list[str] = Field(default_factory=list)  # 上轮失败的维度名（通用经验 select 用）


class GenOutcome(BaseModel):
    """Generator 一轮的产出。"""

    attempt_id: str
    test_variable: TestVariable | None
    baseline_ref: str | None
    gen_mode: str
    prompt: str
    size: str
    reference_image_refs: list[str]  # 相对 run 目录（agent 看到的参考集，供上下文）
    reference_ids: list[str] = Field(default_factory=list)  # 实际传给生图 API 的参考标识符(stem)；[]=纯文生图。供 creativity tuner
    output_image_refs: list[str]  # 相对 run 目录（三视图一张图）
    model: str
    model_family: str
    improved_from_feedback: bool = False  # 本轮 prompt 是否基于上轮反馈改进
    delta_note: str | None = None  # 本轮相对上轮的改动说明（供经验闭环沉淀）
    meaning: str | None = None  # 一句话图片含义解释（风格神韵迁移场景；供 Critic 判概念表达）


def _new_attempt_id(round: int) -> str:
    return f"a{round:03d}_{uuid.uuid4().hex[:6]}"


class Generator:
    """生图 agent。Router 与 chat model 注入；每轮内部构建 deepagent 跑完。"""

    def __init__(
        self,
        router: Router,
        *,
        chat_model: BaseChatModel,
        system_prompt: str | None = None,
        skills_dir: Path | str | None = None,
    ) -> None:
        self.router = router
        self.chat_model = chat_model
        self.system_prompt = system_prompt or load_system_prompt("generator", _DEFAULT_GENERATOR_SYS)
        self.skills_dir = str(skills_dir) if skills_dir else None
        # 跨 loop 知识由本 benchmark 的蒸馏技能包提供（skills_dir → SkillsMiddleware 按需加载）。
        # in-loop 经验（conclusions.json）由 generate_round 每轮 load_conclusions + render_conclusions_brief
        # 精简注入 user message；query_experience 工具作全文钻取。

    def generate_round(
        self,
        *,
        sample: Sample,
        out_dir: Path,
        run_dir: Path,
        round: int,
        baseline_ref: str | None = None,
        prior_feedback: PriorFeedback | None = None,
        escalated_warnings: list[str] | None = None,
        model_hint: ModelFamily | None = None,
        config: RunnableConfig | None = None,
        extra_hints: list[str] | None = None,
    ) -> GenOutcome:
        """跑一轮：deepagent 构造/改进 prompt 并出图。

        size 由考题固定（不动）。首轮纯构造；后续轮据 prior_feedback + 经验库改进 prompt。
        agent 用 response_format=GeneratorOutput 约束最终输出。
        """
        spec = sample.spec
        task = spec.task
        size_str = (task.output.get("size") if task and task.output else None) or "2K"

        # 参考：image_edit/multiview 用 target 作风格锚；style_transfer 注入参考集多图供 agent 抽象风格共性。
        # D2：style_transfer 解禁——agent 通过 generate_image(reference_images=[...]) 主动选参考子集传给生图 API
        # （make_generator_tools 建 ref_registry）；默认 [] = 纯文生图。reference_images 这里只是 agent 看到的上下文。
        reference_images: list[Path] = []  # agent 在 user content 里看到的参考（≠传给 API 的）
        gen_mode = "text_to_image"
        if task and task.mode == "style_transfer":
            refs = [sample.sample_dir / a for a in (task.input_assets or [])]
            reference_images = [p for p in refs if p.exists()]
            gen_mode = "text_to_image"
        elif task and task.mode in ("image_edit", "multiview") and sample.target_path.exists():
            reference_images = [sample.target_path]
            gen_mode = "image_edit"

        attempt_id = _new_attempt_id(round)
        attempt_out = out_dir / attempt_id
        attempt_out.mkdir(parents=True, exist_ok=True)

        sink: dict[str, Any] = {}
        tools = make_generator_tools(
            router=self.router, sample=sample, out_dir=attempt_out, run_dir=run_dir,
            model_hint=model_hint, sink=sink,
        )
        # 跨 loop 蒸馏经验：标准 deepagents skills（progressive disclosure）。
        # skills_dir 非空（本 bench 已蒸馏）→ 接有界 FS：read_file 钉死在技能包内、放回 read_file（Generator 专用窄集），
        # 让 SkillsMiddleware 真能让模型 read_file SKILL.md。无技能包 → 裸跑（原 narrow 全集，剥 read_file）。
        # sample 含 article.md（考题文章正文素材）→ 一并挂载（只读），user message 给虚拟路径让 agent read_file。
        agent_fs = generator_agent_fs(self.skills_dir, sample.sample_dir)
        if agent_fs is not None:
            agent_kwargs: dict[str, Any] = {
                "model": self.chat_model, "tools": tools,
                "system_prompt": self.system_prompt,
                "response_format": provider_structured(GeneratorOutput),
                "checkpointer": None, "name": "generator",
                "middleware": narrow_tools_middleware(excluded=_GENERATOR_NARROW_EXCLUDED),
                "permissions": agent_fs.permissions, "backend": agent_fs.backend,
            }
            if agent_fs.skills_sources:
                agent_kwargs["skills"] = agent_fs.skills_sources
            agent = create_deep_agent(**agent_kwargs)
        else:
            agent = create_deep_agent(
                model=self.chat_model, tools=tools,
                system_prompt=self.system_prompt,
                response_format=provider_structured(GeneratorOutput), checkpointer=None, name="generator",
                middleware=narrow_tools_middleware(),
            )

        # in-loop 经验精简摘要：round>1 时读 conclusions.json，按本轮失败维度选定 ineffective/effective
        # 注入 user message（与 query_experience 工具的全文钻取互补）。读失败兜底为空——绝不中断闭环。
        conclusions_brief = ""
        if round > 1:
            try:
                from ..memory.knowledge import load_conclusions, render_conclusions_brief
                _kb = load_conclusions(run_dir, sample_id=sample.sample_id)
                _fdims = (prior_feedback.failed_dims if prior_feedback else None) or []
                conclusions_brief = render_conclusions_brief(_kb, failed_dims=_fdims)
            except Exception:  # noqa: BLE001
                conclusions_brief = ""

        user_content = self._build_user_content(
            sample, round, prior_feedback, reference_images,
            escalated_warnings=escalated_warnings, conclusions_brief=conclusions_brief,
            extra_hints=extra_hints,
            article_path=agent_fs.article_path if agent_fs is not None else None,
        )
        invoke_cfg = self._merge_recursion(config, AGENT_RECURSION_LIMIT)

        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=user_content)]},
            config=invoke_cfg, label="generator",
        )
        out: GeneratorOutput = (result.get("structured_response") if result else None) or GeneratorOutput(prompt="")

        prompt = (out.prompt or "").strip() or self._base_prompt_text(sample)
        delta_note = (out.delta_note or "").strip() or None
        meaning = (out.meaning or "").strip() or None

        # 出图：agent 调了 generate_image → sink 有 ref；否则用 prompt 现场出图（兜底）
        if sink.get("ref"):
            output_refs = [sink["ref"]]
            model_used = sink.get("model", "")
            model_family = sink.get("family", "?")
            reference_ids_out = list(sink.get("reference_ids") or [])
        else:
            fb_size = _size_from_str(size_str)
            layout = (sample.spec.task.output.get("layout")
                      if sample.spec.task and sample.spec.task.output else None)
            if layout == "three_view_single_image":
                fb_size.ratio = "16:9"  # 三视图宽幅，避免 1:1 挤变形
            # style_transfer 兜底=纯文生图（与 D2 agent 默认一致，避免 7 图全锚定→同质化）；
            # image_edit/multiview 兜底仍用 target 锚定。
            fb_refs: list[Path] = ([] if (task and task.mode == "style_transfer")
                                   else list(reference_images))
            req = GenRequest(
                prompt=prompt, size=fb_size,
                reference_images=fb_refs, model_hint=model_hint,
            )
            # 兜底出图也加重试+退避：dmxapi 间歇性连接中断时，单次 router.generate 抛 APIConnectionError
            # 会直接炸掉整轮（乃至整 sample）。这里重试扛过网关抖动，与 agent.invoke 的重试对齐。
            img = None
            last_exc: Exception | None = None
            for _attempt in range(4):
                try:
                    img = self.router.generate(req, out_dir=attempt_out, config=config)
                    break
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    if _attempt < 3:
                        print(f"[generator] 兜底出图异常({type(e).__name__})，重试 {_attempt + 1}/3",
                              flush=True)
                        time.sleep(2.0)
            if img is None:
                assert last_exc is not None
                raise last_exc
            ext = img.image_path.suffix or ".png"
            dest = attempt_out / f"three_view{ext}"
            if img.image_path != dest:
                img.image_path.rename(dest)
            output_refs = [str(dest.relative_to(run_dir))]
            model_used = img.model
            model_family = img.meta.get("family", "?")
            reference_ids_out = [p.stem for p in fb_refs]

        improved = round > 1 and prior_feedback is not None and bool(prior_feedback.failed_items)

        return GenOutcome(
            attempt_id=attempt_id,
            test_variable="prompt" if round > 1 else None,
            baseline_ref=baseline_ref,
            gen_mode=gen_mode,
            prompt=prompt,
            delta_note=delta_note,
            meaning=meaning,
            size=size_str,
            reference_image_refs=[str(p.resolve()) for p in reference_images],
            reference_ids=reference_ids_out,
            output_image_refs=output_refs,
            model=model_used,
            model_family=model_family,
            improved_from_feedback=improved,
        )

    # --- 辅助 ---

    def _build_user_content(
        self, sample: Sample, round: int,
        prior_feedback: PriorFeedback | None, reference_images: list[Path],
        *,
        escalated_warnings: list[str] | None = None,
        conclusions_brief: str = "",
        extra_hints: list[str] | None = None,
        article_path: str | None = None,
    ) -> list[dict] | str:
        """构造初始 HumanMessage 内容：指令+约束(+文章素材路径)+上轮失败项+本题已验证经验+经验闭环警告+参考图(image_url)。"""
        instr = (sample.spec.task.instruction if sample.spec.task else None) or "生成产品白底三视图素材图"
        constraints = self._extract_constraints(sample)
        text = instr
        if constraints:
            text += "\n约束: " + json.dumps(constraints, ensure_ascii=False)
        if article_path:
            text += (
                "\n\n【文章原文】本题文章正文已挂载（只读），路径："
                f"{article_path}\n构造 prompt 前**先 read_file 读它**（长文用 offset/limit 翻页读完整），"
                "吃透文章核心概念后再构思视觉隐喻；**不要凭训练记忆臆测文章内容**。"
            )
        if round > 1 and prior_feedback and prior_feedback.failed_items:
            fails = "\n".join(f"- [{it.id}] {it.reason}" for it in prior_feedback.failed_items)
            cont = ""
            if prior_feedback.continuous_notes:
                cont = "\n连续维度低分:\n" + "\n".join(f"- {n}" for n in prior_feedback.continuous_notes)
            text += (
                f"\n\n【上轮 Critic 失败项（需改进）】\n{fails}{cont}\n"
                "针对性改进 prompt（参考下方【本题已验证经验】，或调 query_experience 取全文），"
                "再调 generate_image 出图，最后结构化输出 prompt + delta_note。\n"
                "**delta_note 必填**：逐条写出本轮针对每个失败项（引用其 id，如 C3/S1）做了"
                "什么具体的正向改动——这是经验闭环判定『改动是否有效』的依据，缺失会让整条经验链断掉。"
            )
        else:
            text += (
                "\n请把指令精炼成清晰 prompt，"
                "调 generate_image 出图，最后结构化输出 prompt + delta_note。"
            )

        # 经验闭环警告（C）：升级/复发维度直接塞 user message，强制 agent 看见（不依赖其自觉调工具）。
        # 必达 + trace 可见；升级维度要求换根本思路，复发维度提示勿重复上轮思路。
        if escalated_warnings:
            warn = "\n".join(f"- {w}" for w in escalated_warnings)
            text += (
                "\n\n【经验闭环警告（必须遵守）】\n"
                f"{warn}\n"
                "上述维度已被验证 prompt 微调无效或反复失败。本轮必须："
                "(a) 参考【本题已验证经验】里已试过的失败思路，勿重复；"
                "(b) 对⚠️升级维度换根本性方向（换 test_variable 如 reference_images/size，或上报人工），"
                "不要再微调同一 prompt 思路。"
            )

        # in-loop 经验精简摘要（A）：round>1 时由 generate_round 从 conclusions.json 渲染注入。
        # 必达 + trace 可见——让 agent 每轮都看到本题已验证的 ineffective（勿重复）/ effective（保持）。
        # 与 escalated_warnings 互补：警告给「换思路」指令，摘要给「具体试过什么、为何无效/有效」明细。
        if conclusions_brief:
            text += "\n\n" + conclusions_brief

        # 风格神韵迁移专属：要求 generator 产出一句话图片含义（供 Critic 判概念表达 + 最终图文方案文字）
        if sample.spec.task and sample.spec.task.mode == "style_transfer":
            text += (
                "\n\n【风格神韵迁移 · 必填 meaning】"
                "除 prompt + delta_note 外，**必须**用 meaning 字段输出一句话（≤40 字）图片含义解释："
                "这张图如何用视觉隐喻表达给定文章主题的概念。它是产出的一部分，也会被 Critic 用来判断概念表达。"
            )

        # 人工补充要求（运行时人工介入；每轮都加入，与 escalated_warnings 同级强制）
        if extra_hints:
            text += "\n\n【人工补充要求（必须遵守）】\n" + "\n".join(f"- {h}" for h in extra_hints)

        if not reference_images:
            return text
        parts: list[dict] = [{"type": "text", "text": text}]
        for p in reference_images:
            if p.exists():
                parts.append({"type": "image_url", "image_url": {"url": file_to_data_uri(p)}})
        return parts

    def _base_prompt_text(self, sample: Sample) -> str:
        """兜底 prompt（agent 出错时用）：指令+约束，无 LLM 润色。"""
        instr = (sample.spec.task.instruction if sample.spec.task else None) or "生成产品白底三视图素材图"
        constraints = self._extract_constraints(sample)
        base = instr
        if constraints:
            base += "\n约束: " + json.dumps(constraints, ensure_ascii=False)
        return base

    def _extract_constraints(self, sample: Sample) -> dict:
        """从 content_spec 读原始 constraints 字段（若存在）。"""
        try:
            raw = sample.spec.model_dump()
            return raw.get("constraints") or {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
        """把 recursion_limit 合进 config（透传 trace 嵌套用）。"""
        cfg: dict = dict(config) if config else {}
        cfg["recursion_limit"] = limit
        return cfg


__all__ = ["GenOutcome", "Generator", "PriorFeedback"]
