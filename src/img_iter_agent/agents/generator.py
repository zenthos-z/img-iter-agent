"""Generator：基于 Critic 反馈自动迭代优化 prompt 的生图调度者。

迭代核心（用户明确）：**自动优化生图提示词**，不是改尺寸。size 由考题固定，不动。
每轮：
  - 首轮：用考题 instruction + 约束，构造确定性 base prompt
  - 后续轮：读**上轮 Critic 的失败项反馈**，用 LLM 针对性地改进 prompt
    （如上轮"悬浮无阴影(A4)"→ 本轮 prompt 加"添加真实接地阴影"）
  - size 始终固定（来自 content_spec），test_variable 始终记为 "prompt"
  - baseline_ref 指向上轮 attempt（对比迭代效果）

LLM 走依赖注入：无 LLM 时用确定性 prompt（首轮可控、可离线测）；
有 LLM 时首轮润色、后续轮据反馈改进。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from ..data.benchmark import Sample
from ..generation.base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..generation.router import Router
from ..llm import LlmClient
from ..memory.schema import CriticItemJudgment, TestVariable
from .agent_config_loader import load_system_prompt

# 代码默认提示词（agents_config/generator.md 缺失时回退用）
_DEFAULT_REFINE_PROMPT = (
    "你是生图提示词工程师。把下面的生图指令精炼成一段清晰的生图 prompt"
    "（英文优先），保留所有关键约束，不要多余解释。"
)
_DEFAULT_IMPROVE_PROMPT = (
    "你是生图提示词优化师。下面是上一轮生图 prompt 和 Critic 指出的失败项。"
    "请基于失败项针对性改进 prompt（直接针对每个问题加具体描述来规避），"
    "输出改进后的完整 prompt（英文优先）。保留原有正确部分，只针对失败项改进。"
    "不要解释，直接输出新 prompt。"
)


class PriorFeedback(BaseModel):
    """上轮 Critic 反馈的摘要，供 Generator 改进 prompt。"""

    failed_items: list[CriticItemJudgment] = Field(default_factory=list)
    continuous_notes: list[str] = Field(default_factory=list)  # 连续维度的低分理由


class GenOutcome(BaseModel):
    """Generator 一轮的产出。"""

    attempt_id: str
    test_variable: TestVariable | None
    baseline_ref: str | None
    gen_mode: str
    prompt: str
    size: str
    reference_image_refs: list[str]  # 相对 run 目录
    output_image_refs: list[str]  # 相对 run 目录（三视图一张图）
    model: str
    model_family: str
    improved_from_feedback: bool = False  # 本轮 prompt 是否基于上轮反馈改进
    delta_note: str | None = None  # 本轮相对上轮的改动说明（供经验闭环沉淀）


def _new_attempt_id(round: int) -> str:
    return f"a{round:03d}_{uuid.uuid4().hex[:6]}"


class Generator:
    """生成器。Router 与 LlmClient 注入。"""

    def __init__(self, router: Router, *, llm: LlmClient | None = None) -> None:
        self.router = router
        self.llm = llm

    def generate_round(
        self,
        *,
        sample: Sample,
        out_dir: Path,
        run_dir: Path,
        round: int,
        baseline_ref: str | None = None,
        prior_feedback: PriorFeedback | None = None,
        model_hint: ModelFamily | None = None,
    ) -> GenOutcome:
        """跑一轮：构造/改进 prompt → 单次生成【一张三视图排版图】→ 记录参数。

        size 由考题固定（不动）。首轮用确定性 prompt；后续轮据 prior_feedback 改进 prompt，
        并读经验知识库（effective/ineffective）注入上下文，让改进"知道之前什么有效/无效"。
        """
        spec = sample.spec
        task = spec.task
        size_str = (task.output.get("size") if task and task.output else None) or "2K"
        size = _size_from_str(size_str)  # 固定，不随轮次变

        # 参考：image_edit 模式用 target 作风格锚
        reference_images: list[Path] = []
        gen_mode = "text_to_image"
        if task and task.mode in ("image_edit", "multiview") and sample.target_path.exists():
            reference_images = [sample.target_path]
            gen_mode = "image_edit"

        # 构造/改进 prompt + 记录本轮改动说明（delta_note）
        base_prompt = self._base_prompt(sample)
        improved = False
        delta_note: str | None = None  # 本轮相对上轮改了什么（供经验闭环沉淀）
        # 多策略基建：按 test_variable 分派改进策略。
        # 当前只实现 prompt 策略；未来 reference_images/size/generation_mode 在此扩展分支。
        if round > 1 and prior_feedback and prior_feedback.failed_items:
            # === 策略: prompt（当前唯一实现）===
            # 读经验知识库：让改进知道"之前什么有效/无效"
            knowledge_ctx = self.knowledge_context(run_dir)
            prompt, delta_note = self._improve_prompt(base_prompt, prior_feedback, knowledge_ctx)
            improved = True
        else:
            # === 策略: 首轮基线（无改进）===
            prompt = base_prompt

        attempt_id = _new_attempt_id(round)
        attempt_out = out_dir / attempt_id
        attempt_out.mkdir(parents=True, exist_ok=True)

        # 单次生成一张三视图排版图
        req = GenRequest(
            prompt=prompt,
            size=size,
            reference_images=reference_images,
            model_hint=model_hint,
        )
        img: GeneratedImage = self.router.generate(req, out_dir=attempt_out)
        ext = img.image_path.suffix or ".png"
        dest = attempt_out / f"three_view{ext}"
        if img.image_path != dest:
            img.image_path.rename(dest)
        output_refs = [str(dest.relative_to(run_dir))]

        # 参考图（benchmark target）不在 run 目录内，存绝对路径；产出图存相对 run 目录
        ref_refs = [str(p.resolve()) for p in reference_images]

        return GenOutcome(
            attempt_id=attempt_id,
            test_variable="prompt" if round > 1 else None,  # 始终在优化 prompt
            baseline_ref=baseline_ref,
            gen_mode=gen_mode,
            prompt=prompt,
            delta_note=delta_note,
            size=size_str,
            reference_image_refs=ref_refs,
            output_image_refs=output_refs,
            model=img.model,
            model_family=img.meta.get("family", "?"),
            improved_from_feedback=improved,
        )

    def _base_prompt(self, sample: Sample) -> str:
        """首轮确定性 prompt：考题 instruction + 约束。可选用 LLM 润色措辞。"""
        spec = sample.spec
        instr = (spec.task.instruction if spec.task else None) or "生成产品白底素材图"
        constraints = self._extract_constraints(sample)
        base = instr
        if constraints:
            base += "\n约束: " + json.dumps(constraints, ensure_ascii=False)
        if self.llm is None:
            return base
        # 首轮用 LLM 润色成生图友好的 prompt（一次性，不引入逐轮随机）
        msgs = [
            {"role": "system", "content": load_system_prompt("generator", _DEFAULT_REFINE_PROMPT)},
            {"role": "user", "content": base},
        ]
        return self.llm.complete(msgs).strip() or base

    def _improve_prompt(
        self, current_prompt: str, feedback: PriorFeedback, knowledge_ctx: str = ""
    ) -> tuple[str, str]:
        """基于上轮 Critic 失败反馈改进 prompt。

        返回 (new_prompt, delta_note)：delta_note 描述本轮针对哪些问题做了什么改动，
        供经验闭环沉淀（Summarizer 用它 + Critic 前后 verdict 判定改动是否有效）。
        knowledge_ctx：经验知识库上下文（effective/ineffective），引导改进方向。
        """
        if self.llm is None:
            # 无 LLM 时，确定性补强：把失败项理由追加为正向要求
            fixes = [f"确保: {it.reason}" for it in feedback.failed_items]
            delta_note = "针对失败项 " + ", ".join(it.id for it in feedback.failed_items) + " 追加正向要求"
            return current_prompt + "\n改进点:\n" + "\n".join(fixes), delta_note

        fails = "\n".join(f"- [{it.id}] {it.reason}" for it in feedback.failed_items)
        cont = ""
        if feedback.continuous_notes:
            cont = "\n连续维度低分:\n" + "\n".join(f"- {n}" for n in feedback.continuous_notes)
        # 要求 LLM 同时输出「改动说明」+「新 prompt」，用分隔符解析
        sys_prompt = load_system_prompt("generator_improve", _DEFAULT_IMPROVE_PROMPT)
        knowledge_block = f"\n\n【已验证的经验】\n{knowledge_ctx}\n" if knowledge_ctx else ""
        msgs = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"【上一轮 prompt】\n{current_prompt}\n\n"
                    f"【上轮失败项(需改进)】\n{fails}{cont}{knowledge_block}\n\n"
                    "请输出两部分，用一行 '---PROMPT---' 分隔：\n"
                    "第一部分：本次改动的简要说明（针对哪些失败项做了什么；参考有效经验、避开无效尝试）\n"
                    "第二部分：改进后的完整 prompt"
                ),
            },
        ]
        raw = self.llm.complete(msgs).strip() or current_prompt
        # 解析两部分
        if "---PROMPT---" in raw:
            note_part, prompt_part = raw.split("---PROMPT---", 1)
            delta_note = note_part.strip()
            return prompt_part.strip() or current_prompt, delta_note
        # 兜底：LLM 没按格式，整体当 prompt，delta_note 粗略描述
        delta_note = "基于失败项改进 prompt（LLM 未返回结构化改动说明）"
        return raw, delta_note

    def _extract_constraints(self, sample: Sample) -> dict:
        """从 content_spec 读原始 constraints 字段（若存在）。"""
        try:
            raw = sample.spec.model_dump()
            return raw.get("constraints") or {}
        except Exception:  # noqa: BLE001
            return {}

    def knowledge_context(self, run_dir: Path) -> str:
        """读取经验知识库，构造给下轮 prompt 改进的上下文。

        闭环关键：让 Generator 知道"之前试过什么有效/无效"。
        - effective：已验证有效的改动，提示保持该方向
        - ineffective：已验证无效的改动，提示别再试、需换思路
        无结论时返回空串（首轮或无经验）。
        """
        from ..memory.knowledge import load_conclusions

        kb = load_conclusions(run_dir)
        groups = kb.verified_for_generator()
        effective, ineffective = groups["effective"], groups["ineffective"]
        if not effective and not ineffective:
            return ""
        lines = []
        if effective:
            lines.append("【已验证有效的经验（建议保持）】")
            for c in effective:
                lines.append(f"- [{c.dim}] {c.change} → {c.lesson}")
        if ineffective:
            lines.append("【已验证无效的尝试（勿重复，需换思路）】")
            for c in ineffective:
                lines.append(f"- [{c.dim}] {c.change} → {c.lesson}")
        return "\n".join(lines)


def _size_from_str(s: str) -> SizeSpec:
    """'2K' / '2048x2048' / '2048*2048' → SizeSpec。"""
    s = s.strip()
    if s.upper() in {"1K", "2K", "3K", "4K"}:
        return SizeSpec(tier=s.upper())
    for sep in ("x", "*"):
        if sep in s:
            try:
                w, h = s.split(sep)
                return SizeSpec(pixels=(int(w), int(h)))
            except ValueError:
                break
    return SizeSpec(tier="2K")


__all__ = ["GenOutcome", "Generator", "PriorFeedback"]
