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
from dataclasses import dataclass, field
from pathlib import Path

from ..data.benchmark import Sample
from ..generation.base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..generation.router import Router
from ..llm import LlmClient
from ..memory.schema import CriticItemJudgment, TestVariable


@dataclass
class PriorFeedback:
    """上轮 Critic 反馈的摘要，供 Generator 改进 prompt。"""

    failed_items: list[CriticItemJudgment] = field(default_factory=list)
    continuous_notes: list[str] = field(default_factory=list)  # 连续维度的低分理由


@dataclass
class GenOutcome:
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

        size 由考题固定（不动）。首轮用确定性 prompt；后续轮据 prior_feedback 改进 prompt。
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

        # 构造/改进 prompt
        base_prompt = self._base_prompt(sample)
        improved = False
        if prior_feedback and prior_feedback.failed_items:
            prompt = self._improve_prompt(base_prompt, prior_feedback)
            improved = True
        else:
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
            {"role": "system", "content": "你是生图提示词工程师。把下面的生图指令精炼成一段清晰的生图 prompt（英文优先），保留所有关键约束，不要多余解释。"},
            {"role": "user", "content": base},
        ]
        return self.llm.complete(msgs).strip() or base

    def _improve_prompt(self, current_prompt: str, feedback: PriorFeedback) -> str:
        """基于上轮 Critic 失败反馈改进 prompt。

        把失败项（带理由）喂给 LLM，让它针对性补强 prompt（如缺阴影→加阴影描述）。
        """
        if self.llm is None:
            # 无 LLM 时，确定性补强：把失败项理由追加为正向要求
            fixes = [f"确保: {it.reason}" for it in feedback.failed_items]
            return current_prompt + "\n改进点:\n" + "\n".join(fixes)

        fails = "\n".join(f"- [{it.id}] {it.reason}" for it in feedback.failed_items)
        cont = ""
        if feedback.continuous_notes:
            cont = "\n连续维度低分:\n" + "\n".join(f"- {n}" for n in feedback.continuous_notes)
        msgs = [
            {
                "role": "system",
                "content": (
                    "你是生图提示词优化师。下面是上一轮生图 prompt 和 Critic 指出的失败项。"
                    "请基于失败项针对性改进 prompt（直接针对每个问题加具体描述来规避），"
                    "输出改进后的完整 prompt（英文优先）。保留原有正确部分，只针对失败项改进。"
                    "不要解释，直接输出新 prompt。"
                ),
            },
            {
                "role": "user",
                "content": f"【上一轮 prompt】\n{current_prompt}\n\n【上轮失败项(需改进)】\n{fails}{cont}",
            },
        ]
        return self.llm.complete(msgs).strip() or current_prompt

    def _extract_constraints(self, sample: Sample) -> dict:
        """从 content_spec 读原始 constraints 字段（若存在）。"""
        try:
            raw = sample.spec.model_dump()
            return raw.get("constraints") or {}
        except Exception:  # noqa: BLE001
            return {}


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
