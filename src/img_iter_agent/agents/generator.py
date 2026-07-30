"""Generator：测试调度者 + prompt 构造 + 调生图。

作为「测试调度者」（ARCH §2.5），每轮做控制变量法：
  - 声明本轮变了哪个维度 test_variable（prompt/reference_images/size/generation_mode/model_params；
    模型本身固定，不在此列）
  - 指向对照组 baseline_ref（其余维度固定的那次 attempt_id）
  - 构造 GenRequest（三视图任务：对每个视角生成一张）
  - 调 Router 出图，记录参数

prompt 构造可选用 LLM（把考题 instruction + 约束 + 经验揉成生图指令），
但**默认走确定性构造**（直接用 content_spec.instruction），这样无 key 也能测、
且首轮可控。LLM 走依赖注入。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..data.benchmark import Sample
from ..generation.base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..generation.router import Router
from ..llm import LlmClient
from ..memory.schema import TestVariable


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
    output_image_refs: list[str]  # 相对 run 目录（三视图）
    model: str
    model_family: str


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
        test_variable: TestVariable | None = None,
        size_spec: SizeSpec | None = None,
        model_hint: ModelFamily | None = None,
    ) -> GenOutcome:
        """跑一轮：构造 prompt → 对每个视角出图 → 记录参数。

        三视图任务：对 content_spec.task.output.views 里每个视角各生成一张。
        （多数模型单次出 1 张，故按视角循环调用。）
        """
        spec = sample.spec
        task = spec.task
        views = (task.output.get("views") if task and task.output else None) or ["main"]
        size_str = (task.output.get("size") if task and task.output else None) or "2K"
        size = size_spec or _size_from_str(size_str)

        # 参考：image_edit 模式用 target 作风格锚
        reference_images: list[Path] = []
        gen_mode = "text_to_image"
        if task and task.mode in ("image_edit", "multiview") and sample.target_path.exists():
            reference_images = [sample.target_path]
            gen_mode = "image_edit"

        prompt = self._build_prompt(sample, views)
        attempt_id = _new_attempt_id(round)
        attempt_out = out_dir / attempt_id
        attempt_out.mkdir(parents=True, exist_ok=True)

        output_refs: list[str] = []
        used_model = ""
        used_family = ""
        for view in views:
            view_prompt = f"{prompt}\n[视角: {view}]"
            req = GenRequest(
                prompt=view_prompt,
                size=size,
                reference_images=reference_images,
                model_hint=model_hint,
            )
            img: GeneratedImage = self.router.generate(req, out_dir=attempt_out)
            # 重命名为 <view>.<ext>
            ext = img.image_path.suffix or ".png"
            dest = attempt_out / f"{view}{ext}"
            if img.image_path != dest:
                img.image_path.rename(dest)
            output_refs.append(str(dest.relative_to(run_dir)))
            used_model = img.model
            used_family = img.meta.get("family", "?")

        # 参考图（benchmark 的 target）不在 run 目录内，存绝对路径；产出图存相对 run 目录路径
        ref_refs = [str(p.resolve()) for p in reference_images]

        return GenOutcome(
            attempt_id=attempt_id,
            test_variable=test_variable,
            baseline_ref=baseline_ref,
            gen_mode=gen_mode,
            prompt=prompt,
            size=size_str,
            reference_image_refs=ref_refs,
            output_image_refs=output_refs,
            model=used_model,
            model_family=used_family,
        )

    def _build_prompt(self, sample: Sample, views: list[str]) -> str:
        """构造生图指令。默认确定性；若注入了 LLM 则用它润色（带经验时更有效）。"""
        spec = sample.spec
        instr = (spec.task.instruction if spec.task else None) or "生成产品白底素材图"
        # 取 content_spec 的约束（若存在；约束在 content_spec.constraints）
        constraints = self._extract_constraints(sample)
        base = instr
        if constraints:
            base += "\n约束: " + json.dumps(constraints, ensure_ascii=False)

        if self.llm is None:
            return base
        # 用 LLM 把指令+约束揉成更精炼的生图提示词
        msgs = [
            {"role": "system", "content": "你是生图提示词工程师。把下面的生图指令精炼成一句清晰的英文/中文生图 prompt，不要多余解释。"},
            {"role": "user", "content": base},
        ]
        return self.llm.complete(msgs).strip() or base

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


__all__ = ["GenOutcome", "Generator"]
