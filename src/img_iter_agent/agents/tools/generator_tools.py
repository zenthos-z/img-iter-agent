"""Generator deepagent 的工具集（每轮由 generate_round 构建，闭包捕获每轮上下文）。

工具 = 策略扩展点。新增策略（reference_images / size / generation_mode / model_params）
= 新增一个 `make_*_tool` 工厂 + 在 generate_round 注册。不改外层图、不改 RunState。

当前工具：
  - generate_image(prompt, size)：包 Router.generate，出三视图并落盘。
  - query_experience(dim)：包经验知识库（conclusions.json），返回已验证有效/无效经验。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ...data.benchmark import Sample
from ...generation.base import GenRequest, ModelFamily, SizeSpec
from ...generation.router import Router
from ...memory.knowledge import load_conclusions


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


def _format_experience(run_dir: Path, dim: str | None = None) -> str:
    """读 conclusions.json，格式化已验证有效/无效/已升级经验。

    escalated 分组（B 复发检测）：该 dim 连续失败已撞模型能力上限，标注连续失败轮数，
    提示 generator 勿再 prompt 微调、需换根本方向。
    """
    kb = load_conclusions(run_dir)
    groups = kb.verified_for_generator()
    effective = groups["effective"]
    ineffective = groups["ineffective"]
    escalated = groups.get("escalated", [])
    if dim:
        effective = [c for c in effective if c.dim == dim]
        ineffective = [c for c in ineffective if c.dim == dim]
        escalated = [c for c in escalated if c.dim == dim]
    if not effective and not ineffective and not escalated:
        return "（暂无已验证经验）"
    lines: list[str] = []
    if escalated:
        lines.append("【已升级（连续失败疑似模型上限，勿重复微调，需换根本方向）】")
        for c in escalated:
            streak = kb.fail_streaks.get(c.dim, 0)
            lines.append(f"- [{c.dim}] (连续失败 {streak} 轮) {c.change} → {c.lesson}")
    if effective:
        lines.append("【已验证有效（建议保持）】")
        for c in effective:
            lines.append(f"- [{c.dim}] {c.change} → {c.lesson}")
    if ineffective:
        lines.append("【已验证无效（勿重复，需换思路）】")
        for c in ineffective:
            streak = kb.fail_streaks.get(c.dim, 0)
            tag = f" (连续失败 {streak} 轮)" if streak > 0 else ""
            lines.append(f"- [{c.dim}]{tag} {c.change} → {c.lesson}")
    return "\n".join(lines)


class GenerateImageArgs(BaseModel):
    """generate_image 工具的参数 schema（显式 args_schema，避免 Gemini 严格后端拒 anyOf）。

    reference_images 用 list[str] + default_factory=list（非 list[str] | None），生成干净的
    array schema；[] = 纯文生图。详见 _agent_output.py:36-42 的 anyOf 教训。
    """

    prompt: str = Field(description="生图提示词（本轮最终采用的 prompt，应与结构化输出里的 prompt 一致）")
    size: str = Field(default="2K", description="尺寸，如 '2K' 或 '2048x2048'；默认 '2K'")
    reference_images: list[str] = Field(
        default_factory=list,
        description=(
            "【风格迁移场景可选】参考图标识符子集，如 ['hand-abacus', 'object-laptop']。"
            "Gemini 把参考图作为 inline_data 风格条件。传 [] 或省略 = 纯文生图（合法且常更原创）。"
            "推荐 0-2 张：>2 张会过度锚定到参考的具体 motif、压制原创（creativity 的 reference_independence "
            "维度会扣分）。可用标识符见用户消息；未知标识符会被忽略并告警。"
        ),
    )


def make_generate_image_tool(
    *,
    router: Router,
    out_dir: Path,
    run_dir: Path,
    fallback_refs: list[Path],
    ref_registry: dict[str, Path] | None,
    model_hint: ModelFamily | None,
    sink: dict[str, Any],
    aspect_ratio: str | None = None,
) -> BaseTool:
    """生成三视图并落盘的工具。sink 用于把 model/family/ref/reference_ids 回传给 generate_round。

    - fallback_refs：image_edit/multiview 模式下作固定风格锚的参考（如 target）；style_transfer 不用。
    - ref_registry：style_transfer 模式下「参考标识符(stem) → Path」映射，让 agent 通过工具参数
      `reference_images` 主动选参考子集（解禁）；为 None 表示该模式不由 agent 控制参考（用 fallback_refs）。
    - aspect_ratio：三视图等宽幅任务传 "16:9"，让 family D 用宽 aspectRatio（避免 1:1 把三视图挤变形）。
    """

    @tool(args_schema=GenerateImageArgs)
    def generate_image(
        prompt: str, size: str = "2K", reference_images: list[str] | None = None,
    ) -> str:
        """生成一张三视图排版图并落盘到本轮目录。

        Args:
            prompt: 生图提示词（本轮最终采用的 prompt，应与最终结构化输出里的 prompt 一致）。
            size: 尺寸，如 '2K' 或 '2048x2048'；默认 '2K'（实际由考题决定，工具按传入值执行）。
            reference_images: 风格迁移场景可选，传参考图标识符子集（如 ['hand-abacus']）。Gemini 把它们
                作为 inline_data 风格条件；推荐 0-2 张，>2 张过度锚定 motif、压制原创（reference_independence
                维度会扣分）。传 [] 或省略 = 纯文生图。未知标识符忽略并告警。
        返回生成图相对 run 目录的路径。
        """
        size_spec = _size_from_str(size)
        if aspect_ratio:
            size_spec.ratio = aspect_ratio
        # 解析参考图：style_transfer 由 agent 选标识符（ref_registry）；image_edit/multiview 用固定 fallback_refs
        requested_ids = list(reference_images) if reference_images else []
        if ref_registry is not None:
            selected: list[Path] = []
            unknown: list[str] = []
            for ident in requested_ids:
                p = ref_registry.get(ident)
                if p is not None:
                    selected.append(p)
                else:
                    unknown.append(ident)
            if unknown:
                sink.setdefault("warnings", []).append(
                    f"generate_image: 未知参考标识符 {unknown}（可用：{sorted(ref_registry)}），已忽略"
                )
            refs_to_use = selected
        else:
            refs_to_use = list(fallback_refs)
        sink["reference_ids"] = [p.stem for p in refs_to_use]  # 供 trajectory + creativity tuner
        req = GenRequest(
            prompt=prompt,
            size=size_spec,
            reference_images=refs_to_use,
            model_hint=model_hint,
        )
        img = router.generate(req, out_dir=out_dir)
        ext = img.image_path.suffix or ".png"
        dest = out_dir / f"three_view{ext}"
        if img.image_path != dest:
            img.image_path.rename(dest)
        ref = str(dest.relative_to(run_dir))
        sink["ref"] = ref
        sink["model"] = img.model
        sink["family"] = img.meta.get("family", "?")
        return f"已生成三视图：{ref}"

    return generate_image


def make_query_experience_tool(*, run_dir: Path) -> BaseTool:
    @tool
    def query_experience(dim: str = "") -> str:
        """查询经验知识库里已验证的有效/无效经验。

        Args:
            dim: 可选，只看某个评分维度（如 'consistency'）；留空看全部。
        无经验时返回提示文本。
        """
        return _format_experience(run_dir, dim or None)

    return query_experience


def make_generator_tools(
    *,
    router: Router,
    sample: Sample,
    out_dir: Path,
    run_dir: Path,
    model_hint: ModelFamily | None,
    sink: dict[str, Any],
) -> list[BaseTool]:
    """组装本轮 Generator 工具集。sink 由调用方持有，工具运行时回写生成结果元信息。

    query_experience = 本 loop 单题经验（conclusions.json 全文钻取）。跨 loop 通用经验已改由
    本 benchmark 的蒸馏技能包承载（SkillsMiddleware 加载），不再走工具。
    """
    # 参考：image_edit/multiview 用 target 作固定风格锚（fallback_refs）。
    # style_transfer：解禁——agent 可通过 generate_image(reference_images=[...]) 主动选参考子集
    # （ref_registry: 标识符(stem) → Path）；不强制传，[] = 纯文生图。创造力维度反制过度锚定。
    fallback_refs: list[Path] = []
    ref_registry: dict[str, Path] = {}
    mode = sample.spec.task.mode if sample.spec.task else None
    if mode in ("image_edit", "multiview") and sample.target_path.exists():
        fallback_refs = [sample.target_path]
    if mode == "style_transfer" and sample.spec.task and sample.spec.task.input_assets:
        for a in sample.spec.task.input_assets:
            p = sample.sample_dir / a
            if p.exists():
                ref_registry[p.stem] = p
    # 三视图单图 → 宽幅，避免 1:1 把三个视图挤变形（比例失真）
    layout = (sample.spec.task.output.get("layout")
              if sample.spec.task and sample.spec.task.output else None)
    aspect_ratio = "16:9" if layout == "three_view_single_image" else None
    return [
        make_generate_image_tool(
            router=router, out_dir=out_dir, run_dir=run_dir,
            fallback_refs=fallback_refs,
            ref_registry=ref_registry or None,
            model_hint=model_hint, sink=sink,
            aspect_ratio=aspect_ratio,
        ),
        make_query_experience_tool(run_dir=run_dir),
    ]


__all__ = [
    "make_generate_image_tool",
    "make_generator_tools",
    "make_query_experience_tool",
]
