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

from ...data.benchmark import Sample
from ...generation.base import GenRequest, ModelFamily, SizeSpec
from ...generation.router import Router
from ...memory.experience import load_general_experience, render_experience_skill_md
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


def make_generate_image_tool(
    *,
    router: Router,
    out_dir: Path,
    run_dir: Path,
    reference_images: list[Path],
    model_hint: ModelFamily | None,
    sink: dict[str, Any],
    aspect_ratio: str | None = None,
) -> BaseTool:
    """生成三视图并落盘的工具。sink 用于把 model/family/ref 回传给 generate_round。

    aspect_ratio：三视图等宽幅任务传 "16:9"，让 family D 用宽 aspectRatio（避免 1:1 把三视图挤变形）。
    """

    @tool
    def generate_image(prompt: str, size: str = "2K") -> str:
        """生成一张三视图排版图并落盘到本轮目录。

        Args:
            prompt: 生图提示词（本轮最终采用的 prompt，应与最终结构化输出里的 prompt 一致）。
            size: 尺寸，如 '2K' 或 '2048x2048'；默认 '2K'（实际由考题决定，工具按传入值执行）。
        返回生成图相对 run 目录的路径。
        """
        size_spec = _size_from_str(size)
        if aspect_ratio:
            size_spec.ratio = aspect_ratio
        req = GenRequest(
            prompt=prompt,
            size=size_spec,
            reference_images=list(reference_images),
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


def make_query_general_experience_tool(data_root: Path | None, bench_id: str | None) -> BaseTool:
    @tool
    def query_general_experience(dim: str = "", category: str = "") -> str:
        """查询【跨 loop 通用经验库】（多 run 蒸馏出的 dos/donts，跨题复用）。

        比 query_experience 更通用：从许多 loop 综合出的先验知识，本题首次跑也用得上。
        只返回 active 经验（refuted/superseded/archived 不消费）。
        Args:
            dim: 可选，只看某维度（如 'consistency'）。
            category: 可选，只看某粗类（如 '材质色彩'，对应 system prompt 索引里的类目）。
        无匹配经验时返回提示（可在总览点「重新蒸馏」生成）。
        """
        if data_root is None or bench_id is None:
            return "（跨 loop 经验未配置：无 data_root/bench_id）"
        exp = load_general_experience(data_root, bench_id)
        lessons = [l for l in exp.lessons if l.status == "active"]
        if dim:
            lessons = [l for l in lessons if l.dim == dim]
        if category:
            lessons = [l for l in lessons if (l.category or "(未分类)") == category]
        if not lessons:
            return "（暂无匹配的跨 loop 通用经验；可在总览点「重新蒸馏」生成）"
        return render_experience_skill_md(
            exp.model_copy(update={"lessons": lessons}), frontmatter=False
        ).strip()

    return query_general_experience


def make_generator_tools(
    *,
    router: Router,
    sample: Sample,
    out_dir: Path,
    run_dir: Path,
    model_hint: ModelFamily | None,
    sink: dict[str, Any],
    data_root: Path | None = None,
    bench_id: str | None = None,
) -> list[BaseTool]:
    """组装本轮 Generator 工具集。sink 由调用方持有，工具运行时回写生成结果元信息。

    query_experience = 本 loop 单题经验；query_general_experience = 跨 loop 通用经验（先验）。
    """
    # 参考：image_edit/multiview 用 target 作风格锚；style_transfer 不传参考给生图 API
    # （参考只通过 _build_user_content 给 agent 看，避免固定参考锚定生图 → 同质化）
    reference_images: list[Path] = []
    mode = sample.spec.task.mode if sample.spec.task else None
    if mode in ("image_edit", "multiview") and sample.target_path.exists():
        reference_images = [sample.target_path]
    # 三视图单图 → 宽幅，避免 1:1 把三个视图挤变形（比例失真）
    layout = (sample.spec.task.output.get("layout")
              if sample.spec.task and sample.spec.task.output else None)
    aspect_ratio = "16:9" if layout == "three_view_single_image" else None
    return [
        make_generate_image_tool(
            router=router, out_dir=out_dir, run_dir=run_dir,
            reference_images=reference_images, model_hint=model_hint, sink=sink,
            aspect_ratio=aspect_ratio,
        ),
        make_query_experience_tool(run_dir=run_dir),
        make_query_general_experience_tool(data_root, bench_id),
    ]


__all__ = [
    "make_generate_image_tool",
    "make_generator_tools",
    "make_query_experience_tool",
    "make_query_general_experience_tool",
]
