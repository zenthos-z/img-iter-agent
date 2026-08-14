"""Generator deepagent 的工具集（每轮由 generate_round 构建，闭包捕获每轮上下文）。

工具 = 策略扩展点。新增策略（reference_images / size / generation_mode / model_params）
= 新增一个 `make_*_tool` 工厂 + 在 generate_round 注册。不改外层图、不改 RunState。

当前工具：
  - generate_image(prompt, size, reference_images, model_id, edit_previous,
                   negative_prompt, seed, steps)：包 Router.generate，出三视图并落盘。
    动作空间 A 的 8 个策略杠杆：prompt / size / reference_images / model_id /
    edit_previous(多轮改图) / negative_prompt / seed / steps（各族尽力翻译，不支持静默忽略）。
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


def _clip(s: str | None, n: int) -> str:
    """压平并截断：经验渲染的条目是「指引」不是档案，全文动辄上千字符会灌爆上下文。"""
    s = (s or "").strip().replace("\n", "；")
    return s if len(s) <= n else s[: n - 1] + "…"


# query_experience 有界渲染：分组条数上限 + 每条截断 + 总字符预算。
# 背景：无界版实测 34 条结论 → 12 万字符一次性进上下文，模型没有余力完整判断考题要求。
_EXP_GROUP_CAPS = {"escalated": 4, "ineffective": 6, "effective": 4}
_EXP_ENTRY_CLIP = {"change": 80, "lesson": 160}
_EXP_CHAR_BUDGET = 3500


def _digest_section(digest: str, dim: str) -> str:
    """从 digest 提取与 dim 相关的条目（带所在小节标题），保持原文、不做截断。"""
    out: list[str] = []
    header = ""
    for line in digest.splitlines():
        s = line.strip()
        if s.startswith("#"):
            header = s
            continue
        if s and (f"[{dim}]" in s or dim in s):
            if header and (not out or out[-1] != header):
                out.append(header)
            out.append(s)
    return "\n".join(out)


def _format_experience(run_dir: Path, dim: str | None = None) -> str:
    """本题经验查询——**只读 LLM 压缩面（kb.digest）**，不读 raw conclusions 全文。

    digest 由 summarizer 每轮增量重写（语义合并/证伪同步/浓缩），篇幅由压缩 prompt 约束
    （≤800 字）——有界靠「源头就短」，消费侧不做字符截断。dim 参数 → 提取对应维度小节。
    旧 run 无 digest（summarizer 尚未跑过新代码）→ 一次性有界兜底渲染，下一轮即被 digest 取代。
    """
    kb = load_conclusions(run_dir)
    if kb.digest:
        if not dim:
            return kb.digest
        section = _digest_section(kb.digest, dim)
        return section or f"（digest 中无 [{dim}] 相关条目；不传 dim 可看全部摘要）"
    # —— legacy 兜底（无 digest 的旧 run，仅过渡一轮）——
    # 有界：分组上限+条目截断+总预算。raw 全文动辄 12 万字符，直接灌入会淹掉考题要求。
    groups = kb.verified_for_generator()
    effective = groups["effective"]  # 已只含 verified_effective（refuted 不会进）
    ineffective = groups["ineffective"]
    escalated = [c for c in groups.get("escalated", []) if c.status != "refuted"]
    if dim:
        effective = [c for c in effective if c.dim == dim]
        ineffective = [c for c in ineffective if c.dim == dim]
        escalated = [c for c in escalated if c.dim == dim]
    if not effective and not ineffective and not escalated:
        return "（暂无已验证经验）"

    def _recency_key(c: Any) -> tuple[int, int]:
        return (c.verified_round or c.created_round or 0, c.created_round or 0)

    sections: list[tuple[str, list[Any]]] = [
        ("escalated", escalated), ("ineffective", ineffective), ("effective", effective),
    ]
    headers = {
        "escalated": "【已升级（连续失败疑似模型上限，勿重复微调，需换根本方向）】",
        "ineffective": "【已验证无效（勿重复，需换思路）】",
        "effective": "【已验证有效（建议保持）】",
    }
    lines: list[str] = []
    omitted = 0
    used = 0
    for key, cs in sections:
        if not cs:
            continue
        ranked = sorted(cs, key=_recency_key, reverse=True)
        cap = _EXP_GROUP_CAPS[key]
        take, drop = ranked[:cap], ranked[cap:]
        omitted += len(drop)
        entries: list[str] = []
        for c in take:
            streak = kb.fail_streaks.get(c.dim, 0)
            tag = (f" (连续失败 {streak} 轮)" if key == "escalated"
                   else f" (连续失败 {streak} 轮)" if (key == "ineffective" and streak > 0) else "")
            entry = (f"- [{c.dim}]{tag} 「{_clip(c.change, _EXP_ENTRY_CLIP['change'])}」"
                     f"→ {_clip(c.lesson, _EXP_ENTRY_CLIP['lesson'])}")
            # 总预算：装不下的条目直接省略（保条目完整可读，不在句中腰斩）
            if used + len(entry) > _EXP_CHAR_BUDGET:
                omitted += len(take) - len(entries)
                break
            entries.append(entry)
            used += len(entry) + 1
        if entries:
            lines.append(headers[key])
            lines.extend(entries)
    if omitted > 0:
        lines.append(f"（已省略 {omitted} 条较早经验；传 dim='维度名' 可精查单维度，全文见 lessons/conclusions.json）")
    return "\n".join(lines)


def build_ref_registry(sample: Sample) -> dict[str, Path]:
    """{标识符: 路径}：target + 所有 input_assets。

    供 generate_image 的 reference_images 解析 + generator 提示展示可用参考图清单。
    image_edit/multiview 的 target 也在此注册（不再按 mode 单独强制注入工具）。
    """
    reg: dict[str, Path] = {}
    if sample.target_path.exists():
        reg["target"] = sample.target_path
    task = sample.spec.task
    if task and task.input_assets:
        for a in task.input_assets:
            p = sample.sample_dir / a
            if p.exists():
                reg.setdefault(p.stem, p)  # 'target' 已占位时不覆盖
    return reg


def _resolve_reference_images(
    requested: list[str], ref_registry: dict[str, Path], sample_dir: Path,
) -> tuple[list[Path], list[str]]:
    """把 agent 传的参考图标识符/路径解析成实际 Path。

    每项依次尝试：(1) ref_registry 标识符（如 'target'）；(2) 相对 sample 目录的路径；(3) 绝对路径。
    解析不到的进 unknown（工具忽略并告警）。去重。返回 (resolved_paths, unknown_ids)。
    """
    resolved: list[Path] = []
    unknown: list[str] = []
    seen: set[Path] = set()
    for ident in requested:
        p: Path | None = None
        if ident in ref_registry:
            p = ref_registry[ident]
        else:
            cand = sample_dir / ident
            if cand.exists():
                p = cand
            else:
                ap = Path(ident)
                if ap.exists():
                    p = ap
        if p is None or not p.exists():
            unknown.append(ident)
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            resolved.append(p)
    return resolved, unknown


class GenerateImageArgs(BaseModel):
    """generate_image 工具的参数 schema（显式 args_schema，避免 Gemini 严格后端拒 anyOf）。

    可选字段一律用「空值哨兵」而非 Optional：str="" / int=-1 = 未设置，生成干净的单一 type schema
    （与 reference_images=list[str] 同理；详见 _agent_output.py:36-42 的 anyOf 教训）。
    """

    prompt: str = Field(description="生图提示词（本轮最终采用的 prompt，应与结构化输出里的 prompt 一致）")
    size: str = Field(default="2K", description="尺寸，如 '2K' 或 '2048x2048'；默认 '2K'")
    reference_images: list[str] = Field(
        default_factory=list,
        description=(
            "参考图标识符或路径（由你主动决定是否用、用哪几张）。传 [] 或省略 = 纯文生图（不传任何参考图）；"
            "要用就给出标识符（如 ['target']）或图片路径，工具自动读取并上传给生图 API。可用标识符见用户消息。"
            "image_edit/multiview 等需还原 target 的任务**建议**传入 target；未知标识符/路径会被忽略并告警。"
        ),
    )
    # --- 动作空间 A 的非 prompt 杠杆（哨兵默认值；各族尽力翻译，不支持静默忽略）---
    model_id: str = Field(
        default="",
        description=(
            "【可选】显式选生图模型 model_id（限定用户消息给出的已配置集合）；空=按默认路由。"
            "不同模型实际尺寸/比例行为不同——三视图 16:9 宽幅只有 Gemini 生效，其余族落成方图。"
        ),
    )
    edit_previous: bool = Field(
        default=False,
        description=(
            "【可选】True=在上一版图基础上多轮改图（仅 D/Gemini 族生效；若实际用到非 D 族会自动"
            "退化为重新生成并告警）。False(默认)=从零重新生成。微调（只改局部）建议 True。"
        ),
    )
    negative_prompt: str = Field(
        default="",
        description="【可选】反向提示，要避免什么（如 'blurry, extra legs, distorted proportions'）。仅 Qwen 族生效，其余静默忽略；空=不设。",
    )
    seed: int = Field(
        default=-1,
        description="【可选】随机种子（可复现）。仅 Qwen 族生效，其余静默忽略；-1=不设。",
    )
    steps: int = Field(
        default=-1,
        description="【可选】采样步数。当前各族均不支持、静默忽略（预留扩展）；-1=不设。",
    )


def make_generate_image_tool(
    *,
    router: Router,
    out_dir: Path,
    run_dir: Path,
    sample_dir: Path,
    ref_registry: dict[str, Path],
    model_hint: ModelFamily | None,
    sink: dict[str, Any],
    aspect_ratio: str | None = None,
    prev_image: Path | None = None,
) -> BaseTool:
    """生成三视图并落盘的工具。sink 用于把 model/family/ref/reference_ids/非prompt杠杆 回传给 generate_round。

    - ref_registry：「参考标识符 → Path」映射（target + input_assets，见 build_ref_registry）。
    - sample_dir：sample 目录，用于解析 agent 传的相对图片路径。
    - 参考图**完全由 agent 决定**：reference_images 传标识符/路径 → 工具读取上传；[]/省略 = 纯文生图
      （工具不再因 image_edit/multiview 模式自动塞 target）。
    - aspect_ratio：三视图等宽幅任务传 "16:9"，让 family D 用宽 aspectRatio（避免 1:1 把三视图挤变形）。
    - prev_image：上一版生成图路径（baseline_ref 解析）。edit_previous=True 时作 conversation_history
      传给 D 族多轮改图；为 None（首轮）则 edit_previous 退化为重新生成并告警。
    """

    @tool(args_schema=GenerateImageArgs)
    def generate_image(
        prompt: str, size: str = "2K", reference_images: list[str] | None = None,
        model_id: str = "", edit_previous: bool = False,
        negative_prompt: str = "", seed: int = -1, steps: int = -1,
    ) -> str:
        """生成一张三视图排版图并落盘到本轮目录。

        Args:
            prompt: 生图提示词（本轮最终采用的 prompt，应与最终结构化输出里的 prompt 一致）。
            size: 尺寸，如 '2K' 或 '2048x2048'；默认 '2K'（实际由考题决定，工具按传入值执行）。
            reference_images: 参考图标识符或路径（由你决定是否用、用哪几张）。传 []/省略 = 纯文生图
                （不传任何参考图）；要用就给标识符（如 ['target']）或路径，工具自动读取上传。可用标识符见用户消息。
            model_id: 可选，显式选生图模型（限定用户消息给出的已配置集合）；空=按默认路由。
                不同模型实际尺寸/比例行为不同（三视图 16:9 宽幅仅 Gemini 生效）。
            edit_previous: 可选，True=在上一版图基础上改图（仅 D/Gemini 生效；非 D 自动退化重画并告警）。
            negative_prompt: 可选，反向提示（仅 Qwen 生效，其余忽略）；空=不设。
            seed: 可选，随机种子（仅 Qwen 生效）；-1=不设。
            steps: 可选，采样步数（当前各族均不支持，静默忽略）；-1=不设。
        返回生成图相对 run 目录的路径。
        """
        # —— 出图熔断 ——
        # 模型看不到生成的图、拿不到质量反馈，出图成功后常"换个 prompt 再试一版"打转
        # （生产实测 round5 连续 4+ 次 generate_image，每次 45~350s，还撞 recursion_limit
        # 后被 invoke_with_retry 整轮重放放大）。提示词的"只调一次"对 lite 模型不够硬，
        # 这里工具级硬拦：成功一次后拒绝再出图、连续失败封顶，逼模型立即收尾出结构化结果。
        if sink.get("ref"):
            return (
                f"本轮已成功生成图片（{sink['ref']}）。流程规定 generate_image 只成功调用一次，"
                "再次出图已被拒绝。请立即停止调用任何工具，直接以最终回复输出 GeneratorOutput "
                "JSON（prompt/meaning/delta_note/strategy_note）结束本轮。"
            )
        attempts = sink.get("generate_attempts", 0) + 1
        sink["generate_attempts"] = attempts
        if attempts > 3:
            return (
                f"generate_image 已连续失败 {attempts - 1} 次且无成功图，达到重试上限，不再出图。"
                "请基于已有信息直接以最终回复输出 GeneratorOutput JSON"
                "（prompt/meaning/delta_note/strategy_note）结束本轮。"
            )
        size_spec = _size_from_str(size)
        if aspect_ratio:
            size_spec.ratio = aspect_ratio
        # 参考图完全由 agent 决定：传标识符/路径 → 工具解析上传；[] = 纯文生图（不传任何参考图）
        requested = list(reference_images) if reference_images else []
        refs_to_use, unknown = _resolve_reference_images(requested, ref_registry, sample_dir)
        if unknown:
            sink.setdefault("warnings", []).append(
                f"generate_image: 未识别的参考图 {unknown}（可用标识符：{sorted(ref_registry)}，"
                "或给 sample 目录下的图片路径），已忽略"
            )
        sink["reference_ids"] = [p.stem for p in refs_to_use]  # 供 trajectory + creativity tuner

        # 多轮改图（edit_previous）：把上一版图作 conversation_history 传入（仅 D 族消费）
        conv_history: list[Path] = []
        if edit_previous:
            if prev_image is not None and prev_image.exists():
                conv_history = [prev_image]
            else:
                sink.setdefault("warnings", []).append(
                    "generate_image: edit_previous=True 但无上一版图可改（首轮或上轮未出图），改为重新生成"
                )

        req = GenRequest(
            prompt=prompt,
            size=size_spec,
            reference_images=refs_to_use,
            conversation_history=conv_history,
            model_hint=model_hint,
            model_id=model_id or None,
            negative_prompt=negative_prompt or None,
            seed=seed if seed >= 0 else None,
            steps=steps if steps >= 0 else None,
        )
        img = router.generate(req, out_dir=out_dir)

        # edit_previous 退化检测：想改图但实际用了非 D 族 → 自动退化为重画，告警
        effective_family = img.meta.get("family", "?")
        edit_effective = edit_previous and bool(conv_history) and effective_family == "D"
        if edit_previous and conv_history and effective_family != "D":
            sink.setdefault("warnings", []).append(
                f"generate_image: edit_previous=True 但实际用 {effective_family} 族（非 D），"
                "多轮改图不可用，已退化为重新生成"
            )

        ext = img.image_path.suffix or ".png"
        dest = out_dir / f"three_view{ext}"
        if img.image_path != dest:
            img.image_path.rename(dest)
        ref = str(dest.relative_to(run_dir))
        sink["ref"] = ref
        sink["model"] = img.model
        sink["family"] = effective_family
        # 动作空间 A 的非 prompt 杠杆落 sink（供 GenOutcome/trajectory/记忆捕获）
        sink["edit_previous"] = edit_effective
        sink["negative_prompt"] = negative_prompt or None
        sink["seed"] = seed if seed >= 0 else None
        sink["steps"] = steps if steps >= 0 else None
        sink["model_id_chosen"] = model_id or None
        sink["had_conversation_history"] = bool(img.meta.get("had_conversation_history"))
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
    prev_image: Path | None = None,
) -> list[BaseTool]:
    """组装本轮 Generator 工具集。sink 由调用方持有，工具运行时回写生成结果元信息。

    query_experience = 本 loop 单题经验（conclusions.json 全文钻取）。跨 loop 通用经验已改由
    本 benchmark 的蒸馏技能包承载（SkillsMiddleware 加载），不再走工具。
    """
    # 参考图清单：target + input_assets → {标识符: 路径}。agent 经 generate_image(reference_images=[...])
    # 主动决定用哪些（含 image_edit 是否带 target）；[] = 纯文生图。不再按 mode 强制注入 target。
    ref_registry = build_ref_registry(sample)
    # 三视图单图 → 宽幅，避免 1:1 把三个视图挤变形（比例失真）
    layout = (sample.spec.task.output.get("layout")
              if sample.spec.task and sample.spec.task.output else None)
    aspect_ratio = "16:9" if layout == "three_view_single_image" else None
    return [
        make_generate_image_tool(
            router=router, out_dir=out_dir, run_dir=run_dir,
            sample_dir=sample.sample_dir,
            ref_registry=ref_registry,
            model_hint=model_hint, sink=sink,
            aspect_ratio=aspect_ratio,
            prev_image=prev_image,
        ),
        make_query_experience_tool(run_dir=run_dir),
    ]


__all__ = [
    "build_ref_registry",
    "make_generate_image_tool",
    "make_generator_tools",
    "make_query_experience_tool",
]
