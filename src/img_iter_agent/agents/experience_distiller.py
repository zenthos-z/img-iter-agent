"""独立「经验蒸馏」deepagent：跨 loop 总结通用经验（视觉 + 全档案）。

与 in-loop Summarizer 解耦——本 agent 纯离线，由 CLI/web 触发，跨 run 归纳可复用的通用经验，
写到 `<data_root>/experience/<bench>/general.json`。

**视觉 + 全档案**：每个 loop 注入完整档案——逐轮【生图 prompt + Critic 逐项判断(C1✓/C3✗理由) +
验证过的 effective/ineffective 结论 + 还原度】，并附该 loop **最佳轮 & 最差轮的生成图**（视觉证据）。
让蒸馏器「眼见为实」——数字分数有偏差时，它能看图判断真实质量。用视觉模型（gemini）。

实现：数据预取进 prompt + 无探索工具（曾用 4 工具让 agent 自查，deepseek 陷入 query_run 退化循环）；
response_format=DistilledExperience 结构化输出。agent 跑飞 → 安全降级。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from ..data.benchmark import LoadedBenchmark
from ..data.trajectory import TrajectoryReader
from ..memory.experience import (
    AuthoredSkill,
    DistilledLesson,
    GeneralExperience,
    RenoItem,
    RenovationPlan,
    assemble_skill_package,
    new_lesson_id,
    render_lessons_reference_md,
    skill_package_dir,
    slugify_bench,
)
from ..memory.knowledge import load_conclusions
from ..memory.schema import Benchmark
from ._narrow_tools import DISTILLER_RECURSION_LIMIT, invoke_with_retry, narrow_tools_middleware
from .agent_config_loader import load_system_prompt

_DEFAULT_DISTILLER_SYS = (
    "你是生图经验蒸馏员（视觉），做**带反馈的迭代翻新**而非从头重抽。用户消息含：① 上一版通用经验"
    "（active + 已废弃）；② 每个 loop 的逐轮档案（prompt + Critic 逐项判断 + 验证结论 + 还原度）+ "
    "最佳/最差轮生成图；③ 每个 dim 的近期失败率趋势。\n"
    "**核心交付物是 renovation（翻新计划）**：对【上一版每条 active lesson】逐一判定 "
    "`keep`（证据仍支持，原样保留）/ `revise`（需修订：旧将标 superseded）/ `retire`（已证伪/失效，标 refuted）；"
    "并追加 `new` 覆盖档案里**未被旧经验捕捉**的全新反复失败。\n"
    "**关键：避免重复堆积**——若新归纳与某条上一版 active lesson **同 dim 且语义相近/是其精修**，"
    "**必须用 revise**（existing_id=那条的 id）而非 new；只有真正全新的 dim/主题才用 new。\n"
    "**每条 renovation 的 lesson 字段必填**（含 keep/retire）：keep 时复制旧 lesson 内容，retire 时给其 dim+insight。\n"
    "每条 lesson 填齐：dim、insight、dos、donts、evidence（来源 run/轮）、confidence（0-1）、"
    "category（粗类，**复用上一版给出的 category 清单**，无合适项才新增）、"
    "applies_when（construction=首轮构造用 / fix=修复失败用 / always=通用）。\n"
    "判定**眼见为实**：结合图真实质量 + Critic 逐项判断 + 验证过的 effective/ineffective + 近期失败率趋势"
    "（某 dim 仍高失败→相关 lesson 可能 revise/retire；已稳定→keep）。已废弃(refuted/archived)的旧 lesson"
    " **不要复活**，除非有强证据。不要调用任何工具，直接结构化输出 summary + renovation。"
)

# skill-author 技能包目录（魔改版 skill-creator，经验 skill 专用）。
# 内容（SKILL.md + 写作指南 + 质量 checklist）注入 authoring agent 上下文——NarrowToolsMiddleware
# 剥掉了 read_file，agent 不能运行时读这些文件，故调用前由代码读入拼进 system prompt。
_SKILL_AUTHOR_DIR = Path(__file__).parent / "skill_authoring" / "skill-author"


def _read_skill_author_file(rel: str) -> str:
    """读 bundled skill-author 内的文件（SKILL.md / references/*.md）；缺失返回空串。"""
    try:
        return (_SKILL_AUTHOR_DIR / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


# 方法论兜底（bundled 文件意外缺失时用，保证 agent 仍有基本指导）。
_SKILL_AUTHOR_FALLBACK = (
    "你是经验技能编写员，遵循 skill-creator 规范（详见 references/）。产出规范可移植技能包："
    "description（pushy, 做什么+何时触发, 通用不窄化, 无尖括号, <1024）、"
    "skill_md（<500 行, 冷启动可用, 含可执行工作流+输出格式模板(strategy 可扩展), 前景化 3-5 条关键 dos/donts）、"
    "references（域细则, 勿写 lessons.md）、asset_paths（实际用到的资产）。"
    "lessons 是核心精华：SKILL.md 紧跟标题声明必读 references/lessons.md 并内联 3-5 条关键 dos/donts。"
)


def _skill_author_methodology() -> str:
    """拼装 authoring（draft）system prompt：skill-author SKILL.md + 写作指南全文。

    agent 据此被武装完整 skill-creator 写作方法论（魔改成经验 skill 专用）。
    """
    parts = [_read_skill_author_file("SKILL.md")]
    guide = _read_skill_author_file("references/skill_writing_guide.md")
    if guide:
        parts.append("\n\n---\n# 附：写作规范基准（references/skill_writing_guide.md）\n" + guide)
    text = "\n".join(parts).strip()
    return text or _SKILL_AUTHOR_FALLBACK


def _skill_author_review_prompt() -> str:
    """review 阶段 system prompt：评审员角色 + quality checklist 全文。"""
    checklist = _read_skill_author_file("references/quality_checklist.md")
    return (
        "你是经验技能**评审员**，按下方 quality checklist 严格审查草稿，产出**修订版** AuthoredSkill。"
        "**只改有问题的部分，保留草稿有效内容，不要推倒重写**；草稿已合格则原样返回。"
        "输出仍是完整 AuthoredSkill（与草稿同 schema），未改字段照抄草稿。\n\n"
        + (checklist or "（quality checklist 缺失：检查 description 合规 / SKILL.md 冷启动可用 / "
                        "lessons 前景化 / mode 适配）")
    )


# 蒸馏器看图用：缩到最长边 1280，平衡信号与 payload（2K 原图 ×6 张太大）
_VIEW_MAX_DIM = 1280


def _render_checklist(checklist: dict | None) -> str:
    """把 content_spec.checklist（per-dim 逐项判定）渲染成可读文本——这是「生成目标」的细则。

    binary 维度：list[{id, check, anchor?}]；continuous 维度：{_scoring, points[]}。
    """
    if not checklist:
        return "(无)"
    lines: list[str] = []
    for dim, body in checklist.items():
        lines.append(f"#### {dim}")
        if isinstance(body, dict):
            if body.get("_scoring"):
                lines.append(f"_评分：{body['_scoring']}_")
            for p in body.get("points") or []:
                lines.append(f"- {p}")
        elif isinstance(body, list):
            for it in body:
                if isinstance(it, dict):
                    anchor = f"（对照 {it['anchor']}）" if it.get("anchor") else ""
                    lines.append(f"- [{it.get('id', '')}] {it.get('check', '')}{anchor}")
                else:
                    lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines).strip()


def _render_constraints(constraints: dict | None) -> str:
    """把 content_spec.constraints 渲染成可读文本——必须保留/可变/必须避免/禁止 motif。

    forbidden_motifs（禁复制的参考 motif）对原创性维度至关重要，单独高亮。
    """
    if not constraints:
        return "(无)"
    lines: list[str] = []
    labels = [
        ("must_keep", "必须保留"),
        ("may_change", "可变"),
        ("must_avoid", "必须避免"),
        ("forbidden_motifs", "禁止 motif（不得复制的参考 motif——原创性维度的硬约束）"),
    ]
    for key, label in labels:
        vals = constraints.get(key)
        if not vals:
            continue
        lines.append(f"**{label}**：")
        for v in vals:
            lines.append(f"- {v}")
        lines.append("")
    return "\n".join(lines).strip()



def _resized_data_uri(path: Path, max_dim: int = _VIEW_MAX_DIM) -> str | None:
    """读图 → 等比缩到最长边 max_dim → JPEG data-URI（控制注入蒸馏器 prompt 的体积）。"""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        im = Image.open(path)
        im = im.convert("RGB")
        w, h = im.size
        scale = max_dim / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:  # noqa: BLE001
        return None


def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
    cfg: dict = dict(config) if config else {}
    cfg["recursion_limit"] = limit
    return cfg


class ExperienceDistiller:
    """跨 loop 经验蒸馏 deepagent（视觉 + 全档案）。chat model + run_dirs + bench 注入。"""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        run_dirs: list[Path],
        lb: LoadedBenchmark,
        data_root: Path,
        previous: GeneralExperience | None = None,
        system_prompt: str | None = None,
        skills_dir: Path | str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.run_dirs = list(run_dirs)
        self.lb = lb
        self.bench = lb.bench  # 向后兼容（renovation 用）
        self.data_root = data_root
        # 上一版经验：翻新时可见（active 待判定 + refuted/archived 勿复生）；None=首次蒸馏。
        self.previous = previous
        self.system_prompt = system_prompt or load_system_prompt(
            "experience_distiller", _DEFAULT_DISTILLER_SYS,
        )
        self.skills_dir = str(skills_dir) if skills_dir else None

    def distill(self, *, config: RunnableConfig | None = None) -> GeneralExperience:
        """跨 run 蒸馏通用经验，返回 GeneralExperience（含 source_runs/bench_id）。

        公开入口（本身不上报 run）：算好本次蒸馏的 metadata/tags，调 ``@traceable`` 内核
        ``_distill_traced``，使「一次蒸馏 = 一条完整 trace」——逐轮档案 + 最佳/最差轮图 +
        agent run + ChatOpenAI 调用全部嵌套在 ``experience_distiller`` 根 run 之下。

        config：若调用方已包在更大的 trace 里（如批量编排），透传为 parent，本蒸馏嵌套为其子 run。
        """
        ls_extra: dict = {
            "metadata": {
                "bench_id": self.bench.bench_id,
                "source_runs": ",".join(rd.name for rd in self.run_dirs),
                "n_runs": len(self.run_dirs),
            },
            "tags": [f"bench:{self.bench.bench_id}", "distill"],
        }
        if config is not None:
            ls_extra["config"] = config  # parent run：嵌套到调用方的 trace
        exp = self._distill_traced(langsmith_extra=ls_extra, agent_config=config)
        # 第二阶段：编写规范可移植技能包（skill-creator 武装）。失败不阻断 lessons 蒸馏。
        try:
            authored = self.author_skill(exp, agent_config=config)
            if authored is not None:
                assemble_skill_package(
                    self.data_root, self.lb.bench_dir, self.bench.bench_id, authored, exp,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[distiller] 技能包编写失败({type(e).__name__}): {e}", flush=True)
        return exp

    @traceable(name="experience_distiller", run_type="chain")
    def _distill_traced(self, *, agent_config: RunnableConfig | None = None) -> GeneralExperience:
        """蒸馏 trace 根（``experience_distiller`` chain run）。

        - dossier（逐轮 prompt + Critic 逐项判断 + 验证结论 + 最佳/最差轮图）**在此构建**，
          作为 agent 输入消息 → 完整上下文进入 trace（旧实现把 dossier 拼在 trace 外，上下文脱离）。
        - agent run + ChatOpenAI LLM 调用经 contextvar 自动嵌套在本 run 下，无需手建 RunTree。
        - metadata/tags/parent 由 ``distill`` 经 langsmith_extra 注入（@traceable 会消费该 kwarg，
          不透传给本函数）；agent_config 仅用于递归上限 + 原始 config 透传。
        """
        # 在 trace 作用域内构建 dossier + 图 → 进 HumanMessage → trace 里可见完整上下文
        user_content = self._build_user_content()
        agent = create_deep_agent(
            model=self.chat_model, tools=[],
            system_prompt=self.system_prompt,
            skills=[self.skills_dir] if self.skills_dir else None,
            response_format=RenovationPlan, checkpointer=None,
            name="distiller_agent",  # 内层 agent run 名，与外层 experience_distiller trace 根区分
            middleware=narrow_tools_middleware(),
        )
        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=user_content)]},
            config=_merge_recursion(agent_config, DISTILLER_RECURSION_LIMIT), label="distiller",
        )
        plan: RenovationPlan | None = (result.get("structured_response") if result else None)
        plan = plan or RenovationPlan(summary="(蒸馏失败，已降级)", renovation=[])

        return self._merge_renovation(plan)

    # --- 档案构造 ---

    def _critic_line(self, v) -> str:
        """把一轮 Critic verdict 压成一行逐项判断。"""
        if not v:
            return "(无 verdict)"
        segs = []
        for d in v.dimensions:
            if d.scoring_type == "binary":
                its = d.items or []
                marks = " ".join(f"{it.id}{'✓' if it.passed else '✗'}" for it in its)
                fails = "; ".join(f"{it.id}✗:{it.reason}" for it in its if not it.passed and it.reason)
                segs.append(f"{d.dim}[{d.value:.2f}] {marks}" + (f" 失败:{fails}" if fails else ""))
            else:
                segs.append(f"{d.dim}={d.value:.2f}({(d.raw or '')[:30]})")
        return " | ".join(segs)

    def _run_dossier(self, rd: Path, recs: list) -> str:
        """单 loop 的逐轮文本档案：prompt + Critic 逐项 + 改动 + 验证结论 + 还原度。"""
        lines: list[str] = []
        res = [r.verdict.restoration for r in recs if r.verdict]
        best = f"{max(res):.3f}" if res else "?"
        lines.append(f"## {rd.name}: sample={recs[0].sample_id}, model={recs[0].model}, "
                     f"轮数={len(recs)}, 最好还原度={best}")
        for r in recs:
            rest = f"{r.verdict.restoration:.3f}" if r.verdict else "?"
            prompt = (r.prompt or "").replace("\n", " ")[:160]
            delta = (r.delta_note or "").replace("\n", " ").strip()[:120] or "(无改动说明)"
            lines.append(f"  round {r.round}: 还原度={rest}")
            lines.append(f"    prompt: {prompt}")
            lines.append(f"    Critic逐项: {self._critic_line(r.verdict)}")
            lines.append(f"    改动说明: {delta}")
        kb = load_conclusions(rd)
        if kb.conclusions:
            lines.append("  验证过的结论(effective/ineffective):")
            for c in kb.conclusions:
                if c.status in ("verified_effective", "ineffective"):
                    evd = c.critic_evidence.verdict_delta if c.critic_evidence else "?"
                    lines.append(f"    [{c.dim}] 「{c.change}」→ {c.status}（前后分差 {evd}）"
                                 f" lesson: {(c.lesson or '(无)')[:80]}")
        else:
            lines.append("  验证过的结论: (无)")
        return "\n".join(lines)

    def _best_worst(self, recs: list):
        """挑最好 & 最差还原度轮（用于注入图）。"""
        scored = [r for r in recs if r.verdict is not None]
        if not scored:
            return None, None
        scored_sorted = sorted(scored, key=lambda r: r.verdict.restoration)
        return scored_sorted[-1], scored_sorted[0]

    def _build_user_content(self) -> list[dict]:
        """多模态：指令 + 上一版经验 + 趋势提示 + 每 loop(逐轮档案文本 + 最佳/最差轮图)。"""
        dims = ", ".join(d.dim for d in self.bench.score_dimensions)
        run_ids = [rd.name for rd in self.run_dirs]
        parts: list[dict] = [{"type": "text", "text":
            f"benchmark={self.bench.bench_id}，评分维度：{dims}。待分析 run：{run_ids}。\n"
            "任务=**带反馈的迭代翻新**：先看【上一版经验】与【近期趋势】，再结合下方每个 loop 的逐轮档案"
            "（prompt + Critic 逐项判断 + 改动说明 + 验证结论）+ 最佳/最差轮生成图，"
            "对上一版每条 active lesson 判 keep/revise/retire，并按需 new，结构化输出 summary + renovation。"
            "不要调用任何工具。"}]

        # 上一版经验（翻新可见性）：active 待判定 + refuted/archived 勿复生 + category 清单复用
        parts.append({"type": "text", "text": "\n" + self._previous_section()})

        # 每 dim 近期失败率趋势（keep/retire 依据）
        parts.append({"type": "text", "text":
            "\n## 近期维度失败率趋势（判定 keep/revise/retire 的依据）\n" + self._dim_trend_hints()})

        for rd in self.run_dirs:
            tp = rd / "trajectory.jsonl"
            recs = list(TrajectoryReader(tp).iter_records()) if tp.exists() else []
            if not recs:
                parts.append({"type": "text", "text": f"\n## {rd.name}: (无 trajectory)"})
                continue
            parts.append({"type": "text", "text": "\n" + self._run_dossier(rd, recs)})
            best, worst = self._best_worst(recs)
            for label, rec in (("最佳轮", best), ("最差轮", worst)):
                if not rec or not rec.output_image_refs:
                    continue
                img = rd / rec.output_image_refs[0]
                if not img.exists():
                    continue
                uri = _resized_data_uri(img)
                if not uri:
                    continue
                parts.append({"type": "text", "text":
                    f"[{rd.name} {label} round {rec.round} 还原度={rec.verdict.restoration:.3f} 的生成图]"})
                parts.append({"type": "image_url", "image_url": {"url": uri}})
        parts.append({"type": "text", "text":
            "请综合生成图（真实质量）+ Critic 逐项判断（哪些项反复失败）+ 验证结论 + 近期趋势，"
            "产出 renovation：对上一版每条 active lesson 给出 keep/revise/retire（existing_id 对应其 id），"
            "并 new 覆盖未被捕捉的反复失败。revise/retire 的 reason 必填；category 复用上一版清单。"})
        return parts

    def _previous_section(self) -> str:
        """上一版经验摘要：active（待判定）+ refuted/archived（勿复生）+ category 清单。"""
        prev = self.previous
        if not prev or not prev.lessons:
            return "## 上一版经验\n（首次蒸馏，无上一版——全部按 new 产出）"
        active = [l for l in prev.lessons if l.status == "active"]
        retired = [l for l in prev.lessons if l.status in ("refuted", "superseded", "archived")]
        cats = sorted({(l.category or "(未分类)") for l in active}) or ["(未分类)"]
        lines = [f"## 上一版经验（category 清单，复用：{', '.join(cats)}）"]
        if active:
            lines.append("### active（**每条都要在 renovation 里判定 keep/revise/retire，existing_id=其 id**）")
            for l in active:
                lines.append(f"- id={l.id} | [{l.category or '(未分类)'}] [{l.dim}] {l.insight[:60]} "
                             f"(conf={l.confidence:.2f}, applies_when={l.applies_when})")
        else:
            lines.append("### active：(无)")
        if retired:
            lines.append("### 已废弃（**勿复生**，除非强证据）")
            for l in retired:
                tag = f"→{l.successor_id}" if l.status == "superseded" and l.successor_id else l.status
                lines.append(f"- id={l.id} | [{l.dim}] {l.insight[:50]} （{tag}）")
        return "\n".join(lines)

    def _dim_trend_hints(self, n: int = 6) -> str:
        """每 dim 近 n 轮失败率（binary 任一项✗ / continuous <0.7 算失败）→ keep/retire 依据。"""
        per_dim: dict[str, list[bool]] = {}
        for rd in self.run_dirs:
            tp = rd / "trajectory.jsonl"
            if not tp.exists():
                continue
            for r in TrajectoryReader(tp).iter_records():
                if not r.verdict:
                    continue
                for d in r.verdict.dimensions:
                    if d.scoring_type == "binary":
                        failed = any((not it.passed) for it in (d.items or []))
                    else:
                        failed = d.value < 0.7
                    per_dim.setdefault(d.dim, []).append(failed)
        if not per_dim:
            return "(无 trajectory，无法算趋势)"
        lines = []
        for dim, flags in per_dim.items():
            recent = flags[-n:]
            rate = (sum(recent) / len(recent)) if recent else 0.0
            if rate >= 0.34:
                verdict = "仍高失败→相关 lesson 宜 revise/retire"
            elif rate > 0:
                verdict = "偶发失败"
            else:
                verdict = "已稳定→keep 候选"
            lines.append(f"- {dim}: 近 {len(recent)} 轮失败率 {int(rate * 100)}%（{verdict}）")
        return "\n".join(lines)

    def _merge_renovation(self, plan: RenovationPlan) -> GeneralExperience:
        """把 RenovationPlan 合并进上一版，产出带状态链的 GeneralExperience。

        keep→保持 active；revise→旧标 superseded(+successor_id)+追加新 active；retire→旧标 refuted
        (+retire_reason)；new→追加 active（赋新 id）。上一版未被提及的 active lesson 默认保留 active。
        """
        prev = self.previous
        result: list[DistilledLesson] = list(prev.lessons) if prev else []
        by_id: dict[str, int] = {l.id: i for i, l in enumerate(result) if l.id}

        def _activate(l: DistilledLesson) -> DistilledLesson:
            if l.applies_when not in ("construction", "fix", "always"):
                l.applies_when = "always"
            l.status = "active"
            l.successor_id = ""
            l.retire_reason = ""
            return l

        for item in plan.renovation or []:
            action = (item.action or "new").strip().lower()
            lesson = item.lesson  # 必填（schema 已约束非空）
            eid = item.existing_id
            idx = by_id.get(eid)

            if action == "keep" and idx is not None:
                result[idx].status = "active"  # 原样保留，仅确保 active
            elif action == "retire" and idx is not None:
                result[idx].status = "refuted"
                result[idx].retire_reason = item.reason or "蒸馏判定失效"
            elif action == "revise":
                cat = lesson.category or (result[idx].category if idx is not None else "")
                nid = new_lesson_id(cat, lesson.dim)
                if idx is not None:
                    result[idx].status = "superseded"
                    result[idx].successor_id = nid
                    result[idx].retire_reason = item.reason or "被修订"
                new_l = _activate(lesson)
                new_l.id = nid
                result.append(new_l)
                by_id[nid] = len(result) - 1
            elif action == "new":
                # 防重复堆积：若 plan 里已有人类/旧版同 dim 的 active lesson 未被处置，
                # 仍允许新增（LLM 应优先 revise；此处不强行合并，避免误杀）
                nid = new_lesson_id(lesson.category, lesson.dim)
                new_l = _activate(lesson)
                new_l.id = nid
                if nid not in by_id:
                    result.append(new_l)
                    by_id[nid] = len(result) - 1
            # 未知 action 或 existing_id 找不到 → 跳过（防御 LLM 幻觉）

        # 修剪 superseded：它们是修订历史（active 继承者已携带信息），保留会膨胀"已废弃"区。
        # 只留 active（消费）+ refuted/archived（勿复生的负信号）。bounds general.json 增长。
        kept = [l for l in result if l.status != "superseded"]

        return GeneralExperience(
            bench_id=self.bench.bench_id,
            source_runs=[rd.name for rd in self.run_dirs],
            summary=plan.summary,
            lessons=kept,
            scene=self.bench.scene or "",
            dimensions=[d.dim for d in self.bench.score_dimensions],
            bench_description=self.bench.description or "",
        )

    # --- 第二阶段：skill-creator 武装的技能包编写 ---

    def author_skill(
        self, exp: GeneralExperience, *, agent_config: RunnableConfig | None = None
    ) -> AuthoredSkill | None:
        """两阶段编写规范可移植技能包：draft → review-revise。

        - draft：武装 skill-author 方法论的 gemini，吃富化 dossier → 完整 AuthoredSkill。
        - review：按 quality checklist 审查草稿 → 修订版；失败回退 draft。
        失败返回 None（不阻断蒸馏）。skill_name 强制 = slug(bench)。
        """
        draft = self._draft_skill(exp, agent_config=agent_config)
        if draft is None:
            return None
        revised = self._review_and_revise(draft, agent_config=agent_config)
        out = revised or draft
        out.skill_name = slugify_bench(self.bench.bench_id)
        return out

    def _draft_skill(
        self, exp: GeneralExperience, *, agent_config: RunnableConfig | None = None
    ) -> AuthoredSkill | None:
        """阶段一：产完整草稿（武装 skill-author 方法论 + 富化 dossier）。"""
        user_content = self._build_skill_dossier(exp)
        agent = create_deep_agent(
            model=self.chat_model, tools=[],
            system_prompt=_skill_author_methodology(),
            response_format=AuthoredSkill, checkpointer=None,
            name="skill_author_draft", middleware=narrow_tools_middleware(),
        )
        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=user_content)]},
            config=_merge_recursion(agent_config, DISTILLER_RECURSION_LIMIT), label="skill_author_draft",
        )
        return (result.get("structured_response") if result else None)

    def _review_and_revise(
        self, draft: AuthoredSkill, *, agent_config: RunnableConfig | None = None
    ) -> AuthoredSkill | None:
        """阶段二：按 quality checklist 审查草稿并产出修订版；失败返回 None（调用方回退 draft）。"""
        review_input = self._render_authored_for_review(draft)
        agent = create_deep_agent(
            model=self.chat_model, tools=[],
            system_prompt=_skill_author_review_prompt(),
            response_format=AuthoredSkill, checkpointer=None,
            name="skill_author_review", middleware=narrow_tools_middleware(),
        )
        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=review_input)]},
            config=_merge_recursion(agent_config, DISTILLER_RECURSION_LIMIT), label="skill_author_review",
        )
        return (result.get("structured_response") if result else None)

    @staticmethod
    def _render_authored_for_review(draft: AuthoredSkill) -> list[dict]:
        """把 draft 渲染成 review 输入（文本）：description + skill_md + references + asset_paths。"""
        refs = "\n\n".join(
            f"### references/{r.path}\n{r.content}" for r in (draft.references or [])
        ) or "(无)"
        assets = ", ".join(draft.asset_paths) if draft.asset_paths else "(无)"
        text = (
            "# 待审查草稿（AuthoredSkill）\n\n"
            f"## description\n{draft.description}\n\n"
            f"## skill_md\n{draft.skill_md}\n\n"
            f"## references\n{refs}\n\n"
            f"## asset_paths\n{assets}\n\n"
            "请按 quality checklist 审查并产出修订版 AuthoredSkill（只改有问题的部分）。"
        )
        return [{"type": "text", "text": text}]


    def _build_skill_dossier(self, exp: GeneralExperience) -> list[dict]:
        """组装技能编写 dossier（富化版）：benchmark 全要素 + 全量 lessons + 视觉参考图。

        给 authoring agent 充足燃料（修旧版「无米之炊」）：完整 style_brief（风格 spec）、
        代表 target.md（输入样例）、全量 lessons（dos/donts 全文）、完整输入输出契约、
        （style_transfer 类）视觉参考图——参考图即风格 spec，agent 须眼见为实。
        """
        bench = self.bench
        sample = next(iter(self.lb.samples.values()), None)
        spec = sample.spec if sample else None
        task = spec.task if (spec and spec.task) else None
        mode = task.mode if task else "(未知)"
        sample_dir = sample.sample_dir if sample else None

        lines: list[str] = [
            f"# 技能编写任务：benchmark={bench.bench_id}（mode={mode}）",
            f"## 场景\n{bench.scene or bench.description or ''}",
            "## 任务定义",
        ]
        if task:
            if task.instruction:
                lines.append(f"**instruction（输入契约/产出要求）**：\n{task.instruction}")
            if task.article_topic:
                lines.append(f"**article_topic（文章核心概念，输入）**：{task.article_topic}")
            if task.output:
                lines.append(f"**output**：{task.output}")
        # manifest 的 input_contract（若 JSON 里有）
        try:
            task_extra = bench.task.model_dump() if bench.task else {}
            if task_extra.get("input_contract"):
                lines.append(f"**input_contract**：{task_extra['input_contract']}")
        except Exception:  # noqa: BLE001
            pass

        # === ⭐ 生成目标（评分标准——skill 必须命中的目标；这是考题的"要求"）===
        # 旧版只给一行 dim 摘要 + 截断 JSON，agent 漏掉原创性/创造力等高权重维度 → 产出机械。
        # 现：rubric 全文 + dimensions(权重/类型/完整 desc) + checklist 逐项 + constraints(含 forbidden_motifs)。
        spec_d = spec.model_dump() if spec else {}
        lines.append(
            "\n## ⭐ 生成目标（评分标准——本技能必须让产出命中的目标）\n"
            "下方是 benchmark 的评分真源：rubric（人类可读细则）+ score_dimensions（机器真源，含权重）"
            "+ checklist（逐项判定）+ constraints（必须保留/可变/必须避免/禁止 motif）。"
            "**SKILL.md 必须传达这些目标**——尤其高权重维度（注意原创性/创造力常占大权重）与禁止 motif，"
            "别只盯风格还原。"
        )
        rubric_path = self.lb.bench_dir / "rubric.md"
        if rubric_path.exists():
            lines.append("\n### rubric.md（评分细则全文——首要）\n" + rubric_path.read_text(encoding="utf-8"))
        lines.append("\n### score_dimensions（机器真源：dim / 权重 / 类型 / 判定标准）")
        for d in bench.score_dimensions:
            lines.append(
                f"- **{d.dim}**（{getattr(d, 'scoring_type', '?')}, 权重 {getattr(d, 'weight_init', '?')}）："
                f"{d.desc or ''}"
            )
        cl = spec_d.get("checklist")
        if cl:
            lines.append("\n### checklist（逐项判定细则）\n" + _render_checklist(cl))
        cs = spec_d.get("constraints")
        if cs:
            lines.append("\n### constraints\n" + _render_constraints(cs))

        # 头号燃料：style_brief.md 全文（风格 spec——创作风格指南的首要依据）
        if sample_dir is not None:
            sb = sample_dir / "style_brief.md"
            if sb.exists():
                lines.append(
                    "\n## 风格 spec（style_brief.md 全文——创作风格指南的首要依据）\n"
                    + sb.read_text(encoding="utf-8")
                )

        # 代表 target.md（输入文章样例 + 概念映射）
        if sample_dir is not None:
            tg = sample_dir / "target.md"
            if tg.exists():
                lines.append(
                    "\n## 输入样例（代表 target.md——看输入文章长啥样、概念如何映射视觉）\n"
                    + tg.read_text(encoding="utf-8").strip()[:1500]
                )

        # 全量 lessons（dos/donts 全文——核心精华）
        act = [l for l in exp.lessons if l.status == "active"]
        lines.append(
            f"\n## 蒸馏经验 lessons（{len(act)} 条 active；**核心精华**——SKILL.md 必须前景化，"
            "系统自动渲染进 references/lessons.md，你无需写它）\n" + render_lessons_reference_md(exp)
        )

        # 可用资产清单（generator 实际用到的才进 asset_paths）
        asset_lines: list[str] = []
        if task and task.input_assets and sample_dir is not None:
            for a in task.input_assets:
                p = sample_dir / a
                asset_lines.append(f"- {a}（{'存在' if p.exists() else '缺失'}）")
        if asset_lines:
            lines.append(
                "\n## 可用资产（generator 实际用到的才在 asset_paths 列出；style 类参考图通常要打包，"
                "image_edit 类的运行时输入如产品照不要打包）\n" + "\n".join(asset_lines)
            )

        # 上一版技能包（修订而非重写）
        prev_md = self._load_previous_skill_md()
        if prev_md:
            lines.append("\n## 上一版技能包 SKILL.md（优先修订，保留仍有效部分）\n" + prev_md[:5000])
        else:
            lines.append("\n## 上一版技能包\n（首次编写，从零创作）")

        lines.append(
            "\n## 交付要求\n按 skill-author 方法论产出：description（pushy/通用/正确框定为「产出策略」而非"
            "「生成器」/无尖括号/<1024）、skill_md（<500 行/冷启动可用/含可执行工作流+输出格式模板"
            "(strategy 字段可扩展)/前景化 3-5 条关键 dos/donts）、references（域细则, **勿写 lessons.md**）、"
            "asset_paths（实际用到的）。不要调用任何工具。"
        )

        parts: list[dict] = [{"type": "text", "text": "\n".join(lines)}]

        # 视觉参考图（mode-aware）：style_transfer 注入参考图（参考图即风格 spec，眼见为实）；其它 mode 跳过。
        if mode == "style_transfer" and task and task.input_assets and sample_dir is not None:
            parts.extend(self._reference_image_parts(sample_dir, task.input_assets))

        return parts

    def _reference_image_parts(
        self, sample_dir: Path, input_assets: list[str], limit: int = 4
    ) -> list[dict]:
        """把 style 类参考图转 data-URI 注入 dossier（≤limit 张，让 agent 眼见风格 spec）。"""
        parts: list[dict] = []
        for a in input_assets[:limit]:
            p = sample_dir / a
            if not p.exists() or not p.is_file():
                continue
            uri = _resized_data_uri(p)
            if not uri:
                continue
            parts.append({"type": "text", "text": f"[参考图 {a}——风格神韵的视觉依据]"})
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        return parts


    def _load_previous_skill_md(self) -> str | None:
        """读上一版技能包 SKILL.md（供修订参考）；不存在返回 None。"""
        p = skill_package_dir(self.data_root, self.bench.bench_id) / "SKILL.md"
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None


__all__ = ["ExperienceDistiller"]
