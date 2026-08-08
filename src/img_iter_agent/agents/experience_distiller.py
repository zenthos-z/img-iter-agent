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
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..data.trajectory import TrajectoryReader
from ..memory.experience import DistilledExperience, GeneralExperience
from ..memory.knowledge import load_conclusions
from ..memory.schema import Benchmark
from ._narrow_tools import DISTILLER_RECURSION_LIMIT, invoke_with_retry, narrow_tools_middleware
from .agent_config_loader import load_system_prompt

_DEFAULT_DISTILLER_SYS = (
    "你是生图经验蒸馏员（视觉）。用户消息里每个 loop 都附了【逐轮 prompt + Critic 逐项判断 + "
    "验证结论 + 还原度】的完整文本档案，以及该 loop 最佳轮与最差轮的生成图。\n"
    "**核心交付物是 lessons（结构化经验条目），summary 只用 1-2 句点题**。必须产出 **至少 3 条、"
    "最好 4-6 条** lessons，每条填齐：dim（维度名或 'general'）、insight（一句话规律）、"
    "dos（应这样做）、donts（不要这样做）、evidence（来源 run/轮，如 '<run_id>/round2'）、"
    "confidence（0-1）。\n"
    "归纳要**眼见为实**：结合图的真实质量 + Critic 的逐项判断（哪些 C/S/A/B 项反复失败）+ 验证过的 "
    "effective/ineffective。不要调用任何工具，直接结构化输出 summary + lessons。"
)

# 蒸馏器看图用：缩到最长边 1280，平衡信号与 payload（2K 原图 ×6 张太大）
_VIEW_MAX_DIM = 1280


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
        bench: Benchmark,
        system_prompt: str | None = None,
        skills_dir: Path | str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.run_dirs = list(run_dirs)
        self.bench = bench
        self.system_prompt = system_prompt or load_system_prompt(
            "experience_distiller", _DEFAULT_DISTILLER_SYS,
        )
        self.skills_dir = str(skills_dir) if skills_dir else None

    def distill(self, *, config: RunnableConfig | None = None) -> GeneralExperience:
        """跨 run 蒸馏通用经验，返回 GeneralExperience（含 source_runs/bench_id）。"""
        agent = create_deep_agent(
            model=self.chat_model, tools=[],
            system_prompt=self.system_prompt,
            skills=[self.skills_dir] if self.skills_dir else None,
            response_format=DistilledExperience, checkpointer=None,
            name="experience_distiller",
            middleware=narrow_tools_middleware(),
        )
        result, _ok = invoke_with_retry(
            agent, {"messages": [HumanMessage(content=self._build_user_content())]},
            config=_merge_recursion(config, DISTILLER_RECURSION_LIMIT), label="distiller",
        )
        out: DistilledExperience | None = (result.get("structured_response") if result else None)
        out = out or DistilledExperience(summary="(蒸馏失败，已降级)", lessons=[])

        return GeneralExperience(
            bench_id=self.bench.bench_id,
            source_runs=[rd.name for rd in self.run_dirs],
            summary=out.summary,
            lessons=out.lessons,
        )

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
        """多模态：指令 + 每 loop(逐轮档案文本 + 最佳/最差轮图)。"""
        dims = ", ".join(d.dim for d in self.bench.score_dimensions)
        run_ids = [rd.name for rd in self.run_dirs]
        parts: list[dict] = [{"type": "text", "text":
            f"benchmark={self.bench.bench_id}，评分维度：{dims}。待分析 run：{run_ids}。\n"
            f"下面是每个 loop 的完整档案（逐轮 prompt + Critic 逐项判断 + 改动说明 + 验证结论），"
            "并附每个 loop 最佳轮与最差轮的生成图。请结合视觉与逐项判断，跨 run 归纳通用经验，"
            "结构化输出 summary + lessons。不要调用任何工具。"}]
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
            "请综合上面的生成图（真实质量）+ Critic 逐项判断（哪些项反复失败）+ 验证结论，"
            "跨 run 归纳可复用的 dos/donts，结构化输出。"})
        return parts


__all__ = ["ExperienceDistiller"]
