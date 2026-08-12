"""Summarizer：Critic 驱动的经验闭环验证（替代原单轮事实记录）。

闭环核心（ARCH §1.3 演进）：
  不再只记录"本轮参数+分数"的快照，而是做跨轮因果验证——
  上轮改了什么（prev_delta_note）→ Critic 前后 verdict 对比 → 判定有效/无效 → 沉淀。

Critic 是客观裁判：其 verdict（分 + 失败项 + reason）是"改动有效性"的唯一证据。
reason 文本本身是可复用知识（"为什么有效/为什么没用"）。

产出：
  - 更新 lessons/conclusions.json（结构化经验知识库，见 memory/knowledge.py）
  - 返回 conclusions.json 的相对路径（供 AttemptRecord.lesson_ref 指向）
  - 追加 index.json 一条 entry
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from ..memory import index, knowledge
from ..memory.schema import CriticVerdict
from .agent_config_loader import load_system_prompt
from .generator import GenOutcome

# 代码默认（data/agents_config/summarizer.md 缺失时回退）。富化版：强制 ineffective 给「具体替代思路」、
# escalated 标注模型上限，杜绝「需换思路」套话。
_DEFAULT_SUMMARIZE_PROMPT = (
    "你是生图迭代经验归纳员。根据 Critic 前后评分对比，为每条结论提炼『可复用、可执行』的 lesson。\n"
    "要求：\n"
    "1. effective（有效）→ 写『为什么有效』+『后续如何保持/复用』，给可操作要点。\n"
    "2. ineffective（无效）→ 写『为什么失败』（引用 Critic reason）+ **至少一条具体的替代思路**"
    "（禁止只写『需换思路』；必须给出新方向，如换描述角度、换参考图策略、换约束措辞、换 test_variable）。\n"
    "3. escalated（疑似模型能力上限）→ 明确标注『已连续失败多轮，prompt 微调疑似无效，"
    "需根本性换方向（换 test_variable 如 reference_images/size）或上报人工介入』。\n"
    "输出：每条结论一行，格式 `[dim] lesson文本`。只输出要点，不要寒暄。"
)


class Summarizer:
    """经验闭环验证器 + 复发检测（B）+ LLM 富化（A）。

    chat_model 注入（可空）：None 时退化为纯规则（judge_status 套话 + streak 计数仍生效，
    仅 lesson 不富化），保证 Summarizer() 无参在离线测试中可用。
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model

    def summarize(
        self,
        *,
        run_dir: Path,
        round: int,
        outcome: GenOutcome,
        verdict: CriticVerdict,
        sample_id: str,
        prev_verdict: CriticVerdict | None = None,
        prev_delta_note: str | None = None,
        config: RunnableConfig | None = None,
        agent_lessons: list[dict] | None = None,
    ) -> str:
        """做 Critic 驱动的经验闭环验证，更新 conclusions.json，返回其相对 run 目录路径。

        Args:
            outcome/verdict: 本轮产出与 Critic 判定。
            prev_verdict: 上一轮的 Critic 判定（验证上轮改动的"后"证据来源）。
            prev_delta_note: 上一轮 Generator 的改动说明（验证对象）。
                两者需同时提供才能做跨轮验证；首轮无上轮则只登记本轮。
            agent_lessons: critic agent 打分时通过 note_experience 工具写下的第一手经验判断
                （[{dim, judgment: effective|ineffective|escalated, lesson}]）。非空时启用「agent 模式」：
                用 agent 判断替代 judge_status 规则 + _llm_refine 富化（裁判理解最深最准）；为空走旧逻辑
                （_verify_pending + _llm_refine），向后兼容。规则部分（fail_streaks/escalation/register/save）两种模式都跑。
        """
        kb = knowledge.load_conclusions(run_dir, sample_id=sample_id, loop_id=outcome.model)

        # config 透传给 @traceable 子方法（_verify_pending/_llm_refine），保证其 chain run
        # 嵌套在 LangGraph summarizer 节点 run 之下（LangSmith 可观测）。
        ls_extra: Any = {"config": config} if config is not None else {}

        use_agent = bool(agent_lessons)  # critic 工作流模式：agent 已写下第一手判断

        # 1) 验证上轮 pending 结论：
        #    - agent 模式：_apply_agent_lessons 用裁判判断（judgment→status、lesson 直接写入）替代
        #      judge_status 规则 + _llm_refine；critic_evidence 仍走 judge_status 的客观前后快照。
        #    - 旧模式：_verify_pending 走 judge_status 规则（前后分差/项翻转）。不依赖 prev_delta_note
        #      是否被 generator 填写（deepseek 常空，不能让空 delta_note 饿死经验闭环）。
        if use_agent:
            self._apply_agent_lessons(
                kb, agent_lessons=agent_lessons or [],
                prev_verdict=prev_verdict, cur_verdict=verdict, cur_round=round,
            )
        elif prev_verdict is not None:
            self._verify_pending(kb, prev_verdict=prev_verdict, cur_verdict=verdict,
                                 cur_round=round, prev_delta_note=prev_delta_note,
                                 langsmith_extra=ls_extra)

        # 2) 复发检测（B）：按本轮 verdict 更新 per-dim 连续失败计数，对跨阈值的 dim 标升级。
        #    顺序在 _register_round_changes 之前——先算 streak，再让登记的新结论据 escalated dim 打标记。
        streak_changes = knowledge.update_fail_streaks(kb, cur_verdict=verdict)
        knowledge.apply_escalation(kb, cur_round=round)

        # 3) 登记本轮的失败维度为新 pending 结论（待下轮 Critic 验证）。
        #    不再以 delta_note 为门槛——delta_note 空时，change 从 Critic 的失败项理由派生。
        #    agent 模式下，把 agent 对该 dim 的 lesson 填进 pending（首轮裁判理解，下轮验证时更新）。
        self._register_round_changes(
            kb, outcome=outcome, verdict=verdict, round=round,
            streak_changes=streak_changes,
            agent_lessons=agent_lessons if use_agent else None,
        )

        # 4) 可选 LLM 富化（A）：仅旧模式跑；agent 模式下 agent 已写 lesson，跳过。
        if not use_agent and self.chat_model is not None and kb.conclusions:
            self._llm_refine(kb, langsmith_extra=ls_extra)

        lesson_ref = str(knowledge.save_conclusions(run_dir, kb).relative_to(run_dir))

        # 步骤2: discover_standards — 从本 loop critic findings 提炼反复偏差 → 提议标准
        if self.chat_model is not None:
            self._discover_standards(run_dir, kb)

        # 追加 index 条目
        entry = index.make_entry(
            attempt_id=outcome.attempt_id,
            round=round,
            model=outcome.model,
            gen_mode=outcome.gen_mode,
            test_variable=outcome.test_variable,
            baseline_ref=outcome.baseline_ref,
            size=outcome.size,
            restoration=verdict.restoration,
            output_image_refs=outcome.output_image_refs,
            lesson_ref=lesson_ref,
            delta_note=outcome.delta_note,
            prompt=outcome.prompt,
        )
        index.append_entry(run_dir, entry)
        return lesson_ref

    @traceable(name="summarizer.verify_pending", run_type="chain")
    def _verify_pending(
        self, kb: knowledge.KnowledgeBase, *, prev_verdict: CriticVerdict,
        cur_verdict: CriticVerdict, cur_round: int, prev_delta_note: str | None = None,
    ) -> list[dict]:
        """验证上轮 pending 结论：对比 prev→cur Critic 判定，更新 status + critic_evidence。

        返回每条 pending 的判定（dim/status/前后分差）作为 LangSmith chain run output，
        让经验闭环的「上轮改了 X → 这轮分数变了多少 → 判定有效/无效」在 trace 里可见。
        """
        results: list[dict] = []
        for c in kb.pending():
            status, evidence, lesson = knowledge.judge_status(prev_verdict, cur_verdict, c.dim)
            evidence.tested_round = cur_round
            c.status = status
            c.critic_evidence = evidence
            c.lesson = lesson
            c.verified_round = cur_round
            results.append({
                "dim": c.dim,
                "change": c.change,
                "status": status,
                "verdict_delta": evidence.verdict_delta if evidence else None,
            })
        return results

    @traceable(name="summarizer.apply_agent_lessons", run_type="chain")
    def _apply_agent_lessons(
        self, kb: knowledge.KnowledgeBase, *, agent_lessons: list[dict],
        prev_verdict: CriticVerdict | None, cur_verdict: CriticVerdict, cur_round: int,
    ) -> list[dict]:
        """用 critic agent 的第一手判断验证上轮 pending 结论（agent 模式：替代 judge_status + _llm_refine）。

        agent 在打分循环里调 note_experience 写下 [{dim, judgment, lesson}]。对 kb.pending() 每条按 dim
        匹配 agent 判断：judgment→status（effective→verified_effective；ineffective/escalated→ineffective，
        escalated 额外置 c.escalated=True），lesson 直接写入。critic_evidence 仍用 judge_status 的客观前后
        快照（before/after/verdict_delta）——distiller 读 verdict_delta，客观不变；agent 只接管"有效性
        判定 + lesson 文本"。找不到 agent 判断的 pending（agent 漏写）→ 回退 judge_status 规则，不饿死。

        返回每条 pending 的判定作为 chain run output（trace 可见 agent 判断 vs 规则）。
        """
        by_dim: dict[str, dict] = {}
        for al in agent_lessons:
            d = al.get("dim")
            if d and d not in by_dim:
                by_dim[d] = al
        results: list[dict] = []
        for c in kb.pending():
            al = by_dim.get(c.dim)
            # 客观证据（前后快照）始终算（若有 prev_verdict）——与 agent 判断正交，distiller 依赖它
            if prev_verdict is not None:
                status_rule, evidence, lesson_rule = knowledge.judge_status(prev_verdict, cur_verdict, c.dim)
            else:
                status_rule, evidence, lesson_rule = None, None, None
            if al is not None:
                judgment = al.get("judgment")
                c.status = (
                    "verified_effective" if judgment == "effective" else "ineffective"
                )  # ineffective / escalated 都归 ineffective（escalated 叠加标记，与 status 正交）
                if judgment == "escalated":
                    c.escalated = True
                lesson_text = (al.get("lesson") or "").strip()
                if lesson_text:
                    c.lesson = lesson_text
            else:
                # agent 漏写该 dim → 回退 judge_status 规则
                c.status = status_rule or "ineffective"
                if c.lesson is None and lesson_rule:
                    c.lesson = lesson_rule
            if evidence is not None:
                evidence.tested_round = cur_round
                c.critic_evidence = evidence
            c.verified_round = cur_round
            results.append({
                "dim": c.dim, "change": c.change, "status": c.status,
                "agent_judgment": (al or {}).get("judgment"),
                "verdict_delta": evidence.verdict_delta if evidence else None,
            })
        return results

    def _register_round_changes(
        self, kb: knowledge.KnowledgeBase, *, outcome: GenOutcome,
        verdict: CriticVerdict, round: int,
        streak_changes: dict[str, str] | None = None,
        agent_lessons: list[dict] | None = None,
    ) -> None:
        """把本轮 delta_note 登记为新 pending 结论（按失败维度拆分）。

        agent 模式下（agent_lessons 非空）：把 agent 对该 dim 的第一手 lesson 填进新 pending
        （首轮裁判理解，下轮验证时由 _apply_agent_lessons 更新）。
        """
        agent_by_dim = {al.get("dim"): al for al in (agent_lessons or []) if al.get("dim")}
        # 本轮针对的失败维度（来自 Critic 判定）
        failed_dims = {
            d.dim for d in verdict.dimensions
            if d.scoring_type == "binary" and any(not it.passed for it in (d.items or []))
        }
        low_continuous = {d.dim for d in verdict.dimensions if d.scoring_type == "continuous" and d.value < 0.7}
        target_dims = failed_dims or low_continuous or {"general"}
        escalated_dims = kb.escalated_dims()
        for dim in target_dims:
            finding = self._finding_for_dim(verdict, dim)
            # change：优先用 generator 的 delta_note（具体改动叙述）；空则从 Critic 失败项派生
            # （保证空 delta_note 时经验闭环仍登记结论，不被饿死）。
            change = outcome.delta_note or finding or f"{dim} 改进"
            al = agent_by_dim.get(dim)
            agent_lesson = (al.get("lesson") or "").strip() if al else None
            c = knowledge.upsert_conclusion(
                kb, dim=dim, finding=finding, change=change,
                tags=[outcome.test_variable or "prompt"], created_round=round,
                lesson=agent_lesson or None,
            )
            # 该 dim 已达升级阈值 → 新登记的结论直接打 escalated 标记
            if dim in escalated_dims:
                c.escalated = True

    def _finding_for_dim(self, verdict: CriticVerdict, dim: str) -> str:
        """从 verdict 提取该维度的问题描述（失败项理由 / 连续维度低分理由）。"""
        d = next((x for x in verdict.dimensions if x.dim == dim), None)
        if d is None:
            return ""
        if d.scoring_type == "binary":
            reasons = [f"{it.id}: {it.reason}" for it in (d.items or []) if not it.passed]
            return "; ".join(reasons) or f"{dim} 未通过"
        return d.raw or f"{dim} 低分({d.value:.2f})"

    @traceable(name="summarizer.llm_refine", run_type="chain")
    def _llm_refine(self, kb: knowledge.KnowledgeBase) -> str | None:
        """用 chat_model 对刚验证/升级的结论富化 lesson（写到 lesson 字段）。

        effective→为什么有效+如何保持；ineffective→具体替代思路；escalated→模型上限标注。
        容错：invoke 异常 → return None（退化为 judge_status 套话，闭环不炸）。
        返回归纳文本作为 LangSmith chain run output。
        """
        candidates = [c for c in kb.conclusions
                      if c.status in ("verified_effective", "ineffective") or c.escalated]
        if not candidates:
            return None
        facts = "\n".join(self._fact_line(kb, c) for c in candidates)
        msgs = [
            SystemMessage(content=load_system_prompt("summarizer", _DEFAULT_SUMMARIZE_PROMPT)),
            HumanMessage(content=facts),
        ]
        try:
            resp = self.chat_model.invoke(msgs)  # type: ignore[union-attr]
            content = resp.content
            summary = (content if isinstance(content, str) else str(content)).strip()
        except Exception:  # noqa: BLE001  LLM 失败不炸闭环，退化为纯规则 lesson
            return None
        if not summary:
            return None
        # 按 `[dim] text` 解析，回填到对应 conclusion（追加「建议: 」前缀，不覆盖 Critic 判定文本）
        refined = self._parse_dim_lines(summary)
        for c in candidates:
            text = refined.get(c.dim)
            if not text:
                continue
            c.lesson = f"{c.lesson}\n建议: {text}" if c.lesson else text
        return summary

    @traceable(name="summarizer.discover_standards", run_type="chain")
    def _discover_standards(self, run_dir: Path, kb: knowledge.KnowledgeBase) -> str | None:
        """从本 loop Critic findings(reasons) 提炼反复出现、checklist 未显式覆盖的偏差模式 → 提议标准。

        写入 run_dir/proposed_standards.json，供 generator 下轮注入（自主发现盲区 + 反哺）。
        这是「agent 自己设标准、少人参与」的机制：critic 自主对照参考发现的偏差，
        被提炼成持久标准，generator 提前避免。
        """
        import json, re
        findings = [c.finding for c in kb.conclusions if c.finding]
        if len(findings) < 2:
            return None
        facts = "\n".join(f"- {f}" for f in findings)
        prompt = (
            "下面是本 loop Critic 反复发现的问题(reasons)：\n" + facts +
            "\n请提炼出【反复出现、且当前 checklist 未显式覆盖的偏差模式】，转成简洁的『提议标准』——"
            "每条一句话，是生成模型该遵守的规则(如『禁止写实指节/指甲/解剖暗示，须纯线条抽象』)。"
            "只提炼真正的盲区(checklist 已覆盖的不重复)。输出 JSON list[str]，无则 []。"
        )
        try:
            resp = self.chat_model.invoke([HumanMessage(content=prompt)])  # type: ignore[union-attr]
            content = resp.content
            text = content if isinstance(content, str) else str(content)
        except Exception:  # noqa: BLE001  LLM 失败不炸闭环
            return None
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return None
        try:
            proposed = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(proposed, list) or not proposed:
            return None
        proposed = [str(x).strip() for x in proposed if str(x).strip()][:10]
        (run_dir / "proposed_standards.json").write_text(
            json.dumps(proposed, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"proposed {len(proposed)} standards"

    @staticmethod
    def _fact_line(kb: knowledge.KnowledgeBase, c) -> str:
        """构造给 LLM 的单条事实行（含 streak/escalated 上下文）。"""
        delta = c.critic_evidence.verdict_delta if c.critic_evidence else ""
        streak = kb.fail_streaks.get(c.dim, 0)
        if c.escalated:
            return (f"- [{c.dim}] 状态=escalated（已连续失败 {streak} 轮，疑似模型能力上限）"
                    f"；改动「{c.change}」→ {delta}；要求：给根本性换方向建议（换 test_variable 或上报人工）")
        return (f"- [{c.dim}] 状态={c.status}；改动「{c.change}」→ {delta}；"
                f"连续失败 {streak} 轮")

    @staticmethod
    def _parse_dim_lines(summary: str) -> dict[str, str]:
        """解析 LLM 输出里 `[dim] text` 行为 {dim: text}。

        容忍前缀空白、markdown 列表标记（-/*）和数字序号（LLM 常模仿 facts 的 `- [dim]` 格式回写），
        以及全角【】括号。
        """
        out: dict[str, str] = {}
        for line in summary.splitlines():
            m = re.match(r"\s*(?:[-*]|\d+[.)])?\s*[\[【]([\w_]+)[\]】]\s*(.+)", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out


__all__ = ["Summarizer"]
