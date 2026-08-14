"""Critic：对照参考图(target)对生成图打分的 agent（deepagent 版）。

改造前是「逐维度单次 LLM 调用 + 手抠 JSON」；现在是一个真正的 tool-using agent：
  - 生成图 + target 直接注入初始 HumanMessage（多模态），agent 循环每步都看得到；
  - 可用工具 `query_rubric(dim_name)` 按需查某维度的判定标准；
  - 用 `response_format=build_critic_output_schema(bench)` 约束最终输出——每个维度一个
    **具名 required 字段**（二分→items 逐项判定 / 连续→value+reason），结构上杜绝重复
    维度、编造维度名，漏填在校验层即失败；
  - 校验失败/语义问题（items 与 checklist 不对应、缺 value/reason）**不盲目重试**：
    把具体错误反馈给模型续会话修正（`REPAIR_ROUNDS` 轮），仍失败才对可用部分宽松兜底；
  - 代码侧把原始评分映射成 `DimensionScore`，再用权重 `recompute_restoration` 算还原度
    （agent 不知道权重，不返回 restoration）——保住「权重变更不影响打分」的契约。

deepagent 在 `evaluate` 内部构建并同步跑完一轮（checkpointer=None），作为引擎嵌入外层
LangGraph 的 critic 节点。agent 跑飞时退安全默认评分，绝不中断闭环。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ..data.benchmark import Sample
from ..data.weights import recompute_restoration
from ..generation.image_io import _CRITIC_VIEW_MAX_DIM, resized_data_uri
from ..memory.knowledge import load_conclusions
from ..memory.schema import (
    Benchmark,
    CriticVerdict,
    DimensionScore,
)
from ._agent_output import build_critic_output_schema, provider_structured
from ._narrow_tools import AGENT_RECURSION_LIMIT, narrow_tools_middleware
from .agent_config_loader import load_system_prompt
from .tools.critic_tools import _effective_checklist, _load_creativity_overlay, make_critic_tools
from .tools.generator_tools import _format_experience

if TYPE_CHECKING:
    from .generator import GenOutcome

_DEFAULT_CRITIC_SYS = (
    "你是严格的产品图评判员。对照参考图(target)对生成图打分：二分维度逐项判通过/不通过 + 一句理由，"
    "连续维度给 0-1 分 + 一句理由。可用 query_rubric 查某维度的判定标准。所有维度都是"
    "「生成图 vs target」的还原对比，不是绝对评判。拿不准倾向判不通过/给低分。\n"
    "工作流：(1) 看图 → 按需 query_rubric 查判定标准 → 逐维度打分；"
    "(2) 打完分后总结本轮经验——对每个本轮有判断的维度调用 note_experience 写下"
    "『改动是否有效(effective/ineffective/escalated) + 可执行 lesson』，沉淀到经验知识库供后续轮次复用；"
    "你作为裁判对『为什么有效/无效』理解最深最准，这是本职之一，务必执行（首轮无上轮对比时仍可记录本轮新发现的问题与 lesson）；"
    "可用 query_experience 查已沉淀的历史经验辅助判断、避免重复无效思路；"
    "(3) 最后结构化输出每个维度的评分（不要自己算加权和/还原度，权重不在你手上）。\n"
    "【反阿谀/反幻觉（最高优先级）】只描述你确实在两张图上看到的，严禁凭『这类产品通常有』脑补 "
    "target/生成图里不存在的部件（抽屉、床头柜、面板数量等）——这是最常见的虚高分来源。reason 必须"
    "引用能在图上指出的具体点（如『床头板顶部弧度一致』『左数第二根立柱偏左』）；只用『完全对应/高度吻合』"
    "却说不出生成图与 target 具体哪些点匹配 → 视为没认真比对 → 判不通过/给低分。打 product_structure 前"
    "先独立列出 target 部件清单、再列生成图部件清单、再逐件比对，发现多件/少件/形态偏差/穿模即判不通过。"
    "二分维度逐项非过即不过；连续维度有瑕疵 ≤0.6。宁可误判不通过，不可放水判通过。"
)


@dataclass
class CriticInput:
    """一次 Critic 评判的输入（一个 trace）。

    经验总结相关字段（run_dir/outcome/...）由 graph 的 critic_node 注入；缺省时 evaluate 跳过
    经验总结（向后兼容旧测试）。生产路径下，critic 在打分循环里调 note_experience 写下第一手
    经验判断，evaluate 拿到结构化评分后由 Summarizer 用它走规则落盘。
    """

    sample: Sample
    generated_images: list[Path] = field(default_factory=list)  # 三视图等生成图
    weights: dict[str, float] = field(default_factory=dict)  # 当前生效权重
    meaning: str | None = None  # Generator 的一句话图片含义解释（风格迁移场景；判概念表达时参考）
    reference_ids: list[str] = field(default_factory=list)  # 本次实际传给生图 API 的参考标识符；判 reference_independence 用
    # —— 经验总结上下文（evaluate 内部驱动 in-loop 经验总结；缺省则跳过）——
    run_dir: Path | None = None
    round: int = 0
    outcome: GenOutcome | None = None
    prev_verdict: CriticVerdict | None = None
    prev_delta_note: str | None = None
    sample_id: str | None = None


# ---------------------------------------------------------------------------
# 多模态内容构造（生成图 + target 注入 HumanMessage）
# ---------------------------------------------------------------------------


def _build_multimodal_content(
    text: str, images: list[Path], *, max_dim: int = _CRITIC_VIEW_MAX_DIM,
) -> list[dict] | str:
    """构造 OpenAI 兼容的多模态 content：text + 图片(data-URI)。无图时返回纯文本。

    默认用高清 ``_CRITIC_VIEW_MAX_DIM``——critic 要做结构/部件逐件比对，小图会把瑕疵糊掉导致放水。
    trace 体积由 LangSmith 上传侧 compress_images_in_trace 压，不靠这里缩。
    """
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text}]
    for p in images:
        if Path(p).exists():
            parts.append({
                "type": "image_url",
                "image_url": {"url": resized_data_uri(Path(p), max_dim=max_dim)},
            })
    return parts


def _images_block(target: Path, generated: list[Path]) -> str:
    """文字版喂图说明（多模态 content 里也带这段文字锚，便于 LLM 区分图序）。"""
    parts = ["[生成图]"] + [f"  - view {i+1}: {p.name}" for i, p in enumerate(generated)]
    parts += ["[参考图(target 产品实物)]", f"  - {target.name}"]
    return "\n".join(parts)


def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
    cfg: dict = dict(config) if config else {}
    cfg["recursion_limit"] = limit
    return cfg


# ---------------------------------------------------------------------------
# Critic 主体
# ---------------------------------------------------------------------------


class Critic:
    """混合评分 Critic（deepagent 引擎）。chat model + bench 注入。"""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        bench: Benchmark,
        system_prompt: str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.bench = bench
        self.system_prompt = system_prompt or load_system_prompt("critic", _DEFAULT_CRITIC_SYS)
        # 结构化输出 schema：按 bench 维度动态生成「每维度一个具名 required 字段」的
        # CriticAgentOutput——杜绝重复维度/编造维度名，漏填在校验层即失败（详见模块 docstring）。
        self._output_schema, self._dim_to_field = build_critic_output_schema(bench.score_dimensions)
        # 创造力标准 overlay（creativity_tuner 产物）：Critic 实例构建时读一次，覆盖种子 content_spec。
        # 每个 loop 子进程构建一次 → tuner 只在批与批之间生效（不污染在跑的 loop）。
        self._creativity_overlay = _load_creativity_overlay(bench.bench_id)
        # 经验总结（原 in-loop Summarizer 职责）：critic 打分后兼做跨轮因果验证 + lesson 富化。
        # Summarizer 不再作为 graph 独立节点——loop 只剩 generator/critic 两个 agent；其成熟逻辑
        # （distiller 跨 loop 蒸馏依赖的 conclusions.json 产物）作为工具被 critic 调用。
        from .summarizer import Summarizer
        self._summarizer = Summarizer(chat_model=chat_model)

    def evaluate(
        self, inp: CriticInput, *, config: RunnableConfig | None = None,
        extra_hints: list[str] | None = None,
    ) -> CriticVerdict:
        """对一个 trace 打分，产出 CriticVerdict。

        所有维度都对照 target 评判。生成图 + target 以 image_url 注入初始 HumanMessage。
        agent 用 response_format=具名字段 schema 约束最终输出；代码侧映射 + 算 restoration。
        校验失败/语义问题 → 反馈给模型修正（REPAIR_ROUNDS 轮）；最终仍失败 → 逐维度安全
        默认（不中断闭环）。
        """
        spec = inp.sample.spec
        target = inp.sample.target_path
        generated = inp.generated_images

        user_content = self._build_user_content(
            target, generated, spec, meaning=inp.meaning, extra_hints=extra_hints,
            reference_ids=inp.reference_ids,
            prev_delta_note=inp.prev_delta_note, run_dir=inp.run_dir, sample_id=inp.sample_id,
        )
        # note_experience 工具把 agent 打分时的第一手经验判断回写到 sink；evaluate 拿到 verdict 后落盘。
        sink: dict = {}
        tools = make_critic_tools(
            bench=self.bench, spec=spec, overlay=self._creativity_overlay,
            sink=sink, run_dir=inp.run_dir,
        )
        agent = create_deep_agent(
            model=self.chat_model, tools=tools,
            system_prompt=self.system_prompt,
            response_format=provider_structured(self._output_schema), checkpointer=None, name="critic",
            middleware=narrow_tools_middleware(),
        )

        out = self._invoke_with_repair(
            agent, [HumanMessage(content=user_content)], spec=spec,
            config=_merge_recursion(config, AGENT_RECURSION_LIMIT),
        )

        dim_scores = self._to_dimension_scores(out)
        restoration = recompute_restoration(dim_scores, inp.weights)
        verdict = CriticVerdict(
            sample_id=spec.sample_id,
            dimensions=dim_scores,
            weights_used=dict(inp.weights),
            restoration=restoration,
        )

        # 经验总结（原 in-loop Summarizer 职责，现 critic 工作流内由 note_experience 驱动）：
        # 用 agent 写下的判断(sink["lessons"]) + 本轮 verdict 走规则落盘，更新 conclusions.json。
        # 需 run_dir/outcome/sample_id（生产路径）；缺省（旧测试）跳过，向后兼容。总结失败不中断打分。
        if inp.run_dir is not None and inp.outcome is not None and inp.sample_id:
            try:
                verdict.lesson_ref = self._summarizer.summarize(
                    run_dir=inp.run_dir, round=inp.round, outcome=inp.outcome,
                    verdict=verdict, sample_id=inp.sample_id,
                    prev_verdict=inp.prev_verdict, prev_delta_note=inp.prev_delta_note,
                    config=config, agent_lessons=sink.get("lessons"),
                )
            except Exception as e:  # noqa: BLE001  总结失败不中断打分闭环
                print(f"[critic] 经验总结失败({type(e).__name__}: {str(e)[:150]})，跳过", flush=True)
        return verdict

    # --- 辅助 ---

    # 反馈修正轮数：校验失败/语义问题各最多给模型 REPAIR_ROUNDS 次带反馈的修正机会。
    REPAIR_ROUNDS: int = 2
    # 网关瞬时异常（SSL EOF/超时等）的同 payload 退避重试次数（与 invoke_with_retry 语义一致）。
    _TRANSIENT_RETRIES: int = 5

    def _invoke_with_repair(
        self, agent, messages: list[BaseMessage], *, spec, config: dict,
    ) -> Any:
        """invoke agent 并处理结构化输出问题：**带反馈的修正回路**（区别于盲目退避重试）。

        两类失败分开处理：
        - **网关瞬时异常**（SSL EOF/超时/5xx）：同 payload 指数退避重试（旧
          invoke_with_retry 语义，`_TRANSIENT_RETRIES` 次）——重试有效，因为不是模型的错。
        - **结构化输出校验失败 / 语义问题**（缺维度、items 空、id 不对应、缺 value/reason）：
          盲目重试大概率原样再错——改为把**具体错误**反馈给模型，续同一会话让它修正
          （最多 ``REPAIR_ROUNDS`` 轮）。修正后拿全新输出重新校验。

        全部失败时对最后一次坏输出做**宽松解析**逐维度兜底：能解析出的维度保留真实评分，
        解析不出的维度才退安全默认（不是整轮全 0）。

        Returns:
            结构化输出（pydantic 实例 / 宽松解析出的 dict）或 None（完全无可用输出）。
        """
        msgs = list(messages)
        last_bad_ai: AIMessage | None = None
        out: Any = None
        for attempt in range(self.REPAIR_ROUNDS + 1):
            res, sr, feedback, bad_ai = self._invoke_once(agent, msgs, config=config)
            last_bad_ai = bad_ai or last_bad_ai
            if sr is not None:
                errs = self._semantic_errors(sr, spec)
                if not errs:
                    return sr
                out = sr  # 语义有问题但结构可用——兜底时按维度宽松 salvage
                feedback = (
                    "你的结构化评分存在以下问题，请逐项修正后重新输出**完整**的全部维度评分 JSON：\n"
                    + "\n".join(f"- {e}" for e in errs)
                )
            if feedback is None:
                # 瞬时异常重试全灭：无模型输出可反馈，直接走兜底
                break
            if attempt >= self.REPAIR_ROUNDS:
                print(f"[critic] 修正 {self.REPAIR_ROUNDS} 轮仍有问题，退兜底：{feedback[:300]}",
                      flush=True)
                break
            print(f"[critic] 结构化输出有问题，反馈修正（第 {attempt + 1}/{self.REPAIR_ROUNDS} 轮）："
                  f"{feedback[:300]}", flush=True)
            # 续会话：保留本轮全部消息（模型能看到自己刚犯的错）+ 反馈
            msgs = list(res["messages"]) if res else msgs
            if bad_ai is not None:
                msgs = msgs + [bad_ai]  # 校验失败路径 agent 状态里没有那条坏消息，补回
            msgs.append(HumanMessage(content=feedback))
        # 宽松兜底：最后一次（语义有问题的）结构化输出优先；否则从坏 AI 消息里抠 JSON
        if out is not None:
            return out
        return self._lenient_parse(last_bad_ai)

    def _invoke_once(self, agent, messages: list[BaseMessage], *, config: dict):
        """单次 invoke：瞬时异常退避重试；校验错误立即上抛为反馈（不盲目重试）。

        Returns:
            (result, structured_response, feedback, bad_ai)：
            - 成功 → (res, sr, None, None)
            - 校验失败 → (None, None, 反馈文本, 模型的坏 AIMessage)
            - 瞬时异常重试全灭 → (None, None, None, None)
        """
        delay = 2.0
        for i in range(self._TRANSIENT_RETRIES + 1):
            try:
                res = agent.invoke({"messages": messages}, config=config)
                sr = res.get("structured_response") if isinstance(res, dict) else None
                return res, sr, None, None
            except StructuredOutputValidationError as e:
                # 模型输出没通过 schema 校验（JSON 坏 / 缺 required 字段）：重试大概率原样再错，
                # 把校验错误细节反馈给模型修正（不用退避——这不是网关抖动）。
                return None, None, self._validation_feedback(e), e.ai_message
            except Exception as e:  # noqa: BLE001  瞬时异常重试逻辑要吞所有异常
                msg = str(e).replace("\n", " ")[:200]
                if i < self._TRANSIENT_RETRIES:
                    print(f"[critic] invoke 异常({type(e).__name__}: {msg})，"
                          f"重试 {i + 1}/{self._TRANSIENT_RETRIES}", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                else:
                    print(f"[critic] 重试仍异常({type(e).__name__}: {msg})，退兜底", flush=True)
        return None, None, None, None

    @staticmethod
    def _validation_feedback(e: StructuredOutputValidationError) -> str:
        """把 StructuredOutputValidationError 翻译成给模型的修正反馈（具体、可执行）。"""
        src = str(e.source)[:500]
        return (
            "你上一条回复的结构化输出未通过 schema 校验，未被判分系统接受。错误详情：\n"
            f"{src}\n"
            "请对照错误重新输出**完整**的全部维度评分 JSON：每个维度一个字段（字段名见 "
            "response_format schema，不要自造字段/数组），二分维度填逐项判定列表 "
            "[{id, passed, reason}]，连续维度填 {value, reason}。只输出 JSON。"
        )

    def _semantic_errors(self, out: Any, spec) -> list[str]:
        """结构化输出通过 schema 校验后的**语义**检查（schema 表达不了的规则）。

        检查项（都可修复——反馈给模型修正）：
        - 二分维度 items 为空（模型偷懒没逐项判）；
        - items 的 id 集合与该维度生效 checklist 不一致（缺判/多判/自造 id——
          通过率 = passed/len，缺一项分数就错）；
        - 连续维度 reason 为空。
        """
        errs: list[str] = []
        for ddef in self.bench.score_dimensions:
            node = _node_get(out, self._dim_to_field[ddef.dim])
            if node is None:
                errs.append(f"缺少维度「{ddef.dim}」的评分字段")  # required 已拦，防御
                continue
            if ddef.scoring_type == "binary":
                items = node if isinstance(node, list) else []
                if not items:
                    errs.append(
                        f"二分维度「{ddef.dim}」的判定列表为空——须对每一项 checklist 给出 "
                        f"{{id, passed, reason}} 判定"
                    )
                    continue
                cl = _effective_checklist(spec, ddef.dim, self._creativity_overlay, bench=self.bench)
                expected = [it.id for it in (cl if isinstance(cl, list) else [])]
                if expected:
                    got = [str(_item_get(it, "id")) for it in items]
                    missing = sorted(set(expected) - set(got))
                    extra = sorted(set(got) - set(expected))
                    parts = []
                    if missing:
                        parts.append(f"缺判定 id: {missing}")
                    if extra:
                        parts.append(f"多出/自造 id: {extra}")
                    if parts:
                        errs.append(
                            f"二分维度「{ddef.dim}」判定项与 checklist 不对应（{'；'.join(parts)}）——"
                            f"必须逐项对应（应有 {expected}），不得合并/省略/自造"
                        )
            else:
                reason = _node_get(node, "reason")
                if not str(reason or "").strip():
                    errs.append(f"连续维度「{ddef.dim}」缺 reason（须给一句评分理由）")
        return errs

    @staticmethod
    def _lenient_parse(ai_msg: AIMessage | None) -> dict | None:
        """兜底：从模型最后一条坏消息里宽松抠出 JSON dict（schema 校验不过也没关系）。

        _to_dimension_scores 对 dict/pydantic 一视同仁、逐维度取用（取不到 → 安全默认），
        所以哪怕只解析出一部分维度，其余维度也能逐个兜底而不是整轮全 0。
        """
        if ai_msg is None:
            return None
        content = ai_msg.content
        text = content if isinstance(content, str) else "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def _build_user_content(
        self, target: Path, generated: list[Path], spec, meaning: str | None = None,
        extra_hints: list[str] | None = None, reference_ids: list[str] | None = None,
        prev_delta_note: str | None = None, run_dir: Path | None = None,
        sample_id: str | None = None,
    ) -> list[dict] | str:
        """初始 HumanMessage：喂图说明 + 每维度**逐项 checklist** + 评分指令 + 生成图与 target(image_url)。

        关键：把每个二分维度的 checklist 项（id+判定+anchor）直接列出来，强制 agent 逐项判，
        返回与项数相等、id 逐项对应的 items——否则 agent 会偷懒只给一个聚合判断（导致 passed/1=满分级虚高）。
        """
        dim_lines: list[str] = []
        for d in self.bench.score_dimensions:
            cl = _effective_checklist(spec, d.dim, self._creativity_overlay, bench=self.bench)
            if d.scoring_type == "binary":
                items = list(cl) if isinstance(cl, list) else []
                item_lines = "\n".join(
                    f"    - {getattr(it, 'id', '?')}: {getattr(it, 'check', '')}"
                    + (f"（{it.anchor}）" if getattr(it, 'anchor', None) else "")
                    for it in items
                ) or "    (未定义 checklist 项)"
                dim_lines.append(
                    f"- {d.dim}（二分，逐项 ✓/✗ + 理由）：{d.desc or ''}\n"
                    f"  必须为下面**每一项**返回一条 item（id 严格对应）：\n{item_lines}"
                )
            else:
                pts = getattr(cl, "points", None) or []
                ptstr = "; ".join(pts) if pts else "(无)"
                dim_lines.append(
                    f"- {d.dim}（连续，0-1 分 + 理由）：{d.desc or ''}。评分要点：{ptstr}"
                )
        task = (
            "对照参考图(target)，对生成图按下列维度逐一打分（所有维度都是 生成图 vs target 的还原对比）。\n"
            "二分维度：为列出的**每一项** checklist 返回一条 {id, passed, reason}——"
            "items 数量必须等于列出的项数、id 逐项对应，**不得合并、不得省略**（通过率=通过项/总项，少一项分数就错）。"
            "连续维度：给 0-1 的 value + 一句 reason。\n"
            "**严格**：拿不准、有瑕疵、与 target 不完全一致时，倾向判不通过/给低分；只有确无问题才判通过。\n"
            "维度清单：\n" + "\n".join(dim_lines)
            + "\n\n最终结构化输出：每个维度一个字段（字段名=维度名，见 schema）——"
            "二分维度填逐项判定列表，连续维度填 {value, reason}（两字段都必须填）。"
            "同一维度只输出一次，不得重复、不得自造维度名。"
        )
        # 参考图使用情况：让 critic 能判 reference_independence（对照「实际传入」的参考，而非全 7 张）
        if reference_ids is not None:
            ids_str = ", ".join(reference_ids) if reference_ids else "（空 = 纯文生图，未用参考图）"
            task += (
                f"\n\n【参考图使用情况】本次 Generator 实际传给生图 API 的参考图(reference_ids)：{ids_str}。"
                f"reference_independence 维度据此判定：只判生成图是否复制了**这些实际传入**参考图的 motif；"
                f"reference_ids 为空（纯文生图）时 reference_independence 默认通过。"
            )
        # 人工补充评分准则（运行时人工介入；与 checklist 同等效力，必须执行）
        if extra_hints:
            task += (
                "\n\n【额外评分准则（人工补充，与 checklist 同等效力，必须执行）】\n"
                + "\n".join(f"- {h}" for h in extra_hints)
            )
        if meaning:
            task = f"【Generator 的图片含义解释（它声称这张图想表达的概念）】{meaning}\n\n" + task
        if spec.task and spec.task.mode == "style_transfer":
            task = (
                "【风格神韵迁移 · 自主对照参考集判一致性（最高优先级）】\n"
                "严格对照参考集(target)：生成图任何与参考集风格不一致的细节——"
                "写实的解剖/纹理(指甲/指纹/关节纹/皮肤)、参考集里没有的元素、写实阴影/高光——"
                "即使下列 checklist 未明确写出，也**必须在对应 spirit_* 维度判不通过 / 给低分**，并在 reason 写明你发现的偏差。\n"
                "你要主动发现 checklist 之外的偏差(参考集是纯线条抽象、零解剖细节；生成图出现任何写实细节就是不一致)，不要只盯列出的项。\n\n"
            ) + task
        # 经验总结上下文：让 critic 打分时看到历史经验 + 上轮改动，写出更准的 note_experience。
        # 连续失败维度（判 escalated 时参考）+ 已沉淀经验（避免重复无效思路）+ 上轮改动（验证对象）。
        if run_dir is not None and sample_id:
            try:
                kb = load_conclusions(run_dir, sample_id=sample_id)
            except Exception:  # noqa: BLE001
                kb = None
            exp_ctx: list[str] = []
            if kb is not None:
                streaks = {d: s for d, s in kb.fail_streaks.items() if s > 0}
                if streaks:
                    exp_ctx.append(
                        "【连续失败维度（疑似模型能力上限，写 note_experience 时该维度判 escalated）】"
                        + "、".join(f"{d}({s}轮)" for d, s in sorted(streaks.items(), key=lambda kv: -kv[1]))
                    )
                try:
                    exp_hist = _format_experience(run_dir)
                except Exception:  # noqa: BLE001
                    exp_hist = ""
                if exp_hist and "暂无已验证经验" not in exp_hist:
                    exp_ctx.append("【本题已沉淀经验（参考，写 note_experience 时避免重复无效思路）】\n" + exp_hist)
            if prev_delta_note:
                exp_ctx.append(f"【上轮改动（note_experience 的验证对象：判断该改动是否有效）】{prev_delta_note}")
            if exp_ctx:
                task += "\n\n" + "\n\n".join(exp_ctx)
        text = _images_block(target, generated) + "\n\n" + task
        images = list(generated) + ([target] if target.exists() else [])
        return _build_multimodal_content(text, images)

    def _to_dimension_scores(self, out: Any) -> list[DimensionScore]:
        """把 agent 结构化输出映射成 DimensionScore 列表（按 bench 维度顺序，缺失→安全默认）。

        out 是具名字段 schema 的实例（或兜底宽松解析出的 dict）：每个 bench 维度对应一个
        字段，直接按 ``_dim_to_field`` 取——**没有重复维度问题**（对象键唯一）。
        某维度字段缺失/形态不对 → 该维度退安全默认，其余维度照常（部分 salvage）。
        """
        scores: list[DimensionScore] = []
        for ddef in self.bench.score_dimensions:
            node = _node_get(out, self._dim_to_field[ddef.dim])
            if node is None:
                scores.append(self._safe_dim(ddef.dim, ddef.scoring_type))
                continue
            if ddef.scoring_type == "binary":
                items = node if isinstance(node, list) else []
                value = (sum(1 for it in items if _item_get(it, "passed")) / len(items)) if items else 0.0
                scores.append(DimensionScore(
                    dim=ddef.dim, scoring_type="binary", value=value, items=items,
                ))
            else:
                v = _node_get(node, "value")
                if v is None:
                    scores.append(self._safe_dim(ddef.dim, ddef.scoring_type))
                else:
                    scores.append(DimensionScore(
                        dim=ddef.dim, scoring_type="continuous",
                        value=min(max(float(v), 0.0), 1.0), raw=str(_node_get(node, "reason") or ""),
                    ))
        return scores

    @staticmethod
    def _safe_dim(dim: str, scoring_type: str) -> DimensionScore:
        if scoring_type == "binary":
            return DimensionScore(dim=dim, scoring_type="binary", value=0.0, items=[])
        return DimensionScore(dim=dim, scoring_type="continuous", value=0.0, raw="(eval failed)")


def _node_get(out: Any, key: str) -> Any:
    """从结构化输出（pydantic 实例或 dict）里取字段，取不到返回 None（不抛）。"""
    if out is None:
        return None
    if isinstance(out, dict):
        return out.get(key)
    return getattr(out, key, None)


def _item_get(item: Any, key: str) -> Any:
    """从一条判定项（CriticItemJudgment 实例或 dict）里取字段，取不到返回 None。"""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


__all__ = ["Critic", "CriticInput"]
