"""LangGraph 闭环 A：generator → critic → summarizer → human_review(interrupt) → 条件边。

节点职责（ARCH §2）：
  - generator_node：控制变量法构造 GenRequest → 三视图出图
  - critic_node：对三视图打混合评分
  - summarizer_node：写经验 MD + index；记 AttemptRecord
  - human_review_node：interrupt() 等人工裁决（continue/stop/调方向），不自动收敛（ADR-007）

条件边 route：decision==stop → END，否则回到 generator。
checkpointer 用 SqliteSaver（持久化，可断点续跑）；测试可用 InMemorySaver。

节点函数返回 dict（部分 state 更新）；累加器字段（images/verdicts/attempts）用 list 触发 operator.add。
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agents.critic import Critic, CriticInput
from ..agents.generator import Generator, GenOutcome
from ..agents.summarizer import Summarizer
from ..data.benchmark import LoadedBenchmark
from ..data.runstore import RunStore
from ..data.trajectory import TrajectoryWriter
from ..data.weights import load_weights
from ..memory.schema import AttemptRecord, CriticVerdict, TestVariable
from .state import CompiledGraph, RunState


def build_graph(
    *,
    bench: LoadedBenchmark,
    run_store: RunStore,
    generator: Generator,
    critic: Critic,
    summarizer: Summarizer,
    sample_id: str,
    checkpointer=None,
) -> CompiledGraph:
    """构建闭环 A 图。checkpointer 不传则用 InMemorySaver（测试）。"""

    run_dir = run_store.run_dir
    sample = bench.sample(sample_id)

    def generator_node(state: RunState) -> dict:
        round_n = state.get("round", 0) + 1
        baseline_ref = None
        # 若有历史 attempt，取还原度最高的作对照（控制变量法）。
        # 同一 run 内模型固定（ADR-007），故无需按 model 过滤。
        from ..memory.index import recall
        prior = recall(run_dir, limit=1)
        if prior:
            baseline_ref = prior[0].get("attempt_id")

        outcome = generator.generate_round(
            sample=sample, out_dir=run_dir / "out", run_dir=run_dir,
            round=round_n, baseline_ref=baseline_ref,
            test_variable=_pick_variable(state),
        )
        # 透传本轮非 state 字段（用 dict 暂存到 state 的临时键，靠 total=False 容纳）
        return {
            "round": round_n,
            "_outcome": outcome,  # 临时，供 critic/summarizer 节点取用（不走 reducer）
            "images": list(outcome.output_image_refs),
        }

    def critic_node(state: RunState) -> dict:
        outcome: GenOutcome = state["_outcome"]  # type: ignore[typeddict-item]
        weights = load_weights(bench.bench, run_dir=run_dir)
        # 生成的图绝对路径
        gen_imgs = [run_dir / r for r in outcome.output_image_refs]
        verdict = critic.evaluate(CriticInput(
            sample=sample, generated_images=gen_imgs, weights=weights,
        ))
        return {"_verdict": verdict, "verdicts": [verdict]}

    def summarizer_node(state: RunState) -> dict:
        outcome: GenOutcome = state["_outcome"]  # type: ignore[typeddict-item]
        verdict: CriticVerdict = state["_verdict"]  # type: ignore[typeddict-item]
        lesson_ref = summarizer.summarize(
            run_dir=run_dir, round=state["round"], outcome=outcome,
            verdict=verdict, sample_id=sample_id,
        )
        # 写 trajectory
        rec = AttemptRecord(
            attempt_id=outcome.attempt_id, run_id=run_store.meta.run_id if run_store.meta else "",
            round=state["round"], sample_id=sample_id, bench_id=bench.bench.bench_id,
            model=outcome.model, test_variable=outcome.test_variable,
            baseline_ref=outcome.baseline_ref, gen_mode=outcome.gen_mode,
            prompt=outcome.prompt, reference_image_refs=list(outcome.reference_image_refs),
            size=outcome.size, output_image_refs=list(outcome.output_image_refs),
            verdict=verdict, lesson_ref=lesson_ref,
        )
        TrajectoryWriter(run_dir / "trajectory.jsonl").append(rec)
        return {"attempts": [rec]}

    def human_review_node(state: RunState) -> dict:
        verdict: CriticVerdict | None = state.get("_verdict")  # type: ignore[assignment]
        restoration = verdict.restoration if verdict else None
        round_n = state.get("round", 0)
        # 失败项摘要给人工看
        failed: list[str] = []
        if verdict:
            for d in verdict.dimensions:
                if d.scoring_type == "binary":
                    failed += [it.id for it in (d.items or []) if not it.passed]
        payload = {
            "round": round_n,
            "restoration": round(restoration, 4) if restoration is not None else None,
            "failed_items": failed,
            "images": state.get("images", [])[-3:],
            "prompt": "回复 continue 继续下一轮 / stop 停止 / 或输入调整方向",
        }
        decision = interrupt(payload)
        return {"decision": decision}

    def route(state: RunState) -> Literal["generator", "__end__"]:
        if state.get("decision") == "stop":
            return "__end__"
        return "generator"

    builder = StateGraph(RunState)
    builder.add_node("generator", generator_node)
    builder.add_node("critic", critic_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("human_review", human_review_node)
    builder.add_edge(START, "generator")
    builder.add_edge("generator", "critic")
    builder.add_edge("critic", "summarizer")
    builder.add_edge("summarizer", "human_review")
    builder.add_conditional_edges("human_review", route, {"generator": "generator", END: END})

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def _pick_variable(state: RunState) -> TestVariable | None:
    """决定本轮变哪个维度。首轮 None；之后在 prompt/size 间轮换（简化策略）。"""
    round_n = state.get("round", 0)
    if round_n == 0:
        return None
    # 简化：偶数轮调 prompt，奇数轮调 size（真实策略可由 Generator LLM 决策）
    return "prompt" if round_n % 2 == 0 else "size"


def run_loop(
    *,
    bench: LoadedBenchmark,
    run_store: RunStore,
    sample_id: str,
    generator: Generator,
    critic: Critic,
    summarizer: Summarizer,
    decisions: list[str],
    thread_id: str | None = None,
) -> RunState:
    """便捷驱动：跑完给定的 decisions 序列（每次 resume 用一个 decision）。

    decisions[0] 对应首轮 resume（首轮自动跑到 interrupt，第一个 decision 续上）。
    最后一个 decision 通常是 'stop'。
    用于测试与 CLI。
    """
    import uuid
    tid = thread_id or f"{run_store.run_dir.name}-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": tid}}
    app = build_graph(bench=bench, run_store=run_store, generator=generator,
                      critic=critic, summarizer=summarizer, sample_id=sample_id)

    # 首轮：给初始输入跑到第一个 interrupt
    state = app.invoke({"round": 0, "model": (run_store.meta.model if run_store.meta else ""),
                        "bench_id": bench.bench.bench_id, "sample_id": sample_id,
                        "run_id": tid}, config=cfg)
    # 逐个用 decision 续跑
    for d in decisions:
        state = app.invoke(Command(resume=d), config=cfg)
        if state.get("decision") == "stop":
            break
    return state


__all__ = ["build_graph", "run_loop"]
