"""CLI 入口：跑闭环 A（生成迭代）/ 闭环 B（校准）/ 分析。

子命令：
  run        闭环 A：生成→评分→总结→人工审批（逐轮 interrupt）
  calibrate  闭环 B：用人工排序拟合维度权重（learning-to-rank）
  analyze    策略对比：跨 run 汇总还原度，画图

真实生图与 LLM 评分需 .env 配好 key 与 model_id。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agents.critic import Critic
from .agents.generator import Generator
from .agents.summarizer import Summarizer
from .config import get_settings
from .data.benchmark import load_benchmark
from .data.runstore import RunStore
from .generation.client import DmxapiClient
from .generation.router import Router
from .llm.openai_compat import OpenAiCompatLlm
from .pipeline.graph import build_graph


# Agent LLM client（OpenAiCompatLlm，含 langsmith.wrap_openai）已抽到 llm/openai_compat.py，
# 消除 web.services.loop_runner → cli 的反向依赖。


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    lb = load_benchmark(args.bench, settings=settings)
    # 一题一 loop：loop_id = <bench>-<sample>（或用户指定的 --run-id）。
    # 已有则续跑（open），否则新建（create）。
    loop_id = args.run_id or f"{args.bench}-{args.sample}"
    run_dir = settings.run_dir(loop_id)
    if run_dir.exists():
        store = RunStore.open(loop_id, settings=settings)
        print(f"[run] 续跑已有 loop: {loop_id}（当前 {len(list(run_dir.glob('out/a*')))} 轮历史）")
    else:
        store = RunStore.create(loop_id, args.bench,
                                model=args.model or settings.model_seedream_pro,
                                settings=settings, note=args.note)

    # 构造 generator/critic/summarizer
    router = Router(settings=settings, client=DmxapiClient(settings))
    gen_llm = OpenAiCompatLlm(settings, model=settings.generator_model) if settings.generator_model else None
    generator = Generator(router, llm=gen_llm)
    critic = Critic(OpenAiCompatLlm(settings, model=settings.critic_model), bench=lb.bench)
    summarizer = Summarizer()

    # SqliteSaver 持久化 checkpoint（可断点续跑）
    import sqlite3
    conn = sqlite3.connect(store.run_dir / "checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_graph(bench=lb, run_store=store, generator=generator, critic=critic,
                      summarizer=summarizer, sample_id=args.sample, checkpointer=checkpointer)

    cfg = {"configurable": {"thread_id": store.run_dir.name}}
    # 首轮：跑到第一个 interrupt
    assert store.meta is not None  # create() 必已设置
    fixed_model = store.meta.model
    print(f"[run] {args.bench}/{args.sample} | model={fixed_model} | run_id={store.meta.run_id}")
    state = app.invoke({"round": 0, "model": fixed_model, "bench_id": args.bench,
                        "sample_id": args.sample, "run_id": store.run_dir.name}, config=cfg)

    round_done = 0
    for i in range(args.rounds):
        verdict = state.get("_verdict")
        r = state.get("round", 0)
        rest = verdict.restoration if verdict else None
        print(f"\n[round {r}] 还原度={rest:.4f} | 经验见 lessons/conclusions.json")
        print("  回复 continue 继续下一轮 / stop 停止 / 或输入调整方向:")
        try:
            decision = input("  > ").strip() or "continue"
        except EOFError:
            decision = "stop"
        state = app.invoke(Command(resume=decision), config=cfg)
        round_done = i + 1
        if state.get("decision") == "stop":
            print("[run] 已停止。")
            break

    store.finish(note=f"跑完 {round_done} 轮")
    print(f"[run] 完成。trajectory: {store.trajectory_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """闭环 B：用人工排序拟合维度权重。

    人工排序通过 --ranks 传入（与 --traces 对应，越大越好），
    或用 --use-restoration-as-rank 演示（注意：循环论证，仅供测试）。
    """
    import glob

    from .calibration.fit_weights import RankedTrace, fit_weights, save_calibrated_weights
    from .calibration.report import write_report
    from .data.benchmark import load_benchmark

    settings = get_settings()
    bench = load_benchmark(args.bench, settings=settings).bench

    # 收集 trace
    run_dirs = [Path(p) for p in args.runs] if args.runs else \
               [Path(p) for p in glob.glob(str(settings.runs_dir / "*"))]
    traces = []
    for rd in run_dirs:
        tp = rd / "trajectory.jsonl"
        if not tp.exists():
            continue
        import json
        for line in tp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("verdict") or {}
            feats = {d["dim"]: d["value"] for d in v.get("dimensions", [])}
            if not feats:
                continue
            rank = v.get("restoration", 0.0) if args.use_restoration_as_rank else 0.0
            traces.append(RankedTrace(trace_id=f"{rd.name}/r{r.get('round')}",
                                      features=feats, human_rank=rank))

    if not args.use_restoration_as_rank and args.ranks:
        ranks = [float(x) for x in args.ranks]
        if len(ranks) != len(traces):
            print(f"错误: --ranks 给了 {len(ranks)} 个值，但找到 {len(traces)} 条 trace",
                  file=sys.stderr)
            return 1
        for t, r in zip(traces, ranks):
            t.human_rank = r

    print(f"[calibrate] {len(traces)} 条 trace")
    result = fit_weights(traces, bench)
    print(f"[calibrate] 排序吻合度: {result.pairwise_accuracy:.1%}")
    for d in bench.score_dimensions:
        name = d.dim
        print(f"  {name:20} {result.prior_weights[name]:.3f} → {result.weights[name]:.3f}")

    out_dir = run_dirs[0] if run_dirs else settings.runs_dir
    save_calibrated_weights(result, out_dir)
    rep = write_report(result, out_dir, bench_label=args.bench)
    print(f"[calibrate] 权重→ {out_dir}/calibrated_weights.json | 报告→ {rep}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """策略对比：跨 run 读 trajectory，汇总还原度，画图。"""
    import glob

    from .analysis.strategy_compare import (
        load_trajectories,
        plot_restoration_by_round,
        summarize_by_round,
        summarize_by_sample,
    )

    settings = get_settings()
    run_dirs = [Path(p) for p in args.runs] if args.runs else \
               [Path(p) for p in glob.glob(str(settings.runs_dir / "*"))]
    df = load_trajectories(run_dirs)
    if df.empty:
        print("[analyze] 无 trajectory 数据", file=sys.stderr)
        return 1
    print(f"[analyze] {len(df)} 条 trace\n=== 按样本汇总 ===")
    print(summarize_by_sample(df).to_string())
    print("\n=== 各样本×轮次还原度 ===")
    print(summarize_by_round(df).to_string())
    if args.plot:
        out = plot_restoration_by_round(df, Path(args.plot))
        print(f"\n[analyze] 图→ {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="img-iter-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="跑闭环 A（生成迭代）")
    p_run.add_argument("--bench", default="furniture_product_whitebg")
    p_run.add_argument("--sample", default="s001")
    p_run.add_argument("--rounds", type=int, default=3)
    p_run.add_argument("--model", default=None, help="覆盖 RunStore 记录的固定模型")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--note", default=None)
    p_run.set_defaults(func=cmd_run)

    p_cal = sub.add_parser("calibrate", help="闭环 B：用人工排序拟合维度权重")
    p_cal.add_argument("--bench", default="furniture_product_whitebg")
    p_cal.add_argument("--runs", nargs="*", default=None, help="指定 run 目录（默认全部）")
    p_cal.add_argument("--ranks", nargs="*", default=None, help="人工排序值（与 trace 顺序对应，越大越好）")
    p_cal.add_argument("--use-restoration-as-rank", action="store_true",
                       help="用 Critic 还原度当排序替身（循环论证，仅供演示）")
    p_cal.set_defaults(func=cmd_calibrate)

    p_ana = sub.add_parser("analyze", help="策略对比（跨 run 汇总还原度）")
    p_ana.add_argument("--runs", nargs="*", default=None, help="指定 run 目录（默认全部）")
    p_ana.add_argument("--plot", default=None, help="输出折线图路径（可选）")
    p_ana.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
