"""CLI 入口：跑闭环 A（生成迭代）/ 闭环 B（校准）/ 分析 / 经验蒸馏。

子命令：
  run        闭环 A：生成→评分→总结→人工审批（逐轮 interrupt）
  calibrate  闭环 B：用人工排序拟合维度权重（learning-to-rank）
  analyze    策略对比：跨 run 汇总还原度，画图
  summarize  跨 loop 经验蒸馏：独立 Summarizer 读一批 run 的 trajectory+conclusions
             → 通用经验 → experience/<bench>/general.json（不跑 loop、不动 conclusions.json）

真实生图与 LLM 评分需 .env 配好 key 与 model_id。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .data.benchmark import load_benchmark
from .data.runstore import RunStore
from .pipeline.runner import build_loop_context, close_checkpointer, run_loop_session

# Generator/Critic 的 LLM 由 build_loop_context 用 build_chat_model（ChatOpenAI，指向 dmxapi）
# 构造并注入 deepagent；本模块不直接触碰 LLM client。


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

    # 一处收口：agent 配方 + checkpointer + build_graph + 标准 config。
    # loop 整体跑在 run_loop_session（@traceable）下 → 1 loop = 1 LangSmith trace。
    assert store.meta is not None  # create() 必已设置
    ctx = build_loop_context(lb, store, args.sample, loop_model=store.meta.model)
    try:
        return run_loop_session(
            ctx.app, ctx.cfg, store,
            rounds=args.rounds, bench_id=args.bench, sample_id=args.sample,
            prompt_decision=lambda _r, _v: input("  > "),
            langsmith_extra={"metadata": {
                "loop_id": loop_id, "bench_id": args.bench,
                "sample_id": args.sample, "model": store.meta.model,
            }},
        )
    finally:
        close_checkpointer(ctx.checkpointer)


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


def cmd_summarize(args: argparse.Namespace) -> int:
    """跨 loop 经验蒸馏：独立 Summarizer 读一批 run 的 trajectory+conclusions → 通用经验。

    不跑 loop、不动 conclusions.json。产物写 <data_root>/experience/<bench>/general.json。
    """
    import glob

    from .agents.experience_distiller import ExperienceDistiller
    from .llm.chat_model import build_chat_model
    from .memory.experience import save_general_experience
    from .pipeline.runner import _skills_dir

    settings = get_settings()
    # 确定 run_dirs：显式 --runs，或 --bench 取 runs_dir 下 <bench>-*
    if args.runs:
        run_dirs = [Path(p) for p in args.runs]
    elif args.bench:
        run_dirs = [Path(p) for p in glob.glob(str(settings.runs_dir / f"{args.bench}-*"))]
    else:
        print("错误：需要 --bench 或 --runs", file=sys.stderr)
        return 1
    run_dirs = [rd for rd in run_dirs if (rd / "trajectory.jsonl").exists()]
    if not run_dirs:
        print("[summarize] 没有含 trajectory 的 run", file=sys.stderr)
        return 1

    # bench_id：显式或从首条 trajectory 推断
    if args.bench:
        bench_id = args.bench
    else:
        from .data.trajectory import TrajectoryReader
        first = next(iter(TrajectoryReader(run_dirs[0] / "trajectory.jsonl").iter_records()), None)
        bench_id = first.bench_id if first else "unknown"
    bench = load_benchmark(bench_id, settings=settings).bench

    chat = build_chat_model(settings, role="summarizer")
    distiller = ExperienceDistiller(
        chat, run_dirs=run_dirs, bench=bench,
        skills_dir=_skills_dir("experience-distiller"),
    )
    exp = distiller.distill()
    path = save_general_experience(settings.data_root, bench.bench_id, exp)
    print(f"[summarize] {len(exp.lessons)} 条通用经验（来自 {len(exp.source_runs)} 个 run）→ {path}")
    print(f"summary: {exp.summary}")
    for i, ls in enumerate(exp.lessons, 1):
        print(f"  {i}. [{ls.dim}] {ls.insight} (conf={ls.confidence:.2f})")
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

    p_sum = sub.add_parser("summarize", help="跨 loop 经验蒸馏（独立 Summarizer，不跑 loop）")
    p_sum.add_argument("--bench", default=None, help="bench_id（取 runs_dir 下 <bench>-* 全部 run）")
    p_sum.add_argument("--runs", nargs="*", default=None, help="显式指定 run 目录")
    p_sum.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
