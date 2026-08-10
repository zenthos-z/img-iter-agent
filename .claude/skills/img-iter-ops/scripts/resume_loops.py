#!/usr/bin/env python3
"""续跑已有 loop（追加 N 轮），支持多 sample 并发。

与 run_loop_auto 不同：不新建 loop，而是 ``RunStore.open`` 已有 loop，从 checkpoint 续跑。
适配线程当前态（同 web LoopRunner._invoke_round）：
  - interrupt 态（awaiting_review）→ ``Command(resume=decision)`` 跑下一轮；
  - END 态（finished）→ 从 START 重入（不带 round key，generator 自增 N→N+1）追新一轮。

续跑的轮自动用**最新 rubric**（content_spec 的 spirit 5 项等）+ **创造力 overlay**（load_weights tier 2.5
+ _effective_checklist）+ **持久 human_hints**（build_loop_context 启动时 load_effective_hints 注入 cfg）。

用法（项目根下）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/resume_loops.py \\
      --bench anthropic_og_style --samples s001 s002 s003 s004 s005 s006 \\
      --tag creative-gemini --rounds 2 --concurrency 6
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def project_root() -> Path:
    p = HERE
    for _ in range(8):
        if (p / "data" / "runs").exists():
            return p
        p = p.parent
    return Path.cwd()


def main() -> int:
    ap = argparse.ArgumentParser(description="续跑已有 loop，追加 N 轮（多 sample 并发）")
    ap.add_argument("--bench", default="anthropic_og_style")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=2, help="每个 loop 追加的轮数")
    ap.add_argument("--tag", default=None, help="loop_id 后缀；不传则取该 sample 最新 loop")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--decision", default="continue", help="每轮 resume 的 decision")
    args = ap.parse_args()

    root = project_root()
    sys.path.insert(0, str(root / "src"))
    from langgraph.types import Command  # noqa: E402

    from img_iter_agent.config import get_settings  # noqa: E402
    from img_iter_agent.data.benchmark import load_benchmark  # noqa: E402
    from img_iter_agent.data.runstore import RunStore  # noqa: E402
    from img_iter_agent.pipeline.runner import build_loop_context, close_checkpointer  # noqa: E402

    settings = get_settings()
    lb = load_benchmark(args.bench, settings=settings)

    def resume_one(sid: str) -> tuple[str, int, int]:
        if args.tag:
            loop_id = f"{args.bench}-{sid}-{args.tag}"
        else:
            cands = sorted(settings.runs_dir.glob(f"{args.bench}-{sid}-*"))
            if not cands:
                print(f"[resume] {sid}: 无已有 loop 目录，跳过", flush=True)
                return (sid, 1, 0)
            loop_id = cands[-1].name
        run_dir = settings.run_dir(loop_id)
        if not run_dir.exists():
            print(f"[resume] {sid}: loop 目录缺失 {loop_id}，跳过", flush=True)
            return (sid, 1, 0)
        print(f"[resume] {loop_id}: 续跑 {args.rounds} 轮", flush=True)
        ctx = None
        try:
            store = RunStore.open(loop_id, settings=settings)
            ctx = build_loop_context(lb, store, sid, loop_model=store.meta.model, settings=settings)
            appended = 0
            for _ in range(args.rounds):
                # 适配线程态：interrupt → Command(resume)；END → 从 START 重入
                try:
                    snap = ctx.app.get_state(ctx.cfg)
                    at_end = tuple(snap.next or ()) == ()
                except Exception:  # noqa: BLE001
                    at_end = False
                if at_end:
                    md = ctx.cfg.get("metadata") or {}
                    inputs = {"model": md.get("model", ""), "bench_id": md.get("bench_id", ""),
                              "sample_id": sid, "run_id": loop_id}
                    ctx.app.invoke(inputs, config=ctx.cfg)
                else:
                    ctx.app.invoke(Command(resume=args.decision), config=ctx.cfg)
                appended += 1
            print(f"[resume] {loop_id}: 完成，追加 {appended} 轮", flush=True)
            return (sid, 0, appended)
        except Exception as e:  # noqa: BLE001
            print(f"[resume] {loop_id}: 失败 {type(e).__name__}: {e}", flush=True)
            return (sid, 1, 0)
        finally:
            if ctx is not None and ctx.checkpointer is not None:
                try:
                    close_checkpointer(ctx.checkpointer)
                except Exception:  # noqa: BLE001
                    pass

    print(f"[resume] 并发度={args.concurrency} | samples={args.samples} | "
          f"追加轮数={args.rounds} | decision={args.decision}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        results = list(ex.map(resume_one, args.samples))

    print(f"\n{'=' * 50}\n[resume] 汇总\n{'=' * 50}")
    for sid, rc, n in results:
        print(f"  {sid}: {'OK' if rc == 0 else 'FAIL'} | 追加 {n} 轮")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
