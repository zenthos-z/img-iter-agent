#!/usr/bin/env python3
"""新起一个 loop 并自动跑 N 轮（无人值守），作为一条 LangSmith trace。

为什么单独写这个脚本（而不是用 CLI / web）：
  - CLI 的 `img-iter run` 每轮阻塞等 stdin（prompt_decision=input），无法无人值守批量跑。
  - web 的 LoopRunner 用固定 loop_id=<bench>-<sample>，对已有 sample 会续跑、污染正式数据。
  - 本脚本用「全新 loop_id + prompt_decision=None 自动 continue」实现干净、可复现的测试运行。

屏蔽底层瑕疵：run_loop_session(rounds=R) 实际生成 R+1 轮（首轮 invoke + R 次 resume）且最后一轮
不 print。本脚本接收「想要几轮」，内部换算成 resume 次数，并跑完校验 trajectory 行数 == 目标。

用法（项目根下）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/run_loop_auto.py \\
      --bench furniture_product_whitebg --sample s003 --rounds 6 --tag exp6
  .venv/bin/python .../run_loop_auto.py --bench B --sample s003 --rounds 4 --loop-id 自定义id

产出：data/runs/<loop_id>/{trajectory.jsonl, lessons/conclusions.json, out/a00N/}
跑完会用 diagnose_loop.py 诊断（脚本末尾会提示命令）。
"""
from __future__ import annotations

import argparse
import json
import sys

from img_iter_agent.config import get_settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.pipeline.runner import (
    build_loop_context,
    close_checkpointer,
    run_loop_session,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="新起 loop 自动跑 N 轮（无人值守，防污染）")
    ap.add_argument("--bench", default="furniture_product_whitebg")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--rounds", type=int, default=6,
                    help="想要生成的总轮数；脚本保证恰好生成这么多（默认 6）")
    ap.add_argument("--tag", default=None,
                    help="loop_id 后缀，生成 <bench>-<sample>-<tag>（不传则用 -auto）")
    ap.add_argument("--loop-id", default=None, help="显式 loop_id（覆盖 --tag）")
    ap.add_argument("--note", default=None, help="写入 meta.json 的备注")
    ap.add_argument("--model", default=None,
                    help="生图 model_id（默认 settings.model_seedream_pro）；如 gemini-3.1-flash-image / gpt-image-2-03 / qwen-image-2.0-pro")
    args = ap.parse_args()

    settings = get_settings()
    lb = load_benchmark(args.bench, settings=settings)

    loop_id = args.loop_id or f"{args.bench}-{args.sample}-{args.tag or 'auto'}"
    run_dir = settings.run_dir(loop_id)
    if run_dir.exists():
        print(
            f"[FATAL] loop 已存在: {loop_id}（{run_dir}）。\n"
            f"  · 续跑请走 web UI(8765) 的 resume；\n"
            f"  · 新起测试请换 --tag 或 --loop-id，避免污染。",
            file=sys.stderr,
        )
        return 2

    store = RunStore.create(
        loop_id, args.bench,
        model=args.model or settings.model_seedream_pro,
        settings=settings,
        note=args.note or f"img-iter-ops auto run, target {args.rounds} rounds",
    )
    print(f"[run] 新建 loop: {loop_id}")
    print(f"[run] sample={args.sample} | 生图模型={store.meta.model} | 目标轮数={args.rounds}")
    print(f"[run] agent: generator={settings.generator_model} | "
          f"critic={settings.critic_model} | summarizer={settings.summarizer_model}")
    print(f"[run] 开始跑（无人值守，自动 continue），预计 {2*args.rounds}-{4*args.rounds} 分钟……\n")

    # run_loop_session(rounds=R) 实际生成 R+1 轮（首轮 invoke + R 次 resume）。
    # 用户要 N 轮 → resume 次数 = N-1。N=1 → resume=0（只首轮）。
    session_rounds = max(0, args.rounds - 1)
    ctx = build_loop_context(lb, store, args.sample, loop_model=store.meta.model)
    try:
        run_loop_session(
            ctx.app, ctx.cfg, store,
            rounds=session_rounds,
            bench_id=args.bench, sample_id=args.sample,
            prompt_decision=None,  # None = 自动 continue（无人值守）
        )
    finally:
        close_checkpointer(ctx.checkpointer)

    # 跑完校验 + 打印完整还原度曲线（底层最后一轮不 print，这里从 trajectory 补全）
    traj = store.trajectory_path
    recs = [json.loads(l) for l in traj.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n[run] 完成，实际 {len(recs)} 轮（目标 {args.rounds}）")
    if len(recs) != args.rounds:
        print(f"[WARN] 轮数不符！期望 {args.rounds} 实得 {len(recs)}（可能首轮/末轮边界）", file=sys.stderr)
    for r in recs:
        v = r.get("verdict") or {}
        rest = v.get("restoration")
        print(f"  R{r['round']}: 还原度={rest:.4f}" if rest is not None else f"  R{r['round']}: 还原度=N/A")
    print(f"[run] trajectory : {traj}")
    print(f"[run] conclusions: {store.run_dir}/lessons/conclusions.json")
    here = sys.argv[0]
    print(f"[run] 下一步诊断: .venv/bin/python {here}  # 即 diagnose_loop.py {loop_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
