#!/usr/bin/env python3
"""批量跑多个 sample 的 loop（串行，API 限流安全），每个全新 loop_id 不污染。

为什么串行而非并发：dmxapi 生图 + 多模态 Critic 打分每轮 1-3 分钟，并发易触发限流/网关抖动；
串行稳定，每个 loop 独立进程隔离崩溃（一个挂了不影响下一个），且各自一条 LangSmith trace 便于对比。
每个 sample 用 <bench>-<sample>-<tag> 全新 loop_id，绝不续跑默认 id（一题一 loop 语义）。

用法（项目根下）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/batch_loops.py \\
      --bench furniture_product_whitebg --samples s001 s002 s003 --rounds 6 --tag batch1

跑完汇总每个 loop 的还原度曲线（首/末/峰），并提示用 diagnose_loop.py 逐个诊断。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_AUTO = HERE / "run_loop_auto.py"


def project_root() -> Path:
    p = HERE
    for _ in range(8):
        if (p / "data" / "runs").exists():
            return p
        p = p.parent
    return Path.cwd()


def read_rest_series(run_dir: Path) -> list[float | None]:
    traj = run_dir / "trajectory.jsonl"
    if not traj.exists():
        return []
    out: list[float | None] = []
    for line in traj.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = (json.loads(line).get("verdict") or {}).get("restoration")
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="批量跑多 sample loop（串行，防污染）")
    ap.add_argument("--bench", default="furniture_product_whitebg")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tag", default="batch")
    ap.add_argument("--python", default=".venv/bin/python",
                    help="python 解释器（默认项目根 .venv/bin/python）")
    args = ap.parse_args()

    root = project_root()
    py = args.python if Path(args.python).is_absolute() else str(root / args.python)
    results: list[tuple[str, int, list[float | None]]] = []
    for s in args.samples:
        loop_id = f"{args.bench}-{s}-{args.tag}"
        print(f"\n{'='*60}\n[batch] 开始 {loop_id}（{args.rounds} 轮）\n{'='*60}")
        rc = subprocess.run(
            [py, str(RUN_AUTO), "--bench", args.bench, "--sample", s,
             "--rounds", str(args.rounds), "--tag", args.tag],
            cwd=str(root),
        ).returncode
        rest = read_rest_series(root / "data" / "runs" / loop_id)
        results.append((loop_id, rc, rest))

    print(f"\n{'='*60}\n[batch] 汇总\n{'='*60}")
    for loop_id, rc, rest in results:
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        valid = [x for x in rest if x is not None]
        first = f"{rest[0]:.3f}" if rest and rest[0] is not None else "?"
        last = f"{rest[-1]:.3f}" if rest and rest[-1] is not None else "?"
        peak = f"{max(valid):.3f}" if valid else "?"
        print(f"  {loop_id}: {status} | {len(rest)} 轮 | {first}→{last}（峰 {peak}）")
    print(f"\n[batch] 逐个诊断：{py} {HERE}/diagnose_loop.py <loop_id>")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
