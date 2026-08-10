#!/usr/bin/env python3
"""跨 loop 跑一轮「创造力对抗调参」（离线，全自动应用 + 留痕）。

读 data/runs/<bench>-* 的 trajectory → 提炼对抗信号 → LLM renovator 翻新创造力子标准 +
有界调权 → 写版本化 overlay（data/benchmarks/<bench>/creativity_criteria.json）。
下一批 loop 启动时 Critic/load_weights 自动读 overlay（批与批之间生效）。

安全：永不改种子 content_spec；权重每轮±0.05、clamp[0.02,0.40]；判别力不足的维度跳过调权；
历史全留可回滚（overlay.history）。

用法（项目根下，跑完一批 loop 之后）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/tune_creativity.py --bench anthropic_og_style

无含创造力维度的 run（n_records=0）时直接退出不写（先跑 batch_loops.py）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def project_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "data" / "runs").exists():
            return p
        p = p.parent
    return Path.cwd()


def main() -> int:
    ap = argparse.ArgumentParser(description="创造力对抗调参（cross-loop，全自动+留痕）")
    ap.add_argument("--bench", default="anthropic_og_style")
    args = ap.parse_args()

    root = project_root()
    if not (root / "data" / "runs").exists():
        print(f"[FATAL] 找不到 data/runs（project_root={root}）", file=sys.stderr)
        return 2

    # 项目内导入：把 src 加进 sys.path（与 run_loop_auto 同策略，依赖 .venv 已装 editable）
    sys.path.insert(0, str(root / "src"))
    from img_iter_agent.calibration.creativity_tuner import tune_creativity  # noqa: E402

    print(f"[tune] bench={args.bench} | 取证所有 data/runs/{args.bench}-* ……")
    summary = tune_creativity(args.bench)

    print(f"\n{'=' * 60}\n[tune] 结果\n{'=' * 60}")
    print(f"  source_runs: {summary.get('source_runs')}")
    print(f"  n_records: {summary.get('n_records')}")
    sig = summary.get("signals", {})
    print(f"  信号: copy_reward_corr={sig.get('copy_reward_corr')} "
          f"noise_as_creative={sig.get('noise_as_creative_count')} "
          f"over_strict={sig.get('over_strict_count')} "
          f"discriminates={sig.get('discriminates')} "
          f"cd_var={sig.get('creative_departure_var')}")

    if not summary.get("acted"):
        print(f"  未应用: {summary.get('note', '无有效 renovation')}")
        return 0

    print(f"  ✓ 已写 overlay v{summary.get('version')}: {summary.get('overlay_path')}")
    print(f"  新创造力权重: {summary.get('new_weights')}")
    print(f"  renovation: {summary.get('renovation_summary')}")
    print(f"  子标准处置数: {summary.get('n_reno_items')}")
    print(f"\n[tune] 下一批 loop 启动时自动生效（Critic/load_weights 读 overlay）。")
    print(f"[tune] 回滚: 删 {summary.get('overlay_path')} 即恢复种子 content_spec。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
