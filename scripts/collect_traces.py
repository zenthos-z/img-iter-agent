"""批量攒 trace：seedream × 多样本 × 每样本 2 轮（首轮基线 + 1 轮控制变量）。

非交互式（用 run_loop 直接吃 decisions=["continue","stop"]），适合后台批量跑。
真实生图 + 真实 Gemini 评分，需 .env 配好 key 与 model_id。

用法：
    .venv/bin/python scripts/collect_traces.py
可选参数：--samples s001 s002 s003  --rounds 2  --model <model_id>

产出：每个样本一个 run 目录，含 trajectory.jsonl + 经验 MD + 三视图。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.agents.summarizer import Summarizer
from img_iter_agent.llm.chat_model import build_chat_model
from img_iter_agent.config import get_settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.generation.client import DmxapiClient
from img_iter_agent.generation.router import Router
from img_iter_agent.pipeline.graph import run_loop
from img_iter_agent.pipeline.runner import _skills_dir

BENCH = "furniture_product_whitebg"


def run_one_sample(sample_id: str, rounds: int, model: str) -> dict:
    """跑一个样本 N 轮，返回该 run 的还原度序列。"""
    settings = get_settings()
    lb = load_benchmark(BENCH, settings=settings)
    ts = time.strftime("%m%d-%H%M%S")
    run_id = f"collect-{sample_id}-{ts}"
    store = RunStore.create(run_id, BENCH, model=model, settings=settings,
                            note=f"批量攒trace {sample_id}")

    router = Router(settings=settings, client=DmxapiClient(settings))
    # deepagent 引擎：chat_model 用 ChatOpenAI 指向 dmxapi（build_chat_model 按 role 取 settings 的 model_id）
    generator = Generator(
        router, chat_model=build_chat_model(settings, role="generator"),
        skills_dir=_skills_dir("generator"),
        data_root=settings.data_root, bench_id=lb.bench.bench_id,
    )
    critic = Critic(
        build_chat_model(settings, role="critic"),
        bench=lb.bench, skills_dir=_skills_dir("critic"),
    )
    summarizer = Summarizer()

    # decisions: 第1轮跑到 interrupt 后 continue(进第2轮), 第2轮后 stop
    decisions = ["continue"] * (rounds - 1) + ["stop"]

    print(f"\n{'='*60}\n[collect] {sample_id} | model={model} | rounds={rounds}\n{'='*60}")
    t0 = time.time()
    run_loop(
        bench=lb, run_store=store, sample_id=sample_id,
        generator=generator, critic=critic, summarizer=summarizer,
        decisions=decisions, thread_id=run_id,
    )
    elapsed = time.time() - t0

    # 汇总每轮还原度
    from img_iter_agent.data.trajectory import TrajectoryReader
    recs = TrajectoryReader(store.trajectory_path).read_all()
    scores = [(r.round, r.verdict.restoration if r.verdict else None) for r in recs]
    store.finish(note=f"{sample_id} 跑完 {len(recs)} 轮")

    print(f"[collect] {sample_id} 完成 ({elapsed:.0f}s) | run_id={run_id}")
    for rnd, rest in scores:
        print(f"    round {rnd}: 还原度={rest:.4f}" if rest is not None
              else f"    round {rnd}: 无评分")

    return {"sample_id": sample_id, "run_id": run_id, "scores": scores,
            "trajectory_path": str(store.trajectory_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量攒 trace")
    parser.add_argument("--samples", nargs="+", default=["s001", "s002", "s003"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--model", default=None, help="生图 model_id（默认用 seedream）")
    args = parser.parse_args()

    settings = get_settings()
    model = args.model or settings.model_seedream_pro
    if not model:
        print("错误：未配置生图 model_id（检查 .env 的 model_seedream_pro）", file=sys.stderr)
        return 1

    print(f"[collect] 计划：{args.samples} × {args.rounds}轮 | model={model}")
    all_results = []
    for sid in args.samples:
        try:
            res = run_one_sample(sid, args.rounds, model)
            all_results.append(res)
        except Exception as e:  # noqa: BLE001
            print(f"[collect] {sid} 失败: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n{'='*60}\n[collect] 全部完成。汇总：\n{'='*60}")
    for r in all_results:
        scores_str = " → ".join(f"{rest:.3f}" if rest else "?" for _, rest in r["scores"])
        print(f"  {r['sample_id']}: {scores_str}  ({r['run_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
