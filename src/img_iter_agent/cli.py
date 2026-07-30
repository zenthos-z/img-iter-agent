"""CLI 入口：跑闭环 A（生成迭代）/ 后续闭环 B（校准）/ 分析。

当前实现闭环 A 的 `run` 子命令：
  img-iter-agent run --bench furniture_product_whitebg --sample s001 --rounds 3

每轮 interrupt 等人工裁决（continue/stop/调方向）。
真实生图与 LLM 评分需 .env 配好 key 与 model_id。
"""

from __future__ import annotations

import argparse
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agents.critic import Critic
from .agents.generator import Generator
from .agents.summarizer import Summarizer
from .config import Settings, get_settings
from .data.benchmark import load_benchmark
from .data.runstore import RunStore
from .generation.client import DmxapiClient
from .generation.router import Router
from .llm import LlmClient
from .pipeline.graph import build_graph


def _make_llm(settings: Settings, *, multimodal: bool) -> LlmClient:
    """按 settings 构造 agent LLM client。

    当前先用 OpenAI 兼容客户端（protocol=openai）。
    TODO: 支持 gemini/doubao/claude 协议族（各族 chat 端点不同）。
    暂用最小实现：调 /v1/chat/completions。
    """
    # 暂返回一个轻量 OpenAI 兼容 client（httpx 直调）。
    # Critic 的多模态图片注入需 client 支持；此处先给文本 capable 的实现。
    return _OpenAiCompatLlm(settings)


class _OpenAiCompatLlm:
    """OpenAI 兼容端点的最小 LLM client（/v1/chat/completions）。

    多模态图片注入：Critic 在 content 里放图片路径时，这里转 data-URI。
    （Step 4 先支持文本；图片注入完整支持留待 Critic 与 client 联调。）
    """

    def __init__(self, settings: Settings, *, model: str | None = None,
                 multimodal: bool = False) -> None:
        import httpx
        self._httpx = httpx
        self.settings = settings
        self.multimodal = multimodal
        # model 由调用方指定（critic/generator/summarizer 各自的字段）
        self._model_override = model

    def complete(self, messages: list[dict]) -> str:
        # 调用方未在 messages 里带 model；用注入的或默认
        model = self._model_override or self.settings.critic_model
        url = f"{self.settings.dmxapi_host.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.dmxapi_key}",
                   "Content-Type": "application/json"}
        body = {"model": model, "messages": messages}
        with self._httpx.Client(timeout=120.0) as c:
            r = c.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    lb = load_benchmark(args.bench, settings=settings)
    store = RunStore.create(args.run_id or f"{args.bench}-{args.sample}",
                            args.bench, model=args.model or settings.model_seedream_pro,
                            settings=settings, note=args.note)

    # 构造 generator/critic/summarizer
    router = Router(settings=settings, client=DmxapiClient(settings))
    gen_llm = _OpenAiCompatLlm(settings, model=settings.generator_model) if settings.generator_model else None
    generator = Generator(router, llm=gen_llm)
    critic = Critic(_OpenAiCompatLlm(settings, model=settings.critic_model), bench=lb.bench)
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
        print(f"\n[round {r}] 还原度={rest:.4f} | 失败项见 lessons")
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
