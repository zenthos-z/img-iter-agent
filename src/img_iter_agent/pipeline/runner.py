"""驱动核心：收口「构造 agent 配方 + 开 checkpointer + build_graph + 标准化 config」。

cli.cmd_run / loop_runner._build_app / collect_traces 三处驱动统一调用 build_loop_context，
各自只保留独有逻辑。消灭三处重复的「构造 agent + 开 checkpointer + build_graph」胶水。

- checkpointer：open_checkpointer 显式 setup()（替代裸 SqliteSaver(conn) 的隐式建表），
  由调用方在 loop 终态（finished/error）时 close_checkpointer。
- config：make_loop_config 统一带 metadata（loop_id/bench_id/sample_id/model）+ tags（loop:<id>），
  让同一 loop 多轮 invoke 的 trace 在 LangSmith 里能按 loop_id 聚合。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from ..agents.critic import Critic
from ..agents.generator import Generator
from ..agents.summarizer import Summarizer
from ..config import Settings, get_settings
from ..data.benchmark import LoadedBenchmark
from ..data.runstore import RunStore
from ..generation.client import DmxapiClient
from ..generation.router import Router
from .graph import CompiledGraph, build_graph


@dataclass
class LoopContext:
    """一个 loop 的运行上下文：编译好的图 + 标准 config + 持有的 checkpointer。"""

    app: CompiledGraph
    cfg: dict
    checkpointer: object | None  # 持有以便 close；InMemorySaver（persist=False）时为 None


def open_checkpointer(run_dir: Path) -> SqliteSaver:
    """开 SqliteSaver 并显式 setup()（建 checkpoints/writes 表）。

    替代裸 `SqliteSaver(conn)` 的隐式建表行为。返回的 saver 暴露 .conn，
    由调用方在 loop 终态时 close_checkpointer(saver) 关连接。
    """
    conn = sqlite3.connect(run_dir / "checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def close_checkpointer(saver: object | None) -> None:
    """关闭 checkpointer 的 sqlite 连接（幂等，容忍 None/已关）。"""
    if saver is None:
        return
    try:
        saver.conn.close()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001, S110
        pass


def make_loop_config(loop_id: str, bench_id: str, sample_id: str, model: str) -> dict:
    """构造标准 LangGraph config：thread_id + metadata + tags。

    metadata/tags 让同一 loop 的多轮 invoke 在 LangSmith 里按 loop_id 聚合/过滤。
    """
    return {
        "configurable": {"thread_id": loop_id},
        "metadata": {"loop_id": loop_id, "bench_id": bench_id,
                     "sample_id": sample_id, "model": model},
        "tags": [f"loop:{loop_id}"],
    }


def build_loop_context(
    lb: LoadedBenchmark,
    store: RunStore,
    sample_id: str,
    *,
    loop_model: str | None = None,
    persist: bool = True,
    settings: Settings | None = None,
) -> LoopContext:
    """构造 loop 运行上下文：agent 配方 + checkpointer + build_graph + 标准 config。

    - persist=True（默认，生产）：SqliteSaver 持久化（可断点续跑 + get_state 可查状态）。
    - persist=False（批量/测试）：InMemorySaver（不落盘）。
    - loop_model：启动时指定的生图 model_id，反查成 ModelFamily 作为 model_hint 强制路由。
    """
    from ..llm.openai_compat import OpenAiCompatLlm

    settings = settings or get_settings()
    router = Router(settings=settings, client=DmxapiClient(settings))
    gen_llm = OpenAiCompatLlm(settings, model=settings.generator_model) if settings.generator_model else None
    generator = Generator(router, llm=gen_llm)
    critic = Critic(OpenAiCompatLlm(settings, model=settings.critic_model), bench=lb.bench)
    summarizer = Summarizer()

    checkpointer = open_checkpointer(store.run_dir) if persist else InMemorySaver()
    app = build_graph(
        bench=lb, run_store=store, generator=generator, critic=critic,
        summarizer=summarizer, sample_id=sample_id,
        checkpointer=checkpointer, loop_model=loop_model,
    )
    model = store.meta.model if store.meta else (loop_model or "")
    cfg = make_loop_config(store.run_dir.name, lb.bench.bench_id, sample_id, model)
    return LoopContext(app, cfg, checkpointer if persist else None)


__all__ = [
    "LoopContext",
    "build_loop_context",
    "close_checkpointer",
    "make_loop_config",
    "open_checkpointer",
]
