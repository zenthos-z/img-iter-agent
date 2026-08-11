"""驱动核心：收口「构造 agent 配方 + 开 checkpointer + build_graph + 标准化 config」。

cli.cmd_run / loop_runner._build_app / collect_traces 三处驱动统一调用 build_loop_context，
各自只保留独有逻辑。消灭三处重复的「构造 agent + 开 checkpointer + build_graph」胶水。

- checkpointer：open_checkpointer 显式 setup()（替代裸 SqliteSaver(conn) 的隐式建表），
  由调用方在 loop 终态（finished/error）时 close_checkpointer。
- config：make_loop_config 统一带 metadata（loop_id/bench_id/sample_id/model）+ tags（loop:<id>），
  让同一 loop 多轮 invoke 的 trace 在 LangSmith 里能按 loop_id 聚合。
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langsmith import traceable

from ..agents.critic import Critic
from ..agents.generator import Generator
from ..config import Settings, get_settings
from ..data.benchmark import LoadedBenchmark
from ..data.human_hints import load_effective_hints
from ..data.runstore import RunStore
from ..generation.client import DmxapiClient
from ..generation.router import Router
from ..llm.chat_model import build_chat_model
from ..memory.experience import generator_skills_source
from ..memory.schema import CriticVerdict
from .graph import CompiledGraph, build_graph


@dataclass
class LoopContext:
    """一个 loop 的运行上下文：编译好的图 + 标准 config + 持有的 checkpointer。"""

    app: CompiledGraph
    cfg: dict
    checkpointer: object | None  # 持有以便 close；InMemorySaver（persist=False）时为 None


def open_checkpointer(run_dir: Path) -> SqliteSaver:
    """开 SqliteSaver 并显式 setup()（建 checkpoints/writes 表）+ 注册 state 自定义类型。

    替代裸 `SqliteSaver(conn)` 的隐式建表行为。把 state 里的自定义类型
    （CriticVerdict/AttemptRecord/GenOutcome）注册到 msgpack allowlist，消除
    "Deserializing unregistered type ... This will be blocked in a future version"
    warning（未来 strict msgpack 默认开启会直接 block）。返回的 saver 暴露 .conn，
    由调用方在 loop 终态时 close_checkpointer(saver) 关连接。
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from ..agents.generator import GenOutcome
    from ..memory.schema import AttemptRecord, CriticVerdict

    conn = sqlite3.connect(run_dir / "checkpoints.sqlite", check_same_thread=False)
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[CriticVerdict, AttemptRecord, GenOutcome]
    )
    saver = SqliteSaver(conn, serde=serde)
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


def create_loop_trace(loop_id: str, bench_id: str, sample_id: str, model: str) -> object:
    """创建 loop 顶层 RunTree（LangSmith trace root），跨多轮 invoke/resume 持久。

    供 web loop_runner 用：web 跨 HTTP 请求，无法像 cli 的 run_loop_session 那样在一个
    @traceable 调用里包住整个 loop，所以手动建一个 loop root，每次 invoke 嵌套其下。
    返回 RunTree，但调用方以 object 持有（避免在 loop_runner 直接引用 RunTree，触发 test_tracing 守卫）。
    """
    from langsmith import Client, RunTree

    client = Client()
    root = RunTree(
        name="loop", run_type="chain", client=client,
        metadata={"loop_id": loop_id, "bench_id": bench_id,
                  "sample_id": sample_id, "model": model},
    )
    root.post()
    return root


@contextlib.contextmanager
def loop_trace_context(root: object | None):
    """在 loop root 下嵌套执行：web 每次 invoke 包裹此 context，使其 graph run 嵌套到 loop trace。"""
    if root is None:  # tracing 未开或测试时不嵌套
        yield
        return
    from langsmith import tracing_context

    with tracing_context(parent=root):  # type: ignore[arg-type]
        yield


def end_loop_trace(root: object | None, *, error: bool = False, error_msg: str | None = None) -> None:
    """结束 loop trace（loop 终态 finished/error 时调用）。幂等，容忍 None。"""
    if root is None:
        return
    try:
        if error:
            root.end(error=error_msg)  # type: ignore[attr-defined]
        else:
            root.end(outputs={})  # type: ignore[attr-defined]
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
    settings = settings or get_settings()
    router = Router(settings=settings, client=DmxapiClient(settings))
    # Generator/Critic：deepagent 引擎（tool-using agent）。chat_model 指向 dmxapi OpenAI 兼容端点。
    gen_chat = build_chat_model(settings, role="generator")
    generator = Generator(
        router, chat_model=gen_chat,
        skills_dir=generator_skills_source(settings.data_root, lb.bench.bench_id),
        data_root=settings.data_root, bench_id=lb.bench.bench_id,
    )
    critic_chat = build_chat_model(settings, role="critic")
    critic = Critic(critic_chat, bench=lb.bench)

    checkpointer = open_checkpointer(store.run_dir) if persist else InMemorySaver()
    app = build_graph(
        bench=lb, run_store=store, generator=generator, critic=critic,
        sample_id=sample_id,
        checkpointer=checkpointer, loop_model=loop_model,
    )
    model = store.meta.model if store.meta else (loop_model or "")
    cfg = make_loop_config(store.run_dir.name, lb.bench.bench_id, sample_id, model)
    # 启动时加载持久化人工提示词（sample 文件 + loop meta）注入 config——让 CLI / run_loop_auto
    # 路径也带上持久 hints；web runner 的 _cfg_with_round 会用 handle.hints 覆盖此快照（运行中实时改）。
    cfg["configurable"]["human_hints"] = load_effective_hints(
        settings.data_root, store, lb.bench.bench_id, sample_id
    )
    return LoopContext(app, cfg, checkpointer if persist else None)


@traceable(name="loop", run_type="chain")
def run_loop_session(
    app: CompiledGraph,
    cfg: dict,
    store: RunStore,
    *,
    rounds: int,
    bench_id: str,
    sample_id: str,
    prompt_decision: Callable[[int, CriticVerdict | None], str] | None = None,
) -> int:
    """跑整个闭环 loop（首轮 + rounds 次 resume）作为**一条** LangSmith trace。

    所有 app.invoke 嵌套在本函数的 loop run 之下（Python≥3.11 的 contextvar 继承），
    实现「1 loop = 1 trace」——而非 LangGraph 默认的每轮 invoke 一条 trace。
    prompt_decision(round, verdict)→decision：CLI 传交互 input；None=自动 continue（脚本/批量）。
    """
    assert store.meta is not None
    fixed_model = store.meta.model
    loop_id = store.run_dir.name
    # 跨进程「正在跑」标记：CLI/批量脚本起 loop 时在另一进程，web 内存 LoopRunner
    # 不知道它；写 running.pid 让 web「运行中」页能识别（finally 必清，SIGKILL 由
    # run_is_alive 的存活探测兜底）。
    store.mark_running()
    try:
        print(f"[run] {bench_id}/{sample_id} | model={fixed_model} | run_id={store.meta.run_id}")
        state: dict = app.invoke(
            {"round": 0, "model": fixed_model, "bench_id": bench_id,
             "sample_id": sample_id, "run_id": loop_id},
            config=cfg,
        )
        round_done = 0
        for i in range(rounds):
            verdict = state.get("_verdict")
            r = state.get("round", 0)
            rest = verdict.restoration if verdict else None
            print(f"\n[round {r}] 还原度={rest:.4f} | 经验见 lessons/conclusions.json")
            print("  回复 continue 继续下一轮 / stop 停止 / 或输入调整方向:")
            if prompt_decision is not None:
                try:
                    decision = prompt_decision(r, verdict).strip() or "continue"
                except EOFError:
                    decision = "stop"
            else:
                decision = "continue"
            state = app.invoke(Command(resume=decision), config=cfg)
            round_done = i + 1
            if state.get("decision") == "stop":
                print("[run] 已停止。")
                break
        store.finish(note=f"跑完 {round_done} 轮")
        print(f"[run] 完成。trajectory: {store.trajectory_path}")
        return 0
    finally:
        store.clear_running()


__all__ = [
    "LoopContext",
    "build_loop_context",
    "close_checkpointer",
    "create_loop_trace",
    "end_loop_trace",
    "loop_trace_context",
    "make_loop_config",
    "open_checkpointer",
    "run_loop_session",
]
