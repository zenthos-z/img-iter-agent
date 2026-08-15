"""驱动核心：收口「构造 agent 配方 + 开 checkpointer + build_graph + 标准化 config」。

cli.cmd_run / loop_runner._build_app / collect_traces 三处驱动统一调用 build_loop_context，
各自只保留独有逻辑。消灭三处重复的「构造 agent + 开 checkpointer + build_graph」胶水。

- checkpointer：open_checkpointer 显式 setup()（替代裸 SqliteSaver(conn) 的隐式建表），
  由调用方在 loop 终态（finished/error）时 close_checkpointer。
- config：make_loop_config 统一带 metadata（loop_id/bench_id/sample_id/model）+ tags（loop:<id>），
  让同一 loop 多轮 invoke 的 trace 在 LangSmith 里能按 loop_id 聚合。
- tracing：create_round_trace + round_trace_context 把**每一轮** invoke 包成独立 LangSmith trace
  （名字标 bench/sample/loop/round），一个 loop = 多条短 trace，而非一条过长的 loop trace。
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

from ..agent_events import LoopEventEmitter
from ..agents.agent_config_loader import load_agent_model
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
    except Exception:  # noqa: BLE001
        pass


def _round_trace_name(bench_id: str, sample_id: str, loop_id: str, round_n: int) -> str:
    """每轮 trace 的简短清晰名字：``<bench>/<sample>[-<loop 后缀>] R<round>``。

    loop_id 通常 = ``<bench>-<sample>``（一题一 loop）；带后缀时（如 ``-auto``/``-b2``/``-exp6``）
    原样追加，让同 bench/sample 的不同 loop 仍可区分。名字说清四件事：benchmark / sample / loop / 轮次。

      furniture_product_whitebg-s001        R3 → ``furniture_product_whitebg/s001 R3``
      furniture_product_whitebg-s003-exp6   R5 → ``furniture_product_whitebg/s003-exp6 R5``
    """
    base = f"{bench_id}-{sample_id}"
    suffix = loop_id[len(base):] if loop_id.startswith(base) else ""
    return f"{bench_id}/{sample_id}{suffix} R{round_n}"


def _langsmith_tracing_enabled() -> bool:
    """LangSmith 自动 tracing 是否开启（SDK 读 LANGSMITH_TRACING / LANGCHAIN_TRACING_V2）。

    用于 create_round_trace 的短路：未开启时返回 None（round_trace_context(None) 直接 yield），
    与 @traceable 在 tracing 关闭时的 no-op 行为一致——避免未配 LangSmith 时 root.post() 崩溃。
    """
    import os

    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" or \
        os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def create_round_trace(
    loop_id: str, bench_id: str, sample_id: str, round_n: int,
    model: str, phase: str = "round",
) -> object | None:
    """创建**单轮**的 LangSmith trace root（RunTree），并 post() 上报。

    每轮 invoke 各建一个独立 root（各自独立 trace_id）→ 同一 loop 的不同轮次在 LangSmith 里是
    分开的多条 trace（名字标明 bench/sample/loop/round），避免整条 loop trace 过长难观测。
    轮次间靠 extra.metadata.loop_id / tag ``loop:<id>`` 关联，可按 loop 过滤回放。

    供 web loop_runner / run_loop_session 用：调用方以 object 持有返回值（不在 loop_runner 直接
    引用 RunTree，触发 test_tracing 的 AST 守卫），再用 round_trace_context 包住本轮 invoke。
    tracing 未开启时返回 None（整个 round trace 机制静默 no-op）。
    """
    if not _langsmith_tracing_enabled():
        return None
    from langsmith import Client, RunTree

    client = Client()
    root = RunTree(
        name=_round_trace_name(bench_id, sample_id, loop_id, round_n),
        run_type="chain", client=client,
        # RunTree 无独立 metadata 字段：走 extra.metadata（旧版 metadata= 被构造器静默丢弃，从没落库）。
        extra={"metadata": {"loop_id": loop_id, "bench_id": bench_id,
                            "sample_id": sample_id, "model": model,
                            "round": round_n, "phase": phase}},
        tags=[f"loop:{loop_id}", f"round:{round_n}", f"phase:{phase}"],
    )
    root.post()
    return root


@contextlib.contextmanager
def round_trace_context(root: object | None):
    """在本轮 trace root 下嵌套执行，退出时自动收尾（end + patch）。

    每次 invoke 包裹此 context：graph run 嵌套到本轮 root 下；invoke 返回（正常退出）→
    ``root.end(outputs={})`` + ``patch()``；invoke 抛异常 → ``root.end(error=...)`` + ``patch()``
    后原样 raise。这样「一轮 = 一条独立 trace」且每条 trace 都被正确关闭。

    注意：``RunTree.end()`` 只改本地状态，必须再 ``patch()`` 才会落库（旧版仅 end 不 patch，
    trace 会停留在 running 态）。root=None（tracing 未开 / 测试）时直接 yield，不嵌套。
    """
    if root is None:
        yield
        return
    from langsmith import tracing_context

    try:
        with tracing_context(parent=root):  # type: ignore[arg-type]
            yield root
    except Exception as e:
        try:
            root.end(error=str(e))  # type: ignore[attr-defined]
            root.patch()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001  tracing 收尾失败不影响业务异常
            pass
        raise
    else:
        try:
            root.end(outputs={})  # type: ignore[attr-defined]
            root.patch()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def make_loop_config(
    loop_id: str, bench_id: str, sample_id: str, model: str,
    *, callbacks: list | None = None,
) -> dict:
    """构造标准 LangGraph config：thread_id + metadata + tags。

    metadata/tags 让同一 loop 的多轮 invoke 在 LangSmith 里按 loop_id 聚合/过滤。
    callbacks（如 LoopEventEmitter）经 LangGraph ContextVar 自动透传到子 LLM/工具 run，
    覆盖 CLI / 技能脚本 / web 三路径——在 build_loop_context 注入一次即可。
    """
    cfg: dict = {
        "configurable": {"thread_id": loop_id},
        "metadata": {"loop_id": loop_id, "bench_id": bench_id,
                     "sample_id": sample_id, "model": model},
        "tags": [f"loop:{loop_id}"],
    }
    if callbacks:
        cfg["callbacks"] = list(callbacks)
    return cfg


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
    # model 外部化到 data/agents_config/<agent>.json（web 配置页可改）；读不到回退 settings 默认（.env）。
    gen_chat = build_chat_model(
        settings, role="generator",
        model_override=load_agent_model("generator", settings.generator_model),
    )
    generator = Generator(
        router, chat_model=gen_chat,
        skills_dir=generator_skills_source(settings.data_root, lb.bench.bench_id),
    )
    critic_chat = build_chat_model(
        settings, role="critic",
        model_override=load_agent_model("critic", settings.critic_model),
    )
    critic = Critic(critic_chat, bench=lb.bench)

    checkpointer = open_checkpointer(store.run_dir) if persist else InMemorySaver()
    app = build_graph(
        bench=lb, run_store=store, generator=generator, critic=critic,
        sample_id=sample_id,
        checkpointer=checkpointer, loop_model=loop_model,
    )
    model = store.meta.model if store.meta else (loop_model or "")
    # 活动流事件采集：emitter 写 events.jsonl，callbacks 经 ContextVar 透传到 deepagent 内部
    # 的每次 LLM/工具调用。CLI/脚本/web 三路径都过 build_loop_context，故此处注入即全覆盖；
    # web 的 _cfg_with_round 浅拷贝保留 callbacks 键，跨轮复用同一 emitter（同一 events.jsonl）。
    emitter = LoopEventEmitter(store.run_dir)
    cfg = make_loop_config(
        store.run_dir.name, lb.bench.bench_id, sample_id, model, callbacks=[emitter],
    )
    # 启动时加载持久化人工提示词（sample 文件 + loop meta）注入 config——让 CLI / run_loop_auto
    # 路径也带上持久 hints；web runner 的 _cfg_with_round 会用 handle.hints 覆盖此快照（运行中实时改）。
    cfg["configurable"]["human_hints"] = load_effective_hints(
        settings.data_root, store, lb.bench.bench_id, sample_id
    )
    return LoopContext(app, cfg, checkpointer if persist else None)


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
    """跑整个闭环 loop（首轮 + rounds 次 resume），**每一轮是一条独立 LangSmith trace**。

    每个 app.invoke 各建一个 round trace root（名字标 bench/sample/loop/round）并嵌套其下；
    invoke 返回即由 round_trace_context 收尾该轮 trace。同一 loop 的多轮在 LangSmith 里是分开的
    多条短 trace，按 metadata.loop_id / tag ``loop:<id>`` 关联，便于按单轮观测（而非一条过长的 loop trace）。
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
        # 首轮 = 第 1 轮（inputs round:0 → generator 自增到 1）
        with round_trace_context(
            create_round_trace(loop_id, bench_id, sample_id, 1, fixed_model, "first")
        ):
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
            # 本轮 resume 产出第 r+1 轮 → trace 标 R{r+1}
            with round_trace_context(
                create_round_trace(loop_id, bench_id, sample_id, r + 1, fixed_model, "resume")
            ):
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
    "create_round_trace",
    "make_loop_config",
    "open_checkpointer",
    "round_trace_context",
    "run_loop_session",
]
