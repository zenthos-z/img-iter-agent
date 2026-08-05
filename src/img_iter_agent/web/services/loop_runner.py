"""loop 远程控制服务：事件驱动的「跑一轮」执行模型。

不常驻阻塞进程。每次 start/resume 在后台线程里跑一轮 invoke，
跑到 human_review 的 interrupt 就停（状态=awaiting_review），等下次 resume。
状态存内存，可被 status 接口查询。

每个 loop 强制用 SqliteSaver（保证 graph.get_state 可查状态 + 可断点续跑）。
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from langgraph.types import Command

from ...config import get_settings
from ...data.benchmark import load_benchmark
from ...data.runstore import RunStore
from ...pipeline.runner import (
    build_loop_context,
    close_checkpointer,
    create_loop_trace,
    end_loop_trace,
    loop_trace_context,
)


@dataclass
class LoopHandle:
    """一个 loop 的运行态。"""

    loop_id: str
    phase: str = "idle"  # idle / running / awaiting_review / finished / error
    round: int | None = None
    interrupt_payload: dict | None = None
    last_error: str | None = None
    rounds_remaining: int = 0  # 自动连跑剩余轮数；0=不自动连跑（首停审批）
    auto_mode: bool = False  # True=自动连跑模式（跑满后自动 stop 结束，不停等审批）
    future: Future | None = field(default=None, repr=False)
    app: object | None = field(default=None, repr=False)
    cfg: dict | None = field(default=None, repr=False)
    store: RunStore | None = field(default=None, repr=False)
    loop_root: object | None = field(default=None, repr=False)  # LangSmith loop trace root（web 跨请求 1-loop-1-trace）


class LoopRunner:
    """事件驱动的 loop 执行器。max_workers=1 保证同 loop 不并发 invoke。"""

    def __init__(self) -> None:
        self._handles: dict[str, LoopHandle] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)

    def get(self, loop_id: str) -> LoopHandle | None:
        with self._lock:
            return self._handles.get(loop_id)

    # --- 启动（一题一条：同 sample 已有 loop 则续跑，否则新建）---
    def start(
        self,
        *,
        bench_id: str,
        sample_id: str,
        model: str | None = None,
        note: str | None = None,
        rounds: int | None = None,
    ) -> str:
        """对一道题（sample×model）启动/继续 loop。返回 loop_id。

        一题一条语义：先 find_loop，找到则续跑（复用 run_dir+checkpoint），
        无则新建。loop_id = `<bench>-<sample>`（不带时间戳；时间记 meta.started_at）。

        rounds：自动连跑轮数。None/1=首轮跑到等审批就停（等人工 resume）；
        >1=后台连跑，每轮到 interrupt 自动 resume，跑满后置 finished。
        """
        settings = get_settings()
        lb = load_benchmark(bench_id, settings=settings)
        loop_id = f"{bench_id}-{sample_id}"
        # 自动连跑：rounds>1 时启用，首轮跑完后还剩 rounds-1 轮要 resume
        auto_mode = bool(rounds and rounds > 1)
        rounds_remaining = max(0, (rounds or 1) - 1) if auto_mode else 0

        existing = self._find_existing_loop(settings, loop_id)
        if existing:
            # 已有 loop：续跑下一轮（不新建）
            store = RunStore.open(loop_id, settings=settings)
            return self._continue_existing(
                loop_id, store, lb, sample_id, rounds_remaining, auto_mode
            )

        # 新建 loop
        RunStore.create(
            loop_id, bench_id,
            model=model or settings.model_seedream_pro or "unknown",
            settings=settings, note=note,
        )
        store = RunStore.open(loop_id, settings=settings)
        handle = LoopHandle(
            loop_id=loop_id, phase="running",
            rounds_remaining=rounds_remaining, auto_mode=auto_mode,
        )
        with self._lock:
            self._handles[loop_id] = handle
        self._submit(handle, self._run_first_round(settings, lb, store, sample_id))
        return loop_id

    def _find_existing_loop(self, settings, loop_id: str) -> bool:
        """该 loop_id 是否已有 run 目录（一题一条：有则续跑）。"""
        run_dir = settings.run_dir(loop_id)
        return run_dir.exists() and (run_dir / "trajectory.jsonl").exists()

    def _continue_existing(
        self, loop_id: str, store: RunStore, lb, sample_id: str,
        rounds_remaining: int = 0, auto_mode: bool = False,
    ) -> str:
        """在已有 loop 上续跑：跑下一轮（resume）到下一个 interrupt。"""
        settings = get_settings()
        handle = LoopHandle(
            loop_id=loop_id, phase="running",
            rounds_remaining=rounds_remaining, auto_mode=auto_mode,
        )
        with self._lock:
            # 复用或新建 handle
            old = self._handles.get(loop_id)
            if old and old.app is not None:
                handle = old  # 已有 graph 实例，直接 resume
                handle.phase = "running"
                handle.rounds_remaining = rounds_remaining
                handle.auto_mode = auto_mode
                self._submit(handle, self._run_resume(handle, "continue"))
                return loop_id
            self._handles[loop_id] = handle
        # 无 graph 实例（进程重启等）：重建 graph + resume 续跑
        self._submit(handle, self._run_first_round(settings, lb, store, sample_id, resume_existing=True))
        return loop_id

    # --- 继续 / 停止 ---
    def resume(self, loop_id: str, decision: str) -> bool:
        """用一个 decision 续跑一轮（到下个 interrupt 或 END）。返回是否已提交。

        一题一 loop：awaiting_review（等审批）或 finished（已结束想再加一轮）都可 resume。
        """
        handle = self.get(loop_id)
        if handle is None:
            return False
        if handle.phase not in ("awaiting_review", "finished"):
            return False
        handle.phase = "running"
        handle.interrupt_payload = None
        self._submit(handle, self._run_resume(handle, decision))
        return True

    # --- 实际执行任务（在线程池里跑）---
    def _run_first_round(self, settings, lb, store, sample_id, *, resume_existing: bool = False):
        loop_id = store.run_dir.name

        def task() -> None:
            handle = self.get(loop_id)
            try:
                app, cfg = self._build_app(settings, lb, store, sample_id)
                handle.app = app
                handle.cfg = cfg
                handle.store = store  # 供 _post_invoke 在结束时调 store.finish()
                # 建 loop trace root（web 跨请求 1-loop-1-trace）：所有轮 invoke 嵌套其下
                handle.loop_root = create_loop_trace(
                    loop_id, lb.bench.bench_id, sample_id,
                    store.meta.model if store.meta else "",
                )
                inputs = {
                    "round": 0,
                    "model": store.meta.model if store.meta else "",
                    "bench_id": lb.bench.bench_id,
                    "sample_id": sample_id,
                    "run_id": loop_id,
                }
                # 每次 invoke 包在 loop_trace_context 下 → graph run 嵌套到 loop trace root，
                # 实现一个 loop（跨多轮 invoke/resume）一条 trace。round/phase 在 config metadata 辨认轮次。
                with loop_trace_context(handle.loop_root):
                    if resume_existing:
                        # 已有 loop：用 checkpoint 续跑下一轮（不重跑首轮）
                        state = app.invoke(
                            Command(resume="continue"),
                            config=self._cfg_with_round(handle, (handle.round or 0) + 1, "resume"),
                        )
                    else:
                        state = app.invoke(inputs, config=self._cfg_with_round(handle, 1, "first"))
                self._post_invoke(handle, state)
            except Exception as e:  # noqa: BLE001
                self._fail(handle, f"首轮失败: {e}\n{traceback.format_exc()}")
        return task

    def _run_resume(self, handle: LoopHandle, decision: str):
        def task() -> None:
            try:
                with loop_trace_context(handle.loop_root):
                    state = handle.app.invoke(
                        Command(resume=decision),
                        config=self._cfg_with_round(handle, (handle.round or 0) + 1, "resume"),
                    )
                self._post_invoke(handle, state)
            except Exception as e:  # noqa: BLE001
                self._fail(handle, f"resume 失败: {e}\n{traceback.format_exc()}")
        return task

    def _build_app(self, settings, lb, store, sample_id):
        """构造图 + checkpointer（SqliteSaver 强制，显式 setup）+ 标准 config（带 metadata/tags）。

        收口到 pipeline.runner.build_loop_context（agent 配方 + open_checkpointer + build_graph
        + make_loop_config）。返回 (app, cfg) 保持不变，供 _run_first_round 解包；checkpointer 挂在
        app.checkpointer 上，由 _post_invoke/_fail 在 finished/error 时关闭。
        """
        ctx = build_loop_context(
            lb, store, sample_id,
            loop_model=store.meta.model if store.meta else None,
        )
        return ctx.app, ctx.cfg

    @staticmethod
    def _cfg_with_round(handle: LoopHandle, round_n: int, phase: str) -> dict:
        """在标准 config 上叠加 round/phase，让 LangSmith 里每一轮 trace 可辨认（按 round/phase 过滤、列表加 metadata/tags 列）。"""
        base = handle.cfg or {"configurable": {"thread_id": handle.loop_id}}
        cfg = dict(base)
        cfg["metadata"] = {**(base.get("metadata") or {}), "round": round_n, "phase": phase}
        cfg["tags"] = list(base.get("tags") or []) + [f"round:{round_n}", f"phase:{phase}"]
        return cfg

    def _post_invoke(self, handle: LoopHandle, state: dict) -> None:
        """invoke 返回后判定 phase：到 END=finished，到 interrupt=awaiting_review。"""
        # 用 graph.get_state 拿权威状态
        try:
            snapshot = handle.app.get_state(handle.cfg)
            round_now = (snapshot.values or {}).get("round")
            handle.round = round_now
            # interrupts 非空 = 卡在 human_review 等审批
            interrupts = getattr(snapshot, "interrupts", ()) or ()
            if interrupts:
                handle.interrupt_payload = (
                    interrupts[0].value if interrupts else None
                )
                # 自动连跑：还有剩余轮数则不等人工审批，直接 resume 进下一轮
                if handle.rounds_remaining > 0:
                    handle.rounds_remaining -= 1
                    handle.phase = "running"
                    self._submit(handle, self._run_resume(handle, "continue"))
                elif handle.auto_mode:
                    # 自动模式最后一轮（跑满 N 轮）：resume(stop) 让 graph 走到 END
                    handle.phase = "running"
                    self._submit(handle, self._run_resume(handle, "stop"))
                else:
                    handle.phase = "awaiting_review"
            elif (snapshot.next or ()) == ():
                # next 为空 = 到 END，结束 + 收尾 loop trace
                handle.phase = "finished"
                handle.store and handle.store.finish(note="web runner 结束")
                end_loop_trace(handle.loop_root)
                handle.loop_root = None
            else:
                # 还在执行中（一般不会，invoke 是同步跑到 interrupt/END）
                handle.phase = "running"
        except Exception:  # noqa: BLE001
            # get_state 失败时退化为按 decision 判
            if state.get("decision") == "stop":
                handle.phase = "finished"
                handle.store and handle.store.finish(note="web runner 停止")
                end_loop_trace(handle.loop_root)
                handle.loop_root = None

    def _fail(self, handle: LoopHandle, msg: str) -> None:
        handle.phase = "error"
        handle.last_error = msg
        _close_app_checkpointer(handle)  # error 不可 resume，关闭 checkpointer 避免 sqlite 连接泄漏
        end_loop_trace(handle.loop_root, error=True, error_msg=msg)
        handle.loop_root = None
        # 持久化错误到 meta.json（extras.last_error），重启后仍可见
        if handle.store is not None:
            try:
                handle.store.mark_error(msg)
            except Exception:  # noqa: BLE001, S110  持久化失败不影响错误上报
                pass

    def _submit(self, handle: LoopHandle, task) -> None:
        handle.future = self._pool.submit(task)


def _close_app_checkpointer(handle: LoopHandle) -> None:
    """关闭 loop 持有的 checkpointer（仅在 _fail/error 时；finished/awaiting 保留以支持后续 resume）。

    幂等，容忍无 app/checkpointer（测试的 fake _build_app 可能不挂 checkpointer）。
    """
    app = getattr(handle, "app", None)
    checkpointer = getattr(app, "checkpointer", None) if app else None
    close_checkpointer(checkpointer)



# 单例
_runner: LoopRunner | None = None


def get_runner() -> LoopRunner:
    global _runner
    if _runner is None:
        _runner = LoopRunner()
    return _runner
