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
from ...data.human_hints import (
    load_effective_hints,
    load_loop_hints,
    load_sample_hints,
    new_hint_id,
    remove_sample_hint,
    save_loop_hints,
    save_sample_hints,
)
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
    hints: list = field(default_factory=list)  # 当前生效的人工提示词（loop+sample 合并视图，运行中可改）


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
        hints: list[dict] | None = None,
    ) -> str:
        """对一道题（sample×model）启动/继续 loop。返回 loop_id。

        一题一条语义：先 find_loop，找到则续跑（复用 run_dir+checkpoint），
        无则新建。loop_id = `<bench>-<sample>`（不带时间戳；时间记 meta.started_at）。

        rounds：自动连跑轮数。None/1=首轮跑到等审批就停（等人工 resume）；
        >1=后台连跑，每轮到 interrupt 自动 resume，跑满后置 finished。

        hints：启动时附加的人工提示词，每条 {agent, text, scope}。按 scope 追加落盘
        （sample→该考题共享文件，loop→本 loop meta），再刷新 handle.hints 为合并视图。
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
                loop_id, store, lb, sample_id, rounds_remaining, auto_mode, hints, settings
            )

        # 新建 loop
        RunStore.create(
            loop_id, bench_id,
            model=model or settings.model_gemini_image or "unknown",
            settings=settings, note=note,
        )
        store = RunStore.open(loop_id, settings=settings)
        handle = LoopHandle(
            loop_id=loop_id, phase="running",
            rounds_remaining=rounds_remaining, auto_mode=auto_mode,
        )
        # 启动 hints 按 scope 追加落盘 + 加载 effective 合并视图进 handle.hints
        self._apply_start_hints(handle, store, settings, bench_id, sample_id, hints)
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
        hints: list[dict] | None = None, settings=None,
    ) -> str:
        """在已有 loop 上续跑：跑下一轮（resume）到下一个 interrupt。

        handle 优先复用内存中已有 graph 实例（old.app）；无则新建（进程重启等）。
        无论哪条路径，都先 _apply_start_hints 追加启动 hints + 刷新 handle.hints。
        """
        settings = settings or get_settings()
        with self._lock:
            old = self._handles.get(loop_id)
            if old and old.app is not None:
                handle = old  # 已有 graph 实例，直接 resume
                handle.phase = "running"
                handle.rounds_remaining = rounds_remaining
                handle.auto_mode = auto_mode
                reuse = True
            else:
                handle = LoopHandle(
                    loop_id=loop_id, phase="running",
                    rounds_remaining=rounds_remaining, auto_mode=auto_mode,
                )
                self._handles[loop_id] = handle
                reuse = False
        # 启动 hints 追加落盘 + 刷新 handle.hints（IO 在锁外）
        self._apply_start_hints(handle, store, settings, lb.bench.bench_id, sample_id, hints)
        if reuse:
            self._submit(handle, self._run_resume(handle, "continue"))
        else:
            # 无 graph 实例（进程重启等）：重建 graph + resume 续跑
            self._submit(handle, self._run_first_round(settings, lb, store, sample_id, resume_existing=True))
        return loop_id

    # --- 继续 / 停止 ---
    def resume(self, loop_id: str, decision: str) -> bool:
        """用一个 decision 续跑一轮（到下个 interrupt 或 END）。返回是否已提交。

        一题一 loop：awaiting_review（等审批）或 finished（已结束想再加一轮）都可 resume。

        无内存 handle 时（外部 run_loop_auto 起的 loop、server 重启后），若盘上 run_dir 存在
        则**收养**：重建 graph 续跑（与 start() 续跑同一条 _run_first_round(resume_existing=True)
        路径）。否则返回 False（loop 不存在）。
        """
        handle = self.get(loop_id)
        if handle is not None:
            if handle.phase not in ("awaiting_review", "finished"):
                return False
            handle.phase = "running"
            handle.interrupt_payload = None
            self._submit(handle, self._run_resume(handle, decision))
            return True
        # 无内存 handle：尝试收养盘上已存在的 loop
        settings = get_settings()
        if not self._find_existing_loop(settings, loop_id):
            return False
        self._adopt_and_resume(loop_id, decision, settings)
        return True

    def _adopt_and_resume(self, loop_id: str, decision: str, settings) -> None:
        """收养一个无内存 handle 的 loop（外部进程起 / server 重启后）并续跑下一轮。

        复用 start() 续跑的同一机制：建新 handle → _run_first_round(resume_existing=True)
        重建 graph + invoke（END 态重入跑 N+1 轮，interrupt 态 Command(resume)）。
        一键手动续跑：rounds_remaining=0（跑一轮停审批）、auto_mode=False。
        decision 仅 interrupt 态生效；手动续跑恒为 "continue"，这里不透传（保持与
        _run_first_round 既有签名兼容）。
        """
        store = RunStore.open(loop_id, settings=settings)
        bench_id = store.meta.bench_id if store.meta else ""
        _, sample_id = self._resolve_bench_sample(loop_id, store)
        lb = load_benchmark(bench_id, settings=settings)
        handle = LoopHandle(
            loop_id=loop_id, phase="running",
            rounds_remaining=0, auto_mode=False,
        )
        handle.hints = load_effective_hints(settings.data_root, store, bench_id, sample_id)
        with self._lock:
            self._handles[loop_id] = handle
        self._submit(handle, self._run_first_round(settings, lb, store, sample_id, resume_existing=True))

    # --- 人工提示词（hints）管理 ---

    def _apply_start_hints(self, handle, store, settings, bench_id, sample_id, hints):
        """启动时把传入的 hints 按 scope 追加落盘，再刷新 handle.hints 为 effective 合并视图。

        - scope=sample：追加到该考题共享文件（跨 loop 生效）。
        - scope=loop：追加到本 loop meta.extras（仅本 loop）。
        无 hints 时仍刷新 handle.hints（加载已落盘的 sample/loop 持久提示词）。
        """
        if hints:
            sample_new = [h for h in hints if h.get("scope") == "sample"]
            loop_new = [h for h in hints if h.get("scope") == "loop"]
            if sample_new:
                save_sample_hints(
                    settings.data_root, bench_id, sample_id,
                    load_sample_hints(settings.data_root, bench_id, sample_id) + sample_new,
                )
            if loop_new:
                save_loop_hints(store, load_loop_hints(store) + loop_new)
        handle.hints = load_effective_hints(settings.data_root, store, bench_id, sample_id)

    @staticmethod
    def _resolve_bench_sample(loop_id: str, store: RunStore) -> tuple[str, str]:
        """推 (bench_id, sample_id)。

        优先从 trajectory 取（record 含真实 sample_id/bench_id）——loop_id 可能带 batch
        后缀（如 `<bench>-<sample>-b2`），直接按 `<bench>-<sample>` 解析会得到错误的 sample_id
        （s002-b2 而非 s002）。trajectory 缺失时退回 loop_id 解析。
        """
        bench_id = store.meta.bench_id if store.meta else ""
        try:
            from ...data.trajectory import TrajectoryReader
            recs = TrajectoryReader(store.run_dir / "trajectory.jsonl").read_all()
            if recs:
                return recs[-1].bench_id or bench_id, recs[-1].sample_id
        except Exception:  # noqa: BLE001
            pass
        if bench_id and loop_id.startswith(bench_id + "-"):
            sample_id = loop_id[len(bench_id) + 1:]
        else:
            bench_id, _, sample_id = loop_id.partition("-")
        return bench_id, sample_id

    def get_hints(self, loop_id: str) -> list[dict] | None:
        """当前生效 hints 合并视图。

        handle 在内存→返回 handle.hints（权威）；否则从存储读（兼容 run_loop_auto 等
        外部进程启动的 loop——web 内存无其 handle，但 sample/loop 存储仍在）。
        """
        handle = self.get(loop_id)
        if handle is not None:
            return list(handle.hints)
        try:
            settings = get_settings()
            store = RunStore.open(loop_id, settings=settings)
            bench_id, sample_id = self._resolve_bench_sample(loop_id, store)
            return load_effective_hints(settings.data_root, store, bench_id, sample_id)
        except Exception:  # noqa: BLE001  loop 不存在/读取失败→空
            return []

    def add_hint(self, loop_id: str, agent: str, text: str, scope: str) -> dict | None:
        """新增一条 hint：按 scope 落盘 + 刷新内存 handle.hints。返回新建的 hint，失败返回 None。"""
        settings = get_settings()
        try:
            store = RunStore.open(loop_id, settings=settings)
        except Exception:  # noqa: BLE001
            return None
        if store.meta is None:  # loop 不存在（meta.json 缺）→ 不写
            return None
        bench_id, sample_id = self._resolve_bench_sample(loop_id, store)
        hint = {"id": new_hint_id(), "agent": agent, "text": text, "scope": scope}
        if scope == "sample":
            save_sample_hints(
                settings.data_root, bench_id, sample_id,
                load_sample_hints(settings.data_root, bench_id, sample_id) + [hint],
            )
        else:
            save_loop_hints(store, load_loop_hints(store) + [hint])
        handle = self.get(loop_id)
        if handle is not None:
            handle.hints = load_effective_hints(settings.data_root, store, bench_id, sample_id)
        return hint

    def remove_hint(self, loop_id: str, hint_id: str) -> bool:
        """删一条 hint：从对应 scope 存储移除 + 刷新内存 handle.hints。"""
        settings = get_settings()
        try:
            store = RunStore.open(loop_id, settings=settings)
        except Exception:  # noqa: BLE001
            return False
        bench_id, sample_id = self._resolve_bench_sample(loop_id, store)
        handle = self.get(loop_id)
        current = list(handle.hints) if handle is not None else load_effective_hints(
            settings.data_root, store, bench_id, sample_id
        )
        target = next((h for h in current if h.get("id") == hint_id), None)
        if target is None:
            return False
        if target.get("scope") == "sample":
            remove_sample_hint(settings.data_root, bench_id, sample_id, hint_id)
        else:
            save_loop_hints(store, [h for h in load_loop_hints(store) if h.get("id") != hint_id])
        if handle is not None:
            handle.hints = load_effective_hints(settings.data_root, store, bench_id, sample_id)
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
                        # 已有 loop：续跑下一轮（interrupt 态 resume；END/finished 态自动重入跑新轮）
                        state = self._invoke_round(handle, "continue")
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
                    # interrupt 态 resume；若 loop 已 finished（END），_invoke_round 自动重入跑新轮
                    state = self._invoke_round(handle, decision)
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

    def _invoke_round(self, handle: LoopHandle, decision: str):
        """推进一轮的 invoke，自动适配线程当前态：

        - interrupt 态（awaiting_review）：``Command(resume=decision)`` 跑下一轮（常规续跑）。
        - END 态（finished）：``Command(resume)`` 对已结束线程是**空操作**（直接 finished、不出新轮）。
          改用「不带 round 的 inputs」从 START 重入 —— generator 里 ``round_n = state.round + 1``
          自增到 N+1，历史 images/verdicts/attempts 经 reducer 累加保留。这样「继续一个已结束的 loop」
          能真正追加新一轮（回归：tests/test_loop_runner_auto.py::test_continue_finished_loop_runs_next_round）。
        """
        app, cfg = handle.app, handle.cfg
        try:
            snap = app.get_state(cfg)
            at_end = tuple(snap.next or ()) == ()
            cur_round = (snap.values or {}).get("round") or 0
        except Exception:  # noqa: BLE001  get_state 失败时退化为按 interrupt 态 resume
            at_end = False
            cur_round = handle.round or 0
        cfg_r = self._cfg_with_round(handle, cur_round + 1, "resume")
        if at_end:
            return app.invoke(self._continue_inputs(handle), config=cfg_r)
        return app.invoke(Command(resume=decision), config=cfg_r)

    @staticmethod
    def _continue_inputs(handle: LoopHandle) -> dict:
        """END 态追加新一轮的 invoke inputs。

        故意**不带 round**：保留 checkpoint 里的 round（=N），由 generator 自增到 N+1；
        若传 round:0 会把 round 通道覆盖成 0 → 跑成第 1 轮（重置，丢失轮次）。
        model/bench_id/sample_id/run_id 仅满足 START→generator 的入参约定（节点实际从闭包取 bench/sample）。
        """
        md = (handle.cfg or {}).get("metadata") or {}
        return {
            "model": md.get("model", ""),
            "bench_id": md.get("bench_id", ""),
            "sample_id": md.get("sample_id", ""),
            "run_id": handle.loop_id,
        }

    @staticmethod
    def _cfg_with_round(handle: LoopHandle, round_n: int, phase: str) -> dict:
        """在标准 config 上叠加 round/phase + 当前人工提示词。

        - metadata/tags：让 LangSmith 里每一轮 trace 可辨认（按 round/phase 过滤）。
        - configurable.human_hints：把 handle.hints（最新合并视图）注入，供 generator/critic
          node 每轮读取。新建 configurable dict（不 mutate base），用 handle.hints 覆盖
          build_loop_context 启动时注入的快照——从而「运行中改提示词，下一轮立即生效」。
        """
        base = handle.cfg or {"configurable": {"thread_id": handle.loop_id}}
        cfg = dict(base)
        cfg["configurable"] = {**(base.get("configurable") or {}), "human_hints": list(handle.hints)}
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
                else:
                    # 跑满 N 轮（或逐轮模式首轮）：停在 human_review interrupt 等审批。
                    # 不自动 stop→END——保持 graph 挂在 interrupt，可由用户继续 N+1 轮或手动停止。
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
