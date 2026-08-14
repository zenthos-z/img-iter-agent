"""数据聚合服务：把现有 data/ 层的纯函数包装成前端友好的模型。

只读、无网络依赖。复用 TrajectoryReader / RunStore / load_benchmark / knowledge 等。
不重写数据读取，只做「读取 → 转成 API 模型」。
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import Settings, get_settings
from ...data.benchmark import load_benchmark
from ...data.runstore import RunStore, run_is_alive
from ...data.weights import apply_weights, compute_features, load_weights, weighted_restoration
from ...data.trajectory import TrajectoryReader
from ...memory.schema import AttemptRecord, CriticVerdict
from ..models import (
    BenchOverview,
    DimensionScoreOut,
    DistilledLessonOut,
    GeneralExperienceOut,
    LessonEdit,
    LoopDetail,
    LoopSummary,
    OverviewResponse,
    SampleOverview,
    TraceOut,
    VerdictOut,
)

# sample_id 从 loop 目录名解析：目录名形如 collect-s001-0731-115537
# 但 live-verify-001 这类没有 sample 段，需从 trajectory.jsonl 取 sample_id 字段。


# ---------------------------------------------------------------------------
# agent 活动流事件（events.jsonl）
# ---------------------------------------------------------------------------


def read_events_since(run_dir: Path, since: int = 0) -> tuple[list[dict], int]:
    """读 ``run_dir/events.jsonl``，返回 ``(since 之后的行, 总行数)``。

    since = 文件行号游标（1-based 计数，0 起始过滤）。seq 用文件行号语义、emitter 不写 seq，
    故即便 web 重启后新 emitter 追加，游标仍单调不错位。文件不存在 → ``([], 0)``。
    不依赖内存 LoopHandle——CLI/脚本起的 loop（web 内存无 handle）也能读。
    """
    p = Path(run_dir) / "events.jsonl"
    if not p.exists():
        return [], 0
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return [], 0
    events: list[dict] = []
    total = 0
    for line in text.splitlines():
        total += 1
        if total <= since:
            continue
        s = line.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except json.JSONDecodeError:  # noqa: PERF203  坏行跳过（不丢其他行）
            continue
    return events, total


# ---------------------------------------------------------------------------
# human_scores（人工排序）读写
# ---------------------------------------------------------------------------


def _human_scores_path(settings: Settings, sample_id: str) -> Path:
    """人工排序按 sample 存一份（跨 loop 聚合），放 benchmark 目录下。"""
    # 放 data/runs/_human_scores/<sample_id>.json，避免污染单个 loop
    d = settings.runs_dir / "_human_scores"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample_id}.json"


def load_human_ranks(settings: Settings, sample_id: str) -> dict[str, float]:
    """读该 sample 已提交的人工排序：{trace_id(attempt_id): rank}。"""
    p = _human_scores_path(settings, sample_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, float] = {}
    for item in data.get("ranks", []):
        tid = item.get("trace_id")
        if tid:
            out[tid] = float(item.get("rank", 0))
    return out


# ---------------------------------------------------------------------------
# trace / verdict → API 模型
# ---------------------------------------------------------------------------


def _verdict_to_out(
    verdict: CriticVerdict,
    *,
    rescored: bool = False,
    restoration_original: float | None = None,
) -> VerdictOut:
    dims: list[DimensionScoreOut] = []
    for d in verdict.dimensions:
        # 二分维度：透出全量逐项判定（含通过项 + reason），前端逐项 ✓/✗ 展示。
        items = (
            [{"id": it.id, "passed": it.passed, "reason": it.reason} for it in (d.items or [])]
            if d.scoring_type == "binary"
            else []
        )
        failed = [it for it in items if not it["passed"]]
        dims.append(
            DimensionScoreOut(
                dim=d.dim,
                scoring_type=d.scoring_type,
                value=float(d.value),
                raw=d.raw,
                items=items,
                failed_items=failed,
            )
        )
    return VerdictOut(
        restoration=float(verdict.restoration),
        rescored=rescored,
        restoration_original=restoration_original,
        weights_used={k: float(v) for k, v in verdict.weights_used.items()},
        dimensions=dims,
    )


def _trace_to_out(
    rec: AttemptRecord,
    *,
    loop_id: str,
    human_rank: float | None = None,
    weights_now: dict[str, float] | None = None,
) -> TraceOut:
    """weights_now：当前生效权重。与 trace 冻结的 weights_used 不同时，restoration
    按当前权重重算（rescored=True，原始分放 restoration_original），落盘数据不动。
    """
    verdict_out: VerdictOut | None = None
    if rec.verdict is not None:
        v, rescored, original = rec.verdict, False, None
        if weights_now is not None and _weights_differ(weights_now, v.weights_used):
            v = apply_weights(v, weights_now)
            rescored, original = True, float(rec.verdict.restoration)
        verdict_out = _verdict_to_out(v, rescored=rescored, restoration_original=original)
    return TraceOut(
        loop_id=loop_id,
        trace_id=rec.attempt_id,
        round=rec.round,
        sample_id=rec.sample_id,
        bench_id=rec.bench_id,
        model=rec.model,
        ts=rec.ts,
        test_variable=rec.test_variable,
        baseline_ref=rec.baseline_ref,
        gen_mode=rec.gen_mode,
        prompt=rec.prompt,
        size=rec.size,
        output_image_refs=list(rec.output_image_refs),
        reference_image_refs=list(rec.reference_image_refs),
        verdict=verdict_out,
        lesson_ref=rec.lesson_ref,
        delta_note=rec.delta_note,
        human_rank=human_rank,
    )


def _read_traces(run_dir: Path) -> list[AttemptRecord]:
    """读一个 loop 的 trajectory.jsonl（坏行自动跳过）。"""
    reader = TrajectoryReader(run_dir / "trajectory.jsonl")
    return reader.read_all()


def _loop_meta(run_dir: Path) -> dict:
    """读 loop 的 meta.json（容错）。"""
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _loop_sample_id(run_dir: Path) -> str:
    """取 loop 的 sample_id。优先 trajectory，其次 loop_id 解析（<bench>-<sample>）。

    注意：绝不能在 trajectory 为空时直接用 run_dir.name——当 loop 命名与
    `<bench>-<sample>` 一致时，那等于把 bench 前缀也带进来，造出形如
    「furniture_product_whitebg-s001」的幽灵 sample。此处用 meta.bench_id 剥前缀。
    """
    traces = _read_traces(run_dir)
    if traces:
        return traces[0].sample_id
    # 退化：loop_id = "<bench>-<sample>"，用 bench_id 剥前缀
    bench_id = _loop_meta(run_dir).get("bench_id", "")
    name = run_dir.name
    if bench_id and name.startswith(bench_id + "-"):
        return name[len(bench_id) + 1:]
    return name


def _loop_bench_id(run_dir: Path) -> str:
    traces = _read_traces(run_dir)
    if traces:
        return traces[0].bench_id
    # 退化：从 meta 读
    return _loop_meta(run_dir).get("bench_id", "")


def _weights_differ(a: dict[str, float], b: dict[str, float]) -> bool:
    """权重是否实质不同（键集不同，或任一维差 > 1e-9）。

    冻结的 weights_used 常带归一化浮点尘差（0.25000000000000006 vs 0.25），
    直接 dict 比较会把「权重未变」误报成 rescored。
    """
    if set(a) != set(b):
        return True
    return any(abs(a[k] - b[k]) > 1e-9 for k in a)


def _current_weights(
    settings: Settings, bench_id: str | None, sample_id: str | None, *, run_dir: Path | None
) -> dict[str, float] | None:
    """取当前生效权重（run 级校准 > sample 级人工校准 > 先验，见 weights.load_weights）。

    用于对历史 trace 冻结的 restoration 按当前权重重算：手动排序产生新权重后，
    前端展示的历史分数随之更新（展示层重算，不动 trajectory.jsonl 落盘数据）。
    bench 缺失/损坏时返回 None，调用方退回冻结分。
    """
    if not bench_id:
        return None
    try:
        bench = load_benchmark(bench_id, settings=settings).bench
        return load_weights(bench, run_dir=run_dir, sample_id=sample_id)
    except Exception:  # noqa: BLE001, S110  bench 加载失败 → 不重算
        return None


# ---------------------------------------------------------------------------
# 通用经验（跨 loop 蒸馏）
# ---------------------------------------------------------------------------


def build_general_experience(
    bench_id: str, *, settings: Settings | None = None
) -> GeneralExperienceOut:
    """读某 bench 的跨 loop 通用经验（general.json）→ API 模型。无则空。"""
    from ...memory.experience import load_general_experience

    settings = settings or get_settings()
    exp = load_general_experience(settings.data_root, bench_id)
    return GeneralExperienceOut(
        bench_id=exp.bench_id,
        summary=exp.summary,
        lessons=[DistilledLessonOut(**ls.model_dump()) for ls in exp.lessons],
        source_runs=list(exp.source_runs),
        updated_at=exp.updated_at or None,
        scene=exp.scene,
        dimensions=list(exp.dimensions),
        bench_description=exp.bench_description,
        categories=list(exp.categories),
    )


def mutate_lesson(
    bench_id: str,
    lesson_id: str,
    *,
    edit: LessonEdit | None = None,
    refute_reason: str | None = None,
    archive: bool = False,
    settings: Settings | None = None,
) -> GeneralExperienceOut | None:
    """人工改/标无效/归档一条 lesson：load → 按 id 改 → save（自动重渲染 SKILL.md）。

    edit 优先；其次 refute_reason（→refuted）；再次 archive（→archived）。找不到 id 返回 None。
    """
    from ...memory.experience import load_general_experience, save_general_experience

    settings = settings or get_settings()
    exp = load_general_experience(settings.data_root, bench_id)
    target = next((l for l in exp.lessons if l.id == lesson_id), None)
    if target is None:
        return None
    if edit is not None:
        if edit.insight is not None:
            target.insight = edit.insight
        if edit.dos is not None:
            target.dos = list(edit.dos)
        if edit.donts is not None:
            target.donts = list(edit.donts)
        if edit.category is not None:
            target.category = edit.category
        if edit.applies_when in ("construction", "fix", "always"):
            target.applies_when = edit.applies_when
        if edit.confidence is not None:
            target.confidence = max(0.0, min(1.0, float(edit.confidence)))
        target.status = "active"  # 编辑视为重新启用
        target.retire_reason = ""
        target.successor_id = ""
    if refute_reason is not None:
        target.status = "refuted"
        target.retire_reason = refute_reason or "人工标无效"
    elif archive:
        target.status = "archived"
        if not target.retire_reason:
            target.retire_reason = "人工归档"
    save_general_experience(settings.data_root, bench_id, exp)
    return build_general_experience(bench_id, settings=settings)


# ---------------------------------------------------------------------------
# 总览（屏①）
# ---------------------------------------------------------------------------


def _detect_running_loop_ids() -> set[str]:
    """best-effort 扫描活着的 run_loop_auto.py 进程，反解出正在跑的 loop_id 集合。

    兜底用：CLI/批量脚本起 loop 在另一进程，run 目录可能没有 running.pid（如起于本机制
    之前）。只要进程还在跑，就能从命令行参数反解 loop_id。任何异常都返回空集（绝不影响总览）。
    loop_id 拼法复刻 .claude/skills/img-iter-ops/scripts/run_loop_auto.py：--loop-id 优先，
    否则 <bench>-<sample>-<tag 或 'auto'>。
    """
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-ax", "-ww", "-o", "command="],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception:  # noqa: BLE001  ps 不可用/超时 → 跳过，靠 pid 文件
        return set()

    ids: set[str] = set()
    for line in out.splitlines():
        if "run_loop_auto.py" not in line:
            continue
        toks = line.split()
        bench = sample = tag = loop_id = None
        i = 0
        while i < len(toks):
            t = toks[i]
            val = toks[i + 1] if i + 1 < len(toks) else None
            if t == "--bench" and val:
                bench = val
                i += 2
                continue
            if t == "--sample" and val:
                sample = val
                i += 2
                continue
            if t == "--tag" and val:
                tag = val
                i += 2
                continue
            if t == "--loop-id" and val:
                loop_id = val
                i += 2
                continue
            i += 1
        if loop_id:
            ids.add(loop_id)
        elif bench and sample and not sample.startswith("$"):  # 跳过未展开的 $s（shell 文本）
            ids.add(f"{bench}-{sample}-{tag or 'auto'}")
    return ids


def build_overview(settings: Settings | None = None) -> OverviewResponse:
    """聚合所有 bench/sample/loop 的摘要 + 每个 sample 的待打分数。

    即使无 loop，也从 benchmark manifest 预填所有 bench/sample（供启动表单选 sample）。
    """
    settings = settings or get_settings()
    runs_dir = settings.runs_dir

    # 扫描所有 loop 目录（跳过 _human_scores 这类下划线开头的辅助目录）
    loop_dirs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    )

    # 按 bench 聚合
    benches: dict[str, BenchOverview] = {}

    # 预填：从 benchmark manifest 加载所有 bench/sample（即使无 loop 也显示，供选 sample）
    try:
        for bd in settings.benchmarks_dir.iterdir():
            if not bd.is_dir():
                continue
            try:
                lb = load_benchmark(bd.name, settings=settings)
            except Exception:  # noqa: BLE001, S112  损坏的 bench 目录跳过，不影响总览
                continue
            bench = benches.setdefault(
                lb.bench.bench_id,
                BenchOverview(bench_id=lb.bench.bench_id, description=lb.bench.description),
            )
            for sid, smp in lb.samples.items():
                if not any(s.sample_id == sid for s in bench.samples):
                    bench.samples.append(
                        SampleOverview(sample_id=sid, product=smp.spec.product, category=smp.spec.category)
                    )
    except Exception:  # noqa: BLE001, S110  benchmarks 目录容错，无则不预填
        pass

    # 预读 bench 描述
    bench_desc_cache: dict[str, str | None] = {}

    # 实时 loop 运行态：loop_runner 单例的内存 handle.phase。合并到每个 loop 的 status，
    # 否则后台正跑的 loop 在总览里只会是 unknown → 前端「运行中」页过滤不到（空白）。
    # （延迟导入避开循环依赖，与本函数内其它延迟导入风格一致。）
    from .loop_runner import get_runner

    runner = get_runner()
    # 外部进程（CLI/批量脚本）正在跑的 loop_id 集合：ps 扫描兜底（含修复前已起跑、目录
    # 无 running.pid 的 loop）。run_is_alive(pid 文件存活) 在下面 per-loop 再判一次。
    live_ext = _detect_running_loop_ids()

    for ld in loop_dirs:
        bench_id = _loop_bench_id(ld)
        if not bench_id:
            continue
        sample_id = _loop_sample_id(ld)
        if not sample_id:
            continue

        traces = _read_traces(ld)
        store = RunStore.open(ld.name, settings=settings)
        meta = store.meta

        # 按当前生效权重（含人工校准回灌）重算历史 restoration：手动排序产生新权重后，
        # 总览的历史分数随之更新（展示层重算，trajectory.jsonl 冻结值不动）。
        weights_now = _current_weights(settings, bench_id, sample_id, run_dir=ld)
        rescored = False
        restorations: list[float] = []
        for t in traces:
            if not t.verdict:
                continue
            if weights_now is not None and _weights_differ(weights_now, t.verdict.weights_used):
                restorations.append(weighted_restoration(compute_features(t.verdict), weights_now))
                rescored = True
            else:
                restorations.append(t.verdict.restoration)
        best = max(restorations) if restorations else None
        last = restorations[-1] if restorations else None
        thumbnail = traces[-1].output_image_refs[0] if traces and traces[-1].output_image_refs else None

        # status 合并优先级：
        #   1) web LoopRunner 内存态（web 自己起/续的 loop，含 awaiting_review 等暂停态）
        #   2) 外部进程在跑：ps 扫到 run_loop_auto.py 反解出该 loop_id，或 run_dir/running.pid
        #      存活（CLI/批量起，web 内存不知道；ps 兜底也覆盖修复前已起跑、目录无 pid 文件的 loop）
        #   3) 盘上 meta.finished_at → finished
        #   4) 否则 unknown
        handle = runner.get(ld.name)
        if handle is not None and handle.phase in (
            "running",
            "awaiting_review",
            "error",
            "finished",
        ):
            status = handle.phase
        elif ld.name in live_ext or run_is_alive(ld):
            status = "running"
        elif meta and meta.finished_at:
            status = "finished"
        else:
            status = "unknown"

        has_checkpoint = (ld / "checkpoints.sqlite").exists()

        loop_sum = LoopSummary(
            loop_id=ld.name,
            bench_id=bench_id,
            sample_id=sample_id,
            model=meta.model if meta else "",
            started_at=meta.started_at if meta else None,
            finished_at=meta.finished_at if meta else None,
            n_traces=len(traces),
            best_restoration=best,
            last_restoration=last,
            rescored=rescored,
            status=status,
            has_checkpoint=has_checkpoint,
            thumbnail=thumbnail,
        )

        bench = benches.setdefault(
            bench_id, BenchOverview(bench_id=bench_id, description=bench_desc_cache.get(bench_id))
        )
        # 补 bench 描述（懒加载一次）
        if bench.description is None and bench_id not in bench_desc_cache:
            try:
                lb = load_benchmark(bench_id, settings=settings)
                bench.description = lb.bench.description
                bench_desc_cache[bench_id] = lb.bench.description
            except Exception:  # noqa: BLE001
                bench_desc_cache[bench_id] = None

        # 找对应 sample
        sample = next((s for s in bench.samples if s.sample_id == sample_id), None)
        if sample is None:
            sample = SampleOverview(sample_id=sample_id)
            bench.samples.append(sample)
        sample.loops.append(loop_sum)

    # 每个 sample 的聚合统计 + 待打分数
    for bench in benches.values():
        # 跨 loop 通用经验条数（总览入口 badge；只数 active——实际被消费的）
        try:
            from ...memory.experience import load_general_experience

            gen = load_general_experience(settings.data_root, bench.bench_id)
            bench.general_experience_count = sum(1 for l in gen.lessons if l.status == "active")
        except Exception:  # noqa: BLE001
            bench.general_experience_count = 0
        for sample in bench.samples:
            n = sum(l.n_traces for l in sample.loops)
            sample.n_traces = n
            ranked = load_human_ranks(settings, sample.sample_id)
            # 待打分 = 该 sample 下未被排过序的 trace 数
            # 用 trace_id 集合差集，但此处只有 loop 摘要没有 trace_id 列表，
            # 退化为：若该 sample 有人提交过排序，pending=0，否则 pending=n
            sample.pending = 0 if ranked else n
            # 补 product 信息（从 benchmark sample 取）
            try:
                lb = load_benchmark(bench.bench_id, settings=settings)
                smp = lb.samples.get(sample.sample_id)
                if smp:
                    sample.product = smp.spec.product
                    sample.category = smp.spec.category
            except Exception:  # noqa: BLE001, S110  benchmark 元信息容错，失败不影响总览
                pass

    return OverviewResponse(benches=list(benches.values()))


# ---------------------------------------------------------------------------
# loop 详情（屏②）
# ---------------------------------------------------------------------------


def build_loop_detail(
    loop_id: str, *, settings: Settings | None = None, status_extra: dict | None = None
) -> LoopDetail | None:
    """读一个 loop 的完整详情。status_extra 由 loop_runner 注入（运行态）。"""
    settings = settings or get_settings()
    run_dir = settings.run_dir(loop_id)
    if not run_dir.exists():
        return None

    store = RunStore.open(loop_id, settings=settings)
    meta = store.meta
    traces = _read_traces(run_dir)
    sample_id = traces[0].sample_id if traces else ""
    bench_id = traces[0].bench_id if traces else ""

    # 人工排序值（按 attempt_id 取）
    ranks = load_human_ranks(settings, sample_id) if sample_id else {}

    # 当前生效权重：与 trace 冻结权重不同时，展示分按它重算（见 _trace_to_out）
    weights_now = _current_weights(settings, bench_id or None, sample_id or None, run_dir=run_dir)
    trace_outs = [
        _trace_to_out(
            t, loop_id=loop_id, human_rank=ranks.get(t.attempt_id), weights_now=weights_now
        )
        for t in traces
    ]

    # 经验知识库（结构化 conclusions.json，替代原单轮 MD）
    from ...memory.knowledge import load_conclusions

    kb = load_conclusions(run_dir, sample_id=sample_id, loop_id=loop_id)
    conclusions = [c.model_dump() for c in kb.conclusions]

    # target 图
    target_image = None
    target_md = None
    if traces:
        try:
            lb = load_benchmark(bench_id, settings=settings)
            smp = lb.samples.get(sample_id)
            if smp:
                if smp.target_path.exists():
                    target_image = str(smp.target_path.resolve())
                if smp.target_md_path and smp.target_md_path.exists():
                    target_md = smp.target_md_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001, S110  target 图/说明容错，失败则不显示
            pass

    status = "finished" if (meta and meta.finished_at) else "unknown"
    round_now = traces[-1].round if traces else None
    # last_error：优先内存 handle（status_extra），否则从持久化的 meta.extras 读（重启后）
    last_error = (status_extra or {}).get("last_error")
    if last_error is None and meta and meta.extras.get("last_error"):
        last_error = meta.extras.get("last_error")
        # 重启后无内存状态，但有持久化错误 → 推断为 error
        if status == "unknown":
            status = "error"
    if status_extra:  # loop_runner 的实时状态覆盖
        status = status_extra.get("status", status)
        round_now = status_extra.get("round", round_now)
    elif status == "unknown" and run_is_alive(run_dir):
        # 非 web 起的 loop（CLI/批量脚本，无内存 handle）但进程仍在跑 → running
        status = "running"

    return LoopDetail(
        loop_id=loop_id,
        bench_id=(meta.bench_id if meta else ""),
        sample_id=sample_id,
        model=(meta.model if meta else ""),
        started_at=(meta.started_at if meta else None),
        finished_at=(meta.finished_at if meta else None),
        note=(meta.note if meta else None),
        status=status,
        round=round_now,
        interrupt_payload=(status_extra or {}).get("interrupt_payload"),
        last_error=last_error,
        traces=trace_outs,
        conclusions=conclusions,
        target_image=target_image,
        target_md=target_md,
    )
