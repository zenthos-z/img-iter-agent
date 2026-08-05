"""数据聚合服务：把现有 data/ 层的纯函数包装成前端友好的模型。

只读、无网络依赖。复用 TrajectoryReader / RunStore / load_benchmark / knowledge 等。
不重写数据读取，只做「读取 → 转成 API 模型」。
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import Settings, get_settings
from ...data.benchmark import load_benchmark
from ...data.runstore import RunStore
from ...data.trajectory import TrajectoryReader
from ...memory.schema import AttemptRecord, CriticVerdict
from ..models import (
    BenchOverview,
    DimensionScoreOut,
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


def _verdict_to_out(verdict: CriticVerdict) -> VerdictOut:
    dims: list[DimensionScoreOut] = []
    for d in verdict.dimensions:
        failed = (
            [{"id": it.id, "reason": it.reason} for it in (d.items or []) if not it.passed]
            if d.scoring_type == "binary"
            else []
        )
        dims.append(
            DimensionScoreOut(
                dim=d.dim,
                scoring_type=d.scoring_type,
                value=float(d.value),
                raw=d.raw,
                failed_items=failed,
            )
        )
    return VerdictOut(
        restoration=float(verdict.restoration),
        weights_used={k: float(v) for k, v in verdict.weights_used.items()},
        dimensions=dims,
    )


def _trace_to_out(
    rec: AttemptRecord, *, loop_id: str, human_rank: float | None = None
) -> TraceOut:
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
        verdict=_verdict_to_out(rec.verdict) if rec.verdict else None,
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


# ---------------------------------------------------------------------------
# 总览（屏①）
# ---------------------------------------------------------------------------


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

        restorations = [t.verdict.restoration for t in traces if t.verdict]
        best = max(restorations) if restorations else None
        last = restorations[-1] if restorations else None
        thumbnail = traces[-1].output_image_refs[0] if traces and traces[-1].output_image_refs else None

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
            status="finished" if (meta and meta.finished_at) else "unknown",
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

    # 人工排序值（按 attempt_id 取）
    ranks = load_human_ranks(settings, sample_id) if sample_id else {}

    trace_outs = [_trace_to_out(t, loop_id=loop_id, human_rank=ranks.get(t.attempt_id)) for t in traces]

    # 经验知识库（结构化 conclusions.json，替代原单轮 MD）
    from ...memory.knowledge import load_conclusions

    kb = load_conclusions(run_dir, sample_id=sample_id, loop_id=loop_id)
    conclusions = [c.model_dump() for c in kb.conclusions]

    # target 图
    target_image = None
    target_md = None
    if traces:
        bench_id = traces[0].bench_id
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
