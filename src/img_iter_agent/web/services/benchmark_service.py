"""benchmark 管理服务（写操作）：列出 / 详情（结构+消费者）/ 新建 / 删 sample。

``data_access`` 自称「只读」，故建表/删除等写操作集中在本模块。

数据契约要点（见 ``memory/schema.py`` + content_spec 模板）：
- manifest.json 的 ``score_dimensions[].check_items``（list[str]）是二分 checklist 真源，
  所有 sample 自动继承；content_spec.json 的 ``checklist`` 只填连续维度的 per-sample points。
- 删 sample 时连带删其所有 loop + 该题 human_hints + 人工排序，但**不动**跨 loop 蒸馏
  skill 包（``data/experience/<bench>/``，per-bench）。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ...config import Settings, get_settings
from ...data.benchmark import load_benchmark
from ..models import DimensionIn, SampleIn

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class BenchmarkNotFound(KeyError):
    """benchmark 不存在。"""


class SampleNotFound(KeyError):
    """sample 不存在。"""


def _validate_name(name: str, label: str = "bench_id") -> str:
    name = (name or "").strip()
    if not name or not _NAME_RE.match(name) or name.startswith("."):
        raise ValueError(f"非法 {label}: {name!r}（仅允许字母数字 _ . -，且不以点开头）")
    return name


# ---------------------------------------------------------------------------
# 扫描 loop 目录（按 bench/sample 聚合）
# ---------------------------------------------------------------------------


def _scan_all_loops(settings: Settings) -> dict[str, dict[str, list[str]]]:
    """扫 ``runs_dir``，返回 {bench_id: {sample_id: [loop_id, ...]}}。

    复用 ``data_access`` 的 _loop_bench_id / _loop_sample_id（从 trajectory 取，容错）。
    跳过下划线开头的辅助目录（如 _human_scores）。
    """
    from .data_access import _loop_bench_id, _loop_sample_id

    runs_dir = settings.runs_dir
    out: dict[str, dict[str, list[str]]] = {}
    if not runs_dir.exists():
        return out
    for ld in sorted(runs_dir.iterdir()):
        if not ld.is_dir() or ld.name.startswith("_"):
            continue
        bid = _loop_bench_id(ld)
        sid = _loop_sample_id(ld)
        if not bid or not sid:
            continue
        out.setdefault(bid, {}).setdefault(sid, []).append(ld.name)
    return out


def _loops_for_bench(settings: Settings, bench_id: str) -> dict[str, list[str]]:
    return _scan_all_loops(settings).get(bench_id, {})


# ---------------------------------------------------------------------------
# 列出 / 详情
# ---------------------------------------------------------------------------


def list_benchmarks(settings: Settings | None = None) -> list[dict]:
    """所有 benchmark 的摘要（供管理页列表）。损坏的 bench 目录跳过。"""
    from ...memory.experience import skill_package_dir

    settings = settings or get_settings()
    loops_by_bench = _scan_all_loops(settings)
    result: list[dict] = []
    if not settings.benchmarks_dir.exists():
        return result
    for bd in sorted(settings.benchmarks_dir.iterdir()):
        if not bd.is_dir():
            continue
        try:
            lb = load_benchmark(bd.name, settings=settings)
        except Exception:  # noqa: BLE001  损坏目录跳过
            continue
        bench = lb.bench
        n_loops = sum(len(v) for v in loops_by_bench.get(bench.bench_id, {}).values())
        result.append({
            "bench_id": bench.bench_id,
            "description": bench.description,
            "scene": bench.scene,
            "scoring_method": bench.scoring_method,
            "task_type": (bench.task.type if bench.task else None),
            "n_samples": len(bench.samples),
            "n_dims": len(bench.score_dimensions),
            "n_loops": n_loops,
            "has_rubric": (bd / "rubric.md").exists(),
            "has_distill_skill": (skill_package_dir(settings.data_root, bench.bench_id) / "SKILL.md").exists(),
        })
    return result


def get_benchmark_detail(bench_id: str, settings: Settings | None = None) -> dict:
    """单个 benchmark 的结构 + 消费者（供管理页详情）。"""
    from ...memory.experience import load_general_experience, skill_package_dir

    settings = settings or get_settings()
    bench_id = _validate_name(bench_id)
    bench_dir = settings.benchmark_dir(bench_id)
    if not (bench_dir / "manifest.json").exists():
        raise BenchmarkNotFound(bench_id)

    lb = load_benchmark(bench_id, settings=settings)
    bench = lb.bench
    loops_by_sample = _loops_for_bench(settings, bench_id)

    dimensions = []
    for d in bench.score_dimensions:
        dimensions.append({
            "dim": d.dim,
            "desc": d.desc,
            "weight_init": d.weight_init,
            "ref_needed": d.ref_needed,
            "scoring_type": d.scoring_type,
            "n_check_items": len(d.check_items) if d.check_items else 0,
            "rubric_ref": d.rubric_ref,
        })

    samples = []
    for ref in bench.samples:
        smp = lb.samples.get(ref.sample_id)
        sample_loops = loops_by_sample.get(ref.sample_id, [])
        has_target = bool(smp and smp.target_path.exists()) if smp else False
        has_target_md = bool(smp and smp.target_md_path.exists()) if smp else False
        samples.append({
            "sample_id": ref.sample_id,
            "product": ref.product or smp.spec.product if smp else ref.product,
            "category": ref.category or smp.spec.category if smp else ref.category,
            "has_target": has_target,
            "has_target_md": has_target_md,
            "difficulty_note": ref.difficulty_note,
            "n_loops": len(sample_loops),
            "loop_ids": sample_loops,
        })

    # ---- 动态消费信号 ----
    n_loops_bench = sum(len(v) for v in loops_by_sample.values())
    n_running = 0
    try:
        from .loop_runner import get_runner

        runner = get_runner()
        for loops in loops_by_sample.values():
            n_running += sum(1 for lid in loops if runner.is_running(lid))
    except Exception:  # noqa: BLE001
        pass

    has_distill_skill = (skill_package_dir(settings.data_root, bench_id) / "SKILL.md").exists()
    n_active_lessons = 0
    try:
        gen = load_general_experience(settings.data_root, bench_id)
        n_active_lessons = sum(1 for ls in gen.lessons if ls.status == "active")
    except Exception:  # noqa: BLE001
        pass

    has_calibration = any(
        (settings.run_dir(lid) / "calibrated_weights.json").exists()
        for loops in loops_by_sample.values() for lid in loops
    )
    has_creativity = (bench_dir / "creativity_criteria.json").exists()

    consumers = [
        {
            "name": "Generator + Critic（pipeline）",
            "desc": "每轮消费 manifest 维度定义 + 该 sample 的 content_spec checklist + target 参考图",
            "signal": "每轮",
        },
        {
            "name": "LoopRunner",
            "desc": "启动 / 续跑 loop 时加载 benchmark 构建 loop 上下文",
            "signal": f"{n_loops_bench} loop" + (f"（{n_running} 在跑）" if n_running else ""),
        },
        {
            "name": "Distiller（跨 loop 经验蒸馏）",
            "desc": "蒸馏时加载 benchmark 渲染可移植 skill 包",
            "signal": (f"{n_active_lessons} 条 active 经验" if n_active_lessons
                       else ("已有 skill 包" if has_distill_skill else "未蒸馏")),
        },
        {
            "name": "Calibrator（排序校准）",
            "desc": "排序校准时加载 benchmark 维度做权重学习",
            "signal": "已有校准权重" if has_calibration else "未校准",
        },
        {
            "name": "Critic creativity overlay",
            "desc": "创造力场景读 creativity_criteria.json 覆盖打分",
            "signal": "存在" if has_creativity else "无",
        },
        {
            "name": "Agent 设置页",
            "desc": "benchmark 下拉选项来源（per-bench 技能切换）",
            "signal": "始终",
        },
    ]

    return {
        "bench_id": bench.bench_id,
        "description": bench.description,
        "scene": bench.scene,
        "scoring_method": bench.scoring_method,
        "scoring_note": bench.scoring_note,
        "version": bench.version,
        "task": (bench.task.model_dump(exclude_none=True) if bench.task else None),
        "comparative_dims": list(bench.comparative_dims),
        "dimensions": dimensions,
        "samples": samples,
        "file_tree": _file_tree(bench_dir),
        "consumers": consumers,
    }


def _file_tree(bench_dir: Path) -> list[dict]:
    """bench 目录的实际文件结构（top-level 文件 + samples/<id>/ 内文件）。"""
    tree: list[dict] = []
    for p in sorted(bench_dir.iterdir()):
        if p.is_dir() and p.name == "samples":
            tree.append({"path": "samples", "type": "dir"})
            for sd in sorted(p.iterdir()):
                if not sd.is_dir():
                    continue
                tree.append({"path": f"samples/{sd.name}", "type": "dir"})
                for f in sorted(sd.iterdir()):
                    if f.is_file():
                        tree.append({"path": f"samples/{sd.name}/{f.name}", "type": "file"})
                    elif f.is_dir():
                        tree.append({"path": f"samples/{sd.name}/{f.name}/", "type": "dir"})
        elif p.is_file():
            tree.append({"path": p.name, "type": "file"})
        elif p.is_dir():
            tree.append({"path": f"{p.name}/", "type": "dir"})
    return tree


# ---------------------------------------------------------------------------
# 新建 benchmark
# ---------------------------------------------------------------------------


def _task_dict(task_type: str, views: str | None) -> dict:
    views_list = [v.strip() for v in (views or "").split(",") if v.strip()]
    if task_type == "three_view_whitebg_single_image":
        return {
            "type": task_type,
            "layout": "three_view_single_image",
            "views": views_list or ["front", "side", "perspective"],
        }
    if task_type == "style_transfer":
        return {"type": task_type, "layout": "single_image", "views": views_list}
    return {"type": task_type or "custom", "views": views_list}


def _content_spec_scaffold(
    sample: SampleIn, dims: list[DimensionIn], task_type: str, views: str | None
) -> dict:
    """单道考题的 content_spec 脚手架。

    二分维度的 checklist 由 manifest 的 check_items 统一提供（所有 sample 继承），
    此处只给连续维度一个空 points 占位；constraints 留空待用户填。
    """
    views_list = [v.strip() for v in (views or "").split(",") if v.strip()]
    if task_type == "three_view_whitebg_single_image":
        output = {
            "layout": "three_view_single_image",
            "views": views_list or ["front", "side", "perspective"],
            "background": "white",
            "size": "2K",
        }
    elif task_type == "style_transfer":
        output = {"layout": "single_image", "views": views_list}
    else:
        output = {"views": views_list}

    checklist: dict[str, dict] = {}
    for d in dims:
        if d.scoring_type == "continuous":
            checklist[d.dim] = {"_scoring": "continuous", "points": []}

    return {
        "sample_id": sample.sample_id,
        "product": sample.product or "",
        "category": sample.category or "",
        "task": {
            "mode": "image_edit",
            "input_assets": ["target.jpg"],
            "instruction": "",
            "output": output,
            "article_topic": None,
        },
        "constraints": {"must_keep": [], "may_change": [], "must_avoid": []},
        "checklist": checklist,
    }


def _rubric_md(bench_id: str, scene: str | None, dims: list[DimensionIn]) -> str:
    lines = [
        f"# Benchmark: {scene or bench_id}",
        "",
        f"> `bench_id`: {bench_id}",
        "> 评分维度真源 = `manifest.json`（本文件仅人类可读说明，二者不一致以 manifest 为准）。",
        "",
        "## 评分维度",
        "",
    ]
    for d in dims:
        kind = "二分（逐项 ✓/✗）" if d.scoring_type == "binary" else "连续（LLM 0-1 分）"
        lines.append(f"- `{d.dim}`({d.weight_init}) {d.desc or ''} — {kind}")
    lines += ["", "## samples/ 目录约定", "```",
              "samples/<sNNN>/", "├── target.jpg          # 参考锚图（对比型维度评判锚）",
              "├── target.md           # 说明（可选）", "└── content_spec.json   # 任务 + 约束 + 连续维度 points", "```"]
    return "\n".join(lines) + "\n"


def create_benchmark(
    *,
    bench_id: str,
    scene: str | None,
    description: str | None,
    scoring_method: str | None,
    task_type: str,
    views: str | None,
    dimensions: list[DimensionIn],
    samples: list[SampleIn],
    target_files: dict[str, bytes],  # sample_id -> 图片字节
    settings: Settings | None = None,
) -> str:
    """新建一个 benchmark：目录 + manifest + rubric + 每 sample 的 content_spec 脚手架 + target 图。"""
    settings = settings or get_settings()
    bench_id = _validate_name(bench_id)
    if not dimensions:
        raise ValueError("至少需要一个评分维度")
    if not samples:
        raise ValueError("至少需要一道 sample")
    for s in samples:
        _validate_name(s.sample_id, "sample_id")

    bench_dir = settings.benchmark_dir(bench_id)
    if bench_dir.exists():
        raise FileExistsError(f"benchmark 已存在: {bench_id}")

    bench_dir.mkdir(parents=True)

    manifest = {
        "bench_id": bench_id,
        "version": "1.0.0",
        "scene": (scene or "").strip() or bench_id,
        "description": (description or "").strip(),
        "scoring_method": scoring_method or "hybrid_with_rank_calibration",
        "scoring_note": "混合评分(二分+连续) + 排序校准。",
        "score_dimensions": [
            {
                "dim": d.dim,
                "desc": d.desc,
                "weight_init": d.weight_init,
                "ref_needed": d.ref_needed,
                "scoring_type": d.scoring_type,
                "check_items": [c for c in (d.check_items or []) if c.strip()] if d.scoring_type == "binary" else None,
                "rubric_ref": d.rubric_ref if d.scoring_type == "continuous" else None,
            }
            for d in dimensions
        ],
        "comparative_dims": [d.dim for d in dimensions if d.ref_needed],
        "task": _task_dict(task_type, views),
        "samples": [
            {
                "sample_id": s.sample_id,
                "product": s.product or "",
                "category": s.category or "",
                "target": f"samples/{s.sample_id}/target.jpg",
                "difficulty_note": s.difficulty_note or "",
            }
            for s in samples
        ],
    }
    (bench_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bench_dir / "rubric.md").write_text(
        _rubric_md(bench_id, scene, dimensions), encoding="utf-8"
    )

    for s in samples:
        sdir = bench_dir / "samples" / s.sample_id
        sdir.mkdir(parents=True)
        spec = _content_spec_scaffold(s, dimensions, task_type, views)
        (sdir / "content_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        img = target_files.get(s.sample_id)
        if img:
            (sdir / "target.jpg").write_bytes(img)

    return bench_id


# ---------------------------------------------------------------------------
# 删 sample（+ 其所有 loop + human_hints + 人工排序；不动跨 loop 蒸馏 skill）
# ---------------------------------------------------------------------------


def _human_hints_path(settings: Settings, bench_id: str, sample_id: str) -> Path:
    return settings.data_root / "human_hints" / bench_id / f"{sample_id}.json"


def delete_sample(bench_id: str, sample_id: str, settings: Settings | None = None) -> None:
    """删一道 sample：manifest 移除该条 → 删 sample 目录 → 删其所有 loop → 删 human_hints/scores。

    若该 sample 任一 loop 在跑则抛 LoopBusyError（router → 409）。
    **不动** data/experience/<bench>/（跨 loop 蒸馏 skill，per-bench）。
    """
    settings = settings or get_settings()
    bench_id = _validate_name(bench_id)
    sample_id = _validate_name(sample_id, "sample_id")
    bench_dir = settings.benchmark_dir(bench_id)
    manifest_path = bench_dir / "manifest.json"
    if not manifest_path.exists():
        raise BenchmarkNotFound(bench_id)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_samples = data.get("samples", [])
    new_samples = [s for s in old_samples if s.get("sample_id") != sample_id]
    if len(new_samples) == len(old_samples):
        raise SampleNotFound(sample_id)

    # 先收集所有相关 loop 并检查无在跑（删除前整体守卫，避免删一半）
    sample_loops = _loops_for_bench(settings, bench_id).get(sample_id, [])
    from .loop_runner import LoopBusyError, get_runner

    runner = get_runner()
    for lid in sample_loops:
        if runner.is_running(lid):
            raise LoopBusyError(f"sample {sample_id} 的 loop {lid} 正在运行")

    # 1) manifest 移除该 sample（raw-dict 保留所有其它字段含 extra）
    data["samples"] = new_samples
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 删 sample 目录
    sample_dir = bench_dir / "samples" / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    # 3) 删该 sample 所有 loop（复用 delete_loop 的清理：handle/checkpointer/rmtree）
    for lid in sample_loops:
        try:
            runner.delete_loop(lid)
        except Exception:  # noqa: BLE001  best-effort，manifest 已更新
            pass

    # 4) 删 sample 级 human_hints + 人工排序
    hp = _human_hints_path(settings, bench_id, sample_id)
    if hp.exists():
        hp.unlink()
    sp = settings.runs_dir / "_human_scores" / f"{sample_id}.json"
    if sp.exists():
        sp.unlink()
