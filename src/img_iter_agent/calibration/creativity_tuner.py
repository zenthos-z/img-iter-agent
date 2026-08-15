"""对抗式全自动演化「创造力」评分细分标准（cross-loop，离线）。

复用 experience-distiller 的 renovation（keep/revise/retire/new）思路：跨所有
``data/runs/<bench>-*`` 取证 → 提炼对抗信号（参考依赖、噪声判创意、过严、判别力）→
LLM renovator 翻新创造力子标准 + 有界调权 → 写版本化 overlay
（``data/benchmarks/<bench>/creativity_criteria.json``）。

**对抗**体现在：generator 的行为（用几张参考图）与 critic 的创造力打分构成博弈——
信号 ``copy_reward_corr``（参考图数↔创造力分正相关）说明 critic 在奖励抄袭，驱动收紧
``reference_independence``；``noise_as_creative``（高创造力+低原创/概念）说明标准过松。

安全（全自动必需）：永不改种子 ``content_spec.json``；每轮权重 ±0.05、绝对 clamp[0.02,0.40]、
归一；子标准数 clamp[2,6]；判别力不足（方差<0.01）的维度跳过调权；历史全留可回滚。
权重经 ``data/weights.load_weights`` 的 tier 2.5 被 critic 读取（低于 per-sample 人工校准）；
标准经 ``critic._effective_checklist`` 被 critic 读取。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..agents._agent_output import provider_structured
from ..agents._narrow_tools import invoke_with_retry, narrow_tools_middleware
from ..agents.agent_config_loader import load_system_prompt
from ..config import Settings, get_settings
from ..data.benchmark import LoadedBenchmark, load_benchmark
from ..data.trajectory import TrajectoryReader
from ..llm.chat_model import build_chat_model
from ..memory.schema import AttemptRecord, Benchmark, ContinuousRubric

# 创造力的两个固定维度
CREATIVITY_DIMS = ("creative_departure", "reference_independence")
CONTINUOUS_DIM = "creative_departure"
BINARY_DIM = "reference_independence"

# 安全护栏（R3）
WEIGHT_DELTA_CAP = 0.05     # 每轮每维权重微调上限（绝对值）
WEIGHT_FLOOR = 0.02         # 权重下限（防被清零无法再调）
WEIGHT_CEIL = 0.40          # 权重上限（防创造力独占）
SUBCRITERIA_MIN = 2
SUBCRITERIA_MAX = 6
DISCRIMINATION_VAR_FLOOR = 0.01  # 创造力分方差低于此 → 判别力不足 → 跳过调权


# ---------------------------------------------------------------------------
# Renovation schema（LLM 结构化输出；避免 Optional union → Gemini 严格 schema 安全）
# ---------------------------------------------------------------------------


class CreativityRenoItem(BaseModel):
    """对一条创造力子标准（continuous 的一个 point / binary 的一个 check 项）的处置。"""

    dim: Literal["creative_departure", "reference_independence"]
    action: Literal["keep", "revise", "retire", "new"]
    existing_id: str = Field(
        default="",
        description="revise/retire 时：continuous=要改的 point 原文；binary=要改的 item id。new 时留空。",
    )
    text: str = Field(
        default="",
        description="continuous=新的 point 文本；binary=新的 check 判定文本。keep 可留空。",
    )
    item_id: str = Field(default="", description="binary 的 new/revise 目标 item id（如 reference_independence-3）")
    anchor: str = Field(default="", description="binary 的对照锚点（如『对照 reference_ids』）")
    reason: str = Field(default="", description="这条处置的依据（引证据信号/样例）")


class CreativityRenovation(BaseModel):
    """LLM renovator 的结构化输出。"""

    summary: str = Field(default="", description="本轮翻新的总体判断（一段话）")
    items: list[CreativityRenoItem] = Field(default_factory=list, description="子标准级翻新")
    weight_delta: dict[str, float] = Field(
        default_factory=dict,
        description="创造力维度权重微调建议，如 {'creative_departure': +0.03, 'reference_independence': -0.02}；每维±0.05 内。",
    )


_DEFAULT_RENOVATOR_SYS = (
    "你是创造力评分标准对抗式调参员。目标：让创造力维度的子标准能**稳健区分**『有意义的创造性偏离』"
    "与『对参考图的抄袭/拼贴』及『无意义的随机求新』。\n"
    "你会收到：① 当前创造力子标准（continuous 的 points / binary 的 items）；② 跨 loop 的对抗诊断信号"
    "（copy_reward_corr=参考图数↔创造力分相关，正相关=在奖励抄袭；noise_as_creative=高创造力但低原创/概念"
    "的假阳性数；over_strict=低创造力但高原创的假阴性数；discriminates=创造力分是否有判别力）；"
    "③ 若干典型样例（attempt 的参考图数、各维度分、创造力理由）。\n"
    "**输出 renovation**：用 keep/revise/retire/new 翻新子标准——\n"
    "- copy_reward_corr 偏正 → 收紧 reference_independence（revise 加严 / new 一条『不得复刻传入参考的 motif 组合』）；\n"
    "- noise_as_creative 多 → 加严 creative_departure（强调『服务于概念、非随机』）；\n"
    "- over_strict 多 → 放宽 creative_departure（revise 去掉过严措辞）；\n"
    "- discriminates=false（判别力不足）→ 该维度本轮**不要**调权（weight_delta 不含它），先改标准措辞。\n"
    "weight_delta 每维限 ±0.05。子标准总数维持 2-6/维。直接结构化输出，不要调用任何工具。"
)


# ---------------------------------------------------------------------------
# Overlay I/O（版本化、原子写）
# ---------------------------------------------------------------------------


def overlay_path(bench_id: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.benchmark_dir(bench_id) / "creativity_criteria.json"


def load_overlay(bench_id: str, settings: Settings | None = None) -> dict | None:
    p = overlay_path(bench_id, settings)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _atomic_write_json(path: Path, data: dict) -> None:
    """原子写：先写临时文件再 rename，避免半写损坏 overlay。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# 当前创造力标准（overlay 优先，否则从种子 content_spec 取）
# ---------------------------------------------------------------------------


def _seed_criteria(lb: LoadedBenchmark, sample_id: str = "s001") -> dict:
    """从某 sample 的 content_spec checklist 提取创造力种子标准（overlay 格式）。"""
    spec = lb.sample(sample_id).spec
    out: dict[str, Any] = {}
    cd = spec.checklist.get(CONTINUOUS_DIM)
    pts = cd.points if isinstance(cd, ContinuousRubric) else []
    out[CONTINUOUS_DIM] = {"scoring_type": "continuous", "points": list(pts)}
    ri = spec.checklist.get(BINARY_DIM)
    items = []
    if isinstance(ri, list):
        items = [{"id": it.id, "check": it.check, "anchor": it.anchor} for it in ri]
    out[BINARY_DIM] = {"scoring_type": "binary", "items": items}
    return out


def current_criteria(lb: LoadedBenchmark, settings: Settings | None = None) -> tuple[dict, dict | None]:
    """返回 (生效创造力标准, overlay 或 None)。overlay 有则用 overlay，否则用种子。"""
    ov = load_overlay(lb.bench.bench_id, settings)
    if ov and ov.get("criteria"):
        return ov["criteria"], ov
    return _seed_criteria(lb), None


# ---------------------------------------------------------------------------
# 对抗信号提取（纯 Python）
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def extract_signals(records: list[AttemptRecord]) -> dict:
    """从历史 attempts 提炼对抗诊断信号。只统计 verdict 含 creative_departure 的记录。"""
    cds: list[float] = []
    rcounts: list[float] = []
    ods: list[float] = []
    ces: list[float] = []
    ris: list[float] = []
    for r in records:
        if r.verdict is None:
            continue
        f = r.verdict.features
        if CONTINUOUS_DIM not in f:
            continue  # 该 run 跑在加创造力维度之前，跳过
        cds.append(float(f[CONTINUOUS_DIM]))
        rcounts.append(float(len(r.reference_ids or [])))
        ods.append(float(f.get("originality_degree", 0.0)))
        ces.append(float(f.get("concept_expression", 0.0)))
        ris.append(float(f.get(BINARY_DIM, 0.0)))
    n = len(cds)
    if n == 0:
        return {"n_records": 0, "discriminates": False}
    mean_cd = sum(cds) / n
    var_cd = sum((x - mean_cd) ** 2 for x in cds) / n
    copy_corr = _pearson(rcounts, cds)
    noise = sum(1 for c, o, ce in zip(cds, ods, ces) if c > 0.7 and (o < 0.4 or ce < 0.5))
    over_strict = sum(1 for c, o in zip(cds, ods) if c < 0.3 and o > 0.7)
    return {
        "n_records": n,
        "n_with_refs": int(sum(1 for c in rcounts if c > 0)),
        "creative_departure_mean": round(mean_cd, 4),
        "creative_departure_var": round(var_cd, 4),
        "copy_reward_corr": round(copy_corr, 4) if copy_corr is not None else None,
        "noise_as_creative_count": noise,
        "over_strict_count": over_strict,
        "reference_independence_mean": round(sum(ris) / len(ris), 4) if ris else None,
        "discriminates": var_cd >= DISCRIMINATION_VAR_FLOOR,
    }


# ---------------------------------------------------------------------------
# Renovator（LLM 结构化输出）
# ---------------------------------------------------------------------------


def _example_dossier(records: list[AttemptRecord], k: int = 8) -> str:
    """挑若干典型 attempt（高/低创造力各几个）拼成文本证据。"""
    with_cd = [r for r in records if r.verdict and CONTINUOUS_DIM in r.verdict.features]
    if not with_cd:
        return "(暂无含创造力分的 attempt)"
    with_cd.sort(key=lambda r: r.verdict.features[CONTINUOUS_DIM])  # 升序
    picks = []
    if len(with_cd) <= k:
        picks = with_cd
    else:
        half = k // 2
        picks = with_cd[:half] + with_cd[-(k - half):]  # 最低几个 + 最高几个
    lines = []
    for r in picks:
        f = r.verdict.features
        cd_reason = ""
        for d in r.verdict.dimensions:
            if d.dim == CONTINUOUS_DIM and d.raw:
                cd_reason = d.raw[:160]
                break
        lines.append(
            f"- {r.attempt_id} [round {r.round}]: 参考图数={len(r.reference_ids or [])} "
            f"creative_departure={f[CONTINUOUS_DIM]:.2f} originality_degree={f.get('originality_degree', 0):.2f} "
            f"concept_expression={f.get('concept_expression', 0):.2f} reference_independence={f.get(BINARY_DIM, 0):.2f}"
            + (f"｜理由: {cd_reason}" if cd_reason else "")
        )
    return "\n".join(lines)


def _build_renovator_prompt(criteria: dict, signals: dict, records: list[AttemptRecord]) -> str:
    return (
        "## 当前创造力子标准\n```json\n"
        + json.dumps(criteria, ensure_ascii=False, indent=2)
        + "\n```\n\n## 对抗诊断信号\n```json\n"
        + json.dumps(signals, ensure_ascii=False, indent=2)
        + "\n```\n\n## 典型样例（低/高创造力各若干）\n"
        + _example_dossier(records)
        + "\n\n请按系统指令输出 renovation：翻新创造力子标准 + 有界 weight_delta。"
    )


def run_renovator(
    chat_model: BaseChatModel, criteria: dict, signals: dict, records: list[AttemptRecord],
) -> CreativityRenovation | None:
    """跑 LLM renovator，返回 CreativityRenovation；跑飞返回 None。"""
    if signals.get("n_records", 0) == 0:
        return None  # 无证据，不调
    sys_prompt = load_system_prompt("creativity-tuner", _DEFAULT_RENOVATOR_SYS)
    agent = create_deep_agent(
        model=chat_model, tools=[], system_prompt=sys_prompt,
        # 裸 pydantic 会被包成 ToolStrategy（tool_choice 400 风险）且 schema 带 $defs/$ref
        # （items 嵌 CreativityRenoItem）——dmxapi 路由到 Gemini 原生后端时被 400 拒。
        # 走 provider_structured：json_schema response_format + 平化 $defs，与其余 agent 一致。
        response_format=provider_structured(CreativityRenovation),
        checkpointer=None, name="creativity-tuner",
        middleware=narrow_tools_middleware(),
    )
    result, _ok = invoke_with_retry(
        agent, {"messages": [HumanMessage(content=_build_renovator_prompt(criteria, signals, records))]},
        config=None, label="creativity-tuner",
    )
    out = result.get("structured_response") if result else None
    return out if isinstance(out, CreativityRenovation) else None


# ---------------------------------------------------------------------------
# Merge：应用 renovation（带 clamp / 归一 / 判别力跳过）
# ---------------------------------------------------------------------------


def _find_point(points: list[str], target: str) -> int | None:
    if not target:
        return None
    for i, p in enumerate(points):
        if p == target or target in p or p in target:
            return i
    return None


def _find_item(items: list[dict], target_id: str) -> int | None:
    if not target_id:
        return None
    for i, it in enumerate(items):
        if it.get("id") == target_id:
            return i
    return None


def _clamp_subcriteria(lst: list, scoring_type: str, dim: str) -> list:
    """子标准数 clamp 到 [2,6]：超出砍尾部；不足不强行补（保留有效内容）。"""
    return list(lst[:SUBCRITERIA_MAX])


def merge_criteria(current: dict, plan: CreativityRenovation) -> dict:
    """把 renovation 应用到创造力标准，返回新的 criteria dict。"""
    # 深拷贝当前
    new: dict[str, Any] = {}
    for dim, cur in current.items():
        st = cur.get("scoring_type", "continuous")
        key = "points" if st == "continuous" else "items"
        new[dim] = {"scoring_type": st, key: list(cur.get(key, []))}

    for it in plan.items:
        if it.dim not in new:
            continue
        st = new[it.dim]["scoring_type"]
        key = "points" if st == "continuous" else "items"
        lst = new[it.dim][key]
        if it.action == "keep":
            continue
        if it.action == "new":
            if st == "continuous":
                if it.text and it.text not in lst:
                    lst.append(it.text)
            else:
                if it.text:
                    raw_id = it.item_id or f"{it.dim}-tuned-{len(lst) + 1}"
                    clean_id = "".join(c for c in raw_id if c.isalnum() or c in "-_") or f"{it.dim}-tuned-{len(lst) + 1}"
                    lst.append({
                        "id": clean_id,
                        "check": it.text, "anchor": it.anchor or None,
                    })
        elif it.action == "revise":
            if st == "continuous":
                idx = _find_point(lst, it.existing_id)
                if idx is not None and it.text:
                    lst[idx] = it.text
            else:
                idx = _find_item(lst, it.existing_id)
                if idx is not None:
                    cur_item = lst[idx]
                    lst[idx] = {
                        "id": it.item_id or it.existing_id or cur_item.get("id"),
                        "check": it.text or cur_item.get("check"),
                        "anchor": it.anchor or cur_item.get("anchor"),
                    }
        elif it.action == "retire":
            idx = _find_point(lst, it.existing_id) if st == "continuous" else _find_item(lst, it.existing_id)
            if idx is not None and len(lst) > SUBCRITERIA_MIN:  # 不跌破下限
                lst.pop(idx)
        new[it.dim][key] = _clamp_subcriteria(lst, st, it.dim)
    return new


def merge_weights(
    current_creativity_weights: dict[str, float], plan: CreativityRenovation, signals: dict,
) -> dict[str, float]:
    """应用有界 weight_delta；判别力不足的维度跳过；clamp[0.02,0.40]。"""
    out = dict(current_creativity_weights)
    discriminates = signals.get("discriminates", False)
    for dim, delta in (plan.weight_delta or {}).items():
        if dim not in CREATIVITY_DIMS:
            continue
        if not discriminates and dim == CONTINUOUS_DIM:
            continue  # creative_departure 判别力不足 → 跳过调权（先靠改标准）
        d = max(-WEIGHT_DELTA_CAP, min(WEIGHT_DELTA_CAP, float(delta)))
        out[dim] = max(WEIGHT_FLOOR, min(WEIGHT_CEIL, out.get(dim, 0.0) + d))
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def collect_records(bench_id: str, settings: Settings | None = None) -> tuple[list[AttemptRecord], list[str]]:
    """跨所有 data/runs/<bench>-* 读 trajectory，收集含创造力维度的 AttemptRecord + run 名。"""
    s = settings or get_settings()
    run_dirs = sorted(p for p in s.runs_dir.iterdir() if p.is_dir() and p.name.startswith(f"{bench_id}-"))
    records: list[AttemptRecord] = []
    run_names: list[str] = []
    for rd in run_dirs:
        traj = rd / "trajectory.jsonl"
        if not traj.exists():
            continue
        try:
            recs = TrajectoryReader(traj).read_all()
        except Exception:  # noqa: BLE001
            continue
        if any(r.verdict and CONTINUOUS_DIM in r.verdict.features for r in recs):
            records.extend(recs)
            run_names.append(rd.name)
    return records, run_names


def tune_creativity(
    bench_id: str, *, settings: Settings | None = None, chat_model: BaseChatModel | None = None,
) -> dict:
    """跑一轮创造力对抗调参。返回 summary dict（含写入的 overlay 摘要）。

    步骤：取证 → 信号 → renovator → merge → 写 overlay（版本+1，append history）。
    """
    s = settings or get_settings()
    lb = load_benchmark(bench_id, settings=s)
    bench: Benchmark = lb.bench
    prior_weights = bench.init_weights()  # 全维度先验（含创造力 0.30）

    records, run_names = collect_records(bench_id, s)
    signals = extract_signals(records)
    criteria, ov = current_criteria(lb, s)

    # 当前创造力权重 = overlay 有则取，否则先验
    cur_creativity_w = {
        CONTINUOUS_DIM: (ov.get("weights", {}).get(CONTINUOUS_DIM) if ov else prior_weights.get(CONTINUOUS_DIM, 0.2)),
        BINARY_DIM: (ov.get("weights", {}).get(BINARY_DIM) if ov else prior_weights.get(BINARY_DIM, 0.1)),
    }
    if ov and ov.get("weights"):
        cur_creativity_w = {d: float(ov["weights"].get(d, prior_weights.get(d, 0.0))) for d in CREATIVITY_DIMS}

    summary = {
        "bench_id": bench_id,
        "source_runs": run_names,
        "n_records": signals.get("n_records", 0),
        "signals": signals,
        "acted": False,
    }

    if signals.get("n_records", 0) == 0:
        summary["note"] = "无含创造力维度的 run，跳过（先跑 D1 批次）"
        return summary

    cm = chat_model or build_chat_model(s, role="summarizer")  # gemini，结构化输出可靠
    plan = run_renovator(cm, criteria, signals, records)
    if plan is None:
        summary["note"] = "renovator 未产出有效 renovation，跳过"
        return summary

    new_criteria = merge_criteria(criteria, plan)
    new_creativity_w = merge_weights(cur_creativity_w, plan, signals)

    # 组装新 overlay
    version = (ov.get("version", 0) + 1) if ov else 1
    history = list(ov.get("history", [])) if ov else []
    new_overlay = {
        "bench_id": bench_id,
        "version": version,
        "created_at": ov.get("created_at") if ov else datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "source_runs": run_names,
        "weights": {d: round(new_creativity_w[d], 4) for d in CREATIVITY_DIMS},
        "criteria": new_criteria,
        "history": history + [{
            "version": version,
            "ts": datetime.now(UTC).isoformat(),
            "source_runs": run_names,
            "delta_summary": plan.summary,
            "weight_delta": {k: round(v, 4) for k, v in (plan.weight_delta or {}).items()},
            "signals": signals,
            "n_items": len(plan.items),
        }],
    }
    _atomic_write_json(overlay_path(bench_id, s), new_overlay)

    summary.update({
        "acted": True,
        "version": version,
        "new_weights": new_overlay["weights"],
        "renovation_summary": plan.summary,
        "n_reno_items": len(plan.items),
        "overlay_path": str(overlay_path(bench_id, s)),
    })
    return summary


__all__ = [
    "CREATIVITY_DIMS",
    "CreativityRenoItem",
    "CreativityRenovation",
    "collect_records",
    "current_criteria",
    "extract_signals",
    "load_overlay",
    "merge_criteria",
    "merge_weights",
    "overlay_path",
    "run_renovator",
    "tune_creativity",
]
