"""Generator 本 loop 动作记忆（系统托管，按 loop=run 隔离）。

设计决策（用户确认）：
  - **系统托管**：不走 deepagents 原生 ``memory=[files]``（MemoryMiddleware 启动时注入 system prompt）。
    原因：native memory 需 write_file/edit_file 才能「自我改进」，与 NarrowToolsMiddleware
    剥 fs 工具 + 有界 FS 权限冲突。改用**自注入**：本模块把记忆写成 markdown，每轮由
    generator 读回注入 user message（见 generator._load_memory_brief → _build_user_content）。
  - **按 loop 隔离**：记忆文件 ``run_dir/generator_memory.md``，run_dir 天然 per-loop（每 sample/每次跑），
    不同 loop 互不串扰。
  - **追加时机**：critic_node 打完分后（graph.py），这时本轮 Critic verdict 就绪，能记「动作→结果」。
  - **前端可见/可编辑/可删**：纯 markdown 文件，前端读 run_dir 列表即可展示；删 = reset_memory。

记忆内容：每轮一行块，含动作空间 A 的实际杠杆（model/size/reference/edit/params）+ Critic 结果
（restoration + 失败维度 + vs 上轮增减）。让 generator 在后续轮能看到「本 loop 之前试过什么模型/参数、
效果如何」，从而找到更合适的生图模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .schema import CriticVerdict

if TYPE_CHECKING:
    # 仅注解用（断开 agents.generator ↔ memory.loop_memory 循环导入）。
    from ..agents.generator import GenOutcome

_MEMORY_FILENAME = "generator_memory.md"
_HEADER = (
    "# Generator 本 loop 动作记忆（系统托管，每轮追加）\n"
    "# 记录每轮用的模型/参数杠杆 + Critic 结果。供 generator 选更合适的生图模型。\n"
    "# 前端可编辑/删除本文件。\n\n"
)

# 连续维度低于此阈值视为「低分失败」
_LOW_CONT_THRESHOLD = 0.7
# per-dim 涨跌 ≥ 此阈值才记入「改善/退步」（过滤噪声）
_DIM_DELTA_EPS = 0.03


def generator_memory_path(run_dir: Path) -> Path:
    """记忆文件路径（run_dir/generator_memory.md）。"""
    return run_dir / _MEMORY_FILENAME


def load_memory_brief(run_dir: Path) -> str:
    """读记忆正文（去掉头部注释），供 generator 注入 user message。不存在/空 → 空串。

    读失败兜底为空串，绝不中断闭环（与 generator 其它注入项同策略）。
    """
    mf = generator_memory_path(run_dir)
    if not mf.exists():
        return ""
    try:
        text = mf.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""
    # 去掉头部单井号注释行（"# "），保留轮次块标题（"### Round N" 三井号）。
    lines = [ln for ln in text.splitlines() if not ln.startswith("# ")]
    brief = "\n".join(ln for ln in lines if ln.strip()).strip()
    return brief


def reset_memory(run_dir: Path) -> None:
    """清空记忆（前端「删除」=重建空文件）。保留头部注释，便于前端识别。"""
    generator_memory_path(run_dir).write_text(_HEADER, encoding="utf-8")


def read_memory_raw(run_dir: Path) -> str:
    """读记忆原文（含头部注释，供前端编辑器展示）。不存在 → 空串。"""
    mf = generator_memory_path(run_dir)
    if not mf.exists():
        return ""
    try:
        return mf.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


def write_memory_raw(run_dir: Path, content: str) -> None:
    """写记忆原文（前端「编辑」保存）。空内容 → 写头部（等价 reset），避免空文件。"""
    text = content if content.strip() else _HEADER
    generator_memory_path(run_dir).write_text(text, encoding="utf-8")


def append_round(
    run_dir: Path,
    *,
    round_n: int,
    outcome: GenOutcome,
    verdict: CriticVerdict,
    prev_verdict: CriticVerdict | None,
) -> None:
    """追加一轮记忆块。prev_verdict = 上一轮 verdict（算 vs 上轮增减用；首轮 None）。

    写失败仅 print 告警，不抛（记忆是 best-effort 辅助，不能炸闭环）。
    """
    entry = _build_entry(round_n, outcome, verdict, prev_verdict)
    try:
        mf = generator_memory_path(run_dir)
        if not mf.exists():
            mf.write_text(_HEADER + entry + "\n", encoding="utf-8")
        else:
            with mf.open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[loop_memory] 追加记忆失败({type(e).__name__})，已忽略：{e}", flush=True)


def _build_entry(
    round_n: int, outcome: GenOutcome, verdict: CriticVerdict,
    prev_verdict: CriticVerdict | None,
) -> str:
    """构造单轮 markdown 块。"""
    levers: list[str] = [f"size={outcome.size}"]
    levers.append(f"参考={','.join(outcome.reference_ids) or '无'}")
    if outcome.edit_previous:
        levers.append("edit_previous=是")
    if outcome.negative_prompt:
        levers.append(f"negative={outcome.negative_prompt}")
    if outcome.seed is not None:
        levers.append(f"seed={outcome.seed}")
    if outcome.steps is not None:
        levers.append(f"steps={outcome.steps}")

    trend = ""
    if prev_verdict is not None:
        d = verdict.restoration - prev_verdict.restoration
        trend = f"  (vs上轮 {d:+.4f})"

    improved, regressed = _dim_delta(prev_verdict, verdict)
    failed = _failed_dims(verdict)

    lines = [
        f"### Round {round_n}",
        f"- 还原度: {verdict.restoration:.4f}{trend}",
        f"- 模型: {outcome.model_family} ({outcome.model})",
        f"- 杠杆: {' | '.join(levers)}",
    ]
    if outcome.strategy_note:
        lines.append(f"- 策略说明: {outcome.strategy_note.strip()}")
    lines.append(f"- 失败维度: {', '.join(failed) if failed else '无'}")
    if prev_verdict is not None:
        lines.append(
            f"- 改善: {', '.join(improved) or '无'} / 退步: {', '.join(regressed) or '无'}"
        )
    prompt_snip = (outcome.prompt or "").replace("\n", " ").strip()[:140]
    lines.append(f"- prompt 摘要: {prompt_snip}")
    return "\n".join(lines)


def _failed_dims(verdict: CriticVerdict) -> list[str]:
    """本轮失败的维度：二分维度有未通过项 / 连续维度低于阈值。"""
    out: list[str] = []
    for d in verdict.dimensions:
        if d.scoring_type == "binary":
            if any(not it.passed for it in (d.items or [])):
                out.append(d.dim)
        elif d.value < _LOW_CONT_THRESHOLD:
            out.append(f"{d.dim}({d.value:.2f})")
    return out


def _dim_delta(
    prev: CriticVerdict | None, cur: CriticVerdict,
) -> tuple[list[str], list[str]]:
    """vs 上轮的 per-dim 涨跌（过滤 < eps 的噪声）。返回 (改善, 退步)，形如 'color +0.10'。"""
    if prev is None:
        return [], []
    pf, cf = prev.features, cur.features
    improved: list[str] = []
    regressed: list[str] = []
    for dim, now in cf.items():
        old = pf.get(dim)
        if old is None:
            continue
        diff = now - old
        if diff >= _DIM_DELTA_EPS:
            improved.append(f"{dim} +{diff:.2f}")
        elif diff <= -_DIM_DELTA_EPS:
            regressed.append(f"{dim} {diff:+.2f}")
    return improved, regressed


__all__ = [
    "append_round",
    "generator_memory_path",
    "load_memory_brief",
    "read_memory_raw",
    "reset_memory",
    "write_memory_raw",
]
