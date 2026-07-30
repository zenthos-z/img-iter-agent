"""记忆索引（JSON）：记录每次尝试的参数与链接，供召回/过滤。

双层记忆的「机器索引层」（ADR-004）：JSON 只存参数、版本、文件链接，
不存图片 base64、不存经验正文（那些在 MD / 图片文件里）。
索引每条 = 一次 attempt 的摘要，字段对齐 ARCH §3.3.2。

文件：runs/<run_id>/index.json = {"attempts": [entry, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(run_dir: Path) -> dict[str, Any]:
    idx = run_dir / "index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))
    return {"attempts": []}


def _save(run_dir: Path, data: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "index.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_entry(run_dir: Path, entry: dict[str, Any]) -> None:
    """追加一条 attempt 摘要到 index.json。"""
    data = _load(run_dir)
    data.setdefault("attempts", []).append(entry)
    _save(run_dir, data)


def list_entries(run_dir: Path) -> list[dict[str, Any]]:
    return _load(run_dir).get("attempts", [])


def recall(
    run_dir: Path,
    *,
    model: str | None = None,
    gen_mode: str | None = None,
    min_restoration: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按条件召回历史 attempt（Generator 用来参考过往经验）。

    min_restoration：只召回还原度 ≥ 该值的（找高分范例做参考）。
    """
    out = list_entries(run_dir)
    if model is not None:
        out = [e for e in out if e.get("model") == model]
    if gen_mode is not None:
        out = [e for e in out if e.get("gen_mode") == gen_mode]
    if min_restoration is not None:
        out = [e for e in out
               if (e.get("restoration") or 0) >= min_restoration]
    # 按还原度降序（高分范例优先）
    out.sort(key=lambda e: e.get("restoration") or 0, reverse=True)
    if limit is not None:
        out = out[:limit]
    return out


def make_entry(
    *,
    attempt_id: str,
    round: int,
    model: str,
    gen_mode: str | None,
    test_variable: str | None,
    baseline_ref: str | None,
    size: str | None,
    restoration: float | None,
    output_image_refs: list[str],
    lesson_ref: str | None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """构造一条索引 entry（字段对齐 ARCH §3.3.2，剔除大字段）。"""
    return {
        "attempt_id": attempt_id,
        "round": round,
        "model": model,
        "gen_mode": gen_mode,
        "test_variable": test_variable,
        "baseline_ref": baseline_ref,
        "size": size,
        "restoration": restoration,
        "output_image_refs": output_image_refs,
        "lesson_ref": lesson_ref,
        "prompt": prompt,
    }


__all__ = ["append_entry", "list_entries", "make_entry", "recall"]
