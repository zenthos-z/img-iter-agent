"""人工提示词（human hints）存储：运行 loop 时给 generator / critic 追加的自定义提示词。

两种 scope（见 plans/wise-dreaming-shell.md）：
  - loop（临时）：仅本次 loop 生效，存 ``RunMeta.extras["loop_hints"]``（落 ``meta.json``）。
  - sample（持久）：该考题所有 loop 生效，存
    ``<data_root>/human_hints/<bench_id>/<sample_id>.json``。

条目结构：``{"id", "agent"(generator|critic), "text", "scope"(loop|sample)}``。

设计：纯存储读写，无运行时依赖（同 runstore 层）。web ``loop_runner`` 维护内存合并视图
（``LoopHandle.hints``，运行中可改）；``build_loop_context`` 启动时调 ``load_effective_hints``
注入 config，让 CLI / run_loop_auto 路径也带上持久化提示词。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from .runstore import RunStore

AGENTS = ("generator", "critic")
SCOPES = ("loop", "sample")


def _sample_path(data_root: Path | str, bench_id: str, sample_id: str) -> Path:
    return Path(data_root) / "human_hints" / bench_id / f"{sample_id}.json"


def new_hint_id() -> str:
    return "h_" + uuid.uuid4().hex[:8]


def _normalize(hint: dict) -> dict:
    """规整一条 hint：补默认字段、裁剪到合法键、文本去空白。"""
    return {
        "id": hint.get("id") or new_hint_id(),
        "agent": hint.get("agent") if hint.get("agent") in AGENTS else "critic",
        "text": (hint.get("text") or "").strip(),
        "scope": hint.get("scope") if hint.get("scope") in SCOPES else "loop",
    }


def _valid(hint: dict) -> bool:
    return isinstance(hint, dict) and bool((hint.get("text") or "").strip())


# --- sample scope（持久：跨 loop 同考题）---


def load_sample_hints(data_root: Path | str, bench_id: str, sample_id: str) -> list[dict]:
    p = _sample_path(data_root, bench_id, sample_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  损坏文件不阻断 loop
        return []
    raw = data.get("hints") if isinstance(data, dict) else data
    return [_normalize(h) for h in (raw or []) if _valid(h)]


def save_sample_hints(data_root: Path | str, bench_id: str, sample_id: str, hints: list[dict]) -> None:
    """原子写（tmp + replace），避免并发/中断产生半截文件。"""
    p = _sample_path(data_root, bench_id, sample_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hints": [_normalize(h) for h in hints if _valid(h)]}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def add_sample_hint(data_root: Path | str, bench_id: str, sample_id: str, hint: dict) -> dict:
    hints = load_sample_hints(data_root, bench_id, sample_id)
    n = _normalize(hint)
    n["id"] = hint.get("id") or new_hint_id()
    hints.append(n)
    save_sample_hints(data_root, bench_id, sample_id, hints)
    return n


def remove_sample_hint(data_root: Path | str, bench_id: str, sample_id: str, hint_id: str) -> bool:
    hints = load_sample_hints(data_root, bench_id, sample_id)
    left = [h for h in hints if h["id"] != hint_id]
    if len(left) == len(hints):
        return False
    save_sample_hints(data_root, bench_id, sample_id, left)
    return True


# --- loop scope（临时：仅本 loop）---


def load_loop_hints(store: RunStore) -> list[dict]:
    if store.meta is None:
        store._load_meta()
    raw = (store.meta.extras.get("loop_hints") if store.meta else None) or []
    return [_normalize(h) for h in raw if _valid(h)]


def save_loop_hints(store: RunStore, hints: list[dict]) -> None:
    if store.meta is None:
        store._load_meta()
    if store.meta is None:
        return
    store.meta.extras["loop_hints"] = [_normalize(h) for h in hints if _valid(h)]
    store._write_meta(store.meta)


# --- 合并视图 ---


def merge_hints(*groups: list[dict]) -> list[dict]:
    """合并多组 hints，按 id 去重（id 碰撞时后者覆盖）。保留首次出现顺序。"""
    by_id: dict[str, dict] = {}
    for group in groups:
        for h in (group or []):
            if not _valid(h):
                continue
            n = _normalize(h)
            by_id.setdefault(n["id"], n)
    return list(by_id.values())


def load_effective_hints(
    data_root: Path | str, store: RunStore, bench_id: str, sample_id: str
) -> list[dict]:
    """启动时的合并视图：sample 文件 + loop meta（去重）。"""
    return merge_hints(
        load_sample_hints(data_root, bench_id, sample_id),
        load_loop_hints(store),
    )


__all__ = [
    "AGENTS",
    "SCOPES",
    "new_hint_id",
    "load_sample_hints",
    "save_sample_hints",
    "add_sample_hint",
    "remove_sample_hint",
    "load_loop_hints",
    "save_loop_hints",
    "merge_hints",
    "load_effective_hints",
]
