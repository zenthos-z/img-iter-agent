"""Agent 运行事件采集：把 LLM/工具调用钩成结构化事件，append 到 run 目录的 events.jsonl。

供 web「活动流」实时展示「正在跑什么 + 调了什么工具 + 运行详情」。通过 LangChain
BaseCallbackHandler 实现——callbacks 经 LangGraph 的 ContextVar 自动透传（项目自身
LangSmith tracing 即走同一条链），故在 make_loop_config 注入即覆盖 CLI / 技能脚本 / web 三路径。

为什么落文件而非内存：CLI / 批量脚本起的 loop 在独立进程，web 内存 LoopHandle 看不到；
events.jsonl 让 web 直接读文件即可，跨进程可见。seq 采用「文件行号」语义（由读取端
read_events_since 定位 since），emitter 不写 seq——这样即便 web 重启后新 emitter 追加，
since 游标仍单调，不会错位。

事件一行一个 JSON，字段按 type 不同：
  公共: ts, type("tool"|"llm"), round?, phase?, status("running"|"done"|"error"), duration_ms?
  tool running:  tool, node?, args
  tool done:     tool, node?, result
  llm  done:     decided_tools   （不存 messages，避免泄 base64 / 撑爆文件）
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 工具名 → agent 节点（前端分组用；LLM 事件不带 node，分组靠 /status 的 current_node）
_TOOL_NODE: dict[str, str] = {
    "generate_image": "generator",
    "query_rubric": "critic",
    "note_experience": "critic",
    "query_experience": "critic",
}

_BASE64_RE = re.compile(r"data:[^;]+;base64,[A-Za-z0-9+/=]+")
_MAX_STR = 120  # 单字段截断长度
_MAX_LIST = 8  # 列表/字典最多保留前 N 项


def _summarize(value: Any, depth: int = 0) -> Any:
    """递归截断/脱敏：长字符串截断、base64 占位、大列表/字典限长——呼应「包装一下、不全展示」。"""
    if depth > 4:
        return "…"
    if isinstance(value, str):
        v = _BASE64_RE.sub("[base64-image]", value)
        return v if len(v) <= _MAX_STR else v[:_MAX_STR] + "…"
    if isinstance(value, (list, tuple)):
        head = [_summarize(x, depth + 1) for x in value[:_MAX_LIST]]
        if len(value) > _MAX_LIST:
            head.append(f"+{len(value) - _MAX_LIST}")
        return head
    if isinstance(value, dict):
        items = list(value.items())[:_MAX_LIST]
        out = {str(k): _summarize(v, depth + 1) for k, v in items}
        if len(value) > _MAX_LIST:
            out["+more"] = len(value) - _MAX_LIST
        return out
    return value


def _round_phase(metadata: dict | None) -> tuple[int | None, str | None]:
    """从回调 metadata 取 round/phase（make_loop_config 写入、经 langgraph 透传到子 run）。"""
    md = metadata or {}
    r = md.get("round")
    return (int(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else None, md.get("phase"))


class LoopEventEmitter(BaseCallbackHandler):
    """把 agent 的 LLM/工具调用 append 到 ``run_dir/events.jsonl``。

    线程安全（seq/文件写由 _lock 保护）；写失败静默吞掉，绝不拖垮 agent 主流程。
    每个事件 = 一行 JSON，append 后 flush，保证 web 端 tail 读得到。
    """

    def __init__(self, run_dir: Path) -> None:
        self._path = Path(run_dir) / "events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # run_id -> {start, tool, node, round, phase}：配对 start/end + 透传工具名/轮次到 end 事件
        self._pending: dict[str, dict] = {}

    def _emit(self, event: dict) -> None:
        event = {"ts": time.time(), **event}
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:  # noqa: BLE001  写盘失败不阻塞 agent
                pass

    # ---- LLM（ChatOpenAI 走 on_chat_model_start，复用 on_llm_end 收尾）----
    def on_chat_model_start(
        self, serialized, messages, *, run_id=None, parent_run_id=None,
        tags=None, metadata=None, **kwargs,
    ) -> None:
        round_n, phase = _round_phase(metadata)
        self._pending[str(run_id)] = {"start": time.time(), "round": round_n, "phase": phase}
        self._emit({"type": "llm", "status": "running", "round": round_n, "phase": phase})

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None, tags=None,
                   metadata=None, **kwargs) -> None:
        info = self._pending.pop(str(run_id), {})
        round_n, phase = info.get("round"), info.get("phase")
        if round_n is None:  # start 缺席时兜底（理论上不会）
            round_n, phase = _round_phase(metadata)
        decided = self._decided_tools(response)
        self._emit({
            "type": "llm", "status": "done", "round": round_n, "phase": phase,
            "decided_tools": decided,
            "duration_ms": self._dur(info),
        })

    def on_llm_error(self, error, *, run_id=None, metadata=None, **kwargs) -> None:
        info = self._pending.pop(str(run_id), {})
        self._emit({
            "type": "llm", "status": "error", "round": info.get("round"), "phase": info.get("phase"),
            "result": _summarize(str(error)), "duration_ms": self._dur(info),
        })

    # ---- 工具 ----
    def on_tool_start(
        self, serialized_tool, input_str, *, run_id=None, parent_run_id=None,
        tags=None, metadata=None, **kwargs,
    ) -> None:
        name = (serialized_tool or {}).get("name", "") if isinstance(serialized_tool, dict) else ""
        round_n, phase = _round_phase(metadata)
        self._pending[str(run_id)] = {
            "start": time.time(), "tool": name, "node": _TOOL_NODE.get(name),
            "round": round_n, "phase": phase,
        }
        self._emit({
            "type": "tool", "tool": name, "node": _TOOL_NODE.get(name),
            "status": "running", "round": round_n, "phase": phase,
            "args": self._parse_args(input_str),
        })

    def on_tool_end(self, output, *, run_id=None, parent_run_id=None, tags=None,
                    metadata=None, **kwargs) -> None:
        info = self._pending.pop(str(run_id), {})
        result = output if isinstance(output, str) else str(output)
        self._emit({
            "type": "tool", "tool": info.get("tool"), "node": info.get("node"),
            "status": "done", "round": info.get("round"), "phase": info.get("phase"),
            "result": _summarize(result), "duration_ms": self._dur(info),
        })

    def on_tool_error(self, error, *, run_id=None, metadata=None, **kwargs) -> None:
        info = self._pending.pop(str(run_id), {})
        self._emit({
            "type": "tool", "tool": info.get("tool"), "node": info.get("node"),
            "status": "error", "round": info.get("round"), "phase": info.get("phase"),
            "result": _summarize(str(error)), "duration_ms": self._dur(info),
        })

    # ---- 辅助 ----
    @staticmethod
    def _dur(info: dict) -> int | None:
        start = info.get("start")
        return int((time.time() - start) * 1000) if start else None

    @staticmethod
    def _decided_tools(response: Any) -> list[str]:
        """从 on_llm_end 的 response 取模型本轮决定调用的工具名（AIMessage.tool_calls）。"""
        try:
            gens = getattr(response, "generations", None) or []
            if not gens or not gens[0]:
                return []
            msg = gens[0][0].message
            names: list[str] = []
            for tc in getattr(msg, "tool_calls", None) or []:
                name = getattr(tc, "name", None) if not isinstance(tc, dict) else tc.get("name")
                if name:
                    names.append(name)
            return names
        except Exception:  # noqa: BLE001  解析失败不算错（decided_tools 仅辅助）
            return []

    @staticmethod
    def _parse_args(input_str: Any) -> Any:
        """工具入参解析后脱敏。

        ``on_tool_start`` 的 ``input_str`` 在不同 langchain 版本里是 JSON 串或
        Python dict 的 **repr（单引号）**——前者 ``json.loads``，后者 ``ast.literal_eval``
        兜底，最后裸串。优先恢复成 dict 让前端 ``<details>`` 能结构化展示参数。
        """
        if not input_str:
            return {}
        s = input_str if isinstance(input_str, str) else str(input_str)
        try:
            return _summarize(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            pass
        try:  # langchain 常传 Python repr（单引号），literal_eval 安全解析字面量
            v = ast.literal_eval(s)
            return _summarize(v) if isinstance(v, (dict, list)) else _summarize(s)
        except (ValueError, SyntaxError):
            return _summarize(s)


__all__ = ["LoopEventEmitter"]
