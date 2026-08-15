"""Agent 活动流事件采集（LoopEventEmitter）+ 读取端（read_events_since）的离线验证。

不联网、不烧钱：直接拿真 ``BaseChatModel``（FakeToolCallingChatModel）和真 ``@tool``
以 ``callbacks=[emitter]`` 跑一次，验证 handler 的回调签名**与 LangChain 实际调用一致**
（on_chat_model_start↔on_llm_end、on_tool_start↔on_tool_end 都真触发），并断言
events.jsonl 落地的事件结构、脱敏、行号游标语义。

核心保证（呼应需求「包装一下、不全展示」）：
  - 工具调用记 running/done 两条，带工具名 + 结果摘要；
  - LLM 调用的 messages **绝不落盘**（只记 decided_tools 计数），base64 / 长串被截断/占位；
  - seq 用文件行号语义（read_events_since 的 since 游标单调）。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from img_iter_agent.agent_events import LoopEventEmitter
from img_iter_agent.web.services.data_access import read_events_since
from tests._fakes import FakeToolCallingChatModel


def _read(run_dir: Path) -> list[dict]:
    """读回 events.jsonl 全量（坏行跳过，与生产读取端一致）。"""
    p = Path(run_dir) / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# 1. 真 @tool + emitter 回调：跑一次工具，断言 running/done 两条 + 字段
# ---------------------------------------------------------------------------

def test_tool_events_via_real_langchain_tool(tmp_path):
    emitter = LoopEventEmitter(tmp_path)

    @tool
    def add(a: int, b: int) -> int:
        """两数相加（测试用工具）。"""
        return a + b

    add.invoke(
        {"a": 1, "b": 2},
        config={"callbacks": [emitter], "metadata": {"round": 1, "phase": "critic"}},
    )

    events = _read(tmp_path)
    states = [(e["type"], e["status"]) for e in events]
    assert ("tool", "running") in states, f"缺 tool running，实际 {states}"
    assert ("tool", "done") in states, f"缺 tool done，实际 {states}"

    done = next(e for e in events if e["type"] == "tool" and e["status"] == "done")
    assert done["tool"] == "add"
    assert "3" in str(done["result"])  # 工具返回值进 result 摘要
    # config.metadata 里的 round/phase 经 LangChain 透传到回调 metadata（_round_phase 提取）
    assert done["round"] == 1
    assert done["phase"] == "critic"
    assert done["duration_ms"] is not None  # start↔end 配对成功 → 有耗时


# ---------------------------------------------------------------------------
# 2. 真 ChatModel + emitter 回调：on_chat_model_start↔on_llm_end，decided_tools 提取
# ---------------------------------------------------------------------------

def test_llm_events_extract_decided_tools(tmp_path):
    emitter = LoopEventEmitter(tmp_path)
    chat = FakeToolCallingChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "generate_image", "type": "tool_call", "id": "g1",
            "args": {"prompt": "一把椅子"},
        }]),
    ])

    chat.invoke("hi", config={"callbacks": [emitter],
                              "metadata": {"round": 1, "phase": "generator"}})

    events = _read(tmp_path)
    states = [(e["type"], e["status"]) for e in events]
    assert ("llm", "running") in states, f"缺 llm running（on_chat_model_start），实际 {states}"
    assert ("llm", "done") in states, f"缺 llm done（on_llm_end），实际 {states}"

    done = next(e for e in events if e["type"] == "llm" and e["status"] == "done")
    assert "generate_image" in done["decided_tools"], \
        f"应从 AIMessage.tool_calls 提取 generate_image，实际 {done.get('decided_tools')}"
    assert done["round"] == 1


# ---------------------------------------------------------------------------
# 3. 脱敏：base64 占位 + 长串截断；绝不落 messages
# ---------------------------------------------------------------------------

def test_base64_and_long_strings_are_redacted(tmp_path):
    emitter = LoopEventEmitter(tmp_path)

    @tool
    def echo(payload: str) -> str:
        """原样返回（测试长串/base64 脱敏）。"""
        return payload

    b64 = "data:image/png;base64," + "A" * 2000
    echo.invoke({"payload": b64}, config={"callbacks": [emitter]})

    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    # 原始 base64 payload 绝不能进文件（占位成 [base64-image]）
    assert "AAAA" not in raw, "base64 payload 泄漏进 events.jsonl"
    assert "[base64-image]" in raw

    running = next(e for e in _read(tmp_path)
                   if e["type"] == "tool" and e["status"] == "running")
    assert running["args"]["payload"] == "[base64-image]"


def test_summarize_caps_long_string_and_lists():
    from img_iter_agent.agent_events import _summarize

    long = "x" * 500
    assert _summarize(long).endswith("…")
    assert len(_summarize(long)) <= 122  # 120 + 省略号

    big = _summarize(list(range(50)))
    assert isinstance(big, list)
    assert big[-1] == "+42"  # 前 8 项 + 尾标「+42」

    bigd = _summarize({f"k{i}": i for i in range(50)})
    assert bigd["+more"] == 42


def test_llm_events_never_persist_messages(tmp_path):
    """LLM 事件只记 decided_tools，不存 messages（避免泄 base64 / 撑爆文件）。"""
    emitter = LoopEventEmitter(tmp_path)
    chat = FakeToolCallingChatModel(responses=[AIMessage(content="secret answer")])
    chat.invoke("hi", config={"callbacks": [emitter]})

    for e in _read(tmp_path):
        if e["type"] == "llm":
            assert "messages" not in e
            assert "secret answer" not in json.dumps(e, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. 错误路径：on_tool_error / on_llm_error 写 status=error
# ---------------------------------------------------------------------------

def test_error_paths(tmp_path):
    emitter = LoopEventEmitter(tmp_path)
    # 直接调回调（不经 LangChain），验证 error 事件结构
    emitter.on_tool_start({"name": "generate_image"}, "{}",
                          run_id="t1", metadata={"round": 2, "phase": "generator"})
    emitter.on_tool_error(ValueError("boom"), run_id="t1")

    emitter.on_chat_model_start({}, [[]], run_id="l1",
                                metadata={"round": 2, "phase": "critic"})
    emitter.on_llm_error(RuntimeError("llm boom"), run_id="l1")

    events = _read(tmp_path)
    tool_err = next(e for e in events if e["type"] == "tool" and e["status"] == "error")
    assert tool_err["tool"] == "generate_image"
    assert "boom" in tool_err["result"]
    assert tool_err["node"] == "generator"  # _TOOL_NODE 映射
    llm_err = next(e for e in events if e["type"] == "llm" and e["status"] == "error")
    assert "llm boom" in llm_err["result"]


# ---------------------------------------------------------------------------
# 5. read_events_since：行号游标 since 过滤 + 坏行跳过 + 文件缺失
# ---------------------------------------------------------------------------

def test_read_events_since_filtering_and_bad_lines(tmp_path):
    p = tmp_path / "events.jsonl"
    # 6 行：3 合法 tool 事件 + 1 坏 JSON + 1 空行 + 1 合法 llm 事件
    lines = [
        json.dumps({"type": "tool", "status": "running", "tool": "a"}),
        json.dumps({"type": "tool", "status": "done", "tool": "a"}),
        "{NOT JSON",
        "",
        json.dumps({"type": "llm", "status": "done"}),
        json.dumps({"type": "tool", "status": "done", "tool": "b"}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 全量：since=0 → 跳过坏行/空行后 4 条合法，total 计所有 6 行
    all_events, total = read_events_since(tmp_path, since=0)
    assert total == 6
    assert len(all_events) == 4

    # 游标推进：since=2 → 只回第 3 行起的合法事件（坏行被跳过但不回填）
    tail, total2 = read_events_since(tmp_path, since=2)
    assert total2 == 6
    # since=2 跳过前 2 行合法 tool → 剩 2 条合法（llm + tool b）
    assert len(tail) == 2
    assert tail[-1]["tool"] == "b"


def test_read_events_since_missing_file(tmp_path):
    events, total = read_events_since(tmp_path / "nope", since=0)
    assert events == []
    assert total == 0


# ---------------------------------------------------------------------------
# 6. 跨轮复用同一 emitter：events.jsonl 持续追加，since 游标单调不错位
# ---------------------------------------------------------------------------

def test_emitter_appends_across_writes(tmp_path):
    """模拟「同一 loop 多轮」复用同一 emitter：多次写互不覆盖，read_events_since 增量正确。"""
    emitter = LoopEventEmitter(tmp_path)
    emitter.on_tool_start({"name": "generate_image"}, '{"prompt":"r1"}',
                          run_id="r1", metadata={"round": 1, "phase": "generator"})
    emitter.on_tool_end("ok", run_id="r1")

    first, total1 = read_events_since(tmp_path, since=0)
    assert total1 == 2
    assert len(first) == 2

    # 第二轮追加（不覆盖）
    emitter.on_tool_start({"name": "query_rubric"}, "{}",
                          run_id="r2", metadata={"round": 2, "phase": "critic"})
    emitter.on_tool_end("ok", run_id="r2")

    # 用上一轮 total 当 since → 只拿新增的 2 条
    delta, total2 = read_events_since(tmp_path, since=total1)
    assert total2 == 4
    assert len(delta) == 2
    assert delta[0]["tool"] == "query_rubric"
    assert delta[0]["round"] == 2
