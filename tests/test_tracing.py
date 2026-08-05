"""LangSmith 追踪结构的离线验证（不联网、不烧钱）。

验证目标：修复「所有节点都变成 chain、拿不到模型调用信息/耗时」之后，trace 结构正确——

  1. LLM 调用（Critic/Generator/Summarizer 经 OpenAiCompatLlm）→ run_type="llm"，
     带 ls_provider / ls_model_name（模型名）+ usage + 耗时，自动嵌套在 LangGraph 节点下。
     （由 langsmith.wrap_openai 官方机制保证，这里捕获 create_run 断言其元数据。）
  2. 出图调用（generation/client._trace_image_call）→ run_type="tool"（**不是** llm），
     且**原样返回真实响应**给 dispatcher（旧 bug：返回摘要导致生图出空图）。
  3. loop_runner 不再手动拼 RunTree/tracing_context（那种做法会多加一层 chain run 且脆弱）。

捕获原理：langsmith 的 @traceable / wrap_openai 经 Client.create_run / update_run 持久化 run。
把这两个方法 monkeypatch 成 recorder，即可在不联网的前提下拿到完整 run payload
（含 run_type / name / extra.metadata）。/info 连不上是非致命噪音，静默即可。
"""

from __future__ import annotations

import httpx
import pytest

from img_iter_agent.config import Settings
from img_iter_agent.generation.base import ModelFamily


@pytest.fixture
def captured_runs(monkeypatch):
    """开启 LangSmith tracing 到进程内 recorder（无网络）。

    返回一个 list，每个元素是某次 create_run 的 kwargs（含 name/run_type/extra/inputs/outputs）。
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://ls.test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "img-iter-test")
    # 静默 /info 连不上的非致命日志噪音
    import logging
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)

    from langsmith import Client
    runs: list[dict] = []
    monkeypatch.setattr(Client, "create_run", lambda self, *a, **k: runs.append(k))
    monkeypatch.setattr(Client, "update_run", lambda self, *a, **k: None)
    return runs


# ---------------------------------------------------------------------------
# 1. 出图：run_type="tool"，且原样返回真实响应（bug 修复）
# ---------------------------------------------------------------------------

def test_image_gen_traces_as_tool_and_returns_real_response(captured_runs):
    from img_iter_agent.generation.client import _trace_image_call

    real_resp = {"data": [{"b64_json": "ZmFrZQ=="}]}  # dispatcher 期望的真实响应形状

    returned = _trace_image_call(
        ModelFamily.B_DOUBAO, "/v1/responses",
        {"input": "a chair", "image": "b" * 500},  # 长 base64 应被截断成摘要
        lambda: real_resp,
    )

    # (a) 真实响应原样返回——dispatcher 才能拿到 data[].b64_json 落盘（旧 bug 返回摘要→空图）
    assert returned is real_resp

    # (b) trace 里有一条 tool run，名字 image_gen.<family>
    tool_runs = [r for r in captured_runs if r.get("run_type") == "tool"]
    assert any(r["name"].startswith("image_gen.") for r in tool_runs), \
        f"期望 image_gen.* tool run，实际 names={[r.get('name') for r in captured_runs]}"

    # (c) 出图绝不能上报成 llm（否则污染模型调用面板）
    assert not any(r.get("run_type") == "llm" for r in captured_runs)

    # (d) 请求体里的长字段被截断（不把整段 base64 塞进 trace）
    b_run = next(r for r in tool_runs if r["name"].startswith("image_gen.B"))
    req_meta = b_run["extra"]["metadata"]["req"]
    assert req_meta["image"].startswith("<")  # 截断成 "<N chars>"


# ---------------------------------------------------------------------------
# 2. LLM 调用：run_type="llm" + ls_provider/ls_model_name（模型名/usage/耗时由 wrap_openai 保证）
# ---------------------------------------------------------------------------

def _mock_chat_handler(payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)


def test_llm_client_traces_as_llm_with_model_metadata(captured_runs):
    from img_iter_agent.llm.openai_compat import OpenAiCompatLlm

    chat_payload = {
        "id": "c1", "object": "chat.completion", "model": "critic-mm-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "yes"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
    }
    settings = Settings(_env_file=None, dmxapi_host="http://o.test", dmxapi_key="k")
    llm = OpenAiCompatLlm(
        settings, model="critic-mm-1",
        http_client=httpx.Client(transport=_mock_chat_handler(chat_payload)),
    )

    content = llm.complete([{"role": "user", "content": "ping"}])

    # (a) 正常解析返回 content
    assert content == "yes"

    # (b) trace 里有一条 llm run，带 provider + 模型名（这就是旧版缺失的「模型调用信息」）
    llm_runs = [r for r in captured_runs if r.get("run_type") == "llm"]
    assert llm_runs, f"期望至少一条 llm run，实际 {[r.get('run_type') for r in captured_runs]}"
    md = llm_runs[0]["extra"]["metadata"]
    assert md.get("ls_provider") == "openai"
    assert md.get("ls_model_name") == "critic-mm-1"


# ---------------------------------------------------------------------------
# 3. 结构守卫：用 AST 检查实际代码（忽略 docstring/注释里的概念性提及），
#    防止把已删除的错误模式重新加回来。
# ---------------------------------------------------------------------------

def _parse(module_path: str):
    import ast
    import importlib
    from pathlib import Path
    mod = importlib.import_module(module_path)
    return ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))


def _traceable_run_types(tree) -> list[str]:
    """收集所有 @traceable(...) 装饰器里 run_type= 字面量。"""
    import ast
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "traceable":
            for kw in node.keywords:
                if kw.arg == "run_type" and isinstance(kw.value, ast.Constant):
                    out.append(kw.value.value)
    return out


def test_loop_runner_has_no_manual_runtree():
    """loop_runner 不应再手动拼 RunTree / tracing_context（改由 LangGraph 自动 trace +
    metadata.loop_id 关联多轮）。"""
    import ast
    tree = _parse("img_iter_agent.web.services.loop_runner")

    # 不应再定义这些已删除的辅助方法
    removed_methods = {"_new_loop_trace", "_invoke_traced", "_end_loop_trace",
                       "_close_loop_trace"}
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not (removed_methods & defined), f"不应再定义 {removed_methods & defined}"

    # 不应再引用 RunTree / tracing_context（作为名字或属性）
    refs = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            refs.add(n.attr)
    assert "RunTree" not in refs, "loop_runner 不应再引用 RunTree"
    assert "tracing_context" not in refs, "loop_runner 不应再引用 tracing_context"


def test_llm_client_uses_wrap_openai():
    """OpenAiCompatLlm 必须用官方 wrap_openai（不得手写 @traceable 套 LLM 调用）。"""
    import ast
    tree = _parse("img_iter_agent.llm.openai_compat")

    # 必须调用 wrap_openai(...)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "wrap_openai" in calls, "LLM client 必须用 langsmith.wrap_openai"

    # llm 模块不应出现任何 @traceable（LLM 追踪已交给 wrap_openai）
    assert _traceable_run_types(tree) == [], "llm 模块不应手写 @traceable"


def test_image_trace_run_types_are_tool_only():
    """generation/client.py 里所有 @traceable 的 run_type 必须是 tool（不得有 llm）。"""
    rts = _traceable_run_types(_parse("img_iter_agent.generation.client"))
    assert rts, "期望至少一个 @traceable 出图埋点"
    assert all(r == "tool" for r in rts), f"出图埋点 run_type 必须 tool，实际 {rts}"


# ---------------------------------------------------------------------------
# 4. 端到端：跑一轮真实 graph，断言完整 trace 层级——
#    chain(graph) ⊃ chain(节点) ⊃ run_type="llm"(Critic 调用)，且出图是 tool run。
#    LLM 走 OpenAiCompatLlm(wrap_openai + mock transport)，出图走真 Router + mock executor。
# ---------------------------------------------------------------------------

def test_graph_trace_hierarchy(captured_runs, tmp_path, bench_id):
    import json
    from pathlib import Path

    from langgraph.checkpoint.memory import InMemorySaver

    from img_iter_agent.agents.critic import Critic
    from img_iter_agent.agents.generator import Generator
    from img_iter_agent.agents.summarizer import Summarizer
    from img_iter_agent.config import Settings
    from img_iter_agent.data.benchmark import load_benchmark
    from img_iter_agent.data.runstore import RunStore
    from img_iter_agent.generation.client import DmxapiClient
    from img_iter_agent.generation.router import Router
    from img_iter_agent.llm.openai_compat import OpenAiCompatLlm
    from img_iter_agent.pipeline.graph import build_graph

    project_root = Path(__file__).resolve().parents[1]
    settings = Settings(_env_file=None, data_root=tmp_path,
                        dmxapi_host="http://o.test", dmxapi_key="k",
                        model_seedream_pro="seedream-test", model_gpt_image="gpt-image-test",
                        model_gemini_image="gemini-test", model_qwen_image="qwen-test")
    (tmp_path / "benchmarks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "benchmarks" / bench_id).symlink_to(
        project_root / "data" / "benchmarks" / bench_id)
    lb = load_benchmark(bench_id, settings=settings)
    bench = lb.bench

    # Critic LLM：mock openai transport，统一回同时含 judgments/score 的 JSON（解析侧各取所需）
    def chat_handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps({"judgments": [{"id": "X1", "passed": True, "reason": "ok"}],
                              "score": 0.8, "reason": "ok"})
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "model": "critic-mm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    critic_llm = OpenAiCompatLlm(
        settings, model="critic-mm",
        http_client=httpx.Client(transport=httpx.MockTransport(chat_handler)),
    )

    # 出图：真 Router + mock executor（走 _trace_image_call，回带 b64 的真实响应形状）
    class _MockExec:
        def post_json(self, url, headers, body):
            return {"data": [{"b64_json": "ZmFrZQ=="}]}

        def post_multipart(self, url, headers, data, files):
            return {"data": [{"b64_json": "ZmFrZQ=="}]}

    router = Router(settings=settings, client=DmxapiClient(settings, executor=_MockExec()))
    store = RunStore.create("tracetest", bench_id, model="critic-mm",
                            settings=settings, note="t")
    # Generator llm=None → 确定性 prompt（不引入额外 llm run，结构更干净）
    app = build_graph(
        bench=lb, run_store=store,
        generator=Generator(router, llm=None),
        critic=Critic(critic_llm, bench=bench),
        summarizer=Summarizer(),
        sample_id="s001", checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "tracetest"}}
    app.invoke({"round": 0, "model": "critic-mm", "bench_id": bench_id,
                "sample_id": "s001", "run_id": "tracetest"}, config=cfg)

    # 断言：一次 invoke 里 chain / llm / tool 三类 run 同时出现
    runs = captured_runs
    by_type: dict[str, list[dict]] = {}
    for r in runs:
        by_type.setdefault(r.get("run_type"), []).append(r)
    assert {"chain", "llm", "tool"} <= set(by_type), \
        f"应同时出现 chain/llm/tool，实际 {sorted(by_type)}"

    # 断言：llm run 嵌在 chain run 之下（嵌套关系编码在 dotted_order：子 run 的前缀含父 run）
    def _dotted(r):
        return r.get("dotted_order") or ""
    chain_dotted = [_dotted(r) for r in by_type["chain"] if _dotted(r)]
    nested_llm = [
        llm for llm in by_type["llm"]
        if any(_dotted(llm).startswith(d + ".") for d in chain_dotted)
    ]
    assert nested_llm, (
        "至少一条 llm run 应嵌套在 chain(节点) run 下，"
        f"chain_dotted={chain_dotted[:3]}, llm_dotted={[_dotted(r) for r in by_type['llm']][:3]}"
    )

    # 断言：llm run 带模型元数据（旧版缺失的「模型调用信息」）
    md = nested_llm[0]["extra"]["metadata"]
    assert md.get("ls_provider") == "openai"
    assert md.get("ls_model_name") == "critic-mm"

    # 断言：出图 tool run 名字 image_gen.*
    assert any(r["name"].startswith("image_gen.") for r in by_type["tool"])
