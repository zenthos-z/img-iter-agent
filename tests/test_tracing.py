"""LangSmith 追踪结构的离线验证（不联网、不烧钱）。

验证目标：trace 结构正确——

  1. 出图调用（generation/client._trace_image_call）→ run_type="tool"（**不是** llm），
     且**原样返回真实响应**给 dispatcher（旧 bug：返回摘要导致生图出空图）。
  2. loop_runner 不再手动拼 RunTree/tracing_context（那种做法会多加一层 chain run 且脆弱）。
  3. 端到端跑一轮 graph：chain(graph) ⊃ chain(节点) ⊃ llm/tool run，层级正确。
     Generator/Critic 现为 deepagent，其 LLM 调用经 ChatOpenAI（原生 Runnable）自动上报为
     run_type="llm" 并嵌套在节点 run 下；出图仍是 tool run。

捕获原理：langsmith 经 Client.create_run / update_run 持久化 run。把这两个方法 monkeypatch
成 recorder，即可在不联网的前提下拿到完整 run payload（含 run_type / name / extra.metadata）。
"""

from __future__ import annotations

import pytest

from img_iter_agent.config import Settings
from img_iter_agent.generation.base import ModelFamily


@pytest.fixture
def captured_runs(monkeypatch):
    """开启 LangSmith tracing 到进程内 recorder（无网络）。

    返回一个 list，每个元素是某次 create_run 的 kwargs（含 name/run_type/extra/inputs/outputs）。
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
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

    # (d) 请求摘要记在 run inputs（LangSmith UI 的 Input 面板直接可见，不再是 null），
    #     且长字段被截断（不把整段 base64 塞进 trace）
    b_run = next(r for r in tool_runs if r["name"].startswith("image_gen.B"))
    req_in = b_run["inputs"]["req"]
    assert req_in["image"].startswith("<")  # 截断成 "<N chars>"


def test_image_gen_truncates_nested_base64(captured_runs):
    """嵌套在 list/dict 里的 base64（D 族 contents[].parts[].inline_data.data）也要截断——
    早期版本只截顶层 str，base64 整段进 trace 体积爆炸。inputs 里能看到 prompt 与截断占位。"""
    from img_iter_agent.generation.base import ModelFamily
    from img_iter_agent.generation.client import _trace_image_call

    body = {"contents": [{"role": "user", "parts": [
        {"text": "a chair"},
        {"inline_data": {"mime_type": "image/jpeg", "data": "Q" * 5000}},
    ]}]}
    _trace_image_call(ModelFamily.D_GEMINI, "/v1beta/models/m:generateContent", body,
                      lambda: {"candidates": []})

    d_run = next(r for r in captured_runs if r["name"].startswith("image_gen.D"))
    req_in = d_run["inputs"]["req"]
    data = req_in["contents"][0]["parts"][1]["inline_data"]["data"]
    assert data == "<5000 chars>"  # 深度截断，不是 5000 个字符的 base64
    assert req_in["contents"][0]["parts"][0]["text"] == "a chair"


# ---------------------------------------------------------------------------
# 2. 结构守卫：用 AST 检查实际代码（忽略 docstring/注释里的概念性提及），
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
    """loop_runner 不直接拼 RunTree / tracing_context：每轮 trace 的创建/收尾由 pipeline.runner 的
    create_round_trace + round_trace_context 负责（把 RunTree 逻辑关在 runner.py 里），loop_runner 只调用它们。"""
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


def test_image_trace_run_types_are_tool_only():
    """generation/client.py 里所有 @traceable 的 run_type 必须是 tool（不得有 llm）。"""
    rts = _traceable_run_types(_parse("img_iter_agent.generation.client"))
    assert rts, "期望至少一个 @traceable 出图埋点"
    assert all(r == "tool" for r in rts), f"出图埋点 run_type 必须 tool，实际 {rts}"


# ---------------------------------------------------------------------------
# 3. 端到端：跑一轮真实 graph，断言完整 trace 层级——
#    chain(graph) ⊃ chain(节点) ⊃ llm/tool run，且出图是 tool run。
#    Generator/Critic 用 FakeToolCallingChatModel 驱动（原生 Runnable → llm run），
#    出图走真 Router + mock executor（_trace_image_call → tool run）。
# ---------------------------------------------------------------------------

def test_graph_trace_hierarchy(captured_runs, tmp_path, bench_id):
    from pathlib import Path

    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from img_iter_agent.agents.critic import Critic
    from img_iter_agent.agents.generator import Generator
    from img_iter_agent.data.benchmark import load_benchmark
    from img_iter_agent.data.runstore import RunStore
    from img_iter_agent.generation.client import DmxapiClient
    from img_iter_agent.generation.router import Router
    from img_iter_agent.pipeline.graph import build_graph
    from tests._fakes import FakeToolCallingChatModel

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

    # Critic：fake tool-calling chat model 驱动（回 CriticAgentOutput 具名字段结构化输出）
    critic_args: dict = {}
    for d in bench.score_dimensions:
        if d.scoring_type == "binary":
            cl = lb.sample("s001").spec.checklist.get(d.dim, [])
            cl = cl if isinstance(cl, list) else []
            critic_args[d.dim] = [
                {"id": it.id, "passed": True, "reason": "mock"} for it in cl
            ]
        else:
            critic_args[d.dim] = {"value": 0.8, "reason": "ok"}
    critic_chat = FakeToolCallingChatModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "CriticAgentOutput", "type": "tool_call",
                                           "id": "c1", "args": critic_args}]),
    ])

    # 出图：真 Router + mock executor（走 _trace_image_call，回带 b64 的真实响应形状）
    class _MockExec:
        def post_json(self, url, headers, body):
            return {"data": [{"b64_json": "ZmFrZQ=="}]}

        def post_multipart(self, url, headers, data, files):
            return {"data": [{"b64_json": "ZmFrZQ=="}]}

    router = Router(settings=settings, client=DmxapiClient(settings, executor=_MockExec()))
    store = RunStore.create("tracetest", bench_id, model="critic-mm",
                            settings=settings, note="t")
    # Generator：fake tool-calling chat model 驱动（确定性出图 + 结构化输出）
    gen_chat = FakeToolCallingChatModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "generate_image", "type": "tool_call",
                                           "id": "g1",
                                           "args": {"prompt": "p", "size": "2K",
                                                    "reference_images": ["target"]}}]),
        AIMessage(content="", tool_calls=[{"name": "GeneratorOutput", "type": "tool_call",
                                           "id": "s1",
                                           "args": {"prompt": "p", "delta_note": "d"}}]),
    ])
    app = build_graph(
        bench=lb, run_store=store,
        generator=Generator(router, chat_model=gen_chat, skills_dir=None),
        critic=Critic(critic_chat, bench=bench),
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

    # 断言：嵌套的 llm run 带模型元数据（ls_provider/ls_model_name）
    #（Generator/Critic 现走 ChatOpenAI/本测试的 fake model；OpenAiCompatLlm 的模型元数据
    # 由专门的 test_llm_client_traces_as_llm_with_model_metadata 覆盖。）
    assert nested_llm, "至少一条 llm run 嵌套在 chain(节点) run 下"

    # 断言：出图 tool run 名字 image_gen.*
    assert any(r["name"].startswith("image_gen.") for r in by_type["tool"])

    # 断言：router.generate 的 inputs 可见完整请求摘要（prompt + 参考图路径 + 尺寸等），
    # 而不是 null——参考图传没传要在 LangSmith UI 里直接能看到。
    # （本测试走 generate_image 工具路径：image_edit 的参考图由 agent 自决，fake agent
    # 显式传 ['target'] → 工具解析成 target.jpg 真实路径出现在 req_summary 里。）
    rg = next(r for r in captured_runs if r["name"] == "router.generate")
    rs = rg["inputs"]["req_summary"]
    assert rs["prompt"]
    assert rs["reference_images"], "兜底出图应带 target 参考图，且路径要出现在 inputs"
    assert rs["reference_images"][0].endswith("target.jpg")


# ---------------------------------------------------------------------------
# 4. 每轮一条独立 trace：create_round_trace 产独立 trace_id，名字标 bench/sample/loop/round
# ---------------------------------------------------------------------------

def test_round_trace_name_format():
    """_round_trace_name：common case 不带后缀；loop_id 带 batch/tag 后缀时原样追加。"""
    from img_iter_agent.pipeline.runner import _round_trace_name as n

    assert n("furniture_product_whitebg", "s001",
             "furniture_product_whitebg-s001", 3) == "furniture_product_whitebg/s001 R3"
    assert n("furniture_product_whitebg", "s003",
             "furniture_product_whitebg-s003-exp6", 5) == "furniture_product_whitebg/s003-exp6 R5"


def test_each_round_is_separate_trace(captured_runs):
    """每一轮各建一个独立 root（独立 trace_id）→ LangSmith 里是分开的多条 trace，而非一条 loop trace。

    名字标 bench/sample/loop/round；loop_id/round 落在 extra.metadata（RunTree 无独立 metadata 字段，
    走 extra.metadata），tags 含 loop:<id>/round:<n>，便于按 loop 过滤回放。round_trace_context
    退出收尾（end+patch，patch 走 update_run 被 captured_runs fixture 静默，不触网）。
    """
    from img_iter_agent.pipeline.runner import create_round_trace, round_trace_context

    loop_id = "furniture_product_whitebg-s001"
    bench, sample, model = "furniture_product_whitebg", "s001", "gemini-test"

    r1 = create_round_trace(loop_id, bench, sample, 1, model, "first")
    with round_trace_context(r1):  # 模拟一轮 invoke（空 body；真实 graph run 会嵌套其下）
        pass
    r2 = create_round_trace(loop_id, bench, sample, 2, model, "resume")
    with round_trace_context(r2):
        pass

    roots = [r for r in captured_runs if r.get("name")]
    by_name = {r["name"]: r for r in roots}
    assert "furniture_product_whitebg/s001 R1" in by_name
    assert "furniture_product_whitebg/s001 R2" in by_name

    # 两轮 root 各自独立 trace_id（不同）→ 两条独立 trace
    assert by_name["furniture_product_whitebg/s001 R1"]["trace_id"] != \
           by_name["furniture_product_whitebg/s001 R2"]["trace_id"]

    # metadata 落在 extra.metadata
    md1 = by_name["furniture_product_whitebg/s001 R1"]["extra"]["metadata"]
    assert md1["loop_id"] == loop_id and md1["round"] == 1 and md1["bench_id"] == bench
    # tags 关联 loop / round
    assert "loop:furniture_product_whitebg-s001" in by_name["furniture_product_whitebg/s001 R1"]["tags"]
    assert "round:2" in by_name["furniture_product_whitebg/s001 R2"]["tags"]


def test_create_round_trace_noop_when_tracing_off(monkeypatch):
    """LangSmith 未开启时 create_round_trace 返回 None（不建 RunTree、不触网），
    round_trace_context(None) 直接 yield——整条 per-round trace 机制静默 no-op。"""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    from img_iter_agent.pipeline.runner import create_round_trace, round_trace_context

    root = create_round_trace("b-s", "b", "s", 1, "m")
    assert root is None
    with round_trace_context(root):  # None → 不嵌套、不报错
        pass
