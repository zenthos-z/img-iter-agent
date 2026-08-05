"""离线验证 @traceable 子步骤：run_type=chain + 结构化 output。

两个层面：
  1. AST 守卫：agents/ + router 的所有 @traceable run_type 必须是 chain（tool 留给出图，
     llm 留给 wrap_openai）——补 test_tracing 的「client.py 必须 tool」守卫。
  2. 实际调用：generator._improve_prompt（无 LLM 确定性补强分支）作为 chain run 上报，
     且返回值是结构化 dict（new_prompt + delta_note）。
"""

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path


def _parse(module_path: str):
    mod = importlib.import_module(module_path)
    return ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))


def _traceable_run_types(tree) -> list[str]:
    """收集所有 @traceable(...) 装饰器里 run_type= 字面量。"""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "traceable":
            for kw in node.keywords:
                if kw.arg == "run_type" and isinstance(kw.value, ast.Constant):
                    out.append(kw.value.value)
    return out


def test_agent_traceables_are_all_chain() -> None:
    """agents/ + router 的 @traceable 必须全是 chain（观测节点内子步骤；tool/llm 各归其位）。"""
    for mod in ("img_iter_agent.agents.critic",
                "img_iter_agent.agents.generator",
                "img_iter_agent.agents.summarizer",
                "img_iter_agent.generation.router"):
        rts = _traceable_run_types(_parse(mod))
        assert rts, f"{mod} 期望至少一个 @traceable 子步骤"
        assert all(r == "chain" for r in rts), f"{mod} @traceable 必须 chain，实际 {rts}"


def _captured_runs(monkeypatch):
    """开启 LangSmith tracing 到进程内 recorder（同 test_tracing.captured_runs 思路）。"""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://ls.test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "img-iter-test")
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    from langsmith import Client
    runs: list[dict] = []
    monkeypatch.setattr(Client, "create_run", lambda self, *a, **k: runs.append(k))
    monkeypatch.setattr(Client, "update_run", lambda self, *a, **k: None)
    return runs


def test_generator_improve_prompt_traces_as_chain_with_structured_output(monkeypatch) -> None:
    """generator._improve_prompt（无 LLM 确定性补强）作为 chain run，返回结构化 output。"""
    captured_runs = _captured_runs(monkeypatch)
    from img_iter_agent.agents.generator import Generator, PriorFeedback
    from img_iter_agent.config import Settings
    from img_iter_agent.generation.client import DmxapiClient
    from img_iter_agent.generation.router import Router

    settings = Settings(_env_file=None, dmxapi_host="http://o.test", dmxapi_key="k")
    # llm=None → _improve_prompt 走确定性补强（不调 LLM/router），最简验证 @traceable + output 结构
    gen = Generator(Router(settings=settings, client=DmxapiClient(settings)), llm=None)
    config = {"configurable": {"thread_id": "subtest"}}
    result = gen._improve_prompt(
        "a chair", PriorFeedback(failed_items=[]), "",
        langsmith_extra={"config": config},
    )

    # 结构化 output（函数返回值）
    assert result.get("new_prompt")
    assert result.get("delta_note")

    # trace 里有一条 chain run，name=generator.improve_prompt
    chain_runs = [r for r in captured_runs
                  if r.get("run_type") == "chain" and r.get("name") == "generator.improve_prompt"]
    assert chain_runs, (
        f"期望 generator.improve_prompt chain run，实际 "
        f"{[(r.get('run_type'), r.get('name')) for r in captured_runs]}"
    )
