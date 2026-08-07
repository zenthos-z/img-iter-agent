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
    """仍用手写 @traceable 的模块（summarizer/router）必须是 chain。

    Generator/Critic 改造为 deepagent 后，其内部 LLM/工具 run 由 langgraph 图自动上报，
    不再用手写 @traceable，故不在本扫描范围内。
    """
    for mod in ("img_iter_agent.agents.summarizer",
                "img_iter_agent.generation.router"):
        rts = _traceable_run_types(_parse(mod))
        assert rts, f"{mod} 期望至少一个 @traceable 子步骤"
        assert all(r == "chain" for r in rts), f"{mod} @traceable 必须 chain，实际 {rts}"


# 注：原 test_generator_improve_prompt_* 验证旧 _improve_prompt 的 @traceable + 结构化输出。
# Generator 改造为 deepagent 后该方法已删除（prompt 改进由 agent 循环 + GeneratorOutput 结构化
# 输出完成），其 trace 由 langgraph 图自动上报，故移除该用例。生成路径的覆盖见 test_generator_agent。
