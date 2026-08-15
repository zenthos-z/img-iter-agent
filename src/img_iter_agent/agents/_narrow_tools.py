"""窄域 agent 的工具收敛：剥掉 deepagents 默认注入的 fs/subagent/shell 工具。

背景：`create_deep_agent` 默认会注入一批「通用 coding agent」工具（ls/glob/grep/read_file/
write_file/edit_file/delete/execute/task…，由 FilesystemMiddleware + SubAgentMiddleware 提供）。
对本项目的三个窄域 agent（Generator/Critic/ExperienceDistiller）这些工具**无用且有害**——
实测生成 agent 会在空 sandbox 里反复 ls/glob/read_file 找图（参考图其实已附在消息里），
单轮 17~45 步剧烈波动，配上 `generate_round` 较低的 recursion_limit 就抛 GraphRecursionError，
被 `except` 吞成兜底 prompt（agent 实际没起作用）。

解法：通过 `create_deep_agent(..., middleware=[NarrowToolsMiddleware()])`，在 `wrap_model_call`
里过滤 `request.tools`（模型实际可见的工具集），让窄域 agent 只看到自己的自定义工具。
- 这是 deepagents 官方的 `middleware=` 扩展点（非 monkeypatch）；框架自身的
  `_ToolExclusionMiddleware` 走的也是同一条 `wrap_model_call` 过滤路径。
- 只过滤「模型可见集」；ToolNode 注册表不变（无害，模型不调就不会触发）。
- 不依赖 `register_harness_profile`——后者对「预构建的 ChatOpenAI + 自定义 base_url（dmxapi）」
  因 provider 推导不出而匹配不到（deepagents 自身代码标注的已知失败模式）。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.tools import BaseTool

# deepagents 默认注入、但窄域 agent 不需要的内置工具名。
# （generate_image / query_experience / query_rubric /
#   list_runs / query_run / query_dim_history / query_conclusions 是各 agent 自己注册的，不在此列。）
_NARROW_EXCLUDED: frozenset[str] = frozenset({
    "ls", "glob", "grep", "read_file", "read_file_lines", "read_media",
    "write_file", "edit_file", "delete", "execute", "task", "patch", "batch",
})

# Generator 专用窄集：放回 read_file——deepagents SkillsMiddleware 的渐进披露硬性依赖它
# （skills= 只注入 skill 名+描述+路径，要 read_file SKILL.md 才拿得到正文；无 embed 选项）。
# Generator 用 FilesystemBackend + permissions 把 read_file 钉死在本 benchmark 技能包目录内（无图、不可越界），
# 故放回 read_file 不会重演旧版「ls/glob/read_file 在空 sandbox 找图」的死循环（ls/glob/grep 仍剥）。
_GENERATOR_NARROW_EXCLUDED: frozenset[str] = _NARROW_EXCLUDED - {"read_file"}


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """从 BaseTool 或 dict 工具描述里取名字。"""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class NarrowToolsMiddleware(AgentMiddleware[Any, Any, Any]):
    """剥掉 deepagents 内置 fs/subagent/shell 工具，让窄域 agent 只看自定义工具。

    放在 middleware 栈靠后位置（create_deep_agent 会把用户 middleware 接在核心
    tool-injecting middleware 之后），在模型请求前过滤 request.tools。
    """

    def __init__(self, excluded: frozenset[str] = _NARROW_EXCLUDED) -> None:
        self._excluded = excluded

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(tools=filtered)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(tools=filtered)
        return await handler(request)


def narrow_tools_middleware(
    excluded: frozenset[str] = _NARROW_EXCLUDED,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """便捷工厂：返回 `[NarrowToolsMiddleware(excluded=...)]`，供 create_deep_agent(middleware=...) 直接用。

    默认剥全集（Critic/Distiller）；Generator 传 ``_GENERATOR_NARROW_EXCLUDED``（保留 read_file）。
    """
    return [NarrowToolsMiddleware(excluded=excluded)]


def invoke_with_retry(
    agent: Any, payload: dict[str, Any], *, config: Any, label: str, retries: int = 5,
    require_structured: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    """跑一个 deepagent，失败（异常 **或** 无结构化输出）时重试，返回 (result, ok)。

    窄域 agent 经 dmxapi 网关跑 LLM：偶发会撞上 ① 网关瞬时错误（5xx/超时/限流/连接中断）
    ② 模型一次性不收敛（GraphRecursionError 或没产出 structured_response）。这类
    一次性 flake 被各 agent 的 `except` 直接吞成兜底（critic→全 0 安默认、generator→
    兜底 prompt），污染整轮数据。重试 + 退避扛过网关抖动（dmxapi 实测会间歇性 SSL EOF）；
    仍失败才退兜底。

    - ``require_structured=True``（默认，renovation/generator/critic）：ok=True 仅当拿到非空
      ``structured_response``。
    - ``require_structured=False``（全工具 authoring agent，无 ``response_format``）：ok=True 当
      invoke 未抛异常（产物是 write_file 的副作用，无结构化输出可验）——调用方另行检查落盘文件。
    """
    for i in range(retries + 1):
        try:
            res = agent.invoke(payload, config=config)
            if require_structured:
                sr = res.get("structured_response") if isinstance(res, dict) else None
                if sr is not None:
                    return res, True
                if i < retries:
                    print(f"[{label}] 无结构化输出，重试 {i + 1}/{retries}", flush=True)
                else:
                    print(f"[{label}] 重试仍无结构化输出，退兜底", flush=True)
            else:
                return res, True
        except Exception as e:  # noqa: BLE001  重试逻辑要吞所有异常
            msg = str(e).replace("\n", " ")[:200]
            if i < retries:
                print(f"[{label}] invoke 异常({type(e).__name__}: {msg})，重试 {i + 1}/{retries}",
                      flush=True)
            else:
                print(f"[{label}] 重试仍异常({type(e).__name__}: {msg})，退兜底", flush=True)
        if i < retries:
            time.sleep(min(2.0 * (2 ** i), 30.0))  # 指数退避 2,4,8,16,30s：扛 dmxapi 持续性连接抖动窗口（固定 2s 只覆盖 ~8s 窗口，撞持续抖动会 4 连失败退兜底）
    return None, False


# 窄域 agent 的递归上限。剥掉乱逛工具后，generator/critic ~9 步收敛；
# 给充裕余量，避免 LLM 偶发多调几次工具就撞上限（旧值 25/30/40 太低）。
AGENT_RECURSION_LIMIT: int = 50
# 蒸馏器要跨多 run × 多维度查（list_runs + query_run×N + query_dim_history×6 +
# query_conclusions×N），叠加模型并行重复调用 + 综合，步骤远多于 generator/critic。
DISTILLER_RECURSION_LIMIT: int = 120


__all__ = [
    "AGENT_RECURSION_LIMIT",
    "DISTILLER_RECURSION_LIMIT",
    "_GENERATOR_NARROW_EXCLUDED",
    "NarrowToolsMiddleware",
    "invoke_with_retry",
    "narrow_tools_middleware",
]
