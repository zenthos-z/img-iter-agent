"""测试用 BaseChatModel：让 deepagents 在离线（无 dmxapi key）下可驱动。

`langchain_core` 自带的 `FakeMessagesListChatModel` 不实现 `bind_tools`，而 deepagents
构造时必须 `model.bind_tools(...)`。故自写一个最小 BaseChatModel：按预设 `AIMessage` 序列
出队（每条可带 `tool_calls` 模拟工具调用），`bind_tools` / `with_structured_output` 返回 self。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# create_deep_agent(response_format=...) 用的结构化输出 schema 名。
# 生产路径走 ProviderStrategy(json_schema)：最终结构化输出从 AIMessage.content 解析
# （langchain create_agent 只在「无 tool_calls 的最终消息」时解析 content——见
# agents/factory.py 的 ProviderStrategy 分支）。为兼容沿用了旧 ToolStrategy 写法
# （tool_calls=[{name:<schema 名>, args:{...}}]）的测试预设，fake 识别这些 name 并把 args
# 挪到 content；业务工具调用（generate_image 等）原样保留为 tool_call 出队。
_STRUCTURED_SCHEMA_NAMES = frozenset({"CriticAgentOutput", "GeneratorOutput", "RenovationPlan"})


class FakeToolCallingChatModel(BaseChatModel):
    """循环吐预设 AIMessage 的假 chat model，支持 bind_tools。

    构造：`FakeToolCallingChatModel(responses=[AIMessage(...), ...])`。
    - responses: list[AIMessage]，按顺序出队；耗尽后重复最后一个。
      每条 AIMessage 可带 tool_calls=[{"name","args","id","type"}] 模拟工具调用，
      或纯 content 作为终止回答。
    - 结构化输出（ProviderStrategy/json_schema 路径）：测试预设仍可用旧 ToolStrategy 写法
      `tool_calls=[{name:<schema 名>, args:{...}}]` 描述最终交付；fake 自动把它的 args 挪到
      content（create_agent 在无 tool_calls 时从 content 解析）。业务工具调用（generate_image
      等）不受影响、照常作为 tool_call 出队由 agent 执行。
    - bind_tools / with_structured_output 返回 self：忽略 schema，照样吐预设序列。
    - calls: 记录每次 _generate 收到的 messages，便于断言（如初始 HumanMessage 的多模态内容）。
    """

    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "fake-tool-calling"

    def _next_response(self) -> AIMessage:
        if not self.responses:
            return AIMessage(content="")
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    def _generate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        resp = self._next_response()
        # ProviderStrategy 路径适配：name 为结构化输出 schema 的 tool_call → args 挪到 content
        # （create_agent 只对无 tool_calls 的最终消息解析 content 为结构化输出）。
        tcs = getattr(resp, "tool_calls", None) or []
        if tcs:
            tc0 = tcs[0]
            name = tc0.get("name") if isinstance(tc0, dict) else getattr(tc0, "name", "")
            if name in _STRUCTURED_SCHEMA_NAMES:
                args = tc0.get("args") if isinstance(tc0, dict) else getattr(tc0, "args", {})
                resp = AIMessage(content=json.dumps(args, ensure_ascii=False))
        return ChatResult(generations=[ChatGeneration(message=resp)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingChatModel:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeToolCallingChatModel:  # type: ignore[override]
        # 结构化输出走 ProviderStrategy(json_schema)，不经此方法；保留仅为兼容。
        return self


__all__ = ["FakeToolCallingChatModel"]
