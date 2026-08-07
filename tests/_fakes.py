"""测试用 BaseChatModel：让 deepagents 在离线（无 dmxapi key）下可驱动。

`langchain_core` 自带的 `FakeMessagesListChatModel` 不实现 `bind_tools`，而 deepagents
构造时必须 `model.bind_tools(...)`。故自写一个最小 BaseChatModel：按预设 `AIMessage` 序列
出队（每条可带 `tool_calls` 模拟工具调用），`bind_tools` / `with_structured_output` 返回 self。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeToolCallingChatModel(BaseChatModel):
    """循环吐预设 AIMessage 的假 chat model，支持 bind_tools。

    构造：`FakeToolCallingChatModel(responses=[AIMessage(...), ...])`。
    - responses: list[AIMessage]，按顺序出队；耗尽后重复最后一个。
      每条 AIMessage 可带 tool_calls=[{"name","args","id","type"}] 模拟工具调用，
      或纯 content 作为终止回答。
    - bind_tools / with_structured_output 返回 self：忽略工具 schema，照样吐预设序列。
      （Critic 用 ToolStrategy 走工具调用产出结构化输出，不触发 with_structured_output 的 provider 路径。）
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
        return ChatResult(generations=[ChatGeneration(message=self._next_response())])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingChatModel:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeToolCallingChatModel:  # type: ignore[override]
        # 我们用 ToolStrategy（工具调用）产出结构化输出，不走 provider 的 with_structured_output。
        return self


__all__ = ["FakeToolCallingChatModel"]
