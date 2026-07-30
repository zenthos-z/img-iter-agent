"""LLM client 抽象层。

Critic/Generator/Summarizer 三个 agent 都通过 `LlmClient` 调用 LLM。
接口刻意窄化到「给消息列表 → 回文本」，让 agent 逻辑与具体协议族解耦：

- 生产：按 `Settings.*_protocol` 分派到 dmxapi 的对应端点（openai 兼容/gemini/...），
  Step 3 起逐步实现。
- 测试：用 `FakeLlmClient` 回 canned 文本，断言 agent 的分派/解析/加权数学，
  **完全不联网、不需 key**。

多模态（Critic 看图）：消息里可带图片路径，由具体 client 在调用时按需转 base64/URL
（ADR-005：图片全程用路径，只在调用时临时转换）。
"""

from __future__ import annotations

from typing import Protocol


class LlmMessage(dict):
    """一条 chat 消息。键: role('user'|'system'|'assistant'), content。
    content 可以是 str，或多模态 list（具体形态由 client 解释）。这里用 dict 透传，
    避免 agent 层耦合到某一家 API 的 content schema。"""


class LlmClient(Protocol):
    """agent 调用 LLM 的统一接口（结构体，不是抽象基类——用 Protocol 做鸭子类型）。"""

    def complete(self, messages: list[dict]) -> str:
        """同步给一组消息，回一段文本（agent 自行解析结构化结果）。

        messages 每条是 {"role":..., "content":...}；content 里可含图片路径，
        由实现负责在调用时转换。返回纯文本（通常是 JSON，由调用方解析）。
        """
        ...


class FakeLlmClient:
    """测试用假 client：按预设规则回文本。

    用法：`FakeLlmClient(responses=[r1, r2])` 顺序回；或 `func(messages)->str` 动态生成。
    便于 Critic 测试回 canned 的逐项判定 / 连续分 JSON。
    """

    def __init__(self, responses: list[str] | None = None,
                 func=None) -> None:
        self._responses = list(responses) if responses else []
        self._func = func
        self.calls: list[list[dict]] = []  # 记录每次调用入参，便于断言

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self._func is not None:
            return self._func(messages)
        if not self._responses:
            return "{}"
        # 顺序出队；超出则重复最后一个
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


__all__ = ["FakeLlmClient", "LlmClient", "LlmMessage"]
