"""构造 deepagents 所需的 BaseChatModel。

deepagent（`create_deep_agent`）的 `model=` 需要一个支持 `bind_tools` /
`with_structured_output` 的 LangChain `BaseChatModel`。这里用 `langchain_openai.ChatOpenAI`
指向 dmxapi 的 OpenAI 兼容端点（`{dmxapi_host}/v1`）。ChatOpenAI 是原生 langchain Runnable，
LangSmith 自动把每次调用上报为 `run_type="llm"` run（无需 `langsmith.wrap_openai`），且嵌套在
调用方的 `RunnableConfig` 父 run 之下。
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from ..config import Settings

# role → settings 上的 model 字段名
_ROLE_MODEL_ATTR = {
    "generator": "generator_model",
    "critic": "critic_model",
    "summarizer": "summarizer_model",
}

Role = Literal["generator", "critic", "summarizer"]


def build_chat_model(
    settings: Settings, *, role: Role, model_override: str | None = None, **extra,
) -> ChatOpenAI:
    """按 role 从 settings 构造一个 ChatOpenAI（指向 dmxapi 的 OpenAI 兼容端点）。

    Args:
        role: 哪个 agent 用——决定取 generator_model / critic_model / summarizer_model。
        model_override: 非空时覆盖 role 默认 model（来自 data/agents_config/<agent>.json 的外部化配置）；
            为空/None 则用 settings 里该 role 的 model（.env）。默认 None。
        **extra: 透传给 ChatOpenAI 的额外参数（如 temperature、max_tokens）。

    返回的 ChatOpenAI 支持 `.bind_tools(...)` 与 `.with_structured_output(...)`，
    可直接喂给 `create_deep_agent(model=...)`。
    """
    model_id = (model_override or "").strip() or getattr(settings, _ROLE_MODEL_ATTR[role]) or "missing"
    base_url = f"{settings.dmxapi_host.rstrip('/')}/v1"
    return ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=settings.dmxapi_key or "missing",
        timeout=120.0,
        **extra,
    )


__all__ = ["Role", "build_chat_model"]
