"""OpenAI 兼容的 LLM client：走 dmxapi 的 /v1/chat/completions，用 langsmith.wrap_openai 自动追踪。

dmxapi 网关对**所有协议族**暴露统一的 OpenAI 兼容 chat 端点，故一个 client 即可覆盖
Critic(多模态)/Generator/Summarizer，无需按 protocol 分派。

wrap_openai 是 LangSmith 官方推荐的追踪方式：
  - 让每次 chat.completions.create 自动作为 LangSmith 的 run_type="llm" run 上报，
    带标准 I/O（messages → generations）、模型名(ls_model_name)、provider(ls_provider)、
    token usage 与起止时间(耗时)，并自动嵌套在 LangGraph 节点 run 之下。
  - 无需手写 @traceable，也不用手动构造 I/O 格式。

多模态：Critic 在 content 里放图片时构造 OpenAI 标准多模态 content
（[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:..."}}]），openai SDK
原生支持，直接透传即可。

实现 llm.LlmClient Protocol（只需 complete 方法）。本模块从 cli.py 抽出，消除
web.services.loop_runner → cli 的反向依赖。
"""

from __future__ import annotations

from ..config import Settings


class OpenAiCompatLlm:
    """Agent LLM client：走 dmxapi 的 OpenAI 兼容端点（/v1/chat/completions）。

    用官方 openai SDK 调用，并用 langsmith.wrap_openai 包装 client（LangSmith 自动追踪）。
    """

    def __init__(self, settings: Settings, *, model: str | None = None,
                 http_client=None) -> None:
        from langsmith.wrappers import wrap_openai
        from openai import OpenAI

        base_url = f"{settings.dmxapi_host.rstrip('/')}/v1"
        # api_key 为空时用占位串（真实运行 .env 必配 dmxapi_key）；dmxapi 走 Bearer 鉴权。
        # http_client 可注入（测试用 MockTransport 拦截请求、回 canned 响应，不联网）。
        client = OpenAI(base_url=base_url, api_key=settings.dmxapi_key or "missing",
                        timeout=120.0, http_client=http_client)
        self._client = wrap_openai(client)
        self.settings = settings
        # model 由调用方指定（critic/generator/summarizer 各自的 *_model 字段）
        self._model = model or settings.critic_model

    def complete(self, messages: list[dict]) -> str:
        model = self._model
        resp = self._client.chat.completions.create(model=model, messages=messages)  # type: ignore[arg-type]
        return resp.choices[0].message.content or ""


__all__ = ["OpenAiCompatLlm"]
