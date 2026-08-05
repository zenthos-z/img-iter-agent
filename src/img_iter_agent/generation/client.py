"""dmxapi 底层 HTTP 客户端。

封装 httpx + tenacity 重试 + 按协议族换认证头。把「怎么发 HTTP」和「发什么 body」解耦：
body 构造在各 family dispatcher 里，这里只管：鉴权头、超时、重试、返回原始 JSON。

为了可测，httpx client 走依赖注入；测试用 mock client 拦截请求、回 canned 响应，
**不联网**。
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, get_settings
from .base import ModelFamily


def auth_headers(family: ModelFamily, api_key: str) -> dict[str, str]:
    """按协议族生成认证头。

    - A(OpenAI): `Authorization: Bearer <key>`
    - B/C(豆包/Qwen): `Authorization: <key>`（无 Bearer！）
    - D(Gemini): `x-goog-api-key: <key>`（不是 Bearer！）
    """
    if family is ModelFamily.A_OPENAI:
        return {"Authorization": f"Bearer {api_key}"}
    if family is ModelFamily.D_GEMINI:
        return {"x-goog-api-key": api_key}
    # B_DOUBAO / C_QWEN：无 Bearer
    return {"Authorization": api_key}


class HttpExecutor(Protocol):
    """HTTP 执行抽象，便于测试注入 mock。"""

    def post_json(self, url: str, headers: dict, json_body: dict) -> dict[str, Any]:
        ...

    def post_multipart(self, url: str, headers: dict, data: dict,
                       files: list[tuple]) -> dict[str, Any]:
        ...


class HttpxExecutor:
    """真实 httpx 实现：带重试。"""

    def __init__(self, *, timeout: float = 120.0) -> None:
        self._timeout = timeout

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def post_json(self, url: str, headers: dict, json_body: dict) -> dict[str, Any]:
        h = {"Content-Type": "application/json", **headers}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=h, json=json_body)
            resp.raise_for_status()
            return resp.json()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def post_multipart(self, url: str, headers: dict, data: dict,
                       files: list[tuple]) -> dict[str, Any]:
        # multipart 时不要设 Content-Type，httpx 会按 files 自动加 boundary
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            return resp.json()


class DmxapiClient:
    """dmxapi 网关客户端：持有 host/key/executor，提供各族都要用的 post 方法。"""

    def __init__(self, settings: Settings | None = None,
                 executor: HttpExecutor | None = None) -> None:
        self.settings = settings or get_settings()
        self.executor = executor or HttpxExecutor()

    @property
    def host(self) -> str:
        return self.settings.dmxapi_host.rstrip("/")

    @property
    def key(self) -> str:
        return self.settings.dmxapi_key

    def post_json(self, family: ModelFamily, path: str, body: dict) -> dict[str, Any]:
        url = f"{self.host}{path}"
        headers = auth_headers(family, self.key)
        return _trace_image_call(family, path, body, lambda: self.executor.post_json(url, headers, body))

    def post_multipart(self, family: ModelFamily, path: str, data: dict,
                       files: list[tuple]) -> dict[str, Any]:
        url = f"{self.host}{path}"
        headers = auth_headers(family, self.key)
        return _trace_image_call(
            family, path, data,
            lambda: self.executor.post_multipart(url, headers, data, files),
        )


def _trace_image_call(family: ModelFamily, path: str, req_body: dict, fn) -> dict[str, Any]:
    """出图调用埋点：作为 LangSmith 的 run_type="tool" run 上报，并把真实响应原样返回。

    生图是外部图像生成调用，**不是文本 LLM 调用**——不能用 run_type="llm"，否则会把
    一堆无 token/无模型语义的图像调用混进 LLM 面板，污染模型调用统计。用 tool 类型记录
    协议族/路径/请求摘要（截断 base64，避免塞整张图进 trace）/响应键与耗时，与真正的
    LLM 调用清晰区分。

    注意：trace 的 output 只记响应摘要（resp_keys），但本函数**必须把真实响应原样返回**给
    dispatcher——后者要从中取 data[].b64_json/url 落盘。早期版本把摘要当返回值，导致生图
    拿不到图（静默出空图），这里用 holder 旁路：_do 返回摘要供 trace，真实 resp 旁路返回。
    """
    from langsmith import traceable

    # 记录请求体摘要（截断长字段，避免把整个 base64 图片塞进 trace）
    body_summary = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v)
                    for k, v in (req_body or {}).items()}

    holder: dict[str, Any] = {}

    @traceable(name=f"image_gen.{family.value}", run_type="tool")
    def _do() -> dict:
        resp = fn()
        holder["resp"] = resp
        # trace output 只记摘要，不塞整张 base64 图
        return {"status": "ok", "resp_keys": list(resp.keys()) if isinstance(resp, dict) else "n/a"}

    _do(langsmith_extra={"metadata": {"family": family.value, "path": path, "req": body_summary}})
    return holder["resp"]


__all__ = ["DmxapiClient", "HttpExecutor", "HttpxExecutor", "auth_headers"]
