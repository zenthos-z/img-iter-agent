"""生图适配层测试：mock executor 拦截 HTTP，验证各族请求形状 + 尺寸翻译 + 路由。

完全离线、不联网。重点验证四族差异（端点/认证头/prompt 嵌套/size 格式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from img_iter_agent.config import Settings
from img_iter_agent.generation.base import GenRequest, ModelFamily, SizeSpec
from img_iter_agent.generation.client import DmxapiClient, auth_headers
from img_iter_agent.generation.protocols import (
    family_a_openai,
    family_b_doubao,
    family_c_qwen,
    family_d_gemini,
)
from img_iter_agent.generation.router import Router, route

# ---- 测试用 HTTP executor：记录请求，回 canned 响应 ----

class MockExecutor:
    """记录所有 post 调用，按预设响应回。"""

    def __init__(self, *, json_resp: dict | None = None,
                 multipart_resp: dict | None = None) -> None:
        self.calls: list[dict] = []
        self._json_resp = json_resp or {"data": [{"b64_json": "ZmFrZQ=="}]}  # "fake"
        self._multipart_resp = multipart_resp or {"data": [{"b64_json": "ZmFrZQ=="}]}

    def post_json(self, url: str, headers: dict, json_body: dict) -> dict[str, Any]:
        self.calls.append({"kind": "json", "url": url, "headers": dict(headers),
                           "body": json_body})
        return self._json_resp

    def post_multipart(self, url: str, headers: dict, data: dict,
                       files: list[tuple]) -> dict[str, Any]:
        self.calls.append({"kind": "multipart", "url": url, "headers": dict(headers),
                           "data": dict(data), "files": [(f[0], f[1][0]) for f in files]})
        return self._multipart_resp


def _client(executor: MockExecutor, settings: Settings) -> DmxapiClient:
    return DmxapiClient(settings=settings, executor=executor)


def _ref_img(tmp_path: Path) -> Path:
    p = tmp_path / "ref.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")  # PNG 头 + 占位
    return p


def _settings(**kw) -> Settings:
    base = {
        "dmxapi_host": "https://api.dmxapi.cn",
        "dmxapi_key": "sk-test-key",
        "model_seedream_pro": "doubao-seedream-5-0-pro-260628",
        "model_gpt_image": "gpt-image-2-03",
        "model_gemini_image": "gemini-3.1-flash-lite-image",
        "model_qwen_image": "qwen-image-2.0-pro",
        "data_root": Path("/tmp/_nope"),
    }
    base.update(kw)
    return Settings(**base)


# ---- 认证头 ----

def test_auth_headers_per_family():
    assert auth_headers(ModelFamily.A_OPENAI, "sk-x") == {"Authorization": "Bearer sk-x"}
    assert auth_headers(ModelFamily.B_DOUBAO, "sk-x") == {"Authorization": "sk-x"}  # 无 Bearer
    assert auth_headers(ModelFamily.C_QWEN, "sk-x") == {"Authorization": "sk-x"}    # 无 Bearer
    assert auth_headers(ModelFamily.D_GEMINI, "sk-x") == {"x-goog-api-key": "sk-x"} # 非 Bearer


# ---- 尺寸翻译（四族格式不同，关键差异点）----

def test_family_a_size_pixels_and_auto():
    assert family_a_openai.to_size_str(SizeSpec(pixels=(1024, 1024))) == "1024x1024"
    assert family_a_openai.to_size_str(SizeSpec(tier="2K")) == "auto"  # A 无 tier


def test_family_b_size_tier_and_pixels():
    assert family_b_doubao.to_size_str(SizeSpec(tier="2K")) == "2K"
    assert family_b_doubao.to_size_str(SizeSpec(pixels=(2048, 2048))) == "2048x2048"


def test_family_c_size_uses_asterisk():
    # C 用星号！这是与 A/B 的关键差异
    assert family_c_qwen.to_size_str(SizeSpec(pixels=(2048, 2048))) == "2048*2048"
    # C 无 tier，传 tier 也回落到像素
    assert "*" in family_c_qwen.to_size_str(SizeSpec(tier="2K"))


# ---- Family A：generations(JSON) vs edits(multipart) ----

def test_family_a_text_to_image_uses_generations_json(tmp_path):
    ex = MockExecutor()
    client = _client(ex, _settings())
    req = GenRequest(prompt="一把椅子白底", size=SizeSpec(pixels=(1024, 1024)))
    out = family_a_openai.generate(req, client=client, model_id="gpt-image-2-03",
                                   out_dir=tmp_path)
    assert len(ex.calls) == 1
    c = ex.calls[0]
    assert c["kind"] == "json"
    assert c["url"].endswith("/v1/images/generations")
    assert c["headers"]["Authorization"] == "Bearer sk-test-key"
    assert c["body"]["prompt"] == "一把椅子白底"
    assert c["body"]["size"] == "1024x1024"
    assert c["body"]["model"] == "gpt-image-2-03"
    assert out.image_path.exists()
    assert out.endpoint == "generations"
    assert out.model == "gpt-image-2-03"


def test_family_a_edit_uses_multipart_with_image_field(tmp_path):
    ex = MockExecutor()
    client = _client(ex, _settings())
    ref = _ref_img(tmp_path)
    req = GenRequest(prompt="改成白底", reference_images=[ref],
                     size=SizeSpec(pixels=(1024, 1024)))
    family_a_openai.generate(req, client=client, model_id="gpt-image-2-03", out_dir=tmp_path)
    c = ex.calls[0]
    assert c["kind"] == "multipart"
    assert c["url"].endswith("/v1/images/edits")
    # multipart 不应手动设 Content-Type（httpx 自动加 boundary）
    assert "Content-Type" not in c["headers"]
    # 参考图在同名 image 字段
    assert all(f[0] == "image" for f in c["files"])
    assert c["data"]["prompt"] == "改成白底"


# ---- Family B：responses + 无 Bearer + image 重载 ----

def test_family_b_single_ref_image_is_string(tmp_path):
    ex = MockExecutor()
    client = _client(ex, _settings())
    ref = _ref_img(tmp_path)
    req = GenRequest(prompt="三视图", reference_images=[ref], size=SizeSpec(tier="2K"))
    family_b_doubao.generate(req, client=client, model_id="seedream-pro", out_dir=tmp_path)
    c = ex.calls[0]
    assert c["url"].endswith("/v1/responses")
    assert c["headers"]["Authorization"] == "sk-test-key"  # 无 Bearer
    assert c["body"]["input"] == "三视图"        # 提示词是 string
    assert c["body"]["size"] == "2K"
    # 单图 → image 是 string（data-URI）
    assert isinstance(c["body"]["image"], str)
    assert c["body"]["image"].startswith("data:image/")


def test_family_b_multi_ref_image_is_array(tmp_path):
    ex = MockExecutor()
    client = _client(ex, _settings())
    ref1, ref2 = _ref_img(tmp_path), _ref_img(tmp_path / "ref2.png") if False else _ref_img(tmp_path)
    ref2 = tmp_path / "ref2.png"; ref2.write_bytes(b"\x89PNGfake2")
    req = GenRequest(prompt="融合", reference_images=[ref1, ref2])
    family_b_doubao.generate(req, client=client, model_id="seedream-pro", out_dir=tmp_path)
    c = ex.calls[0]
    # 多图 → image 是 array
    assert isinstance(c["body"]["image"], list)
    assert len(c["body"]["image"]) == 2


def test_family_b_text_only_has_no_image_field(tmp_path):
    ex = MockExecutor()
    client = _client(ex, _settings())
    req = GenRequest(prompt="纯文生图")
    family_b_doubao.generate(req, client=client, model_id="seedream-pro", out_dir=tmp_path)
    assert "image" not in ex.calls[0]["body"]


# ---- Family C：同端点但提示词嵌套 + 星号 size ----

def test_family_c_nested_prompt_and_asterisk_size(tmp_path):
    # 返回 data-URI（避免测试联网下载）
    ex = MockExecutor(json_resp={"output": [{"content": [{"text": "data:image/png;base64,ZmFrZQ=="}]}]})
    client = _client(ex, _settings())
    req = GenRequest(prompt="qwen 文生图", size=SizeSpec(pixels=(2048, 2048)))
    out = family_c_qwen.generate(req, client=client, model_id="qwen-pro", out_dir=tmp_path)
    c = ex.calls[0]
    assert c["url"].endswith("/v1/responses")  # 与 B 同端点
    assert c["headers"]["Authorization"] == "sk-test-key"
    # 提示词嵌套在 input.messages[].content[].text
    msgs = c["body"]["input"]["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["text"] == "qwen 文生图"
    # size 在 input.parameters，用星号
    assert c["body"]["input"]["parameters"]["size"] == "2048*2048"
    assert out.image_path.exists()


def test_family_c_edit_image_in_content_array(tmp_path):
    ex = MockExecutor(json_resp={"output": [{"content": [{"text": "data:image/png;base64,ZmFrZQ=="}]}]})
    client = _client(ex, _settings())
    ref = _ref_img(tmp_path)
    req = GenRequest(prompt="编辑", reference_images=[ref])
    family_c_qwen.generate(req, client=client, model_id="qwen-pro", out_dir=tmp_path)
    c = ex.calls[0]
    content = c["body"]["input"]["messages"][0]["content"]
    # content[0]=text, content[1]=image
    assert content[0] == {"text": "编辑"}
    assert "image" in content[1]


# ---- Family D：generateContent + x-goog-api-key + inline_data(snake) ----

def test_family_d_endpoint_auth_and_inline_data(tmp_path):
    ex = MockExecutor(json_resp={
        "candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": "ZmFrZQ=="}}  # camelCase 响应
        ]}}]
    })
    client = _client(ex, _settings())
    req = GenRequest(prompt="gemini 生图", size=SizeSpec(ratio="16:9"))
    out = family_d_gemini.generate(req, client=client, model_id="gemini-flash", out_dir=tmp_path)
    c = ex.calls[0]
    # 端点含模型名 + :generateContent
    assert ":generateContent" in c["url"]
    assert "gemini-flash" in c["url"]
    # 认证头是 x-goog-api-key，不是 Bearer
    assert c["headers"]["x-goog-api-key"] == "sk-test-key"
    assert "Authorization" not in c["headers"]
    parts = c["body"]["contents"][0]["parts"]
    assert parts[0] == {"text": "gemini 生图"}
    # imageConfig 在 generationConfig
    assert c["body"]["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert c["body"]["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert out.image_path.exists()


def test_family_d_reference_image_inline_data_snake_case(tmp_path):
    ex = MockExecutor(json_resp={"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "image/png", "data": "ZmFrZQ=="}}]}}]})
    client = _client(ex, _settings())
    ref = _ref_img(tmp_path)
    family_d_gemini.generate(
        GenRequest(prompt="带参考", reference_images=[ref]),
        client=client, model_id="gemini-flash", out_dir=tmp_path)
    parts = ex.calls[0]["body"]["contents"][0]["parts"]
    # 参考图用 snake_case inline_data
    assert "inline_data" in parts[1]
    assert "mime_type" in parts[1]["inline_data"]


# ---- 路由 ----

def test_route_multiturn_goes_to_d(tmp_path):
    settings = _settings()
    req = GenRequest(prompt="改", conversation_history=[tmp_path / "h.png"])
    d = route(req, settings=settings)
    assert d.family is ModelFamily.D_GEMINI
    assert d.model_id == "gemini-3.1-flash-lite-image"


def test_route_with_reference_prefers_b(tmp_path):
    settings = _settings()
    req = GenRequest(prompt="风格迁移", reference_images=[tmp_path / "r.png"])
    d = route(req, settings=settings)
    assert d.family is ModelFamily.B_DOUBAO


def test_route_text_only_prefers_b(tmp_path):
    settings = _settings()
    d = route(GenRequest(prompt="文生图"), settings=settings)
    assert d.family is ModelFamily.B_DOUBAO


def test_route_falls_back_to_a_when_b_unconfigured(tmp_path):
    settings = _settings(model_seedream_pro="")  # B 未配置
    d = route(GenRequest(prompt="文生图", reference_images=[tmp_path / "r.png"]),
              settings=settings)
    assert d.family is ModelFamily.A_OPENAI


def test_route_respects_model_hint(tmp_path):
    settings = _settings()
    d = route(GenRequest(prompt="x", model_hint=ModelFamily.C_QWEN), settings=settings)
    assert d.family is ModelFamily.C_QWEN


def test_route_raises_when_model_id_missing(tmp_path):
    settings = _settings(model_seedream_pro="", model_gpt_image="", model_qwen_image="",
                         model_gemini_image="")
    with pytest.raises(ValueError, match="未配置"):
        route(GenRequest(prompt="x"), settings=settings)


def test_router_end_to_end_with_mock(tmp_path):
    """Router → 选族 → dispatcher → mock executor 全链路。"""
    ex = MockExecutor()
    settings = _settings()
    router = Router(settings=settings, client=_client(ex, settings))
    out = router.generate(GenRequest(prompt="test", size=SizeSpec(tier="2K")), out_dir=tmp_path)
    assert out.image_path.exists()
    assert ex.calls[0]["url"].endswith("/v1/responses")  # 默认走 B
