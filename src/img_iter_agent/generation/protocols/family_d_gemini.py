"""协议族 D — Gemini 原生（gemini-3.1-flash-lite-image）。

- 端点：POST /v1beta/models/<model>:generateContent  认证头 x-goog-api-key（不是 Bearer！）
- 请求 contents[].parts[]：文本 {text}，图片 {inline_data:{mime_type,data}}（snake_case）
- 大小写差异（易错）：请求 snake_case(inline_data/mime_type)，响应 camelCase(inlineData/mimeType)
- 尺寸：generationConfig.imageConfig.{aspectRatio, imageSize}（aspectRatio 如 "16:9"；Lite 硬限 1K）
        另需 generationConfig.responseModalities 含 "IMAGE"
- 多轮改图：历史轮次按序入 contents；model 轮须同时带 inline_data + thoughtSignature
- 返回：candidates[].content.parts[] → inlineData.{mimeType,data}(base64)（无 fileData 路径）
"""

from __future__ import annotations

import base64
from pathlib import Path

from ..base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..client import DmxapiClient
from ..image_io import guess_mime, save_image_payload

FAMILY = ModelFamily.D_GEMINI

# Lite 模型硬限 1K
_LITE_LIMIT = "1K"


def _image_config(size: SizeSpec) -> dict:
    """SizeSpec → imageConfig（aspectRatio + imageSize）。Lite 硬限 1K。"""
    ratio = size.ratio or _ratio_from_pixels(size.pixels)
    return {"aspectRatio": ratio, "imageSize": _LITE_LIMIT}


def _ratio_from_pixels(pixels: tuple[int, int] | None) -> str:
    if not pixels:
        return "1:1"
    w, h = pixels
    # 近似到 Gemini 支持的几个比例
    r = w / h
    table = [("1:1", 1.0), ("4:3", 4 / 3), ("3:4", 3 / 4), ("3:2", 3 / 2),
             ("2:3", 2 / 3), ("16:9", 16 / 9), ("9:16", 9 / 16)]
    return min(table, key=lambda kv: abs(kv[1] - r))[0]


def generate(req: GenRequest, *, client: DmxapiClient, model_id: str,
             out_dir: Path) -> GeneratedImage:
    # parts：先文本，再各参考图（inline_data）
    parts: list[dict] = [{"text": req.prompt}]
    for p in req.reference_images:
        mime = guess_mime(Path(p))
        b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": _image_config(req.size),
        },
    }

    path = f"/v1beta/models/{model_id}:generateContent"
    data = client.post_json(FAMILY, path, body)
    return _materialize(data, model_id, "generateContent", req, out_dir)


def _materialize(resp: dict, model_id: str, endpoint: str, req: GenRequest,
                 out_dir: Path) -> GeneratedImage:
    # candidates[].content.parts[] → inlineData.{mimeType,data}（camelCase！）
    payload = ""
    candidates = resp.get("candidates", []) or []
    if candidates:
        for part in candidates[0].get("content", {}).get("parts", []):
            # 响应是 camelCase
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                d = inline["data"]
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                # 组成 data-URI 复用落盘逻辑
                payload = f"data:{mime};base64,{d}"
                break
            if part.get("text"):
                # 偶尔返回文本里夹 URL
                payload = part["text"]
    image_path = save_image_payload(payload, out_dir, name_hint="gen", index=0)
    return GeneratedImage(
        image_path=image_path,
        model=model_id,
        endpoint=endpoint,
        meta={"family": "D", "imageConfig": _image_config(req.size),
              "had_reference": bool(req.reference_images)},
    )


__all__ = ["generate"]
