"""协议族 B — 豆包/Seedream Responses（seedream-5.0-pro）。

- 端点：POST /v1/responses  JSON  认证头 Authorization:<key>（无 Bearer！）
- 字段：model / input(提示词 string) / image(单图string 或 多图array) / size /
        output_format / response_format / watermark
- 参考图：image = 公网URL 或 data:image/<fmt>;base64,...（单图 string，多图 array，同名字段重载）
- size：分辨率档 "1K"/"2K"/"3K"/"4K" 或 像素 "2048x2048"（小写 x）
- 返回：data[].url（默认24h有效）或 data[].b64_json
"""

from __future__ import annotations

from pathlib import Path

from ..base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..client import DmxapiClient
from ..image_io import file_to_data_uri, save_image_payload

FAMILY = ModelFamily.B_DOUBAO


def to_size_str(size: SizeSpec) -> str:
    """SizeSpec → B 族 size 字符串（优先 tier，其次像素 小写x）。"""
    if size.tier:
        return size.tier
    if size.pixels:
        w, h = size.pixels
        return f"{w}x{h}"
    return "2K"


def generate(req: GenRequest, *, client: DmxapiClient, model_id: str,
             out_dir: Path) -> GeneratedImage:
    body: dict = {
        "model": model_id,
        "input": req.prompt,
        "size": to_size_str(req.size),
        # 要 base64 便于稳定落盘（避免 url 过期）
        "response_format": "b64_json",
        "watermark": False,
    }
    if req.reference_images:
        # 单图→string；多图→array
        data_uris = [file_to_data_uri(Path(p)) for p in req.reference_images]
        body["image"] = data_uris[0] if len(data_uris) == 1 else data_uris

    data = client.post_json(FAMILY, "/v1/responses", body)
    return _materialize(data, model_id, "responses", req, out_dir)


def _materialize(resp: dict, model_id: str, endpoint: str, req: GenRequest,
                 out_dir: Path) -> GeneratedImage:
    items = resp.get("data", []) or []
    payload = ""
    if items:
        it = items[0]
        payload = it.get("b64_json") or it.get("url") or ""
    image_path = save_image_payload(payload, out_dir, name_hint="gen", index=0)
    return GeneratedImage(
        image_path=image_path,
        model=model_id,
        endpoint=endpoint,
        meta={"family": "B", "size": to_size_str(req.size),
              "had_reference": bool(req.reference_images)},
    )


__all__ = ["generate", "to_size_str"]
