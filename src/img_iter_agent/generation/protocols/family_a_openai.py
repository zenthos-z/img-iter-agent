"""协议族 A — OpenAI Images（gpt-image-2）。

- 文生图：POST /v1/images/generations  JSON  Bearer  字段 model/prompt/n/size/quality/output_format
- 编辑：  POST /v1/images/edits  multipart  参考图同名 image 字段(可多个文件)
- size：精确像素 1024x1024 / 2048x2048 等（或 auto）
- 返回：data[].b64_json 或 data[].url
"""

from __future__ import annotations

from pathlib import Path

from ..base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..client import DmxapiClient
from ..image_io import guess_mime, save_image_payload

FAMILY = ModelFamily.A_OPENAI


def to_size_str(size: SizeSpec) -> str:
    """SizeSpec → A 族的 size 字符串（精确像素或 auto）。"""
    if size.pixels:
        w, h = size.pixels
        return f"{w}x{h}"
    # A 族不支持 tier/ratio 直传，回落 auto
    return "auto"


def generate(req: GenRequest, *, client: DmxapiClient, model_id: str,
             out_dir: Path) -> GeneratedImage:
    """A 族生图。有参考图走 edits(multipart)，否则走 generations(JSON)。"""
    if req.reference_images:
        return _edit(req, client=client, model_id=model_id, out_dir=out_dir)
    return _generate(req, client=client, model_id=model_id, out_dir=out_dir)


def _generate(req: GenRequest, *, client: DmxapiClient, model_id: str,
              out_dir: Path) -> GeneratedImage:
    body = {
        "model": model_id,
        "prompt": req.prompt,
        "n": req.n,
        "size": to_size_str(req.size),
        "quality": req.quality,
    }
    data = client.post_json(FAMILY, "/v1/images/generations", body)
    return _materialize(data, model_id, "generations", req, out_dir)


def _edit(req: GenRequest, *, client: DmxapiClient, model_id: str,
          out_dir: Path) -> GeneratedImage:
    # multipart：参考图同名 image 字段，可多个文件
    files: list[tuple] = []
    for p in req.reference_images:
        mime = guess_mime(Path(p))
        files.append(("image", (Path(p).name, Path(p).read_bytes(), mime)))
    data_form = {
        "model": model_id,
        "prompt": req.prompt,
        "n": str(req.n),
        "size": to_size_str(req.size),
    }
    data = client.post_multipart(FAMILY, "/v1/images/edits", data_form, files)
    return _materialize(data, model_id, "edits", req, out_dir)


def _materialize(resp: dict, model_id: str, endpoint: str, req: GenRequest,
                 out_dir: Path) -> GeneratedImage:
    items = resp.get("data", []) or []
    # 取第一张（三视图多张由上层多次调用处理）
    payload = ""
    if items:
        it = items[0]
        payload = it.get("b64_json") or it.get("url") or ""
    image_path = save_image_payload(payload, out_dir, name_hint="gen", index=0)
    return GeneratedImage(
        image_path=image_path,
        model=model_id,
        endpoint=endpoint,
        meta={"family": "A", "size": to_size_str(req.size), "n": req.n,
              "had_reference": bool(req.reference_images)},
    )


__all__ = ["generate", "to_size_str"]
