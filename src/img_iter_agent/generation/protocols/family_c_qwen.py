"""协议族 C — Qwen Responses（qwen-image-2.0-pro）。

与 B 同端点 /v1/responses，但提示词结构完全不同（嵌套）。
- 字段：model / input(对象) / 顶层可选 size 等
- 提示词：input.messages[].content[].text（仅一个 text，传多个报错；role=user）
- 编辑图：参考图放进同一个 content[] 数组：content[] = [{text:...}, {image:"<url/base64>"}]，1~3 张
- 参数：input.parameters.{negative_prompt, size, n(1-6), prompt_extend, watermark, seed}
- size：宽*高（星号！无分辨率档位），如 2048*2048
- 返回：output[].content[].text（图片URL字符串）
"""

from __future__ import annotations

from pathlib import Path

from ..base import GeneratedImage, GenRequest, ModelFamily, SizeSpec
from ..client import DmxapiClient
from ..image_io import file_to_data_uri, save_image_payload

FAMILY = ModelFamily.C_QWEN


def to_size_str(size: SizeSpec) -> str:
    """SizeSpec → C 族 size 字符串（宽*高 星号；C 无 tier）。"""
    if size.pixels:
        w, h = size.pixels
        return f"{w}*{h}"  # 星号！
    # C 无 tier；给个安全默认
    return "2048*2048"


def generate(req: GenRequest, *, client: DmxapiClient, model_id: str,
             out_dir: Path) -> GeneratedImage:
    # content[]：先 text，再各参考图（image 项）
    content: list[dict] = [{"text": req.prompt}]
    for p in req.reference_images:
        content.append({"image": file_to_data_uri(Path(p))})

    # C 族支持 negative_prompt + seed（见 input.parameters 字段文档）；其余族静默忽略。
    params: dict = {
        "size": to_size_str(req.size),
        "n": req.n,
        "watermark": False,
    }
    if req.negative_prompt:
        params["negative_prompt"] = req.negative_prompt
    if req.seed is not None:
        params["seed"] = req.seed

    body = {
        "model": model_id,
        "input": {
            "messages": [{"role": "user", "content": content}],
            "parameters": params,
        },
    }

    data = client.post_json(FAMILY, "/v1/responses", body)
    return _materialize(data, model_id, "responses", req, out_dir)


def _materialize(resp: dict, model_id: str, endpoint: str, req: GenRequest,
                 out_dir: Path) -> GeneratedImage:
    # 返回 output[].content[].text（图片URL字符串）
    payload = ""
    outputs = resp.get("output", []) or []
    if outputs:
        for part in outputs[0].get("content", []):
            if part.get("text"):
                payload = part["text"]
                break
    image_path = save_image_payload(payload, out_dir, name_hint="gen", index=0)
    return GeneratedImage(
        image_path=image_path,
        model=model_id,
        endpoint=endpoint,
        meta={"family": "C", "size": to_size_str(req.size),
              "had_reference": bool(req.reference_images)},
    )


__all__ = ["generate", "to_size_str"]
