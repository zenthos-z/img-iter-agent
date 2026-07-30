"""生图统一接口 + 数据模型。

屏蔽 dmxapi 四协议族差异（ARCH §4.2）：上层 agent 只面对 `GenRequest`，
适配层负责把它翻译成各族的请求体、把各族返回（url/b64）统一落盘成文件路径（ADR-005）。

四协议族（docs/ARCHITECTURE.md §4.2.3，已对官方 doc.dmxapi.cn 实测核对）：
  A. OpenAI Images   /v1/images/generations(JSON) + /v1/images/edits(multipart)  Bearer  size:1024x1024
  B. 豆包 Responses  /v1/responses  Authorization:<key>(无Bearer)  input:string  image=URL/base64  size:2K/2048x2048
  C. Qwen Responses  /v1/responses(同B端点,提示词嵌套)  input.messages[].content[].text  size:宽*高(星号!)
  D. Gemini 原生     /v1beta/models/<m>:generateContent  x-goog-api-key(非Bearer)  contents[].parts[]  size:imageConfig
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class ModelFamily(str, Enum):
    """四协议族枚举。"""

    A_OPENAI = "A"   # gpt-image-2
    B_DOUBAO = "B"   # seedream-5.0-pro
    C_QWEN = "C"     # qwen-image-2.0
    D_GEMINI = "D"   # gemini-3.1-flash-image


class SizeSpec(BaseModel):
    """统一的尺寸描述。三选一，适配层按各族格式翻译。

    - pixels: (w, h)，如 (1024, 1024)
    - tier:   分辨率档，如 "2K"（B/D 支持）
    - ratio:  宽高比，如 "16:9"（D 的 aspectRatio）
    """

    pixels: tuple[int, int] | None = None
    tier: str | None = None
    ratio: str | None = None


class GenRequest(BaseModel):
    """统一的生图请求（上层 agent 无需知道协议族）。"""

    prompt: str
    size: SizeSpec = Field(default_factory=lambda: SizeSpec(tier="2K"))
    reference_images: list[Path] = Field(default_factory=list)  # 非空→风格锚定
    conversation_history: list[Path] = Field(default_factory=list)  # 非空→Gemini 多轮改图
    model_hint: ModelFamily | None = None  # 偏好；None 时适配层按路由规则选
    quality: str = "high"
    n: int = 1
    # 三视图任务标记：生成多张视图（适配层据此决定是单图 n 还是多次调用）
    output_views: list[str] | None = None


class GeneratedImage(BaseModel):
    """生图产物。始终是文件路径（ADR-005：不存 base64）。"""

    image_path: Path
    model: str  # 实际用的模型 model_id
    endpoint: str  # 实际端点（generations/edits/responses/generateContent）
    meta: dict = Field(default_factory=dict)  # 请求参数快照，便于复现与归因


class ImageGenerator(Protocol):
    """生图适配层统一接口。"""

    def generate(self, req: GenRequest) -> GeneratedImage: ...


__all__ = ["GenRequest", "GeneratedImage", "ImageGenerator", "ModelFamily", "SizeSpec"]
