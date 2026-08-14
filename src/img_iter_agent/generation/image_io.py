"""图片 IO：路径↔base64 转换 + url/b64 响应落盘（ADR-005）。

约定：图片全程用文件路径存储；base64/multipart 只在调用 API 时临时转换，绝不落盘持久化。
URL 类型的响应图会被下载到本地存盘再返回路径。
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx


def file_to_data_uri(path: Path) -> str:
    """读本地图片文件 → `data:image/<fmt>;base64,<...>` data-URI（B/C/D 族用）。"""
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"  # 兜底
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# 注入 agent HumanMessage 的图最长边（控制 LangSmith trace 体积 + 输入 token）。
# 4.2MB PNG 原图经此缩放 → ~0.1MB JPEG，单轮 trace 从 ~14MB 降到 ~1MB，避免上传超时。
_VIEW_MAX_DIM = 1280

# Critic 看的是「高清原图」——结构/部件比对需要细节，不能用 1280 的小图（会把结构瑕疵糊掉、
# 导致 critic 看不清就放水判通过）。trace 体积由 compress_images_in_trace 在 LangSmith 上传侧压，
# 不再靠这里缩图，故 critic 用高分辨率。2560 ≈ 原 three_view(2752) 几乎无损，且远低于各族 API 上限。
_CRITIC_VIEW_MAX_DIM = 2560


def resized_data_uri(path: Path, max_dim: int = _VIEW_MAX_DIM) -> str:
    """读图 → 等比缩到最长边 max_dim → JPEG(q85) data-URI。

    用于把图注入 agent 的 HumanMessage（critic / generator / distiller 共用）。
    与 ``file_to_data_uri``（原图）相对：本函数缩放以控制体积，避免大图把 LangSmith
    trace 撑到十几 MB 导致上传超时。缩放对「整体还原度」类评分无损（distiller 已验证）。
    PIL 不可用或缩放失败时，退回原图 ``file_to_data_uri``（保证总有图可看）。
    """
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return file_to_data_uri(path)
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = max_dim / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:  # noqa: BLE001
        return file_to_data_uri(path)


# LangSmith 上传侧压缩：把 trace 里携带的 data-URI 图片缩到很小，**只影响 LangSmith 上报内容**，
# 不影响真正喂给模型的图（模型拿的是 critic/generator 消息里的高清原图）。
# 由 langsmith Client 的 hide_inputs/hide_outputs 调用（见 config._warm_langsmith_client）。
_LS_TRACE_MAX_DIM = 512
_LS_SKIP_BELOW_BYTES = 80_000  # 已小于 ~80KB 的图不再二次压缩（避免无谓重编码）


def _shrink_data_uri(uri: str, max_dim: int = _LS_TRACE_MAX_DIM, quality: int = 80) -> str:
    """单个 data-URI 图片 → 缩到 max_dim 的 JPEG data-URI。已是小图或解码失败则原样返回（绝不抛）。"""
    if not uri.startswith("data:image/"):
        return uri
    try:
        _header, _, b64 = uri.partition(",")
        raw = base64.b64decode(b64)
    except Exception:  # noqa: BLE001
        return uri
    if len(raw) < _LS_SKIP_BELOW_BYTES:
        return uri
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        scale = max_dim / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        out = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{out}"
    except Exception:  # noqa: BLE001
        return uri


def compress_images_in_trace(obj, max_dim: int = _LS_TRACE_MAX_DIM):
    """递归遍历 langsmith trace 的 inputs/outputs，把其中的 data-URI 图片缩小（仅用于上传压缩）。

    被 ``Client(hide_inputs=compress_images_in_trace, hide_outputs=...)`` 调用：模型仍看高清原图，
    只有上报到 LangSmith 的副本被压缩，从而把带大图的 trace 从十几 MB 压到 ~1MB 避免上传超时。
    任何异常都回退原对象——trace 永不因压缩失败而中断。
    """
    try:
        if isinstance(obj, dict):
            return {k: compress_images_in_trace(v, max_dim) for k, v in obj.items()}
        if isinstance(obj, list):
            return [compress_images_in_trace(v, max_dim) for v in obj]
        if isinstance(obj, str):
            return _shrink_data_uri(obj, max_dim) if obj.startswith("data:image/") else obj
    except Exception:  # noqa: BLE001
        return obj
    return obj


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def save_image_payload(payload: str, dest_dir: Path, *, name_hint: str = "out",
                       index: int = 0) -> Path:
    """把 API 返回的图片 payload 落盘成文件，返回路径。

    payload 可以是：
      - base64 字符串（裸 base64）
      - data:image/...;base64,...  data-URI
      - http(s) URL → 下载存盘
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    s = payload.strip()
    if s.startswith(("http://", "https://")):
        return _download(s, dest_dir, name_hint, index)
    if s.startswith("data:"):
        mime, b64 = _split_data_uri(s)
        ext = _ext_from_mime(mime)
        out = dest_dir / f"{name_hint}_{index}{ext}"
        out.write_bytes(base64.b64decode(b64))
        return out
    # 裸 base64
    out = dest_dir / f"{name_hint}_{index}.png"
    try:
        out.write_bytes(base64.b64decode(s))
    except (ValueError, OSError):
        # base64 解码失败(ValueError 的子类 binascii.Error)或写盘失败 → 原样写文本便于排查
        out = dest_dir / f"{name_hint}_{index}.txt"
        out.write_text(s, encoding="utf-8")
    return out


def _split_data_uri(uri: str) -> tuple[str, str]:
    """`data:image/png;base64,XXXX` → (mime, base64_str)。"""
    header, _, b64 = uri.partition(",")
    mime = "image/png"
    if header.startswith("data:"):
        mime = header[5:].split(";")[0] or "image/png"
    return mime, b64


def _ext_from_mime(mime: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(mime, ".png")


def _download(url: str, dest_dir: Path, name_hint: str, index: int) -> Path:
    """下载 URL 到本地。扩展名按 URL 路径猜，兜底 .png。"""
    path_part = urlparse(url).path
    ext = Path(path_part).suffix or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = ".png"
    out = dest_dir / f"{name_hint}_{index}{ext}"
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out.write_bytes(resp.content)
    return out


__all__ = [
    "compress_images_in_trace",
    "file_to_data_uri",
    "guess_mime",
    "resized_data_uri",
    "save_image_payload",
]
