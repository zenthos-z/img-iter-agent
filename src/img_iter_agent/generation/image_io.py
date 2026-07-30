"""图片 IO：路径↔base64 转换 + url/b64 响应落盘（ADR-005）。

约定：图片全程用文件路径存储；base64/multipart 只在调用 API 时临时转换，绝不落盘持久化。
URL 类型的响应图会被下载到本地存盘再返回路径。
"""

from __future__ import annotations

import base64
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


__all__ = ["file_to_data_uri", "guess_mime", "save_image_payload"]
