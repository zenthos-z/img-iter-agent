"""路由：按 task-mode + model_hint 选协议族 dispatcher + model_id。

路由规则（ARCH §4.2.4）：
  ├─ 多轮改图(conversation_history 非空) → D (Gemini)，唯一支持多轮
  ├─ 参考图风格迁移(reference_images 非空) → 优先 B(seedream, JSON 易异步、融合强)，A(gpt) 备选
  └─ 纯文生图 → A/B/C 皆可（按 model_hint 或默认 B）
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, get_settings
from .base import GeneratedImage, GenRequest, ModelFamily
from .client import DmxapiClient
from .protocols import family_a_openai, family_b_doubao, family_c_qwen, family_d_gemini


@dataclass(frozen=True)
class RouteDecision:
    """路由结果：选了哪族 + 用哪个 model_id + 调哪个 dispatcher。"""

    family: ModelFamily
    model_id: str
    generate: Callable[..., GeneratedImage]


def _model_id_for(family: ModelFamily, settings: Settings) -> str:
    """按族取配置里的 model_id（用户在 .env 填）。"""
    return {
        ModelFamily.A_OPENAI: settings.model_gpt_image,
        ModelFamily.B_DOUBAO: settings.model_seedream_pro,
        ModelFamily.C_QWEN: settings.model_qwen_image,
        ModelFamily.D_GEMINI: settings.model_gemini_image,
    }[family]


_DISPATCHERS = {
    ModelFamily.A_OPENAI: family_a_openai.generate,
    ModelFamily.B_DOUBAO: family_b_doubao.generate,
    ModelFamily.C_QWEN: family_c_qwen.generate,
    ModelFamily.D_GEMINI: family_d_gemini.generate,
}


def route(req: GenRequest, *, settings: Settings | None = None) -> RouteDecision:
    """决定用哪族 + 哪个 model_id。"""
    s = settings or get_settings()

    # 1) 多轮改图 → D
    if req.conversation_history:
        family = ModelFamily.D_GEMINI
    # 2) model_hint 显式指定 → 服从
    elif req.model_hint is not None:
        family = req.model_hint
    # 3) 有参考图 → B 优先（若 B 未配置则 A）
    elif req.reference_images:
        family = ModelFamily.B_DOUBAO if _model_id_for(ModelFamily.B_DOUBAO, s) else ModelFamily.A_OPENAI
    # 4) 纯文生图 → B 优先，其次 A、C
    else:
        family = (
            ModelFamily.B_DOUBAO
            if _model_id_for(ModelFamily.B_DOUBAO, s)
            else (ModelFamily.A_OPENAI if _model_id_for(ModelFamily.A_OPENAI, s)
                  else ModelFamily.C_QWEN)
        )

    model_id = _model_id_for(family, s)
    if not model_id:
        raise ValueError(
            f"路由到 {family.value} 族但未配置对应 model_id（检查 .env 的生图 model 字段）"
        )
    return RouteDecision(family=family, model_id=model_id, generate=_DISPATCHERS[family])


class Router:
    """便捷封装：持有 settings + client，对外只暴露 generate(req, out_dir)。"""

    def __init__(self, settings: Settings | None = None,
                 client: DmxapiClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or DmxapiClient(self.settings)

    def generate(self, req: GenRequest, out_dir: Path) -> GeneratedImage:
        decision = route(req, settings=self.settings)
        return decision.generate(req, client=self.client, model_id=decision.model_id,
                                 out_dir=out_dir)


__all__ = ["RouteDecision", "Router", "route"]
