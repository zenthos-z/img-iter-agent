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
from typing import Any

from langchain_core.runnables import RunnableConfig
from langsmith import traceable

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


def family_for_model_id(model_id: str, settings: Settings) -> ModelFamily | None:
    """反查：一个 model_id 属于哪族（按 .env 配置匹配）。

    用于把 loop 启动时选的 model_id 转成 ModelFamily，作为 model_hint 强制路由，
    避免 Router 按自动规则选到别的模型（用户选的模型不生效）。
    """
    for fam in (ModelFamily.A_OPENAI, ModelFamily.B_DOUBAO, ModelFamily.C_QWEN, ModelFamily.D_GEMINI):
        if _model_id_for(fam, settings) == model_id:
            return fam
    return None


_DISPATCHERS = {
    ModelFamily.A_OPENAI: family_a_openai.generate,
    ModelFamily.B_DOUBAO: family_b_doubao.generate,
    ModelFamily.C_QWEN: family_c_qwen.generate,
    ModelFamily.D_GEMINI: family_d_gemini.generate,
}


def route(req: GenRequest, *, settings: Settings | None = None) -> RouteDecision:
    """决定用哪族 + 哪个 model_id。

    优先级（高→低）：
      0) ``req.model_id`` —— agent 显式选的生图 model_id（限定 .env 已配置的 4 族集合），
         反查族后直接采用。**最高优先级**：edit_previous 若与非 D 族冲突，由调用方（工具层）
         退化为重新生成并告警。
      1) ``conversation_history`` 非空 → D（多轮改图，D 唯一支持）
      2) ``model_hint`` 显式指定 → 服从
      3) 有参考图 → B 优先（B 未配置则 A）
      4) 纯文生图 → B 优先，其次 A、C
    """
    s = settings or get_settings()

    # 0) agent 显式选了 model_id → 服从（动作空间 A 的 model 杠杆）
    if req.model_id:
        fam = family_for_model_id(req.model_id, s)
        if fam is None:
            raise ValueError(
                f"agent 选的 model_id={req.model_id!r} 不在已配置的 4 族集合内（检查 .env 生图 model 字段）"
            )
        return RouteDecision(family=fam, model_id=req.model_id, generate=_DISPATCHERS[fam])

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

    def generate(self, req: GenRequest, out_dir: Path,
                 *, config: RunnableConfig | None = None) -> GeneratedImage:
        """出图，并作为一条 LangSmith chain run 上报「请求→产物」映射。

        出图本身的 tool 埋点在 client.py:_trace_image_call（run_type=tool），这里只多记
        一层映射（family/model_id/prompt→产物路径），便于在 LangSmith 关联请求与产物。

        请求摘要放 `_do(req_summary)` 的**函数入参**——@traceable 把入参记为 run inputs，
        LangSmith UI 的 Input 面板直接可见（含参考图/改图历史的**真实路径**；早期版本
        `_do()` 无参 → inputs 恒为 null，参考图到底传没传在 UI 里根本看不到）。
        真实 GeneratedImage 通过 holder 旁路返回（与 client.py 的 _trace_image_call 同模式，
        避免把含 Path 的大对象塞进 trace output）。
        """
        ls_extra: Any = {"config": config} if config is not None else {}
        holder: dict[str, Any] = {}

        req_summary = {
            "prompt": req.prompt,
            "reference_images": [str(p) for p in req.reference_images],
            "conversation_history": [str(p) for p in req.conversation_history],
            "size": {"tier": req.size.tier, "ratio": req.size.ratio,
                     "pixels": req.size.pixels},
            "model_hint": req.model_hint.value if req.model_hint else None,
            "model_id": req.model_id,
            "negative_prompt": req.negative_prompt,
            "seed": req.seed,
            "steps": req.steps,
        }

        @traceable(name="router.generate", run_type="chain")
        def _do(req_summary: dict) -> dict:
            decision = route(req, settings=self.settings)
            img = decision.generate(req, client=self.client, model_id=decision.model_id,
                                    out_dir=out_dir)
            holder["img"] = img
            return {"family": decision.family.value, "model_id": decision.model_id,
                    "request_prompt": req.prompt, "output_path": str(img.image_path),
                    "model": img.model}

        _do(req_summary, langsmith_extra=ls_extra)
        return holder["img"]


__all__ = ["RouteDecision", "Router", "family_for_model_id", "route"]
