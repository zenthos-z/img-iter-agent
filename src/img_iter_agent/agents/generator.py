"""Generator：基于 Critic 反馈自动迭代优化 prompt 的生图 agent（deepagent 版）。

迭代核心（用户明确）：**自动优化生图提示词**，不是改尺寸。size 由考题固定，不动。
每轮由一个真正的 tool-using agent（`create_deep_agent`）完成：
  - 读考题指令+约束（round>1 还看上轮 Critic 失败项）→ 用工具 `query_experience` 查已验证经验
    → 构造/改进英文 prompt → 调工具 `generate_image` 出图 → 结构化输出 `GeneratorOutput`。
  - baseline_ref 指向上轮 attempt（对比迭代效果），test_variable 始终记为 "prompt"。

deepagent 在 `generate_round` 内部构建并同步跑完一轮（checkpointer=None，无断点），
作为「引擎」嵌入外层 LangGraph 的 generator 节点——`RunState`/checkpointer/trace 编排不变。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..data.benchmark import Sample
from ..generation.base import GenRequest, ModelFamily
from ..generation.image_io import file_to_data_uri
from ..generation.router import Router
from ..memory.schema import CriticItemJudgment, TestVariable
from ._agent_output import GeneratorOutput
from .agent_config_loader import load_system_prompt
from .tools.generator_tools import _size_from_str, make_generator_tools

# 代码默认 system prompt（data/agents_config/generator.md 缺失时回退）。
# 短的「始终生效」角色放这里；更长的诀窍/流程在 skills/generator/SKILL.md（agent 按需读）。
_DEFAULT_GENERATOR_SYS = (
    "你是产品白底三视图的生图提示词工程师。每轮：读考题指令与约束（round>1 时还看上轮 Critic "
    "失败项），可用 query_experience 查【本 loop】已验证经验、query_general_experience 查"
    "【跨 loop 通用经验】（先验，首题也用得上），构造或改进英文优先的生图 prompt，"
    "调 generate_image 出图，最后结构化输出 prompt 与 delta_note（本轮相对上轮改了什么）。"
    "保留原有正确部分，针对每个失败项给出具体的、可执行的正面描述（不要只写『不要 X』）。"
)


class PriorFeedback(BaseModel):
    """上轮 Critic 反馈的摘要，供 Generator 改进 prompt。"""

    failed_items: list[CriticItemJudgment] = Field(default_factory=list)
    continuous_notes: list[str] = Field(default_factory=list)  # 连续维度的低分理由


class GenOutcome(BaseModel):
    """Generator 一轮的产出。"""

    attempt_id: str
    test_variable: TestVariable | None
    baseline_ref: str | None
    gen_mode: str
    prompt: str
    size: str
    reference_image_refs: list[str]  # 相对 run 目录
    output_image_refs: list[str]  # 相对 run 目录（三视图一张图）
    model: str
    model_family: str
    improved_from_feedback: bool = False  # 本轮 prompt 是否基于上轮反馈改进
    delta_note: str | None = None  # 本轮相对上轮的改动说明（供经验闭环沉淀）


def _new_attempt_id(round: int) -> str:
    return f"a{round:03d}_{uuid.uuid4().hex[:6]}"


class Generator:
    """生图 agent。Router 与 chat model 注入；每轮内部构建 deepagent 跑完。"""

    def __init__(
        self,
        router: Router,
        *,
        chat_model: BaseChatModel,
        system_prompt: str | None = None,
        skills_dir: Path | str | None = None,
        data_root: Path | None = None,
        bench_id: str | None = None,
    ) -> None:
        self.router = router
        self.chat_model = chat_model
        self.system_prompt = system_prompt or load_system_prompt("generator", _DEFAULT_GENERATOR_SYS)
        self.skills_dir = str(skills_dir) if skills_dir else None
        # 跨 loop 通用经验库定位（query_general_experience 工具用）；None 时该工具回「未配置」。
        self.data_root = data_root
        self.bench_id = bench_id

    def generate_round(
        self,
        *,
        sample: Sample,
        out_dir: Path,
        run_dir: Path,
        round: int,
        baseline_ref: str | None = None,
        prior_feedback: PriorFeedback | None = None,
        model_hint: ModelFamily | None = None,
        config: RunnableConfig | None = None,
    ) -> GenOutcome:
        """跑一轮：deepagent 构造/改进 prompt 并出图。

        size 由考题固定（不动）。首轮纯构造；后续轮据 prior_feedback + 经验库改进 prompt。
        agent 用 response_format=GeneratorOutput 约束最终输出。
        """
        spec = sample.spec
        task = spec.task
        size_str = (task.output.get("size") if task and task.output else None) or "2K"

        # 参考：image_edit 模式用 target 作风格锚
        reference_images: list[Path] = []
        gen_mode = "text_to_image"
        if task and task.mode in ("image_edit", "multiview") and sample.target_path.exists():
            reference_images = [sample.target_path]
            gen_mode = "image_edit"

        attempt_id = _new_attempt_id(round)
        attempt_out = out_dir / attempt_id
        attempt_out.mkdir(parents=True, exist_ok=True)

        sink: dict[str, Any] = {}
        tools = make_generator_tools(
            router=self.router, sample=sample, out_dir=attempt_out, run_dir=run_dir,
            model_hint=model_hint, sink=sink,
            data_root=self.data_root, bench_id=self.bench_id,
        )
        agent = create_deep_agent(
            model=self.chat_model, tools=tools,
            system_prompt=self.system_prompt,
            skills=[self.skills_dir] if self.skills_dir else None,
            response_format=GeneratorOutput, checkpointer=None, name="generator",
        )

        user_content = self._build_user_content(sample, round, prior_feedback, reference_images)
        invoke_cfg = self._merge_recursion(config, 25)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_content)]}, config=invoke_cfg,
            )
            out: GeneratorOutput = result.get("structured_response") or GeneratorOutput(prompt="")
        except Exception:  # noqa: BLE001  agent 跑飞不能让闭环崩 → 兜底
            out = GeneratorOutput(prompt=self._base_prompt_text(sample))

        prompt = (out.prompt or "").strip() or self._base_prompt_text(sample)
        delta_note = (out.delta_note or "").strip() or None

        # 出图：agent 调了 generate_image → sink 有 ref；否则用 prompt 现场出图（兜底）
        if sink.get("ref"):
            output_refs = [sink["ref"]]
            model_used = sink.get("model", "")
            model_family = sink.get("family", "?")
        else:
            req = GenRequest(
                prompt=prompt, size=_size_from_str(size_str),
                reference_images=list(reference_images), model_hint=model_hint,
            )
            img = self.router.generate(req, out_dir=attempt_out, config=config)
            ext = img.image_path.suffix or ".png"
            dest = attempt_out / f"three_view{ext}"
            if img.image_path != dest:
                img.image_path.rename(dest)
            output_refs = [str(dest.relative_to(run_dir))]
            model_used = img.model
            model_family = img.meta.get("family", "?")

        improved = round > 1 and prior_feedback is not None and bool(prior_feedback.failed_items)

        return GenOutcome(
            attempt_id=attempt_id,
            test_variable="prompt" if round > 1 else None,
            baseline_ref=baseline_ref,
            gen_mode=gen_mode,
            prompt=prompt,
            delta_note=delta_note,
            size=size_str,
            reference_image_refs=[str(p.resolve()) for p in reference_images],
            output_image_refs=output_refs,
            model=model_used,
            model_family=model_family,
            improved_from_feedback=improved,
        )

    # --- 辅助 ---

    def _build_user_content(
        self, sample: Sample, round: int,
        prior_feedback: PriorFeedback | None, reference_images: list[Path],
    ) -> list[dict] | str:
        """构造初始 HumanMessage 内容：指令+约束(+上轮失败项)+参考图(image_url)。"""
        instr = (sample.spec.task.instruction if sample.spec.task else None) or "生成产品白底三视图素材图"
        constraints = self._extract_constraints(sample)
        text = instr
        if constraints:
            text += "\n约束: " + json.dumps(constraints, ensure_ascii=False)
        if round > 1 and prior_feedback and prior_feedback.failed_items:
            fails = "\n".join(f"- [{it.id}] {it.reason}" for it in prior_feedback.failed_items)
            cont = ""
            if prior_feedback.continuous_notes:
                cont = "\n连续维度低分:\n" + "\n".join(f"- {n}" for n in prior_feedback.continuous_notes)
            text += (
                f"\n\n【上轮 Critic 失败项（需改进）】\n{fails}{cont}\n"
                "请先 query_experience / query_general_experience 查经验，再针对性改进 prompt，"
                "调 generate_image 出图，最后结构化输出 prompt + delta_note。"
            )
        else:
            text += (
                "\n请把指令精炼成清晰 prompt（首题可先 query_general_experience 取跨题先验），"
                "调 generate_image 出图，最后结构化输出 prompt + delta_note。"
            )

        if not reference_images:
            return text
        parts: list[dict] = [{"type": "text", "text": text}]
        for p in reference_images:
            if p.exists():
                parts.append({"type": "image_url", "image_url": {"url": file_to_data_uri(p)}})
        return parts

    def _base_prompt_text(self, sample: Sample) -> str:
        """兜底 prompt（agent 出错时用）：指令+约束，无 LLM 润色。"""
        instr = (sample.spec.task.instruction if sample.spec.task else None) or "生成产品白底三视图素材图"
        constraints = self._extract_constraints(sample)
        base = instr
        if constraints:
            base += "\n约束: " + json.dumps(constraints, ensure_ascii=False)
        return base

    def _extract_constraints(self, sample: Sample) -> dict:
        """从 content_spec 读原始 constraints 字段（若存在）。"""
        try:
            raw = sample.spec.model_dump()
            return raw.get("constraints") or {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
        """把 recursion_limit 合进 config（透传 trace 嵌套用）。"""
        cfg: dict = dict(config) if config else {}
        cfg["recursion_limit"] = limit
        return cfg


__all__ = ["GenOutcome", "Generator", "PriorFeedback"]
