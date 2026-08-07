"""独立「经验蒸馏」deepagent：跨 loop 总结通用经验。

与 in-loop Summarizer 解耦——本 agent 纯离线，由 CLI/web 触发，读一批 run 的 trajectory +
conclusions（in-loop Summarizer 机器验证过的 effective/ineffective），用 LLM 跨 run 归纳出
可复用的通用经验（dos/donts + 证据引用），写到共享 store `<data_root>/experience/<bench>/general.json`。

机制与 Generator/Critic 一致：`create_deep_agent` + `response_format=DistilledExperience` + 跨 run
聚合工具 + skill，`checkpointer=None` 每次跑完。agent 跑飞 → 安全降级（空 lessons + 占位 summary）。
"""

from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..memory.experience import DistilledExperience, GeneralExperience
from ..memory.schema import Benchmark
from .agent_config_loader import load_system_prompt
from .tools.experience_tools import make_experience_tools

_DEFAULT_DISTILLER_SYS = (
    "你是生图经验蒸馏员。从一批已完成 loop 的 trajectory + 已验证结论里，跨 run 归纳可复用的通用经验。"
    "以 query_dim_history 看到的「已验证 effective/ineffective」为锚（比单看分数可靠），跨 run 找反复"
    "出现的有效/无效做法，写成 dos/donts。每条经验引用来源 run，诚实给置信度。最后结构化输出 "
    "summary + lessons。更详细方法论见 skills/experience-distiller。"
)


def _merge_recursion(config: RunnableConfig | None, limit: int) -> dict:
    cfg: dict = dict(config) if config else {}
    cfg["recursion_limit"] = limit
    return cfg


class ExperienceDistiller:
    """跨 loop 经验蒸馏 deepagent。chat model + run_dirs + bench 注入。"""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        run_dirs: list[Path],
        bench: Benchmark,
        system_prompt: str | None = None,
        skills_dir: Path | str | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.run_dirs = list(run_dirs)
        self.bench = bench
        self.system_prompt = system_prompt or load_system_prompt(
            "experience_distiller", _DEFAULT_DISTILLER_SYS,
        )
        self.skills_dir = str(skills_dir) if skills_dir else None

    def distill(self, *, config: RunnableConfig | None = None) -> GeneralExperience:
        """跨 run 蒸馏通用经验，返回 GeneralExperience（含 source_runs/bench_id）。"""
        tools = make_experience_tools(self.run_dirs, self.bench)
        agent = create_deep_agent(
            model=self.chat_model, tools=tools,
            system_prompt=self.system_prompt,
            skills=[self.skills_dir] if self.skills_dir else None,
            response_format=DistilledExperience, checkpointer=None,
            name="experience_distiller",
        )
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=self._build_user_content())]},
                config=_merge_recursion(config, 30),
            )
            out: DistilledExperience | None = result.get("structured_response")
        except Exception:  # noqa: BLE001  agent 跑飞 → 安全降级
            out = None
        out = out or DistilledExperience(summary="(蒸馏失败，已降级)", lessons=[])

        return GeneralExperience(
            bench_id=self.bench.bench_id,
            source_runs=[rd.name for rd in self.run_dirs],
            summary=out.summary,
            lessons=out.lessons,
        )

    def _build_user_content(self) -> str:
        dims = ", ".join(d.dim for d in self.bench.score_dimensions)
        run_ids = [rd.name for rd in self.run_dirs]
        return (
            f"benchmark={self.bench.bench_id}，评分维度：{dims}。\n"
            f"待分析 run：{run_ids}。\n"
            "请跨 run 归纳通用经验：先 list_runs 总览，再对每个维度 query_dim_history 看改动史，"
            "归纳出 dos/donts（引用来源 run，如 <run_id>/round3），最后结构化输出 summary + lessons。"
        )


__all__ = ["ExperienceDistiller"]
