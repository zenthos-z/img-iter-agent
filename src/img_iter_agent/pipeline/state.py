"""闭环 A 的跨轮 State（LangGraph）。

状态设计（ARCH §2 + ADR-007）：
  - round：轮次（自增）
  - model：本闭环**固定**的模型（分开独立评测；不累加）
  - images：每轮生成的图路径，operator.add 累加（跨轮保留历史）
  - verdicts：每轮的 CriticVerdict，累加
  - attempts：每轮的 AttemptRecord，累加（写 trajectory 用）
  - decision：人工裁决（continue / stop / 调方向），由 human_review 的 interrupt 注入

用 operator.add reducer：节点返回的 list 会拼到既有 list 上，而非覆盖。

注：generator→critic→summarizer 之间传递的本轮临时产物（GenOutcome / CriticVerdict）
不在此 TypedDict 声明里（它们不持久化、不参与 reducer）。节点函数内用 state.get(...) 取
后做局部类型标注，见 graph.py。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..agents.generator import GenOutcome
from ..memory.schema import AttemptRecord, CriticVerdict


class RunState(TypedDict, total=False):
    """闭环 A 的状态。total=False 让首轮不必填全部字段。"""

    round: int
    model: str
    bench_id: str
    sample_id: str
    run_id: str

    # 累加器：跨轮历史
    images: Annotated[list[str], operator.add]  # 生成图路径（相对 run 目录）
    verdicts: Annotated[list[CriticVerdict], operator.add]
    attempts: Annotated[list[AttemptRecord], operator.add]

    # 本轮临时（节点间传递；不参与 reducer，每轮覆盖）
    # 注意：必须声明在 schema 里，否则 LangGraph 会丢弃未声明的 state 键。
    _outcome: GenOutcome  # 本轮生成产出（generator → critic/summarizer）
    _verdict: CriticVerdict  # 本轮评分（critic → summarizer/human_review）

    decision: str  # continue / stop / <方向调整说明>
    last_error: str  # 本轮异常（若有），便于人工审批时看到


# graph 编译产物的宽松类型（langgraph 的 CompiledStateGraph 泛型繁琐，这里用 Any）
CompiledGraph = Any


__all__ = ["CompiledGraph", "RunState"]
