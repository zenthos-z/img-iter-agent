"""generator 经验闭环警告注入（C）单元测试。

escalated_warnings 直接塞进初始 HumanMessage（强制 agent 看见，不依赖其自觉调工具）。
用 FakeToolCallingChatModel.calls 断言 user message 含警告文本。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from img_iter_agent.agents.generator import Generator
from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.generation.base import GeneratedImage
from img_iter_agent.generation.router import Router
from tests._fakes import FakeToolCallingChatModel


class _FakeRouter(Router):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, req, out_dir: Path, *, config=None) -> GeneratedImage:  # type: ignore[override]
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"f{self.calls}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return GeneratedImage(image_path=p, model="fake-model", endpoint="fake",
                              meta={"family": "B", "size": "2K"})


def _resp(prompt: str) -> list[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[{"name": "generate_image", "type": "tool_call",
                                           "id": "g", "args": {"prompt": prompt, "size": "2K"}}]),
        AIMessage(content="", tool_calls=[{"name": "GeneratorOutput", "type": "tool_call",
                                           "id": "s", "args": {"prompt": prompt, "delta_note": "d"}}]),
    ]


def _msg_text(msgs) -> str:
    """把首条 HumanMessage 的 content（str 或多模态 list）拍平成文本。"""
    text = ""
    for m in msgs:
        c = m.content
        if isinstance(c, str):
            text += c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    text += part.get("text", "")
    return text


@pytest.fixture()
def setup(tmp_path, bench_id):
    settings = Settings(data_root=tmp_path)
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    (tmp_path / "benchmarks" / bench_id).symlink_to(
        Path(__file__).resolve().parents[1] / "data" / "benchmarks" / bench_id)
    lb = load_benchmark(bench_id, settings=settings)
    store = RunStore.create("warn-run", bench_id, "seedream-5.0-pro",
                            settings=settings, note="warn injection test")
    return lb, store


def test_escalated_warnings_injected_into_user_message(setup):
    """带 escalated_warnings 调 generate_round → 首条 HumanMessage 含警告块。"""
    lb, store = setup
    fake = FakeToolCallingChatModel(responses=_resp("p round2"))
    gen = Generator(_FakeRouter(), chat_model=fake, skills_dir=None)
    warnings = [
        "⚠️ [consistency] 已连续失败 2 轮，prompt 微调疑似无效（模型能力上限），必须换根本思路",
    ]
    gen.generate_round(
        sample=lb.sample("s001"), out_dir=store.run_dir / "out", run_dir=store.run_dir,
        round=2, escalated_warnings=warnings,
    )
    text = _msg_text(fake.calls[0])
    assert "⚠️" in text
    assert "已连续失败" in text
    assert "consistency" in text
    assert "经验闭环警告" in text


def test_no_warnings_no_block(setup):
    """无 escalated_warnings → user message 不含警告块（首轮正常路径）。"""
    lb, store = setup
    fake = FakeToolCallingChatModel(responses=_resp("p round1"))
    gen = Generator(_FakeRouter(), chat_model=fake, skills_dir=None)
    gen.generate_round(
        sample=lb.sample("s001"), out_dir=store.run_dir / "out", run_dir=store.run_dir,
        round=1,
    )
    text = _msg_text(fake.calls[0])
    assert "经验闭环警告" not in text
