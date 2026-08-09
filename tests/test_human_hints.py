"""人工提示词（human hints）测试：存储层 + LoopRunner 运行时 + config 注入 + agent 追加。

覆盖（见 plans/wise-dreaming-shell.md）：
  1. data/human_hints.py：sample/loop 两种 scope 的读写、merge 去重、effective 合并、add/remove。
  2. LoopRunner：add_hint/remove_hint/get_hints 改 handle.hints + 落盘；sample-scope 跨 runner 可见。
  3. _cfg_with_round：把 handle.hints 注入 config["configurable"]["human_hints"]（保留 thread_id）。
  4. Generator/Critic._build_user_content：extra_hints 追加【人工补充要求/额外评分准则】段。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from img_iter_agent.agents.critic import Critic
from img_iter_agent.agents.generator import Generator
from img_iter_agent.config import Settings
from img_iter_agent.data import human_hints as hh
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.web.services.loop_runner import LoopHandle, LoopRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OG_BENCH = "anthropic_og_style"


# ---------------------------------------------------------------------------
# 1. 存储层（纯 tmp_path，无 benchmark 依赖）
# ---------------------------------------------------------------------------


def test_sample_hints_roundtrip(tmp_path):
    assert hh.load_sample_hints(tmp_path, "b", "s001") == []
    hh.save_sample_hints(tmp_path, "b", "s001",
                         [{"agent": "critic", "text": "准则1", "scope": "sample"}])
    out = hh.load_sample_hints(tmp_path, "b", "s001")
    assert len(out) == 1
    assert out[0]["text"] == "准则1" and out[0]["scope"] == "sample"
    assert out[0]["id"]  # 自动补 id


def test_loop_hints_roundtrip(tmp_path):
    store = RunStore.create("b-s001", "b", "model", settings=Settings(data_root=tmp_path))
    hh.save_loop_hints(store, [{"agent": "generator", "text": "g1", "scope": "loop"}])
    out = hh.load_loop_hints(store)
    assert len(out) == 1 and out[0]["agent"] == "generator"
    # 落到 meta.json 的 extras
    import json
    meta = json.loads((tmp_path / "runs" / "b-s001" / "meta.json").read_text("utf-8"))
    assert meta["loop_hints"][0]["text"] == "g1"


def test_merge_hints_dedup():
    a = [{"id": "h1", "agent": "critic", "text": "x", "scope": "sample"}]
    b = [{"id": "h1", "agent": "critic", "text": "x", "scope": "sample"},
         {"id": "h2", "agent": "generator", "text": "y", "scope": "loop"}]
    m = hh.merge_hints(a, b)
    assert {h["id"] for h in m} == {"h1", "h2"}  # h1 去重，不重复


def test_load_effective_merges_both_scopes(tmp_path):
    store = RunStore.create("b-s001", "b", "model", settings=Settings(data_root=tmp_path))
    hh.save_sample_hints(tmp_path, "b", "s001",
                         [{"agent": "critic", "text": "sample准则", "scope": "sample"}])
    hh.save_loop_hints(store, [{"agent": "generator", "text": "loop要求", "scope": "loop"}])
    eff = hh.load_effective_hints(tmp_path, store, "b", "s001")
    texts = [h["text"] for h in eff]
    assert "sample准则" in texts and "loop要求" in texts


def test_add_remove_sample_hint(tmp_path):
    h = hh.add_sample_hint(tmp_path, "b", "s001", {"agent": "critic", "text": "t", "scope": "sample"})
    assert hh.load_sample_hints(tmp_path, "b", "s001")[0]["id"] == h["id"]
    assert hh.remove_sample_hint(tmp_path, "b", "s001", h["id"]) is True
    assert hh.load_sample_hints(tmp_path, "b", "s001") == []


def test_normalize_drops_empty_and_clamps(tmp_path):
    # 空 text 被过滤；非法 agent/scope 落到默认
    hh.save_sample_hints(tmp_path, "b", "s001", [
        {"agent": "critic", "text": "", "scope": "sample"},  # 空→丢
        {"agent": "weird", "text": "ok", "scope": "forever"},  # 非法→默认
    ])
    out = hh.load_sample_hints(tmp_path, "b", "s001")
    assert len(out) == 1
    assert out[0]["text"] == "ok"
    assert out[0]["agent"] in hh.AGENTS and out[0]["scope"] in hh.SCOPES


# ---------------------------------------------------------------------------
# 2. LoopRunner hints 管理（monkeypatch get_settings 指向 tmp_path）
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    settings = Settings(data_root=tmp_path)
    monkeypatch.setattr(
        "img_iter_agent.web.services.loop_runner.get_settings", lambda: settings
    )
    RunStore.create("b-s001", "b", "model", settings=settings)
    return settings


def test_runner_add_get_remove_loop_hint(runner_env):
    runner = LoopRunner()
    h = runner.add_hint("b-s001", "critic", "准则X", "loop")
    assert h and h["text"] == "准则X" and h["scope"] == "loop"
    got = runner.get_hints("b-s001")
    assert any(x["text"] == "准则X" for x in got)
    assert runner.remove_hint("b-s001", h["id"]) is True
    assert runner.get_hints("b-s001") == []


def test_runner_sample_scope_visible_to_new_runner(runner_env):
    """sample-scope 提示词落 sample 文件 → 新 runner 实例（模拟新 loop）能读到（跨 loop 同考题）。"""
    LoopRunner().add_hint("b-s001", "critic", "持久准则", "sample")
    got = LoopRunner().get_hints("b-s001")  # 全新实例，无内存 handle
    assert any(x["text"] == "持久准则" and x["scope"] == "sample" for x in got)


def test_runner_remove_unknown_returns_false(runner_env):
    assert LoopRunner().remove_hint("b-s001", "nope") is False


# ---------------------------------------------------------------------------
# 3. _cfg_with_round 注入
# ---------------------------------------------------------------------------


def test_cfg_with_round_injects_hints_and_keeps_thread_id():
    handle = LoopHandle(loop_id="b-s",
                        hints=[{"id": "h1", "agent": "critic", "text": "x", "scope": "loop"}])
    handle.cfg = {"configurable": {"thread_id": "b-s"}, "metadata": {}, "tags": []}
    cfg = LoopRunner._cfg_with_round(handle, 1, "first")
    assert cfg["configurable"]["human_hints"] == handle.hints
    assert cfg["configurable"]["thread_id"] == "b-s"  # 保留原有 configurable
    assert cfg["metadata"]["round"] == 1 and cfg["metadata"]["phase"] == "first"
    # 不 mutate base cfg
    assert "human_hints" not in handle.cfg["configurable"]


# ---------------------------------------------------------------------------
# 4. Generator / Critic _build_user_content 追加 extra_hints
# ---------------------------------------------------------------------------


@pytest.fixture()
def og_setup(tmp_path):
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    real = PROJECT_ROOT / "data" / "benchmarks" / OG_BENCH
    (tmp_path / "benchmarks" / OG_BENCH).symlink_to(real)
    lb = load_benchmark(OG_BENCH, settings=Settings(data_root=tmp_path))
    return lb


def _text(content) -> str:
    return content if isinstance(content, str) else content[0]["text"]


def test_generator_build_user_content_appends_hints(og_setup):
    sample = og_setup.sample("s001")
    gen = Generator(router=None, chat_model=None, data_root=None)  # data_root=None：不读经验文件
    text = _text(gen._build_user_content(sample, 1, None, [], extra_hints=["画面中手应是连续扁平线条"]))
    assert "画面中手应是连续扁平线条" in text
    assert "人工补充要求" in text
    # 无 hints 时不追加该段
    text0 = _text(gen._build_user_content(sample, 1, None, []))
    assert "人工补充要求" not in text0


def test_critic_build_user_content_appends_hints(og_setup):
    lb = og_setup
    sample = lb.sample("s001")
    critic = Critic(chat_model=None, bench=lb.bench)
    text = _text(critic._build_user_content(
        sample.target_path, [], sample.spec, extra_hints=["额外评分准则 XYZ"]
    ))
    assert "额外评分准则 XYZ" in text
    assert "额外评分准则" in text
    text0 = _text(critic._build_user_content(sample.target_path, [], sample.spec))
    assert "额外评分准则" not in text0
