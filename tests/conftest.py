"""pytest 公共夹具。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# 测试默认把 LangSmith 指向「立即拒绝」的本地端点 + 静默日志：tracing 仍开启（语义不变，
# 也不污染 test_tracing 的状态），但后台上传瞬间连接失败、不打日志、不 30s 超时。
# test_tracing 的 captured_runs fixture 会 monkeypatch 成进程内 recorder（不走上传）。
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_API_KEY", "test-disabled")
os.environ.setdefault("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-disabled")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "http://127.0.0.1:9")
import logging

logging.getLogger("langsmith").setLevel(logging.CRITICAL)

# 项目根（tests/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_ID = "furniture_product_whitebg"


@pytest.fixture(scope="session")
def bench_id() -> str:
    return BENCH_ID


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def repo_data_root() -> Path:
    """真实仓库的 data/ 目录（读真实 benchmarks/runs，只读用途）。"""
    return PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def tmp_run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """独立的临时 data_root，run/analyses 写这里，绝不污染真实 data/。"""
    return tmp_path_factory.mktemp("data_root")
