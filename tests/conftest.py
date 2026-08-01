"""pytest 公共夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

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
