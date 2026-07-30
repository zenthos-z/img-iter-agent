"""Live smoke 测试：真正调用 dmxapi 生图。

默认 SKIP（避免 CI/常规测试联网、烧钱）。仅当环境变量 IMG_ITER_LIVE_SMOKE=1 时运行，
且要求 .env 里已配好 dmxapi_key 与对应族 model_id。

手动运行（用真实 .env）：
    IMG_ITER_LIVE_SMOKE=1 .venv/bin/python -m pytest tests/test_live_smoke.py -s

每个测试只生 1 张图、用最便宜的档位，控制成本。
"""

from __future__ import annotations

import os

import pytest

_live = os.environ.get("IMG_ITER_LIVE_SMOKE") == "1"
pytestmark = pytest.mark.skipif(not _live, reason="set IMG_ITER_LIVE_SMOKE=1 to run live dmxapi calls")


@pytest.fixture()
def settings():
    from img_iter_agent.config import Settings
    s = Settings()
    if not s.dmxapi_key:
        pytest.skip("IMG_ITER_DMXAPI_KEY 未配置")
    return s


def test_live_family_b_seedream_text_to_image(settings, tmp_path):
    """B 族(seedream) 纯文生图：验证真实端点/认证/响应落盘。"""
    from img_iter_agent.generation.base import GenRequest, SizeSpec
    from img_iter_agent.generation.router import Router

    if not settings.model_seedream_pro:
        pytest.skip("seedream model_id 未配置")
    router = Router(settings=settings)
    out = router.generate(
        GenRequest(prompt="一把简约白色椅子，纯白背景，电商产品图",
                   size=SizeSpec(tier="2K")),
        out_dir=tmp_path,
    )
    assert out.image_path.exists()
    assert out.image_path.stat().st_size > 1000  # 真实图不会太小
    print(f"\n[live B] 生成图: {out.image_path} ({out.image_path.stat().st_size} bytes)")


def test_live_family_b_seedream_with_reference(settings, tmp_path, bench_id):
    """B 族(seedream) 参考图风格迁移：验证 image 字段(data-URI) 真实可用。"""
    from img_iter_agent.data.benchmark import load_benchmark
    from img_iter_agent.generation.base import GenRequest, SizeSpec
    from img_iter_agent.generation.router import Router

    if not settings.model_seedream_pro:
        pytest.skip("seedream model_id 未配置")
    lb = load_benchmark(bench_id)
    target = lb.sample("s001").target_path
    router = Router(settings=settings)
    out = router.generate(
        GenRequest(prompt="以该产品为参考，生成同一产品的正视图白底素材图",
                   reference_images=[target], size=SizeSpec(tier="2K")),
        out_dir=tmp_path,
    )
    assert out.image_path.exists()
    assert out.image_path.stat().st_size > 1000
    print(f"\n[live B+ref] 生成图: {out.image_path}")
