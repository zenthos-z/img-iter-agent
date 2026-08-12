"""benchmark 管理路由：列出 / 详情（结构+消费者）/ 新建（multipart）/ 删 sample。

benchmark 目录结构见 ``data/benchmark.py``；写操作（建表/删 sample）在
``services/benchmark_service.py``。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import TypeAdapter

from ..models import DimensionIn, SampleIn
from ..services.benchmark_service import (
    BenchmarkNotFound,
    SampleNotFound,
    create_benchmark,
    delete_sample,
    get_benchmark_detail,
    list_benchmarks,
)
from ..services.loop_runner import LoopBusyError

router = APIRouter()


@router.get("/benchmarks")
def list_benches() -> dict:
    """所有 benchmark 摘要（管理页列表）。"""
    return {"benches": list_benchmarks()}


@router.get("/benchmarks/{bench_id}")
def get_bench(bench_id: str) -> dict:
    """单个 benchmark 的结构（维度/题目/文件树）+ 消费者。"""
    try:
        return get_benchmark_detail(bench_id)
    except BenchmarkNotFound:
        raise HTTPException(status_code=404, detail=f"benchmark {bench_id} 不存在")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/benchmarks", status_code=201)
async def create_bench_endpoint(request: Request) -> dict:
    """新建 benchmark（multipart）：表单字段 + 维度/题目 JSON + 各 sample 的 target 图。

    - 表单字段：bench_id / scene / description / scoring_method / task_type / views
    - ``dimensions``、``samples`` 为 JSON 字符串（由前端序列化）
    - target 图文件名形如 ``target_<sample_id>``，按 sample_id 归属
    """
    form = await request.form()
    bench_id = (form.get("bench_id") or "").strip()
    scene = form.get("scene") or None
    description = form.get("description") or None
    scoring_method = form.get("scoring_method") or None
    task_type = (form.get("task_type") or "three_view_whitebg_single_image").strip()
    views = form.get("views") or None

    try:
        dimensions = TypeAdapter(list[DimensionIn]).validate_json(form.get("dimensions") or "[]")
        samples = TypeAdapter(list[SampleIn]).validate_json(form.get("samples") or "[]")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"维度/题目解析失败: {e}")

    target_files: dict[str, bytes] = {}
    for key in form:
        if not key.startswith("target_"):
            continue
        upload = form[key]
        if not getattr(upload, "filename", None):
            continue
        target_files[key[len("target_"):]] = await upload.read()

    try:
        bid = create_benchmark(
            bench_id=bench_id, scene=scene, description=description,
            scoring_method=scoring_method, task_type=task_type, views=views,
            dimensions=dimensions, samples=samples, target_files=target_files,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"bench_id": bid}


@router.delete("/benchmarks/{bench_id}/samples/{sample_id}", status_code=204)
def delete_sample_endpoint(bench_id: str, sample_id: str) -> None:
    """删一道 sample（+ 其所有 loop + human_hints + 人工排序）；不动跨 loop 蒸馏 skill。"""
    try:
        delete_sample(bench_id, sample_id)
    except BenchmarkNotFound:
        raise HTTPException(status_code=404, detail=f"benchmark {bench_id} 不存在")
    except SampleNotFound:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} 不存在")
    except LoopBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
