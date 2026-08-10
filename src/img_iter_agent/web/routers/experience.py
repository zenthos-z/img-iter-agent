"""通用经验路由：展示 / 蒸馏触发 / 状态轮询 / 导出 SKILL.md。

载体 general.json + SKILL.md 在 ``data/experience/<bench_id>/`` 下（per-bench）。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ...config import get_settings
from ...memory.experience import (
    experience_skill_md_path,
    load_general_experience,
    render_experience_skill_md,
    skill_package_dir,
)
from ..models import DistillStatusOut, LessonEdit, LessonRefute
from ..services.data_access import build_general_experience, mutate_lesson
from ..services.distiller_runner import get_distiller_runner

router = APIRouter()


@router.get("/experience/{bench_id}")
def get_experience(bench_id: str) -> dict:
    """读某 bench 的跨 loop 通用经验（general.json → 自描述模型）。"""
    return build_general_experience(bench_id).model_dump()


@router.post("/experience/{bench_id}/distill")
def distill(bench_id: str) -> dict:
    """触发异步经验蒸馏（后台线程；前端轮询 /distill/status）。"""
    get_distiller_runner().trigger(bench_id)
    return {"ok": True, "distill_triggered": True}


@router.patch("/experience/{bench_id}/lessons/{lesson_id}")
def edit_lesson(bench_id: str, lesson_id: str, edit: LessonEdit) -> dict:
    """人工编辑一条 lesson（改内容即重新启用为 active）。"""
    out = mutate_lesson(bench_id, lesson_id, edit=edit)
    if out is None:
        raise HTTPException(status_code=404, detail="lesson 不存在")
    return out.model_dump()


@router.post("/experience/{bench_id}/lessons/{lesson_id}/refute")
def refute_lesson(bench_id: str, lesson_id: str, body: LessonRefute) -> dict:
    """人工标无效（→ refuted，带理由；不再被消费，但翻新时可见勿复生）。"""
    out = mutate_lesson(bench_id, lesson_id, refute_reason=body.reason)
    if out is None:
        raise HTTPException(status_code=404, detail="lesson 不存在")
    return out.model_dump()


@router.delete("/experience/{bench_id}/lessons/{lesson_id}")
def archive_lesson(bench_id: str, lesson_id: str) -> dict:
    """人工归档（→ archived；不再被消费）。"""
    out = mutate_lesson(bench_id, lesson_id, archive=True)
    if out is None:
        raise HTTPException(status_code=404, detail="lesson 不存在")
    return out.model_dump()


@router.get("/experience/{bench_id}/distill/status")
def distill_status(bench_id: str) -> dict:
    """查蒸馏状态（前端轮询）。"""
    st = get_distiller_runner().status(bench_id)
    out = DistillStatusOut(
        bench_id=bench_id,
        state=st.state,
        message=st.message,
        n_lessons=st.n_lessons,
        updated_at=st.updated_at,
        error=st.error,
    )
    return out.model_dump()


@router.get(
    "/experience/{bench_id}/skill.md",
    response_class=PlainTextResponse,
)
def export_skill_md(bench_id: str) -> PlainTextResponse:
    """导出技能包的 SKILL.md（规范 frontmatter + 正文）。

    优先读规范技能包目录（<slug>/SKILL.md）；回落旧单文件；再回落按需渲染（存量经验）。
    """
    settings = get_settings()
    pkg = skill_package_dir(settings.data_root, bench_id) / "SKILL.md"
    if pkg.exists():
        return PlainTextResponse(
            pkg.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8"
        )
    old = experience_skill_md_path(settings.data_root, bench_id)
    if old.exists():
        return PlainTextResponse(
            old.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8"
        )
    exp = load_general_experience(settings.data_root, bench_id)
    if not exp.lessons:
        raise HTTPException(status_code=404, detail="尚无经验，先蒸馏生成")
    return PlainTextResponse(
        render_experience_skill_md(exp), media_type="text/markdown; charset=utf-8"
    )


@router.get("/experience/{bench_id}/skill.zip")
def export_skill_zip(bench_id: str):
    """导出规范技能包 zip（技能目录 zip，根 <slug>/）。

    外部 agent 工具加载/安装即可复现该 benchmark 的生成能力。
    """
    import io
    import zipfile

    from fastapi.responses import Response

    settings = get_settings()
    pkg_dir = skill_package_dir(settings.data_root, bench_id)
    if not pkg_dir.exists():
        raise HTTPException(status_code=404, detail="尚无技能包，先蒸馏生成")
    slug = pkg_dir.name
    exclude_parts = {"__pycache__", ".DS_Store", "evals"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(pkg_dir)
            if any(part in exclude_parts for part in rel.parts) or f.suffix == ".pyc":
                continue
            zf.write(f, arcname=f"{slug}/{rel.as_posix()}")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug}.zip"'},
    )
