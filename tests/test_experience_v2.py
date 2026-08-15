"""通用经验 v2 单测：翻新状态链 + 人工兜底(mutate) + 向后兼容。"""

from __future__ import annotations

from types import SimpleNamespace

from img_iter_agent.agents.experience_distiller import ExperienceDistiller
from img_iter_agent.memory.experience import (
    DistilledLesson,
    GeneralExperience,
    RenoItem,
    RenovationPlan,
)
from img_iter_agent.web.models import LessonEdit
from img_iter_agent.web.services.data_access import mutate_lesson


def _lesson(id, dim, conf, *, category="", status="active", applies_when="always", insight=None):
    return DistilledLesson(
        id=id, dim=dim, insight=insight or f"{dim}-insight", confidence=conf,
        category=category, status=status, applies_when=applies_when,
    )


def _exp(lessons):
    return GeneralExperience(bench_id="b", lessons=lessons)


# ---- 向后兼容：旧 lesson 无 id/category/status → 自动补 ----


def test_old_lesson_backcompat_autofills():
    raw = DistilledLesson(dim="consistency", insight="布局清晰", confidence=0.9)
    assert raw.id  # 自动补
    assert raw.status == "active"
    assert raw.applies_when == "always"
    assert raw.category == ""


# ---- _merge_renovation：keep/revise/retire/new 状态链 ----


def _distiller_with_previous(prev):
    d = ExperienceDistiller.__new__(ExperienceDistiller)
    d.previous = prev
    d.run_dirs = []
    d.bench = SimpleNamespace(bench_id="b", scene="", description="", score_dimensions=[])
    return d


def test_merge_renovation_status_chain():
    prev = _exp([
        _lesson("L1", "consistency", 0.9, category="结构", applies_when="construction"),
        _lesson("L2", "artifact_defect", 0.6, category="瑕疵", applies_when="fix"),
        _lesson("L9", "x", 0.5, category="x", status="refuted"),
    ])
    plan = RenovationPlan(summary="翻新", renovation=[
        RenoItem(existing_id="L1", action="keep", reason="仍有效",
                 lesson=DistilledLesson(dim="consistency", insight="布局清晰")),
        RenoItem(existing_id="L2", action="revise", reason="原版太笼统",
                 lesson=DistilledLesson(dim="artifact_defect", insight="加软阴影更具体",
                                        confidence=0.85, category="瑕疵", applies_when="fix")),
        RenoItem(existing_id="L9", action="retire", reason="已证伪",
                 lesson=DistilledLesson(dim="x", insight="旧的错的")),
        RenoItem(action="new",
                 lesson=DistilledLesson(dim="material_texture", insight="用具体材质词",
                                        confidence=0.9, category="材质", applies_when="always")),
    ])
    exp = _distiller_with_previous(prev)._merge_renovation(plan)
    by_id = {l.id: l for l in exp.lessons}
    assert by_id["L1"].status == "active"
    # L2 被 revise → superseded 后**修剪掉**（修订历史不留，继承者携带信息）
    assert "L2" not in by_id
    # 继承者是 active，承接 dim/confidence
    succ = next(l for l in exp.lessons if l.dim == "artifact_defect" and l.status == "active")
    assert succ.confidence == 0.85
    assert by_id["L9"].status == "refuted" and "证伪" in by_id["L9"].retire_reason
    # new 进来一条 material_texture active
    new = [l for l in exp.lessons if l.dim == "material_texture" and l.status == "active"]
    assert len(new) == 1
    # active：L1 + revised successor + new = 3；refuted：L9；superseded 已修剪
    assert sum(1 for l in exp.lessons if l.status == "active") == 3
    assert all(l.status != "superseded" for l in exp.lessons)


# ---- 人工兜底 mutate_lesson：refute / archive / edit ----


def test_mutate_lesson_refute_archive_edit(tmp_path):
    from img_iter_agent.memory.experience import save_general_experience
    exp = _exp([_lesson("L1", "consistency", 0.9, category="结构")])
    save_general_experience(tmp_path, "b", exp)

    # refute
    out = mutate_lesson("b", "L1", refute_reason="测试证伪", settings=SimpleNamespace(data_root=tmp_path))
    assert out is not None
    assert out.lessons[0].status == "refuted"
    assert out.lessons[0].retire_reason == "测试证伪"

    # edit 重新启用为 active + 改内容
    out = mutate_lesson("b", "L1",
                        edit=LessonEdit(insight="改过的 insight", confidence=0.77, applies_when="fix"),
                        settings=SimpleNamespace(data_root=tmp_path))
    assert out.lessons[0].status == "active"
    assert out.lessons[0].insight == "改过的 insight"
    assert out.lessons[0].confidence == 0.77
    assert out.lessons[0].applies_when == "fix"

    # archive
    out = mutate_lesson("b", "L1", archive=True, settings=SimpleNamespace(data_root=tmp_path))
    assert out.lessons[0].status == "archived"

    # 不存在的 id
    assert mutate_lesson("b", "nope", archive=True, settings=SimpleNamespace(data_root=tmp_path)) is None
