"""魔改版 skill-creator（skill-author）技能包 + 结构校验 单测。

验证：① bundled skill-author 文件齐全；② quick_validate 原样可用（自校验 skill-author）；
③ validate_skill_md（assembly 路径用的轻量校验）规则正确；④ assemble_skill_package 的
description sanitize（剥尖括号——quick_validate 硬规则）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from img_iter_agent.agents.experience_distiller import (
    _SKILL_AUTHOR_DIR,
    _read_skill_author_file,
    _skill_author_methodology,
    _skill_author_review_prompt,
)
from img_iter_agent.memory.experience import (
    AuthoredReference,
    AuthoredSkill,
    GeneralExperience,
    assemble_skill_package,
    validate_skill_md,
)

SKILL_AUTHOR = _SKILL_AUTHOR_DIR
QUICK_VALIDATE = SKILL_AUTHOR / "scripts" / "quick_validate.py"


# ---- bundled 文件齐全 ----


def test_skill_author_bundle_files_exist():
    """skill-author（魔改 skill-creator）四件套齐全。"""
    assert (SKILL_AUTHOR / "SKILL.md").exists()
    assert (SKILL_AUTHOR / "references" / "skill_writing_guide.md").exists()
    assert (SKILL_AUTHOR / "references" / "quality_checklist.md").exists()
    assert QUICK_VALIDATE.exists()


def test_methodology_loaded_into_prompt():
    """方法论全文注入 authoring system prompt（agent 真被武装，非 35 行浓缩）。"""
    m = _skill_author_methodology()
    assert len(m) > 3000, "方法论过短，疑似仍是浓缩版"
    assert "经验技能" in m
    assert "pushy" in m.lower()
    # 写作指南附录拼进来了
    assert "渐进披露" in m or "Progressive" in m or "Anatomy" in m


def test_review_prompt_has_checklist():
    r = _skill_author_review_prompt()
    assert "评审员" in r
    assert "quality checklist" in r or "结构合规" in r


# ---- quick_validate 原样可用（官方脚本，自校验 skill-author）----


def test_quick_validate_runs_and_self_validates():
    """原样复制的 quick_validate.py 能跑，且 skill-author 自身合法。"""
    result = subprocess.run(
        [sys.executable, str(QUICK_VALIDATE), str(SKILL_AUTHOR)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"quick_validate 失败：{result.stdout}\n{result.stderr}"
    assert "valid" in result.stdout.lower()


# ---- validate_skill_md（assembly 路径用的同规则校验）----


def _skill_md(name: str, desc: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n# body\n"


def test_validate_skill_md_accepts_valid():
    ok, msg = validate_skill_md(_skill_md("anthropic-og-style", "输入文章产出策略. 触发于x."))
    assert ok, msg


def test_validate_skill_md_rejects_angle_brackets():
    ok, msg = validate_skill_md(_skill_md("x-y", "a <b> c"))
    assert not ok and "angle bracket" in msg.lower()


def test_validate_skill_md_rejects_non_kebab_name():
    ok, msg = validate_skill_md(_skill_md("Bad_Name", "ok desc"))
    assert not ok and "kebab" in msg.lower()


def test_validate_skill_md_rejects_long_description():
    ok, msg = validate_skill_md(_skill_md("x", "d" * 1025))
    assert not ok and "1024" in msg


def test_validate_skill_md_rejects_missing_frontmatter():
    ok, msg = validate_skill_md("# no frontmatter")
    assert not ok and "frontmatter" in msg.lower()


# ---- assemble_skill_package description sanitize ----


def test_assemble_sanitizes_description_brackets(tmp_path):
    """description 含尖括号 → assembly 剥掉（quick_validate 硬规则），落盘 SKILL.md 过校验。"""
    authored = AuthoredSkill(
        summary="x",
        skill_name="will-be-overwritten",
        description="输入文章→产出<策略>. 触发于y.",  # 含尖括号
        skill_md="# 技能\n\n> 必读 lessons.md\n\n## 工作流\n1. x",
        references=[AuthoredReference(path="style_guide.md", content="# 指南")],
        asset_paths=[],
    )
    exp = GeneralExperience(bench_id="test-bench")
    pkg = assemble_skill_package(tmp_path, tmp_path, "test-bench", authored, exp)
    text = (pkg / "SKILL.md").read_text(encoding="utf-8")
    # 尖括号已剥
    assert "<" not in text.split("---")[1] and ">" not in text.split("---")[1]
    # 落盘后过结构校验
    ok, msg = validate_skill_md(text)
    assert ok, msg


def test_assemble_writes_lessons_reference(tmp_path):
    """references/lessons.md 由 general.json 渲染（单一源），agent 不写。"""
    from img_iter_agent.memory.experience import DistilledLesson

    exp = GeneralExperience(
        bench_id="test-bench",
        lessons=[DistilledLesson(dim="d1", insight="i1", confidence=0.8, category="c1")],
    )
    authored = AuthoredSkill(
        summary="x", skill_name="b", description="ok desc",
        skill_md="# body", references=[], asset_paths=[],
    )
    pkg = assemble_skill_package(tmp_path, tmp_path, "test-bench", authored, exp)
    lessons_md = (pkg / "references" / "lessons.md").read_text(encoding="utf-8")
    assert "d1" in lessons_md and "i1" in lessons_md


def test_read_skill_author_file_missing_returns_empty():
    assert _read_skill_author_file("nonexistent.md") == ""
