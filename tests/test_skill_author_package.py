"""全工具 authoring agent 的技能包装配（prepare/finalize）+ 结构校验 + skill-author 自身合法性 单测。

验证：① bundled skill-author 文件齐全 + 自身过 quick_validate/validate_skill_md（它是真 skill）；
② skill-author 作为 deepagents ``skills=`` 源**隔离加载**（只它一个，无兄弟污染）；
③ ``validate_skill_md`` 规则正确；④ ``prepare_skill_package`` 清空+建目录；
⑤ ``finalize_skill_package`` 读回 agent 写的 SKILL.md → sanitize frontmatter(name=slug/剥尖括号) →
渲染 lessons.md(单一源) → 拷 assets → 过校验；⑥ 缺 SKILL.md → None（优雅降级）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from img_iter_agent.agents.experience_distiller import (
    _REPO_ROOT,
    _SKILL_AUTHOR_PARENT,
    _backend_root,
    _skill_author_source,
)
from img_iter_agent.memory.experience import (
    DistilledLesson,
    GeneralExperience,
    finalize_skill_package,
    prepare_skill_package,
    slugify_bench,
    validate_skill_md,
)

SKILL_AUTHOR = _SKILL_AUTHOR_PARENT / "skill-author"
QUICK_VALIDATE = SKILL_AUTHOR / "scripts" / "quick_validate.py"


# ---- bundled 文件齐全 + skill-author 自身合法 ----


def test_skill_author_bundle_files_exist():
    """skill-author（魔改 skill-creator）四件套齐全。"""
    assert (SKILL_AUTHOR / "SKILL.md").exists()
    assert (SKILL_AUTHOR / "references" / "skill_writing_guide.md").exists()
    assert (SKILL_AUTHOR / "references" / "quality_checklist.md").exists()
    assert QUICK_VALIDATE.exists()


def test_skill_author_skill_self_validates():
    """skill-author 是真 skill（被 deepagents 加载），自身须过 validate_skill_md。"""
    text = (SKILL_AUTHOR / "SKILL.md").read_text(encoding="utf-8")
    ok, msg = validate_skill_md(text)
    assert ok, f"skill-author SKILL.md 自身不合法：{msg}"


def test_quick_validate_runs_and_self_validates():
    """原样复制的 quick_validate.py 能跑，且 skill-author 自身合法。"""
    result = subprocess.run(
        [sys.executable, str(QUICK_VALIDATE), str(SKILL_AUTHOR)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, f"quick_validate 失败：{result.stdout}\n{result.stderr}"
    assert "valid" in result.stdout.lower()


# ---- skill-author 作为 skills 源隔离加载（只它一个，无兄弟污染）----


def test_skill_author_source_isolated():
    """skills 源 = skill_authoring/ 父目录；deepagents 只在该源发现 skill-author（隔离）。"""
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.skills import _list_skills

    root = _backend_root(_REPO_ROOT / "data")  # 默认 data_root 在 repo 内 → root=repo
    backend = FilesystemBackend(root_dir=str(root))
    src = _skill_author_source(root)
    assert src.endswith("skill_authoring"), src
    skills = _list_skills(backend, src)
    names = [s["name"] for s in skills]
    assert names == ["skill-author"], f"应隔离加载仅 skill-author，实际：{names}"


# ---- validate_skill_md（finalize 路径用的同规则校验）----


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


# ---- prepare_skill_package ----


def test_prepare_cleans_and_mkdirs(tmp_path):
    """prepare 清空旧包残留 + 建 references/assets 子目录。"""
    from img_iter_agent.memory.experience import skill_package_dir

    pkg = skill_package_dir(tmp_path, "test-bench")
    pkg.mkdir(parents=True)
    (pkg / "references").mkdir(parents=True)
    (pkg / "references" / "stale.md").write_text("old", encoding="utf-8")  # 旧残留

    out = prepare_skill_package(tmp_path, "test-bench")
    assert out == pkg
    assert (pkg / "references").is_dir()
    assert (pkg / "assets").is_dir()
    assert not (pkg / "references" / "stale.md").exists(), "旧残留应被清空"


# ---- finalize_skill_package ----


def _write_agent_skill_md(pkg: Path, *, name: str = "wrong-name", desc: str = "ok desc") -> None:
    """模拟全工具 agent 直接 write_file 的 SKILL.md（可能 name 不对 / desc 含尖括号）。"""
    (pkg / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n# 技能\n\n> 必读 lessons.md\n\n## 工作流\n1. x\n",
        encoding="utf-8",
    )


def test_finalize_sanitizes_frontmatter_and_renders_lessons(tmp_path):
    """agent 写的 SKILL.md：name 不对→强制 slug；desc 含尖括号→剥掉；并渲染 lessons.md（单一源）。"""
    data_root = tmp_path / "data"
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    bench_id = "test-bench"
    slug = slugify_bench(bench_id)

    pkg = prepare_skill_package(data_root, bench_id)
    _write_agent_skill_md(pkg, name="wrong-name", desc="输入文章→产出<策略>. 触发于y.")  # 含尖括号

    exp = GeneralExperience(
        bench_id=bench_id,
        lessons=[DistilledLesson(dim="d1", insight="i1", confidence=0.8, category="c1")],
    )
    out = finalize_skill_package(data_root, bench_dir, bench_id, exp, asset_paths=[])
    assert out is not None

    text = (pkg / "SKILL.md").read_text(encoding="utf-8")
    fm = text.split("---")[1]
    assert "wrong-name" not in fm, "agent 写错的 name 应被强制覆盖为 slug"
    assert slug in fm
    assert "<" not in fm and ">" not in fm, "尖括号应被剥掉"
    ok, msg = validate_skill_md(text)
    assert ok, msg

    # lessons.md 由 general.json 渲染（agent 不写）
    lessons_md = (pkg / "references" / "lessons.md").read_text(encoding="utf-8")
    assert "d1" in lessons_md and "i1" in lessons_md


def test_finalize_copies_assets(tmp_path):
    """asset_paths（bench 内相对）被拷进 assets/。"""
    data_root = tmp_path / "data"
    bench_dir = tmp_path / "bench"
    asset_src = bench_dir / "reference_style" / "hand.png"
    asset_src.parent.mkdir(parents=True)
    asset_src.write_bytes(b"\x89PNG fake")

    pkg = prepare_skill_package(data_root, "test-bench")
    _write_agent_skill_md(pkg)

    exp = GeneralExperience(bench_id="test-bench")
    finalize_skill_package(
        data_root, bench_dir, "test-bench", exp, asset_paths=["reference_style/hand.png"],
    )
    assert (pkg / "assets" / "hand.png").exists()
    assert (pkg / "assets" / "hand.png").read_bytes() == b"\x89PNG fake"


def test_finalize_overwrites_agent_lessons_md(tmp_path):
    """agent 误写 references/lessons.md → finalize 用 general.json 单一源覆盖。"""
    data_root = tmp_path / "data"
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    pkg = prepare_skill_package(data_root, "test-bench")
    _write_agent_skill_md(pkg)
    (pkg / "references" / "lessons.md").write_text("# agent 误写的 lessons\n应被覆盖", encoding="utf-8")

    exp = GeneralExperience(
        bench_id="test-bench",
        lessons=[DistilledLesson(dim="real-dim", insight="real-insight", confidence=0.9, category="c")],
    )
    finalize_skill_package(data_root, bench_dir, "test-bench", exp, asset_paths=[])
    lessons_md = (pkg / "references" / "lessons.md").read_text(encoding="utf-8")
    assert "real-dim" in lessons_md
    assert "agent 误写" not in lessons_md


def test_finalize_returns_none_when_no_skill_md(tmp_path):
    """agent 未落盘 SKILL.md（失败/死循环）→ finalize 返回 None（优雅降级，exp 照存）。"""
    data_root = tmp_path / "data"
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    prepare_skill_package(data_root, "test-bench")  # 空目录，无 SKILL.md
    exp = GeneralExperience(bench_id="test-bench")
    out = finalize_skill_package(data_root, bench_dir, "test-bench", exp, asset_paths=[])
    assert out is None


def test_finalize_prepends_frontmatter_when_missing(tmp_path):
    """agent 写的 SKILL.md 无 frontmatter → finalize 前置最小合规块。"""
    data_root = tmp_path / "data"
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    pkg = prepare_skill_package(data_root, "test-bench")
    (pkg / "SKILL.md").write_text("# 技能\n\n正文无 frontmatter\n", encoding="utf-8")

    out = finalize_skill_package(data_root, bench_dir, "test-bench", GeneralExperience(bench_id="test-bench"), [])
    assert out is not None
    text = (pkg / "SKILL.md").read_text(encoding="utf-8")
    ok, msg = validate_skill_md(text)
    assert ok, msg
    assert "正文无 frontmatter" in text  # 正文保留
