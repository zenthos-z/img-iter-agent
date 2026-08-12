"""generator_skills_source 单测：per-bench 蒸馏技能解析（SkillsMiddleware source 定位）。

验证 generator 的 skills source 按 benchmark 切换：有 skill_package→返回 bench 目录；
未蒸馏/无 bench_id/仅扁平 SKILL.md→None（generator 裸跑）。
"""

from __future__ import annotations

from pathlib import Path

from img_iter_agent.memory.experience import generator_skill_fs, generator_skills_source, slugify_bench


def _make_skill_pkg(data_root: Path, bench_id: str) -> None:
    """造一个规范 skill_package：<data_root>/experience/<bench>/<slug>/SKILL.md。"""
    pkg = data_root / "experience" / bench_id / slugify_bench(bench_id)
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "SKILL.md").write_text(
        "---\nname: {slug}\ndescription: test\n---\nbody\n".format(slug=slugify_bench(bench_id)),
        encoding="utf-8",
    )


def test_returns_bench_dir_when_skill_package_exists(tmp_path: Path) -> None:
    _make_skill_pkg(tmp_path, "anthropic_og_style")
    src = generator_skills_source(tmp_path, "anthropic_og_style")
    assert src == tmp_path / "experience" / "anthropic_og_style"


def test_returns_none_when_not_distilled(tmp_path: Path) -> None:
    # 目录根本不存在
    assert generator_skills_source(tmp_path, "never_distilled") is None
    # experience/<bench>/ 存在但无 skill_package 子目录
    (tmp_path / "experience" / "empty_bench").mkdir(parents=True)
    assert generator_skills_source(tmp_path, "empty_bench") is None


def test_returns_none_when_no_bench_id(tmp_path: Path) -> None:
    assert generator_skills_source(tmp_path, "") is None


def test_ignores_flat_skill_md(tmp_path: Path) -> None:
    # 扁平 SKILL.md（save_general_experience 产物）不应被识别——SkillsMiddleware 只认子目录
    bench = tmp_path / "experience" / "some_bench"
    bench.mkdir(parents=True)
    (bench / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
    assert generator_skills_source(tmp_path, "some_bench") is None


# ---------------------------------------------------------------------------
# generator_skill_fs：为 Generator 构造有界 FS（read_file 钉死在技能包内）
# ---------------------------------------------------------------------------


def test_skill_fs_bounded_when_package_exists(tmp_path: Path) -> None:
    """有 skill_package → 返回有界 (backend, permissions, source_rel)，read 仅放行 <bench>/<slug>/**。"""
    bench_id = "anthropic_og_style"
    slug = slugify_bench(bench_id)
    _make_skill_pkg(tmp_path, bench_id)
    src = generator_skills_source(tmp_path, bench_id)
    assert src is not None

    fs = generator_skill_fs(src)
    assert fs is not None, "已蒸馏 bench 应返回有界 FS"
    backend, permissions, source_rel = fs

    assert source_rel == bench_id, "skills source 相对 backend root = bench_id 目录名"
    assert len(permissions) == 2, "一条 allow（技能包）+ 一条 deny（兜底）"
    allow, deny = permissions
    assert allow.mode == "allow" and deny.mode == "deny"
    assert allow.operations == ["read"], "Generator 只读、永不写"
    assert allow.paths == [f"/{bench_id}/{slug}/**"], "仅放行本 bench 技能包子树"
    assert deny.paths == ["/**"], "deny /** 兜底（first-match-wins，allow 在前）"
    # backend root（cwd）锚定在 experience/（覆盖技能包即可）
    assert str(backend.cwd).endswith("experience")


def test_skill_fs_none_when_not_distilled(tmp_path: Path) -> None:
    """None / 不存在 / 无技能包 → None（Generator 裸跑）。"""
    assert generator_skill_fs(None) is None
    assert generator_skill_fs(tmp_path / "missing") is None
    # source 存在但无 <slug>/SKILL.md
    empty_src = tmp_path / "experience" / "empty_bench"
    empty_src.mkdir(parents=True)
    assert generator_skill_fs(empty_src) is None


def test_skill_fs_readonly_denies_writes(tmp_path: Path) -> None:
    """deny 规则同时 deny read+write → Generator 无任何写权限。"""
    _make_skill_pkg(tmp_path, "furniture_product_whitebg")
    src = generator_skills_source(tmp_path, "furniture_product_whitebg")
    fs = generator_skill_fs(src)
    assert fs is not None
    _backend, permissions, _rel = fs
    deny = [p for p in permissions if p.mode == "deny"][0]
    assert "write" in deny.operations, "兜底 deny 必须含 write——Generator 永不写"
