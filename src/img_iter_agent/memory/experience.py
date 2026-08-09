"""跨 loop 通用经验：蒸馏产物 schema + 读写 + 可移植 Skill 渲染 + 策略性消费。

与 `memory/knowledge.py` 的 `conclusions.json`（每 run、规则驱动、按 (dim,change) 的
effective/ineffective）区分：
  - conclusions.json = 单 loop 内、Critic 前后对比**机器验证**的结论（in-loop Summarizer 写）。
  - general.json     = 跨 loop、LLM **综合**的通用经验（独立蒸馏器写），是 conclusions 的上层归纳。

承载形式（双轨，单一渲染源）：
  - `<data_root>/experience/<bench_id>/general.json` —— 结构化源（自描述：含 scene/dimensions）。
  - `<data_root>/experience/<bench_id>/SKILL.md`    —— 可移植载体（deepagents 原生 skill）。

消费形式（v2：策略性输入，避免库膨胀→注意力分散）：
  - system prompt 只常驻 `render_experience_index`（每 category 一行，恒定 ~6 行）。
  - 每轮 user message 按 `select_lessons` 注入 ≤K 条选定详情（R1=construction/always；R>1=failed_dims 命中）。
  - 工具 `query_general_experience` 钻取。

lesson 状态机（v2 翻新）：active / refuted / superseded / archived。消费侧只取 active；
翻新时 refuted/superseded 仍可见（避免被重新生成）。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:48] or "x"


def _stable_id(dim: str, insight: str) -> str:
    """向后兼容：旧 lesson 无 id 时，按 dim+insight 哈希补一个稳定 id。"""
    h = hashlib.sha1(f"{dim}|{insight}".encode("utf-8")).hexdigest()[:8]
    return f"{_slug(dim)}-{h}"


def new_lesson_id(category: str, dim: str) -> str:
    """新建 lesson 的 id（蒸馏器 action=new/revise 时用）：category-dim-短uuid。"""
    return f"{_slug(category) or 'general'}-{_slug(dim)}-{uuid.uuid4().hex[:6]}"


class DistilledLesson(BaseModel):
    """一条跨 loop 蒸馏出的通用经验。"""

    model_config = {"extra": "ignore"}

    dim: str = Field(description="关联的评分维度（跨维度共性可填 'general'）")
    insight: str = Field(description="一句话通用经验/规律")
    dos: list[str] = Field(default_factory=list, description="应该这样做")
    donts: list[str] = Field(default_factory=list, description="不要这样做（已验证无效）")
    evidence: list[str] = Field(
        default_factory=list,
        description="支撑来源，如 ['<run_id>/round3', ...']",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度 0-1")
    # v2 字段
    id: str = Field(default="", description="稳定标识；空则按 dim+insight 哈希补")
    category: str = Field(default="", description="粗类索引（LLM 赋，如 材质色彩/结构一致性）")
    status: str = Field(default="active", description="active/refuted/superseded/archived")
    applies_when: str = Field(
        default="always", description="construction=首轮构造 / fix=修复失败时 / always"
    )
    successor_id: str = Field(default="", description="被 revise 时指向新版 id")
    retire_reason: str = Field(default="", description="refuted/superseded 时的理由")

    @model_validator(mode="after")
    def _ensure_id(self) -> "DistilledLesson":
        if not self.id:
            self.id = _stable_id(self.dim, self.insight)
        return self


class DistilledExperience(BaseModel):
    """经验蒸馏 deepagent 的结构化输出（v1 response_format，保留向后兼容）。"""

    summary: str = Field(default="", description="跨 loop 的总体总结")
    lessons: list[DistilledLesson] = Field(default_factory=list, description="蒸馏出的通用经验")


class RenoItem(BaseModel):
    """翻新计划的一条：对某条旧 lesson 的处置（或新增）。

    lesson 必填（非可空）：Gemini function-declaration schema 不接受 anyOf/可空嵌套对象
    （会报 "didn't specify the schema type field"）。keep/retire 时 lesson 仍需给出
    （可复制上一版对应 lesson 的 dim+insight），merge 时其内容被忽略。
    """

    model_config = {"extra": "ignore"}

    existing_id: str = Field(default="", description="处置的旧 lesson id；action=new 时留空")
    lesson: DistilledLesson
    action: str = Field(
        default="new", description="keep(保留) / revise(修订，旧→superseded) / retire(废弃→refuted) / new(新增)"
    )
    reason: str = Field(default="", description="处置理由（retire/revise 必填）")


class RenovationPlan(BaseModel):
    """经验蒸馏器 v2 结构化输出：带反馈的迭代翻新（keep/revise/retire/new）。"""

    summary: str = Field(default="", description="本轮翻新的总体总结")
    renovation: list[RenoItem] = Field(default_factory=list)


class AuthoredReference(BaseModel):
    """技能包内 agent 自撰的一篇参考文档（references/<path>）。"""

    path: str = Field(description="相对 references/ 的文件名（如 style_guide.md），禁绝对路径/../")
    content: str = Field(description="Markdown 正文")


class AuthoredSkill(BaseModel):
    """skill-creator 武装的 author_skill agent 产出：规范可移植技能包内容。

    lessons.md 不在此（由 general.json 确定性渲染，单一源）；asset_paths 指向 benchmark
    实际用到的资产（style_transfer 参考图等），由 assembly 拷进 assets/。
    """

    summary: str = Field(default="", description="这个技能做什么（一句话）")
    skill_name: str = Field(default="", description="技能名（assembly 强制 = slug(bench)，保 name==dir)")
    description: str = Field(description="frontmatter description：做什么+何时触发（pushy）")
    skill_md: str = Field(description="SKILL.md 正文（无 frontmatter；<500 行，渐进披露）")
    references: list[AuthoredReference] = Field(
        default_factory=list, description="agent 自撰的域参考（style_guide/eval_criteria 等）"
    )
    asset_paths: list[str] = Field(
        default_factory=list, description="要打包的 benchmark 资产（相对 bench_dir），用到的才列"
    )


class GeneralExperience(BaseModel):
    """落盘的跨 loop 通用经验库（general.json 内容）。自描述 + 状态机。"""

    bench_id: str
    updated_at: str = ""
    source_runs: list[str] = Field(default_factory=list, description="参与蒸馏的 run_id 列表")
    summary: str = ""
    lessons: list[DistilledLesson] = Field(default_factory=list)
    scene: str = ""
    dimensions: list[str] = Field(
        default_factory=list, description="评分维度名（供消费者理解 lesson.dim 语义）"
    )
    bench_description: str = ""
    categories: list[str] = Field(default_factory=list, description="粗类缓存（派生，渲染/索引用）")


# ---------------------------------------------------------------------------
# 路径 / slug
# ---------------------------------------------------------------------------


def general_experience_path(data_root: Path, bench_id: str) -> Path:
    d = Path(data_root) / "experience" / bench_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "general.json"


def experience_skill_md_path(data_root: Path, bench_id: str) -> Path:
    d = Path(data_root) / "experience" / bench_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "SKILL.md"


def skill_package_dir(data_root: Path, bench_id: str) -> Path:
    """规范技能包目录：<data_root>/experience/<bench_id>/<slug>/（name==dir）。

    slug = slugify_bench(bench_id)，保证 skill-creator validate（name==目录名）通过。
    """
    return Path(data_root) / "experience" / bench_id / slugify_bench(bench_id)


def slugify_bench(bench_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (bench_id or "").lower()).strip("-")
    return slug[:64] or "experience"


# ---------------------------------------------------------------------------
# 策略性消费：索引 + 选择（v2 输入轴）
# ---------------------------------------------------------------------------


def _active(exp: GeneralExperience) -> list[DistilledLesson]:
    return [l for l in exp.lessons if l.status == "active"]


def render_experience_index(exp: GeneralExperience, *, max_lines: int = 8) -> str:
    """精简索引：每个 category 一行（最高置信 active insight），供 system prompt 常驻。

    恒定体量（≤max_lines 行），永不随库膨胀——避免全量 inline 稀释注意力。
    无 active 经验时返回空串。
    """
    act = _active(exp)
    if not act:
        return ""
    # 按 category 聚合，每类取最高置信代表
    by_cat: dict[str, DistilledLesson] = {}
    for l in act:
        cat = l.category or "(未分类)"
        cur = by_cat.get(cat)
        if cur is None or l.confidence > cur.confidence:
            by_cat[cat] = l
    lines = ["**通用经验索引**（详情见 user message 或调 query_general_experience）："]
    # 按置信降序，限行
    for cat, l in sorted(by_cat.items(), key=lambda kv: -kv[1].confidence)[:max_lines]:
        insight = (l.insight or "").replace("\n", " ").strip()
        lines.append(f"- [{cat}] {l.dim}：{insight[:60]}（{int(l.confidence * 100)}%）")
    return "\n".join(lines)


def select_lessons(
    exp: GeneralExperience,
    *,
    round: int,
    failed_dims: list[str] | None = None,
    k: int = 4,
) -> list[DistilledLesson]:
    """按本轮上下文选 ≤k 条 active 详情，注入 user message（策略性输入）。

    - R1（round≤1）：applies_when ∈ {construction, always}，按 confidence 取 top-k。
    - R>1：dim ∈ failed_dims 优先（fix 类更相关），不足补 always 高分。
    - 永远排除非 active。
    """
    act = _active(exp)
    if not act:
        return []
    failed_dims = failed_dims or []

    def by_conf(xs):
        return sorted(xs, key=lambda l: l.confidence, reverse=True)

    if round <= 1:
        pool = [l for l in act if l.applies_when in ("construction", "always")]
        if not pool:  # 兜底：没有标记 construction 的，退化为全 active 高分
            pool = act
        return by_conf(pool)[:k]

    # R>1
    failed = {d for d in failed_dims}
    hit = [l for l in act if l.dim in failed]
    # 失败命中里 fix 类优先，再按置信
    hit.sort(key=lambda l: (l.applies_when != "fix", -l.confidence))
    chosen = hit[:k]
    if len(chosen) < k:
        rest = [l for l in act if l not in chosen and l.applies_when == "always"]
        chosen += by_conf(rest)[: k - len(chosen)]
    return chosen


# ---------------------------------------------------------------------------
# 渲染：结构化经验 → 可移植 SKILL.md（唯一渲染源）
# ---------------------------------------------------------------------------


def render_experience_skill_md(exp: GeneralExperience, *, frontmatter: bool = True) -> str:
    """把通用经验渲染成 deepagents 原生 SKILL.md（按 category 分组，仅 active 进正文）。

    frontmatter=False 时只返回正文（用于工具/导出片段）。
    """
    slug = slugify_bench(exp.bench_id)
    desc = exp.bench_description or exp.scene or f"{exp.bench_id} 系列跨 loop 通用生图经验"
    desc = desc[:1024]

    lines: list[str] = []
    if frontmatter:
        lines += ["---", f"name: {slug}", f"description: {desc}", "---", ""]

    lines.append(f"# 通用经验 · {exp.bench_id}")
    lines += [""]
    if exp.scene or exp.bench_description:
        lines.append(f"> 场景：{exp.scene or exp.bench_description}")
    if exp.dimensions:
        lines.append(f"> 评分维度：{', '.join(exp.dimensions)}")
    if exp.source_runs:
        lines.append(f"> 来源 run：{', '.join(exp.source_runs)}")
    if exp.updated_at:
        lines.append(f"> 更新：{exp.updated_at}")
    lines += [""]

    lines.append("## 总览")
    lines.append(exp.summary or "(无)")
    lines += [""]

    act = _active(exp)
    if act:
        # 按 category 分组
        by_cat: dict[str, list[DistilledLesson]] = {}
        for l in act:
            by_cat.setdefault(l.category or "(未分类)", []).append(l)
        lines.append("## 经验条目（按类索引）")
        lines += [""]
        for cat in sorted(by_cat):
            lines.append(f"### 类目：{cat}")
            lines += [""]
            for ls in by_cat[cat]:
                _append_lesson(lines, ls)
    else:
        lines.append("## 经验条目")
        lines += ["(暂无 active 经验)"]
        lines += [""]

    # 已废弃区（透明可追溯，但不进消费）
    retired = [l for l in exp.lessons if l.status in ("refuted", "superseded", "archived")]
    if retired:
        lines.append("## 已废弃（不消费，仅追溯）")
        lines += [""]
        for ls in retired:
            tag = f"→ {ls.successor_id}" if ls.status == "superseded" and ls.successor_id else ls.status
            reason = f"：{ls.retire_reason}" if ls.retire_reason else ""
            lines.append(f"- [{ls.dim}] {ls.insight[:60]} （{tag}{reason}）")
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


def _append_lesson(lines: list[str], ls: DistilledLesson) -> None:
    conf = f" (置信度 {ls.confidence:.2f}; {ls.applies_when})" if ls.confidence is not None else ""
    lines.append(f"#### [{ls.dim}] {ls.insight}{conf}")
    lines += [""]
    if ls.dos:
        lines.append("**应该这样做：**")
        lines += [f"- {d}" for d in ls.dos]
        lines += [""]
    if ls.donts:
        lines.append("**不要这样做：**")
        lines += [f"- {d}" for d in ls.donts]
        lines += [""]
    if ls.evidence:
        lines.append(f"*证据：{'; '.join(ls.evidence)}*")
        lines += [""]


def render_lessons_detail(lessons: list[DistilledLesson]) -> str:
    """渲染一批 lesson 的详情正文（dos/donts/confidence/applies_when），供 user message 按上下文注入。

    与 SKILL.md 同源（复用 _append_lesson），但只渲染给定子集——策略性输入用。
    """
    if not lessons:
        return ""
    lines: list[str] = []
    for ls in lessons:
        _append_lesson(lines, ls)
    return "\n".join(lines).rstrip()


def render_lessons_reference_md(exp: GeneralExperience) -> str:
    """渲染技能包内 references/lessons.md：active 经验按 category 分组（dos/donts/confidence）。

    从 general.json 确定性渲染——lessons 的单一源（agent 不重复撰写）。
    """
    lines = [
        "# 蒸馏经验（lessons）",
        "",
        f"> 来源 benchmark：{exp.bench_id}　·　更新：{exp.updated_at or '-'}"
        + (f"　·　来源 run：{', '.join(exp.source_runs)}" if exp.source_runs else ""),
        "",
        "跨 loop 蒸馏的可复用 dos/donts（按类索引）。生成时直接遵循；失败时按失败维度查相关类目。",
        "",
    ]
    act = _active(exp)
    if not act:
        lines.append("_(暂无 active 经验)_")
        return "\n".join(lines)
    by_cat: dict[str, list[DistilledLesson]] = {}
    for l in act:
        by_cat.setdefault(l.category or "(未分类)", []).append(l)
    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines += [""]
        for ls in by_cat[cat]:
            _append_lesson(lines, ls)
    return "\n".join(lines).rstrip() + "\n"


def assemble_skill_package(
    data_root: Path,
    bench_dir: Path,
    bench_id: str,
    authored: "AuthoredSkill",
    exp: GeneralExperience,
) -> Path:
    """把 AuthoredSkill 装配成规范技能包目录（SKILL.md + references/ + assets/）。

    - 清空并重建 skill_dir，避免旧 reference/asset 残留。
    - skill_name 强制 = slug(bench)，保 skill-creator validate（name==dir）通过。
    - references/lessons.md 由 general.json 渲染（单一源）；其余 references 用 authored。
    - assets/ 拷贝 authored.asset_paths（相对 bench_dir，仅存在的）。
    返回 skill_dir。
    """
    slug = slugify_bench(bench_id)
    skill_dir = skill_package_dir(data_root, bench_id)
    if skill_dir.exists():
        shutil.rmtree(skill_dir, ignore_errors=True)
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "assets").mkdir(parents=True, exist_ok=True)

    # SKILL.md：frontmatter（name=slug，薄格式化）+ agent 自撰正文（含 lessons 强调，由 prompt 保证）。
    # 代码不注入内容——SKILL.md 全文由 agent 按规范产出；代码只落盘。
    # description 结构 sanitize（剥尖括号——quick_validate 硬规则；纯合规，非创作）+ 截断 1024。
    desc = re.sub(r"[<>]", "", authored.description or "").strip()[:1024]
    frontmatter = (
        "---\n"
        f"name: {slug}\n"
        f"description: {desc}\n"
        "---\n\n"
    )
    skill_md_text = frontmatter + authored.skill_md.strip() + "\n"
    (skill_dir / "SKILL.md").write_text(skill_md_text, encoding="utf-8")

    # 结构校验（复刻 quick_validate 硬规则）：不过则日志告警，不阻断（正文质量归 review 阶段）。
    ok, msg = validate_skill_md(skill_md_text)
    if not ok:
        print(f"[skill_package] 结构校验未过（{slug}）：{msg}", flush=True)

    # references/lessons.md（单一源，确定性渲染）
    (skill_dir / "references" / "lessons.md").write_text(
        render_lessons_reference_md(exp), encoding="utf-8"
    )

    # agent 自撰 references（路径安全校验）
    for ref in authored.references or []:
        safe = _safe_ref_path(ref.path)
        if not safe or not safe.endswith(".md"):
            continue
        (skill_dir / "references" / safe).write_text(ref.content.strip() + "\n", encoding="utf-8")

    # assets（拷 benchmark 实际用到的）。asset_paths 可能是 bench_dir 相对 或 sample 相对
    # （如 reference_style/x.png 实际在 samples/<id>/reference_style/）—— 兼容两种。
    bench_root = Path(bench_dir).resolve()
    for ap in authored.asset_paths or []:
        src = _resolve_asset(bench_root, ap)
        if src is None:
            continue
        shutil.copy2(src, skill_dir / "assets" / src.name)

    return skill_dir


def _resolve_asset(bench_root: Path, ap: str) -> Path | None:
    """在 bench_dir 下解析资产路径：先 bench_dir/ap，再 samples/*/ap（sample 相对）。"""
    p = (bench_root / ap).resolve()
    try:
        p.relative_to(bench_root)  # 防越界
    except ValueError:
        return None
    if p.exists() and p.is_file():
        return p
    # 退化：sample 相对路径（ap 不含 samples/ 前缀）
    samples_dir = bench_root / "samples"
    if samples_dir.is_dir():
        for sd in sorted(samples_dir.iterdir()):
            if not sd.is_dir():
                continue
            cand = (sd / ap).resolve()
            try:
                cand.relative_to(bench_root)
            except ValueError:
                continue
            if cand.exists() and cand.is_file():
                return cand
    return None


def _safe_ref_path(path: str) -> str:
    """references 子路径安全化：取 basename，禁目录穿越/绝对路径。"""
    if not path:
        return ""
    p = Path(path).name  # 仅取 basename，丢弃任何目录部分
    return p if p.endswith(".md") else ""


def validate_skill_md(text: str) -> tuple[bool, str]:
    """结构校验（复刻 skill-creator ``quick_validate`` 的 frontmatter 硬规则，yaml 解析保持一致）。

    用于 ``assemble_skill_package`` 装配后自检 + 测试/router 复用。**仅查 frontmatter 结构**；
    正文（冷启动可用性 / lessons 前景化等）质量由 distiller 的 review 阶段（quality_checklist）把关。
    """
    import yaml  # 局部 import：仅校验时需要（yaml 是本环境传递依赖，已可用）

    if not text.startswith("---"):
        return False, "No YAML frontmatter found"
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return False, "Invalid frontmatter format"
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:  # noqa: PERF203
        return False, f"Invalid YAML in frontmatter: {e}"
    if not isinstance(fm, dict):
        return False, "Frontmatter must be a YAML dictionary"

    allowed = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}
    extra = set(fm.keys()) - allowed
    if extra:
        return False, f"Unexpected frontmatter key(s): {', '.join(sorted(extra))}"

    name = fm.get("name", "")
    name = name.strip() if isinstance(name, str) else ""
    if not name:
        return False, "Missing 'name' in frontmatter"
    if not re.match(r"^[a-z0-9-]+$", name) or name.startswith("-") or name.endswith("-") or "--" in name:
        return False, f"Name '{name}' should be kebab-case (no leading/trailing/double hyphen)"
    if len(name) > 64:
        return False, f"Name too long ({len(name)} > 64)"

    desc = fm.get("description", "")
    desc = desc.strip() if isinstance(desc, str) else ""
    if not desc:
        return False, "Missing 'description' in frontmatter"
    if "<" in desc or ">" in desc:
        return False, "Description contains angle brackets (< or >)"
    if len(desc) > 1024:
        return False, f"Description too long ({len(desc)} > 1024)"
    return True, "Skill is valid"


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------


def load_general_experience(data_root: Path, bench_id: str) -> GeneralExperience:
    p = general_experience_path(data_root, bench_id)
    if not p.exists():
        return GeneralExperience(bench_id=bench_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return GeneralExperience.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return GeneralExperience(bench_id=bench_id)


def save_general_experience(data_root: Path, bench_id: str, exp: GeneralExperience) -> Path:
    """写 general.json + 同步渲染 SKILL.md。双写保证结构化源与可移植载体同步。"""
    exp.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    exp.categories = sorted({(l.category or "(未分类)") for l in exp.lessons if l.status == "active"})
    p = general_experience_path(data_root, bench_id)
    p.write_text(
        json.dumps(json.loads(exp.model_dump_json()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        md_path = experience_skill_md_path(data_root, bench_id)
        md_path.write_text(render_experience_skill_md(exp), encoding="utf-8")
    except OSError:
        pass
    return p


__all__ = [
    "AuthoredReference",
    "AuthoredSkill",
    "DistilledExperience",
    "DistilledLesson",
    "GeneralExperience",
    "RenoItem",
    "RenovationPlan",
    "assemble_skill_package",
    "experience_skill_md_path",
    "general_experience_path",
    "load_general_experience",
    "new_lesson_id",
    "render_experience_index",
    "render_experience_skill_md",
    "render_lessons_detail",
    "render_lessons_reference_md",
    "save_general_experience",
    "select_lessons",
    "skill_package_dir",
    "slugify_bench",
    "validate_skill_md",
]
