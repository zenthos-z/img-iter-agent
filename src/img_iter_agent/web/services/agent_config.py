"""Agent 配置服务：把系统提示词 + 模型 id 外部化到 data/agents_config/。

可改项：system_prompt + model_id（POST 只接受这两个）——仅对 generator/critic 生效。
工具 / 技能 / 其他参数是**代码派生的只读量**，通过 get_agent_config 一并返回，
供前端做只读展示——让用户看清每个 agent 的完整画像。

文件布局：
  data/agents_config/<agent>.md   — 系统提示词正文（可选；缺失则用代码默认）
  data/agents_config/<agent>.json — {"model": "..."}（可选；缺失则用 settings 默认）

系统提示词的「代码默认」单一来源是 agents/*.py 里的 _DEFAULT_*_SYS 常量（完整版），
本模块通过 _code_default_prompt 延迟导入取用——杜绝历史上「web 层简短版 vs agent 完整版」
两套默认值不一致的问题。

技能：
  - generator 的技能 = summarizer 蒸馏出的 skill_package（experience/<bench>/<slug>/SKILL.md），
    跑哪个 benchmark 就激活哪个（generator_skills_source 解析）；未蒸馏 → 无技能（裸跑）。
  - distiller 的技能 = 内置 skill-author（agents/skill_authoring/skill-author/SKILL.md，静态、全局）——
    创作蒸馏技能包的 meta-skill。distiller 的 system prompt 代码内置（_SKILL_AUTHOR_SYS），
    **只读展示**（save 不生效），故 get_agent_config 返回 readonly=True。
  - critic 不使用技能。

支持三个 agent：generator / critic / distiller。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...config import Settings, get_settings
from ...memory.experience import generator_skills_source

AGENTS = ("generator", "critic", "distiller")

# 每个 agent 的可用工具（只读展示用）。须与各 agent 的工具构造保持一致。
_AGENT_TOOLS: dict[str, list[str]] = {
    "generator": ["generate_image", "query_experience", "query_general_experience"],
    "critic": ["query_rubric"],
    # distiller = skill-author 全工具 authoring agent（read_file/write_file/edit_file/ls/glob/grep/task）
    "distiller": ["read_file", "write_file", "edit_file", "ls", "glob", "grep"],
}


def _code_default_prompt(agent: str) -> str:
    """取 agent 代码模块里的完整版默认系统提示词（单一真相源）。

    延迟导入避免循环依赖：agents/generator.py 反向 import 了 agent_config_loader，
    若本模块顶层 import 会在某些加载顺序下成环。web 层不被 agent 模块导入，函数内导入安全。
    """
    if agent == "generator":
        from ...agents.generator import _DEFAULT_GENERATOR_SYS
        return _DEFAULT_GENERATOR_SYS
    if agent == "critic":
        from ...agents.critic import _DEFAULT_CRITIC_SYS
        return _DEFAULT_CRITIC_SYS
    if agent == "distiller":
        from ...agents.experience_distiller import _SKILL_AUTHOR_SYS
        return _SKILL_AUTHOR_SYS
    return ""


def config_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    d = settings.data_root / "agents_config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prompt_path(agent: str, settings: Settings) -> Path:
    return config_dir(settings) / f"{agent}.md"


def _meta_path(agent: str, settings: Settings) -> Path:
    return config_dir(settings) / f"{agent}.json"


def _skills_dir_for(
    agent: str, bench_id: str | None = None, settings: Settings | None = None,
) -> Path | None:
    """agent 的技能 source 目录（runner + 只读展示共用）。

    generator → experience/<bench>/（蒸馏 skill_package，per-benchmark；generator_skills_source 解析）；
                未蒸馏或 bench_id 缺失 → None。
    distiller → agents/skill_authoring/（内置 skill-author，静态全局）。
    critic → None（不使用技能）。
    """
    if agent == "generator" and bench_id:
        settings = settings or get_settings()
        return generator_skills_source(settings.data_root, bench_id)
    if agent == "distiller":
        from ...agents.experience_distiller import _SKILL_AUTHOR_PARENT
        return _SKILL_AUTHOR_PARENT
    return None


def _scan_skill_packages(src_dir: Path | None, source_label: str) -> list[dict]:
    """扫 src_dir 下子目录的 SKILL.md frontmatter，返回 [{name, summary, source}]。

    source_label 是来源前缀（如 experience/<bench> 或 skill_authoring），供前端标注。
    """
    if not src_dir or not src_dir.is_dir():
        return []
    out: list[dict] = []
    for sub in sorted(src_dir.iterdir()):
        skill_md = sub / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        # 优先从 frontmatter（--- 包裹块）取 name + description；否则退化到目录名/首段
        name = sub.name
        summary = ""
        fm = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if fm:
            block = fm.group(1)
            mn = re.search(r"^name:\s*(.+?)\s*$", block, re.MULTILINE)
            if mn:
                name = mn.group(1).strip()
            md_desc = re.search(r"^description:\s*[\"']?(.*?)[\"']?\s*$", block, re.MULTILINE)
            if md_desc:
                summary = md_desc.group(1).strip()
        if not summary:
            for line in text.splitlines():
                t = line.strip()
                if t and not t.startswith("#") and not t.startswith("---"):
                    summary = t
                    break
        out.append({"name": name, "summary": summary, "source": f"{source_label}/{sub.name}"})
    return out


def _list_skills(
    agent: str, bench_id: str | None = None, settings: Settings | None = None,
) -> list[dict]:
    """列出 agent 当前可用的技能（只读展示用）。

    generator：扫 experience/<bench>/ 下子目录的 SKILL.md（per-bench 蒸馏技能）；未蒸馏 → []。
    distiller：扫 agents/skill_authoring/ 下子目录的 SKILL.md（内置 skill-author）。
    critic → []。
    """
    if agent == "generator" and bench_id:
        settings = settings or get_settings()
        bench_dir = generator_skills_source(settings.data_root, bench_id)
        return _scan_skill_packages(bench_dir, f"experience/{bench_id}")
    if agent == "distiller":
        from ...agents.experience_distiller import _SKILL_AUTHOR_PARENT
        return _scan_skill_packages(_SKILL_AUTHOR_PARENT, "skill_authoring")
    return []


def _agent_params(
    agent: str, settings: Settings, bench_id: str | None = None,
) -> dict:
    """收集 agent 级别的只读固定参数（非每轮动态输入），供前端展示。"""
    # invoke_with_retry 的默认重试次数（agents/_narrow_tools.py 默认 retries=5）
    RETRY_COUNT = 5
    # ChatOpenAI 超时（llm/chat_model.py 写死 timeout=120.0）
    TIMEOUT = 120.0

    params: dict = {
        "timeout": TIMEOUT,
        "dmxapi_host": settings.dmxapi_host,
    }
    if agent in ("generator", "critic"):
        # generator/critic 走 narrow_tools + invoke_with_retry
        from ...agents._narrow_tools import AGENT_RECURSION_LIMIT
        params["recursion_limit"] = AGENT_RECURSION_LIMIT
        params["retry_count"] = RETRY_COUNT
    if agent == "generator":
        skills_src = _skills_dir_for(agent, bench_id, settings)
        params["skills_dir"] = str(skills_src) if skills_src else "(未蒸馏，generator 裸跑)"
        params["data_root"] = str(settings.data_root)
    if agent == "distiller":
        # distiller 是全工具 authoring agent（非 narrow_tools）；FS 权限由 _build_skill_author_agent 限定
        skills_src = _skills_dir_for(agent, bench_id, settings)
        params["skills_dir"] = str(skills_src) if skills_src else ""
    return params


def list_bench_ids(settings: Settings | None = None) -> list[str]:
    """扫 data/benchmarks/ 下的子目录名，供前端配置页 benchmark 下拉。"""
    settings = settings or get_settings()
    bench_dir = settings.data_root / "benchmarks"
    if not bench_dir.is_dir():
        return []
    return sorted(d.name for d in bench_dir.iterdir() if d.is_dir())


def get_agent_config(
    agent: str, *, bench_id: str | None = None, settings: Settings | None = None,
) -> dict:
    """读一个 agent 的配置。读不到文件则返回代码默认（完整版）。

    bench_id 仅影响只读派生量（skills 列表 / skills_dir 参数）——系统提示词与 model
    仍按全局 data/agents_config/ 解析（与 benchmark 无关）。

    distiller 的 system prompt 代码内置（_SKILL_AUTHOR_SYS），save 不生效 → readonly=True。
    """
    settings = settings or get_settings()
    prompt_path = _prompt_path(agent, settings)
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = _code_default_prompt(agent)  # 代码完整版默认（单一来源）

    model = ""
    meta_path = _meta_path(agent, settings)
    if meta_path.exists():
        try:
            model = json.loads(meta_path.read_text(encoding="utf-8")).get("model", "")
        except (json.JSONDecodeError, OSError):
            model = ""

    # 若配置里没填 model，回退 settings 默认
    if not model:
        model = _settings_default_model(agent, settings)

    return {
        "agent": agent,
        "system_prompt": system_prompt,
        "model": model,
        "tools": list(_AGENT_TOOLS.get(agent, [])),
        "skills": _list_skills(agent, bench_id, settings),
        "params": _agent_params(agent, settings, bench_id),
        # distiller 配置代码内置，仅展示（save 写文件但 distiller 不读）；前端据此禁用编辑
        "readonly": agent == "distiller",
    }


def save_agent_config(
    agent: str, *, system_prompt: str | None = None, model: str | None = None,
    bench_id: str | None = None, settings: Settings | None = None,
) -> dict:
    """写一个 agent 的配置（只写非 None 的字段）。返回写后的完整配置。

    写入走全局 data/agents_config/（与 benchmark 无关）；bench_id 仅用于返回值的只读 skills 展示。
    对 readonly agent（distiller）写入不生效（其运行时不读此文件），前端应禁用保存。
    """
    settings = settings or get_settings()
    cur = get_agent_config(agent, bench_id=bench_id, settings=settings)

    if system_prompt is not None:
        _prompt_path(agent, settings).write_text(system_prompt, encoding="utf-8")
        cur["system_prompt"] = system_prompt
    if model is not None:
        _meta_path(agent, settings).write_text(
            json.dumps({"model": model}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cur["model"] = model
    return cur


def reset_agent_config(
    agent: str, *, bench_id: str | None = None, settings: Settings | None = None,
) -> dict:
    """恢复代码默认（删配置文件）。"""
    settings = settings or get_settings()
    for p in (_prompt_path(agent, settings), _meta_path(agent, settings)):
        if p.exists():
            p.unlink()
    return get_agent_config(agent, bench_id=bench_id, settings=settings)


def _settings_default_model(agent: str, settings: Settings) -> str:
    """settings 里各 agent 的默认 model 字段。distiller 复用 summarizer_model。"""
    mapping = {
        "generator": settings.generator_model,
        "critic": settings.critic_model,
        "distiller": settings.summarizer_model,
    }
    return mapping.get(agent, "")
