# img-iter-agent — AGENTS.md

面向 coding agent（Claude Code 等）的项目约定。架构与关键设计决策见
`docs/ARCHITECTURE.md`（含 ADR），评分方法论见 `docs/EVALUATION.md`。

## 这是什么

自我迭代的 AI 生图 agent 系统：Generator 与 Critic 两个 deepagent 在 LangGraph
闭环里「生成 → 对抗评判 → 经验沉淀 → 改进 prompt」逐轮迭代，人工在 `human_review`
节点介入裁决；另有独立蒸馏器跨 loop 归纳通用经验、装配成 per-benchmark 技能包。

## 语言与工具链

- **Python ≥ 3.11**，包管理 `uv` 优先（或 `pip` + venv）
- 测试 `pytest`（项目根目录运行，无需任何 API key）
- 代码风格 `ruff` + `mypy`

## 开发约定

- 配置全走环境变量 / `.env`（已被 `.gitignore`，**绝不提交密钥**）
- 三个 agent 的系统提示词外部化在 `data/agents_config/`，改提示词不需要改代码
- benchmark 是声明式数据（`manifest.json` + `rubric.md` + `samples/`），新增场景优先
  加 benchmark 而不是改代码
- 架构决策先写进 `docs/ARCHITECTURE.md` 的 ADR，再写代码

## 不要做

- 不要提交 `data/runs/`、`data/experience/`、`data/calibration/`、`data/human_hints/`
  等运行产物（见 `.gitignore` 三层数据约定）
- 不要把 dmxapi / LangSmith key 写进代码、测试或文档
- 不要在 benchmark 素材目录放未整理的原始图片（用 `samples/_unsorted/`，已 ignore）
