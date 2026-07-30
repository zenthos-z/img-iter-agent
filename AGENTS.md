# img-iter-agent — AGENTS.md

本项目是 `~/Documents/codelib` 工作区下的独立项目，遵循根 `AGENTS.md` 的
「裸镜像 + worktree」约定。本文件只写**本项目专属**的内容。

## 这是什么

一个自我迭代的 AI 生图 agent 系统：通过「生成 → 对抗评判 → 总结 → 优化 prompt」的
闭环，自动产出**风格元素一致、内容可变**的一组图像。详见 `docs/ARCHITECTURE.md`。

## 语言与工具链

- **Python ≥ 3.11**（生图与 agent 生态的事实标准）
- 包管理：`uv` 优先（快），或 `pip` + `venv`
- 跑测试：`pytest`（在 worktree 根目录）
- 代码风格：`ruff` + `mypy`

## 开发约定

- 在 worktree 内开发：`~/Documents/codelib/work/img-iter-agent/<branch>/`
- 配置走环境变量 / `.env`（`.env` 已在 `.gitignore`，绝不提交密钥）
- 生成的图片、模型权重、运行记录**不进 git**（见 `.gitignore`）
- 架构决策先写进 `docs/ARCHITECTURE.md` 或 `docs/ADR/`，再写代码

## 不要做

- 不要在 `github/img-iter-agent.git` 裸镜像里改文件
- 不要提交 `data/outputs`、`data/runs`、`*.safetensors` 等大文件
