---
name: experience-distiller
description: "跨多个生图 loop 总结通用经验时使用。当你需要从一批 run 的 trajectory + 已验证结论里归纳出跨 run 复用的 dos/donts 时，加载本技能。"
---

# Experience Distiller（跨 loop 经验蒸馏）

## Overview
你拿到一批已完成 loop（每 run 一个目录，含 trajectory.jsonl + lessons/conclusions.json）。你的任务
是跨 run 归纳出**通用经验**：哪些做法跨题反复有效、哪些反复无效，写成可复用的 dos/donts。
可用工具：`list_runs`（总览）、`query_run`（单 run 逐轮）、`query_dim_history`（某维度跨 run 改动史，
**最关键**）、`query_conclusions`（单 run 已验证结论）。最后结构化输出 `DistilledExperience`。

## Best Practices
- **以已验证结论为锚**：conclusions.json 里的 effective/ineffective 是 Critic 前后对比**机器验证**过的，
  比单看分数更可靠。归纳优先基于 `query_dim_history`。
- **跨 run 找模式**：同一改动在多个 run 有效 → 高置信 dos；多次无效 → dont。单 run 孤证 → 低置信或不下结论。
- **按维度归纳**：每个评分维度（artifact_defect / consistency / ...）单独蒸馏一条 lesson；发现跨维度共性
  可加一条 dim='general' 的总规律。
- **引用证据**：每条 lesson 的 evidence 填来源 run（如 `<run_id>/round3`），让人能回溯。
- **诚实置信度**：证据强 → confidence 高（>0.7）；孤证或冲突 → 低（<0.4）。宁缺毋滥。

## Process
1. 调 `list_runs` 看本次有哪些 run、各自还原度。
2. 对每个维度调 `query_dim_history(dim)`，看该维度跨 run 的改动 + 验证状态。
3. 必要时调 `query_run(run_id)` 看某 run 的逐轮演化（分数趋势、改动顺序）。
4. 归纳：跨 run 反复出现的有效做法 → dos；反复无效 → donts；写成一句话 insight。
5. 结构化输出：`summary`（整体观察）+ `lessons`（每维度一条，带 dim/insight/dos/donts/evidence/confidence）。

## Common Pitfalls
- 只看单个 run 就下「通用」结论 → 那是单 run 经验，不算通用。
- 把 pending（未验证）结论当事实 → 只采信 effective/ineffective。
- evidence 不填来源 → 无法回溯，可信度存疑。
- 把分数小幅波动当「有效」→ 以 Critic 验证状态为准，分数噪声由 verdict_delta 体现，要甄别。
