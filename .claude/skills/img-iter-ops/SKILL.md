---
name: img-iter-ops
description: 操作 img-iter-agent（自我迭代的 AI 生图 agent：Generator↔Critic 博弈逐轮改图、Critic 多维打分、Summarizer 沉淀经验）的全流程技能——创建标准化 benchmark 考题、启动/批量运行生成-评判闭环 loop、人工介入中断、诊断经验闭环效果（还原度收敛 + 经验沉淀 A/B/C 三环）、管理跨 loop 经验。捆绑了防数据污染的自动驱动脚本、结构化诊断脚本和考题脚手架。每当用户在这个项目里提到 跑 loop / 跑测试 / 看效果 / 新起 loop / 续跑 / 批量跑 / 加考题 / 建 benchmark / 经验沉淀 / 经验诊断 / 还原度 / critic 打分 / 升级 / escalated / fail_streaks / conclusions / general.json / 蒸馏经验 / 效果迭代 / 经验管理，或要做任何「跑一轮看效果」「这题怎么跑」类操作时，务必使用本技能——它避免每次手工拼凑驱动脚本、避免误污染已有 loop、并给出标准化的经验闭环诊断。
---

# img-iter-ops — img-iter-agent 项目操作中枢

## 这个项目一句话
两个 LLM agent（Generator 出图、Critic 多维打分）在一道考题上博弈，每轮 改 prompt → 出图 → Critic 打分 → Summarizer 沉淀经验，逼近还原度收敛。首个场景：家具电商白底产品图（结构/材质/颜色最易翻车）。

**操作前必懂的核心概念：**
- **一题一 loop**：`loop_id = <bench>-<sample>`。同 sample 再跑 = **续跑**（轮数叠加），不是新 loop。
- **经验两层**：`conclusions.json`（单 loop，每轮规则+LLM 写，Generator 读本题经验）+ `general.json`（跨 loop，CLI 蒸馏，Generator 读跨题先验）。两个文件、两层生命周期。
- **闭环图**：`generator → critic → summarizer → human_review(interrupt) → continue/stop`。`human_review` 卡在 interrupt 等决策。
- **还原度** = 权重·维度特征（二分维度=通过率、连续维度=LLM 0-1 分加权求和），不是 LLM 直接吐一个数。

## 四大能力 → 入口（先判断用户意图落点）

| 用户想做的事 | 能力 | 入口 |
|---|---|---|
| 加一道新考题（产品图+验收标准） | 创建 benchmark | `references/benchmark-create.md` + `scripts/new_benchmark.py` |
| 跑一个 loop 看 N 轮效果 | 启动 loop | `references/loop-run.md` + `scripts/run_loop_auto.py` |
| 一次跑多个 sample | 批量 loop | `scripts/batch_loops.py` |
| 看一个 loop 跑得怎样、经验闭环是否生效 | 效果诊断 | `references/experience-diagnosis.md` + `scripts/diagnose_loop.py` |
| 跑完一批后对抗调参「创造力」子标准（全自动+留痕） | 创造力对抗调参 | `scripts/tune_creativity.py`（写 `data/benchmarks/<bench>/creativity_criteria.json`，下批 loop 自动生效） |
| 把多个 loop 的经验归纳成通用 dos/donts | 经验管理 | `references/experience-manage.md` |

不确定就读 `references/experience-diagnosis.md`（最常用）。

## 三条铁律（都是踩过坑的，务必遵守）

1. **测试用全新 loop_id，绝不续跑默认 id。** 默认 `<bench>-<sample>` 是「一题一 loop」的正式数据；测试新起用 `<bench>-<sample>-<tag>`（如 `-exp6`、`-batch1`），避免污染历史 loop、避免和正式数据混在一起难对比。`run_loop_auto.py --tag` 自动这么命名；脚本检测到 loop 已存在会直接拒绝、提示换 tag，不会偷偷续跑。
2. **「跑 N 轮」要真的生成 N 轮。** 底层 `run_loop_session(rounds=R)` 实际生成 **R+1 轮**（首轮 invoke + R 次 resume）且最后一轮不打印——这是已知瑕疵。`run_loop_auto.py` 接收「想要几轮」并内部换算 + 跑完校验 trajectory 行数，屏蔽它。诊断脚本读 trajectory 拿真实轮数，不信 print。
3. **经验闭环有三环，诊断要分开看**（见下），别混为一谈。

## 经验闭环 A/B/C 三环（诊断的核心框架）

这是 2026-08-09 升级（commit `1b1e600`）的目标。`diagnose_loop.py` 按这三环判读 `conclusions.json`：

- **A 环（lesson 富化）**：Summarizer 用 LLM 把干瘪的「建议保持方向」富化成 **具体替代思路 + 模型上限标注**（ineffective 给 ControlNet/换 seed/上报人工等可执行建议，escalated 标注「模型能力上限」）。判据：`lesson` 字段是否实质、是否含具体建议词。
- **B 环（fail_streaks + escalated）**：per-dim 连续失败计数落盘；连续失败 **≥2 轮** → `escalated:true`（撞模型上限，prompt 微调无效，该换根本思路）。判据：`fail_streaks` 字段 + 有无 `escalated:true`；且偶发失败（只败 1 轮就修好）不该升级。
- **C 环（generator 强制警告）**：升级维度警告直接塞进 generator 的 user message（不靠 agent 自觉调工具）。判据：generator 是否真的 **换了 test_variable**（reference_images/size/seed），而非继续改 prompt。

> 实测常态：**A/B 通常生效，C 常「信息送达但 generator 不换思路」**（test_variable 全程是 prompt）——这是诊断报告要重点暴露的执行缺口，也是建议人工介入的主要信号。

## 人工介入策略（默认连续跑完 + 关键点高亮）
默认无人值守跑完 N 轮（`run_loop_auto.py` 自动 continue），跑完 `diagnose_loop.py` 在报告里 **高亮 escalated 维度 / 还原度突崩（>0.1 跌幅）/ 未收敛**，并标注「建议人工介入」。需要逐轮审批时走 web UI（端口 8765）的 human_review——它的 interrupt payload 已透出 `escalated_dims`，是人工「该停/换策略」的官方信号。

## 环境（执行前确认）
- 跑脚本：项目根下 `.venv/bin/python`（脚本假设 cwd=项目根）
- 配置：`.env`（dmxapi key + 四个生图 model_id + `IMG_ITER_GENERATOR/CRITIC/SUMMARIZER_MODEL`）；缺 key 会在首轮 LLM/出图时报错
- web UI（可选）：端口 8765，看 loop/trace/人工排序校准
- LangSmith：每 loop 一条 trace（project=`img-iter-agent`），按 `loop_id` 聚合多轮

## 何时读哪个 reference（progressive disclosure）
- 建考题（图+口述→自动起草 6 维度 checklist）→ `references/benchmark-create.md`
- 跑 loop 的路径选择 / loop_id 规范 / 自动连跑 / 续跑→ `references/loop-run.md`
- 三环诊断框架细节 / 关键点判定阈值 / 报告怎么读 → `references/experience-diagnosis.md`
- 跨 loop 蒸馏 general.json / loop 清理对比 / 状态管理 → `references/experience-manage.md`

`scripts/` 下的脚本可直接执行（用法见各自 `--help` 或文件头注释），不必读源码，除非要改行为。
