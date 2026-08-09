# 效果诊断：A/B/C 三环 + 关键点高亮

跑完一个 loop（或中途想看），诊断它跑得怎样、经验闭环是否生效：

```bash
.venv/bin/python .claude/skills/img-iter-ops/scripts/diagnose_loop.py <loop_id>
```

## 报告 5 段结构
1. **还原度曲线**：每轮还原度 + 各维度失败项（二分❌哪些 id / 连续分<0.7）
2. **关键点高亮**：突崩 / 未收敛 / 末轮回落
3. **B 环**：fail_streaks + escalated dims
4. **A 环**：lesson 富化质量（含具体建议 vs 干瘪模板）
5. **C 环**：generator 是否换 test_variable
+ **总结**：A/B/C 各环 ✅/⚠️ + 是否建议人工介入

## A/B/C 三环详解（2026-08-09 升级 commit 1b1e600 的目标）

### A 环 · lesson 富化
- **是什么**：Summarizer 用 LLM 把规则判定的干瘪 lesson（"建议保持方向"/"需换思路"）富化成**具体替代思路 + 模型上限标注**。
- **看什么**：`conclusions.json` 每条的 `lesson` 字段。
- **富化判据**：含 `ControlNet / reference / test_variable / seed / 上限 / 瓶颈 / 上报人工 / 图生图` 等具体建议词。
- **干瘪判据**：只有"建议保持该方向"且 <60 字。
- **常态**：富化占多数（summarizer=gemini-3.1-flash-lite 可靠）。
- **异常（A⚠️）**：全干瘪 = summarizer 的 LLM 没接上（查 `build_loop_context` 是否传了 `summarizer_model`）或 LLM 调用失败退化纯规则。

### B 环 · fail_streaks + escalated
- **是什么**：per-dim 连续失败计数（binary 有未过项 / 连续 <0.7 = 失败）；连续失败 **≥2 轮** → `escalated:true`（撞模型上限，prompt 微调无效，该换根本思路）。
- **看什么**：`fail_streaks` 字段 + 每条 conclusion 的 `escalated`。
- **不误伤**：只败 1 轮就修好的 dim **不该** escalated（阈值=2 的意义）。
- **常态**：复杂题有 1–4 个 dim 升级（如 consistency 三视图几何一致性）。
- **异常**：全 escalated = 任务/模型太难；无升级 = 题太简单（R1 近满分，没失败项可累计）。

### C 环 · generator 强制警告
- **是什么**：升级 dim 的警告直接塞进 generator 的 user message（不靠 agent 自觉调工具），强制它换思路。
- **看什么**：trajectory 每轮的 `test_variable` 字段。
- **生效判据**：escalated 后 generator **换了 test_variable**（reference_images/size/seed/negative_prompt），而非继续改 prompt。
- **常态（实测）⚠️**：信息送达但 generator 不换——`test_variable` 全程是 `prompt`。**这是当前最大缺口**：升级诊断对了，但 generator 工具/习惯让它只会改 prompt。

## 关键点高亮规则
- **突崩**：相邻轮还原度跌幅 >0.1
- **未收敛**：全程峰值 <0.7
- **末轮回落**：末轮 < 峰值 − 0.05

任一命中 → 报告标"建议人工介入"。

## 怎么下结论（按 A/B/C 组合）
| 组合 | 含义 | 建议 |
|---|---|---|
| A✅ B✅ C✅ | 闭环健康、收敛良好 | 理想，继续迭代 |
| A✅ B✅ C⚠️ | **最常见**：诊断对了但 generator 不换思路 | 人工介入换 test_variable，或给 generator 加换策略工具 |
| A⚠️ | summarizer LLM 没接上 | 查 `build_loop_context` 的 `summarizer_model` |
| B 无升级 | 题太简单 / 模型太强 | 可能不需要经验闭环，或换更难的题 |
| 还原度低 + A/B✅ | 模型/任务瓶颈，闭环已正确定位 | 换 test_variable 或上报人工（非闭环的锅） |

## 真实例子：s003-exp6（2026-08-09 实跑）
还原度 0.40→0.53→0.54→**0.62**→0.50→0.52（R4 峰值、R5 突崩）
- **A✅**：15/18 lesson 富化（ineffective 给出 ControlNet/换 seed/上报人工/放弃单图三视图等具体建议）
- **B✅**：consistency / material_texture / artifact_defect 连续失败 6 轮升级，product_structure 2 轮升级；commercial_focus 只败 1 轮未升级（不误伤）✓
- **C⚠️**：`test_variable` 全程 `prompt`，升级建议未执行
- **结论**：经验闭环**诊断精准**（锁定 consistency 是 seedream 单图三视图的固有几何瓶颈），但**执行端消化不了** → 建议人工介入或给 generator 加换 test_variable 能力。

这个例子说明一个重要区分：**经验闭环的诊断质量**（A/B）和**生图还原度**是两回事。还原度未收敛不一定是闭环的锅——闭环可能已正确诊断出"这是模型能力上限"，只是 loop 内执行不了。
