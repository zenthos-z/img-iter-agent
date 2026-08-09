# 经验管理：跨 loop 蒸馏 + loop 管理

## 两层经验（操作时务必分清）
| | conclusions.json | general.json |
|---|---|---|
| 粒度 | 单 loop（一题一份） | 跨 loop（一 bench 一份） |
| 谁写 | Summarizer 每轮写 | ExperienceDistiller（CLI summarize）|
| 内容 | 逐条 verified/ineffective + lesson | 跨题通用 dos/donts + confidence |
| Generator 怎么读 | `query_experience`（本题经验） | `query_general_experience`（跨题先验，首题也用得上）|
| 前端 | ✅ loop 详情页经验沉淀面板 | ❌ **无端点无 UI**（已知缺口）|

## 跨 loop 蒸馏 general.json
跑过多个 sample 的 loop 后，把零散结论归纳成**通用 dos/donts**：

```bash
python -m img_iter_agent.cli summarize --bench furniture_product_whitebg
```

读该 bench 下所有 run 的 trajectory + conclusions → 蒸馏器（deepagent，视觉模型看图归纳）→ `data/experience/<bench>/general.json`。

**何时蒸馏**：跑过 ≥2 个 sample 的 loop 后（单 loop 蒸馏意义有限）。蒸馏产物比循环内 conclusions 更通用可执行（带具体 dos/donts + confidence），是 Generator 的跨题先验。

**显式指定 run**（避免把测试 loop 混进蒸馏输入）：
```bash
python -m img_iter_agent.cli summarize --runs data/runs/loop-a data/runs/loop-b
```

## 已知缺口：general.json 前端不显示
web UI 没有 general.json 的端点/面板（`docs/EXPERIENCE_FLOW.md` 已记录）。蒸馏后只能 CLI 输出看，或直接读 `data/experience/<bench>/general.json`。要让网页展示需自行加端点+面板。

## loop 状态管理
- **看状态**：web UI 总览页（每 sample 的 loop 状态/轮数/还原度），或直接看 `data/runs/<loop_id>/meta.json`
- **续跑**：web UI loop 详情页「继续下一轮」（run_loop_auto 暂不支持续跑，见 `loop-run.md`）
- **清理测试 loop**：`data/runs/<loop_id>/` 整个目录删掉即可（含 sqlite checkpoint + trajectory + out 图）。带 tag 的测试 loop 跑完确认无用就删，别堆积污染 `data/runs/`、影响 analyze/summarize 的输入。

## 对比多个 loop（跨 run 还原度汇总）
```bash
python -m img_iter_agent.cli analyze --bench furniture_product_whitebg [--plot out.png]
```
按样本/按轮次汇总还原度、画曲线。评估"哪个 sample / 模型 / 策略更好"。

## 经验库污染处理
- **`general` 噪声桩**：旧版当一轮无失败项时会登记 `dim=general` 的废结论（永远 ineffective）。新版已修（无失败项不登记）。若旧数据有噪声桩，手动从 conclusions.json 删掉或重跑。
- **测试 loop 混入蒸馏**：带 tag 的测试 loop 也在 `data/runs/` 下，会被 `summarize --bench` 扫到。蒸馏重要经验前，先用 `--runs` 显式指定正式 loop，或先清理无用测试 loop。

## 评分校准（闭环 B，独立于经验闭环）
若 Critic 还原度与人判断有偏差，做人工排序校准权重：
```bash
python -m img_iter_agent.cli calibrate --bench furniture_product_whitebg --runs ... --ranks <排序值>
```
人只做 listwise 排序（人擅长的），系统用 learning-to-rank 拟合维度权重，修正 LLM 连续打分偏差。详见 `docs/EVALUATION.md`。这是另一个闭环，与经验沉淀（闭环 A）正交。
