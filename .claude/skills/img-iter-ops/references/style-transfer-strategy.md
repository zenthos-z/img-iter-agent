# 策略技能：风格神韵迁移 · 多文章概念表达（style-transfer v2）

> 一种**新考试形式**：6 篇文章 × 风格神韵迁移，考三件事——**风格精髓抓取 + 概念表达 + 方法泛化性**。
> Generator 看参考集（7 张）抽象风格共性，凭「文章核心概念」原创一张同神韵、motif 不复制的封面，
> 并产出一句话图片含义（meaning）；Critic 按三柱打分。
> 配套 benchmark：`data/benchmarks/anthropic_og_style/`（6 sample）。

## 1. 为什么需要这种考试形式

产品图还原 = 「对着 target 越像越好」。本形式有三点根本不同：
1. **学共性、不学 motif**：给 Generator **多张**参考逼它抽象风格共性（神韵），原创靠 `originality_motif` + forbidden_motifs 约束，不靠蒙眼。
2. **考概念表达，不是泛主题**：图必须表达文章的**核心概念**（Generator 还要产出一句话 meaning 自证），不只是「贴主题」。
3. **考泛化性**：6 篇不同概念域的文章，看系统在 A 文章学到的「风格→概念」表达方法能否迁移到 B/C/D。**关键要素（神韵维度）不能开题拍板**，随经验闭环演化（§9）。

## 2. 新考试形式 `style_transfer_single`（多 sample）

| 字段 | 取值 |
|---|---|
| `manifest.scene_type` | `style_transfer` |
| `manifest.task.type` | `style_transfer_single` |
| Generator 输入 | **参考集（7 张）** + 文章核心概念 + 风格 brief（辅助） |
| Generator 产出 | **封面图 + meaning（一句话含义解释）** |
| 评分三柱 | 神韵 0.40 / 原创差异 0.40 / 概念表达 0.20 |
| sample 结构 | 6 个 sample = 一文一题（见 §5） |

参考集布局：`samples/<s>/reference_style/*.png`（7 张共享）+ `target.jpg`（contact sheet）+ `target.md`（参考说明 + 本文角色 + 官方 og 对照）。

## 3. meaning 产出与 concept_expression 判定（本次升级核心）

- **Generator 产出 meaning**：`GeneratorOutput` 新增 `meaning` 字段（一句话，≤40 字），解释「这张图如何用视觉隐喻表达文章概念」。generator 在 style_transfer 模式被强制要求输出。
- **全链路打通**：`GeneratorOutput` → `GenOutcome` → `CriticInput`(传给 critic) + `AttemptRecord`(存 trajectory)。
- **Critic 看 meaning 判概念表达**：critic 的 user content 注入 generator 的 meaning，`concept_expression` 维度判「图 + meaning 是否一致 + 是否清晰表达文章核心概念」（避免 generator 嘴上说一套、图画另一套）。
- meaning 也是**最终图文方案**的那句文字（图 + 一句话含义）。

## 4. 三柱评分（9 维度，权重和=1.000）

- **神韵 0.40**：套系统提炼的 5 要素（`spirit_*`，v1 候选）。
- **原创差异 0.40**：`originality_motif`(0.24, binary, motif 非复制——参考对照类额外不得复制官方 og) + `originality_degree`(0.16, continuous)。
- **概念表达 0.20**：`concept_expression`(0.12, binary, 图+meaning 表达核心概念) + `concept_clarity`(0.08, continuous)。

## 5. 多文章分组与泛化性考察（`generalization_protocol`）

6 sample 分两组，**泛化性是 benchmark 级跨 sample 考察**（不是单 sample 内维度）：

| 组 | sample | 文章 | 设计意图 |
|---|---|---|---|
| **参考对照** | s002, s003 | teaching-claude-why / cryptographic-weaknesses | og 在参考集 → 有官方 motif 作 **ground truth**，验证系统能否逼近官方表达 |
| **非案例考题** | s001, s004, s005, s006 | global-workspace / project-deal / robotics / 81k-interviews | og 不在参考集 → 无官方答案，纯考**泛化与概念原创表达** |

**泛化性指标**（跑完 6 个 loop 后看）：
1. **跨 sample 经验迁移**：后跑 sample 首轮还原度 vs 先跑（冷启动对比）——看 `general.json` 是否真沉淀了「跨文章通用的风格→概念表达方法」。
2. **概念表达泛化**：`concept_expression` 在 6 篇不同概念域上的表现稳定性。
3. **参考对照 vs 非案例差距**：理论参考对照应略高（有官方 motif 参照），差距小说明方法泛化好。

> 跑法建议：先跑参考对照（s002/s003，让 general.json 沉淀「有 ground truth 验证过」的方法），再跑非案例（看迁移效果）。

## 6. 新工具设计（待实施 · 利用 deepagent）

### Generator 侧
- `forbidden_motifs` 已通过 `constraints` 字段注入 generator user content（v2 已修：ContentSpec 加 constraints 字段，`_extract_constraints` 提取）。generator 看到文字 motif 黑名单 + 7 张参考图，双重防撞车。

### Critic 侧
- meaning 已注入 critic user content（v2 已实施）。**参考集多图注入**（替换单张 contact sheet）仍待实施——让 critic 逐张比对判 motif 原创性。

### Summarizer 侧（要素演化，见 §9）
- `discover_spirit_elements(critic_reasons)`：从 Critic 逐轮 reason 蒸馏真关键要素，反哺神韵 checklist。

## 7. test_variable 策略

| test_variable | 策略 | 理由 |
|---|---|---|
| `reference_images` | ✅ 允许（多张） | 多张逼抽象共性、学神韵 |
| `prompt` | ✅ 允许 | 改 motif 隐喻 / 概念表达 |
| `size` / `seed` | ✅ 允许 | |

原创约束靠 `originality_motif` 维度 + forbidden_motifs + 多图抽象三重保证，不靠蒙眼。

## 8. 经验闭环 A/B/C 三环适配

| 环 | 产品图还原 | → 风格神韵迁移 |
|---|---|---|
| **A（lesson 富化）** | 保持方向 / 换 ControlNet | 聚焦：「哪种 motif 隐喻能表达某类概念」「哪种笔触/配色更近神韵」——**跨 sample 沉淀进 general.json = 泛化方法** |
| **B（escalated）** | 撞模型上限 | **先怀疑要素错位再怀疑模型**（§9）。神韵维度连续失败，可能是「要素判定不现实」而非模型不行 |
| **C（generator 警告）** | 换 test_variable | 换 motif 隐喻 / 换 seed；若 escalated 源于要素错位则上升到 §9 要素演化 |

## 9. 要素演化（本形式的核心机制）

### 问题
神韵 5 要素是 qwen3-vl-plus 一次性快照，可能漏关键/权重偏/判定不现实。固化成静态 checklist = 开考前写死答案。

### 实证（首轮 global-workspace 跑出的）
`spirit_ink_line` 连续 2 轮 0.60 → escalated。但它的判定项「线宽 8–12px、**标准差≥1.5px**」是**像素级精确要求**，seedream-pro 文生图根本无法精确控制线宽标准差。**这是要素判定不现实，不是模型瓶颈**——应修订要素（放宽成定性「手工笔触感」），而非判模型死刑。首轮就应验了「escalated 先怀疑要素错位」。

### 机制（四层）
1. **种子 v1**：系统一次性提炼的 5 要素（`content_spec.spirit_*`）。
2. **发现（每 loop）**：Summarizer 蒸馏 N 轮 critic reasons → 真关键 / 噪声 / 漏网 → `conclusions.json`。
3. **沉淀（跨 loop）**：多次 loop 蒸馏进 `general.json` 的 `style_key_elements`。
4. **反哺**：下一 loop 启动前更新 `spirit_*` checklist（v1→v2→…）。骨架稳定，判定项可演化。

### `discover_spirit_elements` LLM 任务草案
输入 N 轮 critic reasons → 输出「真关键要素 / 噪声项 / 漏网点 + 增删调权建议」。

## 10. 跑法（多 sample）
```bash
# 每个 sample 一个 loop，用 -tag 防污染。建议先参考对照、后非案例（验泛化）
for s in s002 s003 s001 s004 s005 s006; do
  .venv/bin/python .claude/skills/img-iter-ops/scripts/run_loop_auto.py \
    --bench anthropic_og_style --sample $s --rounds 4 --tag style-exp2
done
```

## 11. 诊断要点（本形式 + 泛化专属）
- 神韵柱哪个反复低分 → 先查要素错位(§9) 还是 brief/模型问题(A 环)；
- `originality_motif` 频繁不通过 → generator 在抄参考，查是否换 motif（C 环）；
- `concept_expression` 低但神韵高 → 抓了风格但没表达概念，meaning 与图不一致；
- **泛化**：后跑 sample 首轮 vs 先跑冷启动；参考对照 vs 非案例差距。

## 12. 后续可演进
- 落地 `discover_spirit_elements` 自动反哺 + critic 多图注入；
- `new_benchmark.py --scene style_transfer` 一键骨架；
- 泛化性诊断脚本（自动对比 6 sample 的跨 loop 迁移曲线）。
