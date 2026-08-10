# Benchmark: Anthropic OG 封面风格 · 神韵迁移创作

> `bench_id`: anthropic_og_style
> 评分维度真源 = `manifest.json`（本文件仅人类可读说明，二者不一致以 manifest 为准）。
> **场景类型 = 风格神韵迁移（非产品图还原）**。Generator 看参考集（多张）抽象风格共性，凭文章主题原创 motif；神韵要素为 v1 候选，随经验闭环演化。

## 任务契约
- **输入（Generator）**：**参考集（7 张）** + `{article_topic}`（文章主题）+ 风格 brief（v1 候选，辅助）。
- **输出**：单张封面插画。
- **学共性、不学 motif**：多张参考逼模型抽象风格共性（=神韵）；motif 必须原创（forbidden_motifs + `originality_motif` 维度约束），不靠蒙眼。

## 四柱评分（还原度 = Σ wᵢ × feature，feature∈[0,1]）

> v2.1：解禁 generator 传参考图 + 新增「柱 D 创造力」反制参考图过度锚定。权重经 creativity ≈ 0.30 再分配（主要切自原创）。创造力细分标准由 `creativity_tuner` 对抗式全自动演化（overlay：`data/benchmarks/anthropic_og_style/creativity_criteria.json`，永不改种子、留版本痕）。

### 柱 A · 风格神韵（合计 0.36）
套用系统提炼的 5 个神韵要素（权重×0.36 缩放）：
- `spirit_hand_form`(0.090, binary, 5 项) 抽象主体造型——含**极致平面/一笔画/无细节/一点透视**硬约束：整只手是一笔连续粗墨线、无指甲/褶皱/关节纹理、无错落立体堆叠/纵深透视（v2.2 新增 -4/-5 项）
- `spirit_ink_line`(0.079, continuous) 墨线质感
- `spirit_geometric_object`(0.072, continuous) 几何化物件
- `spirit_negative_space`(0.065, continuous) 负空间节奏
- `spirit_symbolic_interaction`(0.054, binary) 象征性互动

### 柱 B · 原创差异（合计 0.20）
- `originality_motif`(0.130, binary) motif 非复制——主体不得雷同任一参考 motif
- `originality_degree`(0.070, continuous) 原创程度——是否在参考 motif 空间之外

### 柱 C · 概念表达（合计 0.14）
- `concept_expression`(0.085, binary) 图 + meaning 共同清晰表达文章核心概念
- `concept_clarity`(0.055, continuous) 概念隐喻清晰度

### 柱 D · 创造力（合计 0.30）—— 反制参考图过度锚定
- `creative_departure`(0.200, continuous) 创造性偏离——在保留神韵前提下提出参考 motif 空间之外的有意义新隐喻（服务于概念，非随机求新）
- `reference_independence`(0.100, binary) 参考独立性——不直接复制本次实际传入参考图（`reference_ids`）的 motif；纯文生图默认通过

权重合计 = 0.36 + 0.20 + 0.14 + 0.30 = **1.000**

## 对比型 vs 绝对型
- 对比型（柱 A + 柱 B + 柱 D，ref_needed=true）：Critic 同时看参考集（target.jpg = contact sheet）+ 生成图。
- 绝对型（柱 C，ref_needed=false）：只看生成图 + article_topic。

## 评分要点
- binary 维度逐项 ✓/✗ → 通过率；**严格倾向**：拿不准/有瑕疵/与参考撞 motif 时判不通过。
- continuous 维度 LLM 给 0-1，偏差由排序校准吸收。
- 「无手变体」容许：object-laptop 证明无手也是合法变体，`spirit_hand_form` 不通过不应一票否决整题（见 target.md）。

## 当前实现的适配点（不改 agent 代码）
1. **Critic 仅注入单张 target.jpg**：已用 7 图 contact sheet 作 target，让 Critic 一图看全参考。完整多参考逐张注入是待实施工具（见策略文档 §3）。
2. **神韵要素 = v1 候选**：5 要素由系统一次性提炼，**不保证是关键要素**；真实关键要素由经验闭环动态发现（见策略文档 §6 要素演化）。神韵维度 escalated 时应先怀疑要素错位、再怀疑模型上限。
