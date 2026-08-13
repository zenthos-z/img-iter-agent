# 创建标准化 benchmark 考题

用户给了「产品图 + 口述验收标准」，要变成一道可被闭环跑的标准考题。目标产物：

```
data/benchmarks/<bench>/
├── manifest.json              # 评分体系真源：6 维度 + 权重 + check_items
├── rubric.md                  # 人类可读说明（维度真源仍是 manifest）
└── samples/<sNNN>/
    ├── target.jpg             # ★产品实物参考图（Critic 的评判锚）
    ├── target.md              # （推荐）结构/材质/颜色说明 + 还原要点
    └── content_spec.json      # 任务 + 约束 + 6 维度 checklist/rubric points
```

## manifest vs content_spec：通用标准 vs sample 专属（先读这块）

一个 benchmark = **1 个 `manifest.json`（全 bench 共享）+ N 个 `content_spec.json`（每 sample 专属）**。两者职责分明——混写就是 bug 源头（见下「消费方式」里的合并规则）。

### `manifest.json` —— benchmark 通用标准（所有 sample 自动继承）
评分体系的**真源**，只写「跟具体 sample 无关、整个 bench 通用的东西」：
- **`score_dimensions`**：6 维度定义（`dim` / `desc` / `scoring_type` / `weight_init` / `ref_needed`）——Critic 的维度清单、顺序、二分/连续类型全从这里来。
- **二分维度的 `check_items`**：通用底线判定项（如 consistency C1–C5、artifact_defect A1–A4）。**所有 sample 自动继承，所以 sample 不再写这些二分项**。
- **`weight_init`**：先验权重（loop 跑起来后由排序校准覆盖）。
- **`task`**：bench 级任务类型（三视图白底等）。
- **`samples[]`**：只是**引用**（`sample_id` / `product` / `target` 路径 / `difficulty_note`），**不含 sample 内容本身**。

### `samples/<s>/content_spec.json` —— 单个 sample 专属描述
只写「**只属于这一道题**」的东西，是 generator/critic 取 sample 专属上下文的入口：
- **`task.input_assets`**：该 sample **独属的输入素材**（指向 sample 目录下要读的文件，如风格迁移参考图、要对照的独属文章/图片）。←「需要读什么独属的内容」走这里。
- **`task.instruction` / `task.output` / `task.article_topic`**：该题的任务指令、输出要求、概念主题。
- **`constraints`**（`must_keep` / `may_change` / `must_avoid` / `forbidden_motifs`）：给 **generator** 的还原指引，运行时直接拼进生图 prompt（特定结构如「折叠关节」「弧形床头板」只进这里，**不进 checklist**）。
- **`checklist`**：只填该 sample **专属**的评分项——
  - 连续维度（material_texture / color_accuracy）的 `points`：manifest 没有连续 points，**必须写在 sample**；
  - 二分维度：**只在确有额外专属项时才追加，绝不重复 manifest 的通用项**（重复 → 渲染两遍，见下）。

### 消费方式（谁读什么）
| 消费方 | 从 `manifest` 读 | 从 `content_spec` 读 |
|---|---|---|
| **Critic 打分** | `score_dimensions`（维度/类型/顺序）+ 二分 `check_items`（通用底线） | `checklist`：连续 `points` + 二分的**额外**项 |
| **Generator 出图** | —（不直接读 manifest） | `task`（`mode`/`input_assets`/`instruction`）+ `constraints`（注入 prompt）+ `target.jpg`/`target.md` |
| **还原度 / 权重** | `init_weights()` 先验 → 排序校准覆盖 | — |

**⚠️ 关键合并规则（上次「标准重复」bug 的根源）**：Critic 的 `_effective_checklist`（`agents/tools/critic_tools.py`）对每个二分维度做的是「**manifest 通用项 + sample 项，直接拼接**」，**没有去重**。所以：
- sample 的二分 `checklist` 一旦重复了 manifest 已有的通用项 → 同一项在 critic 用户提示词里渲染两遍，还会让通过率分母（`passed/len(items)`）翻倍、带偏分数。
- 正确姿势：**通用二分项只在 manifest；sample 二分项只写额外专属项（多数情况为空）；连续 `points` 只在 sample。** 代码侧不做兜底去重，靠 benchmark 数据卫生保证。

## 标准流程

### 1. 搭骨架（脚本一键）
```bash
.venv/bin/python .claude/skills/img-iter-ops/scripts/new_benchmark.py \
    --bench <bench_id> --samples s001 s002 [--scene "场景描述"]
```
生成目录 + manifest（6 维度骨架）+ rubric + 每个 sample 的 content_spec 模板。

### 2. 放参考图
把用户给的产品实物图放到 `samples/<sNNN>/target.jpg`。**这是对比型维度（consistency/structure/material/color）的评判锚**，没它 Critic 没法对照打分。

### 3. LLM 起草 content_spec（核心，由你/Claude 完成）
脚本只复制了模板（`assets/content_spec.template.json`，含 6 维度骨架 + 占位符）。现在要根据**产品图 + 用户口述**，把每个维度的 checklist 填成针对该产品的具体判定项。

**⚠️ 两层标准（manifest 通用 / sample 专属）的区别与消费方式见上方「manifest vs content_spec」，这里只说落地**：sample **不写二分维度**（通用项由 manifest 继承），只填 ① 连续维度 material/color 的 points；② constraints.must_keep/must_avoid（给 generator 的还原指引，含特定结构）。

**起草要点（决定经验闭环质量 + 泛化性）：**
- **二分维度（consistency/product_structure/artifact_defect/commercial_focus）的每一项必须可二分判定**（✓/✗ 明确无歧义）。**但绝不写死特定结构**——S4 别写「长条折叠形态」这类绑定参考图的特征，应写开放式对比「整体造型轮廓忠实还原参考图，对照 target 自由判断」：让 Critic 动态对比参考图，而非机械查"折叠关节在不在"。写死特定结构 = 设计缺陷（lesson 沉淀绑定该 sample、无法泛化）。**特定结构（折叠关节/弧形床头板等）只进 `constraints.must_keep`**（给 generator 的还原指引）。
- **连续维度（material_texture/color_accuracy）填具体 `points`**，让 LLM 知道打分看哪些点。例：material 的 points = ["金属框架质感(非塑料)", "织物有编织纹理(非光滑面)"]。
- **`must_avoid` 写该产品最易翻车的失真**（经验闭环靠它聚焦）。例：折叠椅写「折叠结构变形为普通床腿」「三视角不一致」。
- **A4「接地阴影」项别漏**——历史经验里「悬浮感」是反复出现的关键缺陷（见 `data/experience/.../general.json`）。

起草时直接看产品图，把口述的「关键结构/材质/颜色/易翻车点」映射到对应维度的 check 项。

### 4. 补 manifest 的 samples 描述
打开 `manifest.json`，把每个 sample 的 `product`/`category`/`difficulty_note` 从 TODO 填实（脚本已留好位）。

### 5. （推荐）写 target.md
一段该产品的「结构/材质/颜色 + 还原要点」，给 Critic 当辅助参考。可选但提分。

## 校验（跑之前确认）
- 每个 `samples/<s>/target.jpg` 存在
- content_spec.json 的 checklist **只含连续维度**（material_texture/color_accuracy）的 points；**不含二分维度**（consistency/product_structure/artifact_defect/commercial_focus 由 manifest 继承）
- constraints.must_keep/must_avoid 含该产品的特定结构 + 易翻车点（给 generator）
- manifest.json 无残留 TODO；check_items 含 C5（视图完整性）+ 开放式 product_structure（S1/S4 不绑定具体形态）
- 维度名与 manifest score_dimensions 一致（Critic 按这些 dim 打分）

## 全新场景（非家具白底）怎么办
`new_benchmark.py` 默认沿用家具白底的 6 维度。若场景差异大（如服装、食品、室内场景图），
要手动改 manifest 的 `score_dimensions`（增删维度、调 weight_init 使 Σ=1、改 scoring_type），
并相应改 content_spec 模板的 checklist 结构。这是少见的高级操作，改完务必让 Critic 跑一轮确认维度能打分。
