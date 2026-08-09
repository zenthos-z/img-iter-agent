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

**起草要点（决定经验闭环质量）：**
- **二分维度（consistency/product_structure/artifact_defect/commercial_focus）的每一项必须可二分判定**（✓/✗ 明确无歧义）。例：S4 别写「结构合理」，写「仍是长条折叠形态（非普通床，对照参考）」——Critic 才能客观判。
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
- 每个 content_spec.json 的 checklist id 连续（C1-C4/S1-S4/A1-A4/B1-B3）且 check 文本非占位
- manifest.json 无残留 TODO
- 维度名与 manifest score_dimensions 一致（Critic 按这些 dim 打分）

## 全新场景（非家具白底）怎么办
`new_benchmark.py` 默认沿用家具白底的 6 维度。若场景差异大（如服装、食品、室内场景图），
要手动改 manifest 的 `score_dimensions`（增删维度、调 weight_init 使 Σ=1、改 scoring_type），
并相应改 content_spec 模板的 checklist 结构。这是少见的高级操作，改完务必让 Critic 跑一轮确认维度能打分。
