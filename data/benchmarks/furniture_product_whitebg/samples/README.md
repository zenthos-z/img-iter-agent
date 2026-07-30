# 考题结构说明（samples/ 目录规范）

> 一道**考题** = 一个「生成任务」+ 一个「还原基准」+ 一组「二分验收项」。
> 系统拿任务输入去生成**三视图白底图**，Critic 对每张图逐项做**二分判定（✓/✗）**，
> 汇总成通过率，对照还原基准评还原度。详见 `docs/EVALUATION.md`。

## 任务：生成工业标准三视图白底素材图

每道考题要求生成 **3 张视图**：**正视图 + 侧视图 + 立体图**（白底）。
- 正视图：正面平视；侧视图：90° 侧面平视；立体图：45° 斜侧。
- 三张必须是**同一产品、同色、同比例**（跨视图一致性是重点验收项）。

## 一道考题 = 一个子目录

```
samples/<sample_id>/
├── target.jpg          # ★还原基准图（对比锚）——材质/颜色/结构对照它评
├── target.md           # 产品说明：结构/材质/颜色 + 验收点
└── content_spec.json   # 生成任务定义 + 二分验收 checklist（见下）
```

## content_spec.json 字段（考题的核心定义）

```jsonc
{
  "sample_id": "s001",
  "product": "欧式白色雕花双人床",
  "category": "bed",

  "task": {
    "mode": "image_edit",              // image_edit / multi_ref_fusion
    "input_assets": ["target.jpg"],    // 输入素材（相对本目录）
    "instruction": "以该图为参考，生成正/侧/立体三个视角的白底素材图",
    "output": {
      "views": ["front", "side", "perspective"],   // ★三视图
      "background": "white",
      "size": "2K"
    }
  },

  "constraints": {
    "must_keep": ["弧形床头板", "雕花装饰", "纯白漆面"],
    "may_change": ["背景", "光影"],
    "must_avoid": ["床头板变平直", "雕花消失", "悬浮不接地"]
  },

  // —— ★二分验收 checklist（核心：分类代替量化）——
  // 每项是 LLM 擅长的 ✓/✗ 判定，归入维度组，带判定锚点
  "checklist": {
    "product_structure": [             // 维度组 = manifest 的 score_dimensions
      {"id": "S1", "check": "部件数量与参考一致", "anchor": "对照 target"},
      {"id": "S2", "check": "无部件穿模/重叠"},
      {"id": "S3", "check": "无部件缺失"}
    ],
    "consistency": [                   // ★跨视图一致性（三视图特有，重点）
      {"id": "C1", "check": "三视图是同一产品"},
      {"id": "C2", "check": "三视图颜色一致"},
      {"id": "C3", "check": "侧视宽度=正视宽度（几何一致）"},
      {"id": "C4", "check": "各视图高度比例一致"}
    ],
    "material_texture": [
      {"id": "M1", "check": "材质类型与参考一致", "anchor": "对照 target"},
      {"id": "M2", "check": "有真实纹理（非光滑塑料感）"}
    ]
    // ... 其余维度组见 manifest
  },

  "anchor_for": ["material_texture", "color_accuracy", "product_structure"]
}
```

## 三个关键概念

1. **还原基准（target.jpg）**：生成图的"标准答案"。带 `anchor` 的验收项对照它判。
2. **三视图任务**：每题产出 3 张图（正/侧/立体），白底。跨视图一致性是核心质量信号。
3. **二分 checklist**：每个维度拆成若干 ✓/✗ 项，Critic 分类判定（而非打分）。
   还原度 = Σ(维度权重 × 维度通过率)。

## 怎么加一道考题

1. 复制一个现有考题目录为 `sNNN/`
2. 放入 `target.jpg`（还原基准图）
3. 写 `target.md`（产品结构/材质/颜色 + 该产品特有验收点）
4. 改 `content_spec.json`：task（三视图）+ constraints + checklist（按产品定制二分项）

> 二分项尽量"可客观判定"——判"穿模有无"比判"材质几分"准得多，这正是改量化的原因。

