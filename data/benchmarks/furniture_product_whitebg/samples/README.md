# 考题结构说明（samples/ 目录规范）

> 一道**考题** = 一个「生成任务」+ 一个「还原基准」。系统拿任务输入去生成图，
> Critic 把生成图与还原基准对比，按 manifest.json 的维度打分（还原度）。

## 一道考题 = 一个子目录

```
samples/<sample_id>/
├── target.png          # ★还原基准图（对比锚）——材质/颜色维度对照它评
├── target.md           # 产品说明：结构/材质/颜色/验收点（给人+Critic看）
└── content_spec.json   # 生成任务定义：输入什么→输出什么→约束（见下）
```

## content_spec.json 字段（考题的核心定义）

```jsonc
{
  "sample_id": "s001",

  // —— 产品信息 ——
  "product": "北欧实木餐椅",
  "category": "chair",              // bed/chair/table/sofa/cabinet/...

  // —— ★生成任务（决定 AI 拿什么、生成什么）——
  "task": {
    "mode": "image_edit",           // text_to_image / image_edit / multi_ref_fusion
    "input_assets": ["target.png"], // 输入素材路径（相对本 sample 目录）
    "instruction": "保持产品完全一致，仅将背景替换为纯白",  // 生成指令
    "output": {
      "view": "front",              // 期望输出视角
      "background": "white",        // 期望输出背景
      "size": "2K"                  // 期望尺寸
    }
  },

  // —— 约束（供 Generator 控制变量 + Critic 验收）——
  "constraints": {
    "must_keep": ["原木色", "四条直腿", "靠背竖条数量"],   // 必须保留的特征
    "may_change": ["背景", "光影"],                       // 允许变化的部分
    "must_avoid": ["悬浮不接地", "腿部穿模", "材质塑料感"] // 必须避免的瑕疵
  },

  // —— 对比锚说明（哪些维度要对照 target.png）——
  "anchor_for": ["material_texture", "color_accuracy"]
}
```

## 三个关键概念

1. **还原基准（target.png）**：生成图的"标准答案"。对比型维度（材质/颜色）对照它打分。
   绝对型维度（结构/瑕疵/商业）单看生成图评，但 target.md 里写的结构也要满足。

2. **生成任务（task）**：AI 的输入与目标。常见三种模式：
   - `image_edit`：给产品图，指令改某处（如换背景、换色）——**最常用**
   - `multi_ref_fusion`：给多张参考图，融合生成
   - `text_to_image`：纯文字描述生成（无参考图，此时材质/颜色维度难评）

3. **约束（constraints）**：must_keep / may_change / must_avoid——
   Generator 据此控制变量，Critic 据此验收。

## 怎么加一道考题

1. 复制 `_TEMPLATE/` 为 `s001/`（或 s002、s003…）
2. 放入 `target.png`（还原基准图）
3. 写 `target.md`（产品结构/材质/颜色 + 验收点）
4. 改 `content_spec.json`（任务 + 约束）

> 注意：target.png 既是**风格/还原度锚**，也是**对比型维度的基准**，一图两用。
