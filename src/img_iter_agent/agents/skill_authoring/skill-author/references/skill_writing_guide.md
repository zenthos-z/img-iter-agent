# Skill 写作指南（摘自 skill-creator，规范基准）

本文件是 skill-creator 写作方法论的**完整复制**（仅保留「如何写规范 skill」部分，去掉活 eval-loop）。
经验技能编写员（见上级 SKILL.md）据此写作；本文件讲通用规范，上级 SKILL.md 讲经验技能的特化。

## 技能包结构（Anatomy）

```
skill-name/
├── SKILL.md (必需)
│   ├── YAML frontmatter (name, description 必需)
│   └── Markdown 正文
└── 可选 bundled 资源
    ├── scripts/    - 可执行代码（确定性/重复任务）
    ├── references/ - 按需加载的文档
    └── assets/     - 产出用文件（模板/图标/参考图）
```

## 渐进披露（3 级加载）

1. **metadata**（name + description）—— 常驻上下文（~100 词）。
2. **SKILL.md 正文** —— 触发时加载（**< 500 行**为佳）。
3. **bundled 资源** —— 按需（无上限；scripts 可不加载即执行）。

词数是近似值，必要时可更长。

**关键模式**：
- SKILL.md 控制在 500 行内；接近上限就加一层层级 + 写明「接下来去哪读」。
- 从 SKILL.md 清楚引用各 reference 文件，并说明何时读它。
- 大 reference（>300 行）加目录。

**多域组织**：技能支持多域/多框架时，按变体组织（Claude 只读相关那篇）：
```
cloud-deploy/
├── SKILL.md (工作流 + 选择)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

## 写作模式

- **用祈使句**写指令。

- **定义输出格式**——用模板明确定义：
  ```markdown
  ## 报告结构
  ALWAYS use this exact template:
  # [标题]
  ## 摘要
  ## 关键发现
  ## 建议
  ```

- **示例模式**——给输入/输出示例（若有 Input/Output 字样可略偏离格式）：
  ```markdown
  **Example 1:**
  Input: 加了 JWT 鉴权
  Output: feat(auth): implement JWT-based authentication
  ```

## 写作风格

- **解释 why**，而非堆 MUST。今天的 LLM 很聪明，给它理由比死命令更有效、更人性化。
- 用 theory of mind，让技能**通用**而非窄化到具体例子。
- 想写 `ALWAYS` / `NEVER` 全大写时是黄牌——尽量改成解释原因，让模型理解为何重要。
- 先写草稿，再用新鲜眼光审视改进。

## description（触发机制，最重要的字段）

description 是 Claude 决定是否调用技能的**主要依据**。要同时含【做什么】+【何时触发】——
所有「何时用」的信息都放这里，不放正文。

当前 Claude 倾向 **undertrigger**（该用时不用），所以 description 要略 **pushy**：

- ❌ 「How to build a simple dashboard.」
- ✅ 「How to build a fast dashboard to display internal data. **Use this whenever** the user mentions
  dashboards, data visualization, internal metrics, or wants to display any kind of data, even if they
  don't explicitly ask for a 'dashboard.'」

## name 规范

- kebab-case（小写字母 / 数字 / 连字符）。
- 不以连字符开头/结尾，无连续连字符。
- ≤ 64 字符。
- **name == 目录名**（本系统强制：目录名 = bench slug，frontmatter name 同步）。

## frontmatter 允许的字段

仅允许：`name` / `description` / `license` / `allowed-tools` / `metadata` / `compatibility`。
其它键会被 quick_validate 拒绝。
