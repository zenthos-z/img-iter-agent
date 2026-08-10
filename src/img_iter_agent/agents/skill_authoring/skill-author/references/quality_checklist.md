# 质量 checklist（review 阶段审查 + 修订依据）

review 阶段：你拿到 draft 的全文 + 本 checklist。**只改有问题的部分，保留草稿有效内容，不要推倒重写**；
若草稿已合格，原样返回即可。逐条核对：

## A. 结构合规（quick_validate 硬规则，必过）

- [ ] SKILL.md 有 YAML frontmatter（`---` 包裹）。
- [ ] frontmatter 仅含允许键：`name` / `description` / `license` / `allowed-tools` / `metadata` / `compatibility`。
- [ ] `name` 存在、kebab-case（`^[a-z0-9-]+$`）、≤64 字符、不以连字符开头/结尾、无连续连字符。（name==dir 由系统强制）
- [ ] `description` 存在、≤1024 字符、**无尖括号 `<` `>`**。

## B. description 质量

- [ ] 含【做什么】+【何时触发】。
- [ ] pushy（明确倾向于触发）。
- [ ] **通用，不窄化到具体例子**（无「如 X」「例如某 Y」）。
- [ ] **正确框定产物**：是「产出策略」，不是「是生成器」。是「输入 X → 产出策略」，不是「X 生成器」。

## C. SKILL.md 冷启动可用（拿来即用 = 核心，最重要）

- [ ] 只读 SKILL.md（不读 references）就能产出合格策略。
- [ ] 含【何时用 / 输入契约】（吃什么、吐什么）。
- [ ] 含【核心工作流】（**可执行步骤**，非泛谈口号）。
- [ ] 含【输出格式模板】（结构化对象，`strategy` 字段可扩展）。
- [ ] **前景化 3-5 条关键 dos/donts**（直接内联在 SKILL.md，不依赖读 references）。
- [ ] 有指针指向 `references/lessons.md`（及其它 reference）。
- [ ] < 500 行。

## D. lessons 核心精华（不可弱化）

- [ ] 紧跟标题声明「必读 `references/lessons.md` 并遵循 dos/donts」。
- [ ] 内联 3-5 条最高置信/最关键 dos/donts。
- [ ] 未把 lessons 弱化为可选参考。

## E. mode 自适应

- [ ] style 类：用到的参考图在 `asset_paths`；写了风格指南 reference。
- [ ] image_edit 类：运行时输入不打包；写了编辑流程 + eval criteria。
- [ ] 结构据 mode 自定，非硬编码模板。

## F. 生成目标覆盖（最重要——别漏考题要求；数据驱动，不预设 benchmark 类型）

- [ ] **有「## 评分目标」段**（输入→输出契约 + 评分维度按权重排序 + 每维度判定标准 + 评分意图 why），置于工作流之前。
- [ ] **按 dossier 实际权重讲了高权重维度怎么命中**——权重最大的维度有专门展开。（高权重是什么由数据决定：风格类可能是创造力/原创，还原类可能是 consistency/structure——不预设。）
- [ ] **constraints 硬约束**：若 dossier 有 `forbidden_motifs` 或关键 `must_avoid`，SKILL.md 明确列出 + 解释为何 + 如何规避；**没有就不写，不凭空编造**。
- [ ] checklist 的关键 binary ✓/✗ 判定（用本 benchmark 真实判定项）进了自检步骤。
- [ ] 传达了 benchmark 的评分意图（why，据 rubric 实际写的倾向/意图）。

## G. 写作基调（拒绝机械填模板）

- [ ] 读起来像深度理解 benchmark 的人写的指南，非填模板产物。
- [ ] 规则附 why（用本 benchmark 真实的维度权重/意图作理由，不只是 MUST/禁止罗列）。
- [ ] 有 benchmark-grounded 的具体例子（真实维度/约束/输入），非占位符。
- [ ] 传达了 benchmark 想要什么样的产出（核心目标 + 最看重的维度）。
- [ ] 段落讲思路 + 例子讲做法 + 清单做速查，三者结合；不是清一色规则罗列。

## H. references / assets

- [ ] `references` 路径是相对名（如 `style_guide.md`），**不含 `lessons.md`**（系统渲染）。
- [ ] `asset_paths` 是 benchmark 内相对路径，且 generator 实际用到（不瞎列）。

---

修订产出仍是完整 `AuthoredSkill`（与 draft 同 schema）。把改后的 description / skill_md / references / asset_paths
填全；未改的字段照抄 draft。
