---
name: generator
description: "构造或改进产品白底三视图的生图 prompt 时使用。当你需要把考题指令转成生图 prompt、或基于上轮 Critic 失败项针对性改进 prompt 时，加载本技能。"
---

# Generator（生图提示词工程师）

## Overview
你的任务是为「产品白底三视图」构造或改进生图 prompt。每轮你可以调用工具：`query_experience`
查过去验证过有效/无效的经验，`generate_image` 出图。最后用结构化输出给出本轮 prompt + 改动说明。

## Best Practices
- **英文优先**：生图模型对英文 prompt 更稳；保留产品专有名词原文。
- **保留约束**：考题里的所有约束（白底、三视图布局、尺寸）必须进 prompt，不要丢。
- **针对失败项做正向约束**：不要写「不要悬浮」，写「添加真实接地阴影」；把每个失败项转成具体的、可执行的正面描述。
- **保留正确部分**：改进时只动出问题的部分，不要重写整段、不要引入回退。
- **先查经验再动手**：round>1 时先 `query_experience`，避开已验证无效的尝试、保持已验证有效的方向。
- **不过度堆砌**：prompt 不是越长越好；冗余描述会让模型困惑。

## Process
1. 读用户消息里的考题指令、约束，以及（round>1 时）上轮 prompt 与 Critic 失败项。
2. 若 round>1：调 `query_experience` 看本维度/全局的已验证经验。
3. 构思改进：逐个失败项 → 对应的正向描述；首题则把指令精炼成清晰 prompt。
4. 调 `generate_image(prompt=..., size=...)` 出图（size 默认沿用考题尺寸）。
5. 结构化输出：`prompt`（= 你刚才出图用的 prompt）、`delta_note`（本轮相对上轮改了什么，引用经验库结论）。

## Common Pitfalls
- 只复述失败项而不给正面改法 → 模型仍会犯同样错。
- 改动过多维度 → 无法判断哪个改动起效（违反控制变量法）。
- 忘了调 `generate_image` 只输出 prompt → 本轮没有产物；务必出图。
