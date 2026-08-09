# 经验系统说明 —— 载体 / 流程 / 实跑节点效果

> 回答三个问题：① 经验存在哪个文件、内容是什么、怎么生产与消费；② 一道题 4 轮的完整运转流程与每个节点的效果；③ 「一直没总结出经验」到底是前端显示问题还是真没产出。
>
> 文中所有数字与结论来自 **2026-08-08 的一次真实实跑**：`furniture_product_whitebg / s003`，全新 run `furniture_product_whitebg-s003-live`，4 轮（generator=`deepseek-v4-flash`，critic=`gemini-3.1-flash-lite`，生图=`gemini-3.1-flash-image`）。

---

## 0. 一句话结论

经验**有产出**，但分**两层、两个文件**，且其中一层**前端完全没接入**；循环内那层对「简单题」会产出**噪声桩**。你看到「没有经验」，是 **(前端没显示跨 loop 经验) ＋ (循环内经验对简单题是垃圾)** 两件事叠加的结果，不是系统完全没跑通。

---

## 1. 经验承载本体：两个文件（不是同一个）

系统里有**两套独立**的经验，载体、写法、读者都不同。这是理解一切的前提。

```
                        ┌───────────────────────────┐
                        │   trajectory.jsonl (原料)  │   每轮：prompt + Critic verdict + delta_note
                        └───────────┬───────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
   ┌────────────────────────────┐         ┌──────────────────────────────┐
   │ ① 循环内经验 conclusions.json│         │ ② 跨 loop 经验 general.json   │
   │   data/runs/<loop>/lessons/ │         │   data/experience/<bench>/    │
   │        conclusions.json     │         │        general.json           │
   └────────────────────────────┘         └──────────────────────────────┘
        每轮、规则驱动、单 loop                   离线、LLM 蒸馏、跨 loop
        Summarizer 写                            ExperienceDistiller 写
        Generator 的 query_experience 读          Generator 的 query_general_experience 读
        ✅ 前端展示（经验沉淀面板）                 ❌ 前端零接入
```

| 维度 | ① conclusions.json（循环内） | ② general.json（跨 loop） |
|---|---|---|
| **路径** | `data/runs/<loop_id>/lessons/conclusions.json` | `data/experience/<bench_id>/general.json` |
| **粒度** | 一题一 loop 一份 | 一个 bench 一份（跨所有 loop） |
| **谁写** | Summarizer agent，**每轮**写 | ExperienceDistiller，**仅 CLI** `img-iter summarize` |
| **有无 LLM** | ❌ 纯规则（前后 verdict 对比） | ✅ LLM 综合（视觉模型看图归纳） |
| **内容单位** | `KnowledgeConclusion`：维度/发现/改动/状态/前后证据/lesson | `DistilledLesson`：维度/insight/dos/donts/evidence/confidence |
| **状态机** | `pending` → `verified_effective` / `ineffective` | 无（一次性归纳，带 confidence） |
| **谁消费** | Generator 工具 `query_experience` | Generator 工具 `query_general_experience` |
| **触发** | loop 自动（每轮） | 手动 CLI，**web 无入口** |
| **前端展示** | ✅ loop 详情页「经验沉淀」面板 | ❌ **无端点、无 UI** |

> 「素材载体」除这两个成品外，还有 `trajectory.jsonl`（经验的**原料**，逐轮 prompt+verdict+delta_note）、`index.json`（轮次索引）、以及考题侧的 `content_spec.json` + `target.jpg`。

### 1.1 conclusions.json 里一条结论长什么样（实跑真实内容）

```
[exp_001] dim=consistency   status=verified_effective   登记于round1 → 验证于round2
   finding : C1:生成图是木质藤编扶手椅，参考图是金属折叠行军床，产品完全不同；C2:…；C3:…
   change  : （同 finding——因为 delta_note 为空，见 §5）
   证据     : 分 0.00→1.00; 失败项 ['C1','C2','C3','C4']→[]
   lesson  : [consistency] 改动有效（分 0.00→1.00; 失败项 […]→[]），建议保持该方向
```

一条结论 = 「在维度 X 上，改了 Y → Critic 前后分数从 a 到 b → 判定有效/无效 → 一句可复用 lesson」。**判定完全由 Critic 前后对比驱动，Critic 是唯一裁判。**

### 1.2 general.json 里一条经验长什么样（实跑真实内容）

```
[artifact_defect]  confidence=0.95   evidence: s003/round1
   insight : 产品落地阴影是决定商品真实感与稳定性的关键细节。
   dos     : 在 Prompt 中明确要求 "soft ground shadows" / "casting shadows on the floor"
   donts   : 忽略底部阴影描述，导致产品呈现漂浮感
```

跨 loop 蒸馏的产物明显比循环内结论**更通用、更可执行**（有具体 dos/donts），但**只在 CLI 产出、前端看不到**。

---

## 2. 一道题、4 轮的完整运转流程

### 2.1 单轮管线（每轮都跑一遍这 5 个节点）

```
 ┌──────────┐   ┌────────────────┐   ┌─────────┐   ┌──────────────┐   ┌─────────────────┐
 │  考题     │  │ ① Generator    │   │ ② 出图  │   │ ③ Critic     │   │ ④ Summarizer    │
 │ content_  │▶ │  读经验+构造/  │▶ │ gemini  │▶ │ 对照target   │▶ │ 登记/验证经验   │
 │ spec.json │  │   改进 prompt  │   │  生图   │   │ 逐项打分      │   │ 写conclusions   │
 │ target.jpg│  └───────┬────────┘   └─────────┘   │ → verdict     │   └────────┬────────┘
 └──────┬────┘          │                          └──────────────┘            │
        │      读 conclusions.json(本题)                                       │ 写
        │      读 general.json(跨题)                                            ▼
        │                                              ┌─────────────────────────────┐
        │                                              │  lessons/conclusions.json   │
        │                                              │  ⑤ 经验产物（pending→验证） │
        │                                              └──────────────┬──────────────┘
        │                                                             │
        └───────────  下一轮 ① Generator 回头读经验  ◀─────────────────┘
```

| 节点 | 角色 | 输入 | 输出 | 经验动作 |
|---|---|---|---|---|
| **① Generator** | 构造/改进英文 prompt | 考题指令+约束+上轮失败项+**经验库** | prompt + delta_note | **消费** 经验（query_*） |
| **② 出图** | 调生图模型 | prompt + target(参考图) | 三视图 jpg | — |
| **③ Critic** | 对照 target 逐项打分 | 生成图 + target + checklist | verdict（每维分+失败项+还原度） | **产出证据** |
| **④ Summarizer** | 经验闭环验证 | 本轮 verdict + 上轮 verdict + delta_note | 更新 conclusions.json | **产出/验证** 经验 |
| **⑤ conclusions** | 经验落盘 | — | pending → effective/ineffective | **被下轮消费** |

> 「考试」= 节点 ③ Critic；「产出经验」= 节点 ④ Summarizer；「验证产出」= 下一轮的 ④（拿上轮 pending 和本轮 verdict 对比）。**首轮只登记 pending，从第 2 轮起才能验证出 effective/ineffective**——这就是「两轮才能有经验」的由来。

### 2.2 经验闭环的状态机（核心机制）

```
        round N                                   round N+1
   ┌───────────────┐                         ┌──────────────────┐
   │ Critic 抓到失败 │  ④Summarizer 登记       │ 拿上轮 verdict    │
   │ 维度 X         │ ─────────────────▶     │ 和本轮对比维度 X  │
   │ (如 A4 阴影)   │   status = pending      │                  │
   └───────────────┘                         └────────┬─────────┘
                                                      │  judge_status()
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                 失败项消除/分上升            仍失败/分下降
                                 verified_effective           ineffective
                                 (lesson: 建议保持)           (lesson: 需换思路)
```

**关键前提**：round N 必须有真实失败项，round N+1 才有东西可验证。**若 round 1 就满分 → 没有失败项可登记 → 只能造一条 `general` 噪声桩（见 §5）。**

---

## 3. 实跑节点效果（s003-live，4 轮真实数据）

### 3.1 逐轮总览

| 轮次 | prompt 来源 | 还原度 | Critic 失败项 | 经验动作 |
|---|---|---|---|---|
| **R1** | Generator agent（deepseek，英文 prompt） | **0.275** 🔴 | consistency C1-C4 全 fail；product_structure S1,S3,S4 fail | 登记 2 条 **pending**（consistency、product_structure） |
| **R2** | **兜底**（agent 网关失败→原始中文指令） | **0.988** 🟢 | 无（全通过） | 验证 R1 的 pending → 2 条 **verified_effective** ✅ |
| **R3** | Generator agent（deepseek，英文 prompt） | **0.966** | 无（material 0.85 小幅低） | 无新失败项；R2 的 general 桩判 **ineffective** |
| **R4** | **兜底**（agent 网关失败→原始中文指令） | **0.984** 🟢 | 无 | 无新验证 |

### 3.2 还原度曲线

```
还原度
 1.00 ┤                              ●R2=0.988              ●R4=0.984
      │
 0.75 ┤
      │
 0.50 ┤
      │
 0.25 ┤  ●R1=0.275
      │
 0.00 ┤──────────────────────────────────────────────────────
          R1            R2            R3            R4
       (agent英文)   (兜底中文)    (agent英文)    (兜底中文)
```

### 3.3 这次实跑产出的经验（conclusions.json，3 条）

| id | 维度 | 状态 | 前后证据 | 真伪 |
|---|---|---|---|---|
| exp_001 | **consistency** | **verified_effective** ✅ | 分 0.00→1.00；失败项 C1-C4→[] | **真经验** |
| exp_002 | **product_structure** | **verified_effective** ✅ | 分 0.25→1.00；失败项 S1,S3,S4→[] | **真经验** |
| exp_003 | general | ineffective ❌ | 分 0.00→0.00 | **噪声桩**（见 §5） |

> **结论：经验闭环这次跑通了**——R1 暴露的真实问题（生成图把「金属折叠行军床」画成了「木质藤编扶手椅」，consistency/product_structure 双双崩盘）在 R2 被修复，Summarizer 正确判定为 `verified_effective`。这正是「两轮产出经验」的理想形态。

---

## 4. 节点效果细节（按流程五阶段）

### 阶段 A：考题设置
考题 = `content_spec.json` + `target.jpg`。s003 是「折叠便携躺椅（金属+织物）」，要求生成**一张**白底三视图（正/侧/立体），6 个评分维度、共 17 条 checklist 项。其中 **A4「家具接地有阴影(不悬浮)」** 是历史经验里反复出现的关键项。

### 阶段 B：消费（Generator 读经验 → 改 prompt）
- 首轮：无本题经验可读，可读跨题 `general.json`（4 条 dos/donts，如「明确要求 soft ground shadows」）。
- 工具：`query_experience`（读 conclusions）+ `query_general_experience`（读 general）。
- ⚠️ **本次 R2/R4 的 Generator 因网关失败走了兜底路径，兜底不调这两个工具 → 这两轮没消费经验**（见 §6）。

### 阶段 C：考试（Critic 对照 target 逐项打分）
混合评分：4 个二分维度（逐项 ✓/✗）+ 2 个连续维度（0-1 分）。还原度 = 权重·特征。本次 R1 Critic 抓出了真实且严重的问题（产品都画错了），是经验闭环能跑通的前提。

### 阶段 D：产出经验（Summarizer 写 conclusions）
纯规则、无 LLM。R1 登记失败维度为 pending；R2 拿 R1 vs R2 verdict 对比 → 升级为 effective。

### 阶段 E：验证产出
就是下一轮的 D。本次 R2 验证了 R1 的 pending（→ effective）。`conclusions.json` 里每条结论带 `critic_evidence`（before/after 快照 + verdict_delta），可追溯。

---

## 5. 为什么「一直没总结出经验」？—— 三个叠加原因

### 原因 1：跨 loop 经验（general.json）前端根本不显示 🔴 最大
`general.json` **质量很好**（4 条带 dos/donts 的高置信经验），但：
- web 后端**没有任何端点**读它；
- 前端 `app.js` **没有任何代码**渲染它；
- `img-iter summarize` **只能 CLI 触发，web 无入口**。

→ 你如果跑过 `img-iter summarize` 然后去网页找，**必然找不到**。这是确凿的前端缺口。

### 原因 2：循环内经验对「简单题」产出的是噪声桩 🔴
当一轮生成**太好**（所有二分项 ✓、连续维度 ≥ 0.7）时，Summarizer 找不到失败维度，**回退**登记一条：

```
dim=general   change="general 改进"   分 0.00→0.00   ineffective   lesson:"改动无效…需换思路"
```

这条结论的 `general` 维度在 verdict 里**根本不存在**，所以 `_dim_snapshot` 恒为 0.00，**永远判 ineffective**。它对 Generator 毫无指导意义，却在「经验沉淀」面板里占位 → 你看到「有经验但都是废话」≈「没经验」。

> 这正是 s001/s002（从 R1 起就近乎满分）只产出这种桩、而 s003（R1 有真实失败）能产出真经验的原因。

### 原因 3：delta_note 全程为空 → 经验归因失真 🟡
本次 4 轮 `delta_note` **全部为空**（deepseek-v4-flash 结构化输出弱点，已知问题）。后果：

```
change 字段本应记：「本轮把 prompt 从 X 改成 Y」
实际记成了：「C1:生成图是木质藤编扶手椅…」(= Critic 的失败描述 finding)
```

于是 lesson 只能说「改动有效，建议保持该方向」——**却说不出保持哪个方向**。系统**检测到**了改进，但**归因不到**具体可复用的改动。

---

## 6. 实跑额外暴露的系统脆弱点（真实观察）

| 现象 | 实跑中的体现 | 影响 |
|---|---|---|
| **网关间歇 SSL EOF** | R2/R4 的 Generator agent 3 次重试全失败 → 退兜底 | agent 实际没起作用 |
| **兜底反而比 agent prompt 好** | 兜底（原始中文指令）R2=0.988、R4=0.984；agent（英文 prompt）R1=0.275、R3=0.966 | 讽刺：故障路径效果更好，说明 deepseek 生成的英文 prompt 质量不稳定 |
| **兜底绕过经验消费** | 兜底路径直接 `router.generate`，不调 query_experience | R2/R4 没读经验——改进是「换回中文指令」带来的，不是「读了经验」带来的 |
| **经验归因错误** | effective 结论的 change 记成 Critic finding 而非真实 prompt 改动 | lesson 不可执行（见 §5 原因 3） |
| **Summarizer 无 LLM** | lesson 全是规则模板（「改动有效，建议保持该方向」） | 循环内经验偏干瘪；丰富的归纳只在跨 loop 蒸馏层 |

> 这些不影响「经验被产出」的事实（conclusions.json 确实写了 2 条真 effective），但解释了为什么经验**质量参差、且感觉不到它在驱动迭代**。

---

## 7. 一张图收尾：经验的全生命周期

```
   考题                        单 loop（每轮）                         跨 loop（离线）
 ┌─────────┐   R1..N     ┌──────────────────────┐    img-iter     ┌────────────────────┐
 │content_ │ ──────────▶│ Generator→出图→Critic │ ───summarize──▶│ ExperienceDistiller│
 │spec+tgt │            │   →Summarizer         │    (CLI手动)   │ (LLM看图归纳)      │
 └─────────┘            └──────────┬───────────┘                └─────────┬──────────┘
                                   │ 写每轮                                │ 写一次
                                   ▼                                       ▼
                        ┌──────────────────────┐                ┌──────────────────────┐
                        │ conclusions.json     │                │ general.json         │
                        │ 单loop·规则·每轮     │                │ 跨loop·LLM·离线      │
                        │ ✅前端展示           │                │ ❌前端不展示         │
                        └──────────┬───────────┘                └──────────┬───────────┘
                                   │ query_experience 读                   │ query_general_exp 读
                                   └──────────┬───────────────────────────┘
                                              ▼
                                    下一轮 Generator 消费
```

**两个文件 = 经验的两个生命周期阶段**：conclusions 是「单题逐轮机器验证」的细粒度事实，general 是「跨题 LLM 综合」的可复用先验。前者前端看得到，后者看不到；前者依赖 Critic 抓到失败项，后者依赖手动 CLI 触发。

---

## 8. 给你的判断与建议（待确认，未动代码）

1. **「没产出经验」是误判**：两个文件都在写，conclusions 这次还产出了 2 条真 effective。问题在**展示与质量**，不在「没跑通」。
2. **最该补的是前端**：给 `general.json` 加一个端点 + 一个面板（跨 loop 经验），并让 `img-iter summarize` 能从 web 一键触发。这是你「找不到经验」的直接原因。
3. **干掉 `general` 噪声桩**：当一轮无任何失败项时，Summarizer **不应**登记 `dim=general` 的废结论（直接跳过登记即可），避免污染经验库。
4. **修 delta_note 归因**：兜底路径也应填 delta_note（至少记「兜底：用原始指令」），并把 `change` 与 `finding` 分清；或换更可靠的结构化输出模型。
5. **Generator 兜底策略**：网关抖动让 agent 频繁退兜底，且兜底效果反而更好——值得评估是否默认就用「指令直出」、把 agent 作为可选增强。

> 以上 2–5 均为建议，未改任何代码。如需我落地其中某几条，告诉我优先级。
