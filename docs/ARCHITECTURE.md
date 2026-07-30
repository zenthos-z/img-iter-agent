# img-iter-agent 架构与技术栈分析

> 目标：建立「AI 生图自动迭代 → 风格元素相同、内容可变」的**自我迭代对抗总结** Agent 系统。
> 本文档是当前阶段的核心交付，确定系统形态、闭环机制，并对候选技术栈做选型分析与推荐。

---

## 0. 先把目标拆清楚

用户的原始描述里其实埋了三个互相约束的子目标，必须先分开，否则技术栈会打架：

| # | 子目标 | 含义 | 技术上对应什么 |
|---|--------|------|----------------|
| **G1** | **风格元素相同** | 配色、笔触、材质、构图语言、氛围在多张图之间稳定 | 不是靠 prompt 文字，而是靠**图像条件**锚定（IP-Adapter / ControlNet / LoRA） |
| **G2** | **内容可变** | 主体、场景、主题在风格约束下自由变化 | 由 prompt 的「内容维度」驱动，并与风格条件解耦 |
| **G3** | **自动迭代 + 自我对抗总结** | 生成质量随轮次提升，系统自己归纳经验并改进 | Agent 闭环：生成 → 评判 → 总结 → 改进 |

关键洞察：**G1（风格稳定）本质不是"prompt 工程"能解决的，它是图像条件控制问题**；
而 G3（自我迭代）是 Agent 编排问题。两者必须用不同技术栈，文档第 4 节会把它们拼起来。

---

## 1. 核心机制：把 GAN 思想搬到 Agent 层

用户说"生成对抗总结自我迭代"，这其实是一个非常具体的范式——
**不要训练两个神经网络，而是用两个 LLM agent 在「经验库」上博弈**：

```
                  ┌─────────────────────────────────────────────────┐
                  │            风格条件（锚定 G1，见 §4.2）            │
                  │  参考图 → IP-Adapter/LoRA/ControlNet 风格嵌入      │
                  └───────────────────────┬─────────────────────────┘
                                          │ 稳定不变
                                          ▼
   ┌──────────┐   生成图    ┌──────────┐     评分+批评     ┌──────────┐
   │ Generator│ ──────────▶ │  Image   │ ───────────────▶ │  Critic  │
   │  Agent   │             │  Pool    │                  │  Agent   │
   │（改prompt│ ◀────────── │（本轮产物）│ ◀─── 取最好/最差 ───┤（多模态 │
   │+内容采样)│   经验反馈   └──────────┘                  │ 打分)    │
   └────┬─────┘             └─────▲────┘                  └────┬─────┘
        │                         │                            │
        │   prompt 修改建议         │                            │ 打分维度+理由
        ▼                         │ 写入                        ▼
   ┌──────────┐                   │                   ┌──────────────┐
   │ Summarizer│                  │                   │ Memory Store │
   │  Agent    │──── 经验条目 ────▶│                   │（向量库+结构化）│
   │（归纳跨轮 │                   └───────────────────│ 跨轮累积      │
   │ 规律）    │                                       └──────────────┘
   └──────────┘                                             ▲
        │                                                   │ 检索相关经验
        └───────────────────────────────────────────────────┘
                       每轮开始时注入 Generator/Critic
```

三个 agent 的职责切分（对应"生成对抗总结"四个字）：

- **Generator（生成方）**：负责 **G2 内容可变**。从内容空间采样新主题，结合 Memory 检索到的
  经验，产出 prompt 并调用生图模型。
- **Critic（对抗评判方）**：多模态 LLM 看图打分。**关键设计：评判时同时给"风格一致性"和
  "内容质量"两个分**——前者强约束 G1，后者驱动 G2/G3。这一"对抗"压力倒逼 Generator 改进。
- **Summarizer（总结方）**：跨轮归纳。把零散的"Critic 评分理由"抽象成可复用的经验条目
  （"用 XX 笔触描述时风格漂移，应改用 YY"），写回 Memory，让系统真正"越跑越好"。

> 这就是"自我对抗总结自我迭代"：对抗(Critic) → 总结(Summarizer) → 自我迭代(下一轮 Generator 读 Memory)。

---

## 2. 迭代闭环（一次 epoch）

```
for round t in 1..T:
    1. [内容采样]   Generator 从主题空间采一批新内容（保证"内容可变"）
    2. [经验检索]   从 Memory 检索与当前内容/历史失败相关的经验条目
    3. [生成]       prompt = f(内容, 经验, 固定风格描述)
                    images  = 生图模型(prompt, 风格条件嵌入)   # 风格锚定见 §4.2
    4. [评判]       Critic 对每张图打 style_score / content_score / 总分，给批评
    5. [筛选]       选 top-k 进 Memory 的"标杆池"；选 worst-k 做"反面教材"
    6. [总结]       Summarizer 看本轮 + 历史，提炼新经验条目入 Memory
    7. [收敛判定]   若平均 style_score ≥ 阈值 且 内容多样性达标 → 停；否则下一轮
```

**收敛指标（自动判定停止 / 监控）**：

- `style_consistency`：同批图两两的风格特征距离（CLIP/特征向量 cosine），越高越好；
- `content_diversity`：同批图的内容主题熵 / 聚类数，保证"内容可变"；
- `quality_trend`：`content_score` 的移动平均，看是否还在涨。

---

## 3. 三个核心难点与技术对应

### 3.1 难点 A：风格怎么稳定（G1 的命根子）

这是最容易踩坑的地方。**纯靠 prompt 描述风格是守不住的**——不同内容下，"水彩风"
四个字会漂移。正确做法是给生图模型**图像级条件**，让风格脱离文字、直接以视觉特征注入：

| 手段 | 锚定强度 | 灵活性 | 成本 | 何时用 |
|------|---------|--------|------|--------|
| **IP-Adapter**（参考图嵌入） | ★★★ | 高（内容自由） | 低 | **首选**：1~N 张参考图，即插即用，不训练 |
| **Reference-only ControlNet** | ★★★ | 中 | 低 | 参考图风格强迁移 |
| **LoRA**（风格微调） | ★★★★ | 低 | 中（需训练） | 风格已完全锁定、可复现性要求高 |
| **纯 prompt（style tokens）** | ★ | 高 | 极低 | 仅辅助，不能单独成事 |

**推荐组合：IP-Adapter 做主锚定 + prompt style 描述做微调 + （可选）LoRA 固化最终风格**。
详见 §4.2、ADR-002。

### 3.2 难点 B：内容怎么"可控地变"（G2）

内容采样不能随机乱来，否则 Critic 没法稳定打分。建议给内容建模一个**主题空间**：

- 显式 schema：`{主体, 动作, 场景, 视角, 光照}`，每维有受控取值池；
- Generator 每轮从池里做"受控变异"（改一两个维度，其余固定）——这样跨图既有多样性，
  又能归因"是哪个维度的变化导致了风格漂移"。

### 3.3 难点 C：Memory 怎么不变成垃圾（自我迭代的关键）

自我迭代成败在 Memory。经验条目必须有**结构 + 检索 + 衰减/去重**，否则几轮后全是噪声：

- **结构化条目**：`{触发条件, 经验内容, 适用维度, 置信度, 来源轮次, 验证次数}`；
- **向量检索**：Generator/Critic 工作前，按当前上下文检索 top-k 相关条目注入 prompt；
- **强化/衰减**：被 Critic 多次验证的经验提置信度；长期没用、与高分相悖的衰减或剔除。

---

## 4. 技术栈选型

### 4.1 选型总表

| 层 | 选项 | 推荐 | 理由 |
|----|------|------|------|
| **语言** | Python / TS | **Python ≥3.11** | 生图、agent、向量库生态全在 Py |
| **生图（云端 API）** | Gemini(google-genai) / gpt-image / DALL·E / fal.ai / Replicate | **Gemini 图像 + fal.ai 备选** | Gemini 多模态强、编辑能力强；fal 模型多、支持 IP-Adapter |
| **生图（本地可控）** | diffusers(SDXL/Flux) + IP-Adapter + ControlNet + LoRA | **本地 diffusers**（可控性要求高时） | 风格锚定(G1)需要 IP-Adapter/ControlNet，云端 API 普遍不暴露这些 |
| **多模态评判(Critic)** | Gemini 2.x / GPT-4o / Claude | **Gemini 2.x 多模态** | 评分+批评质量高、价格好；可并行多 judge 投票 |
| **Agent 编排** | LangGraph / 手写状态机 / LangChain | **LangGraph** | 环路(闭环)是一等公民，状态/检查点/可视化原生支持 |
| **LLM 接口统一** | 官方 SDK / LiteLLM | **LiteLLM** | 一套接口切多 provider，评判/生成可换模型 |
| **结构化输出** | Pydantic + 模型原生 JSON | **Pydantic v2 + instructor** | Critic 评分、经验条目需严格 schema |
| **记忆向量库** | ChromaDB(本地) / Qdrant(本地+服务) | **ChromaDB 起步** | 零服务依赖、纯本地；量大再升 Qdrant |
| **结构化经验存储** | SQLite / JSON Lines | **SQLite (SQLModel)** | 经验条目要查改、做衰减统计 |
| **图像/特征** | Pillow + OpenCLIP / transformers | **Pillow + OpenCLIP** | 算 style_consistency 指标用 CLIP 特征 |
| **配置/密钥** | pydantic-settings + .env | **pydantic-settings** | 类型安全、12-factor |
| **异步/并发** | asyncio + httpx | **asyncio** | 批量生图、批量评判是 IO 密集 |
| **重试/限流** | tenacity | **tenacity** | API 调用必须容错 |
| **可观测** | 结构化日志 + run 记录 → 可选 Langfuse | **logging + JSONL run logs** | 先做最小可观测，不引入外部依赖 |
| **实验追踪(可选)** | MLflow / W&B / 纯 JSONL | **JSONL run logs 起步** | 自迭代本身就有完整 run 记录，先不引重型工具 |

### 4.2 风格锚定方案（G1 决定项）—— 关键决策

生图后端的选择直接决定能不能做 G1。分两条路：

**路线 1（推荐起步）：云端 API 生图**
- 优点：零运维、快上手、Gemini/fal 的图像编辑能力强；
- 缺点：**多数云端 API 不暴露 IP-Adapter/ControlNet**，风格稳定主要靠"参考图作为编辑输入"
  （如 Gemini 的 image editing、fal 的 image-to-image）或 prompt 强约束。可控性 < 本地。
- 适用：快速验证闭环、内容多样性优先、对风格 100% 复现要求不高。

**路线 2（可控性优先）：本地 diffusers**
- 完全掌控 IP-Adapter + Reference ControlNet + LoRA，G1 可做到工业级稳定；
- 缺点：要 GPU、要装 torch/diffusers、磁盘大。
- 适用：风格锚定是第一诉求、可接受本地 GPU 成本。

**建议**：MVP 先走**路线 1（Gemini API 生图 + 参考图编辑）** 跑通整个闭环和 agent 逻辑；
G1 精度不够时，把"生图"这层换成路线 2 的 diffusers 后端——架构上生图被抽象成
`ImageGenerator` 接口（见 §5），换后端不动 agent 逻辑。详见 ADR-001。

### 4.3 评判器（Critic）方案 —— 对抗压力的来源

- **多模态打分**：Gemini 2.x 看图，按 §2 的三个维度评分 + 给自然语言批评；
- **多 judge 投票（可选）**：同一张图让 Gemini + GPT-4o 各打一次，取均值，降低单一模型偏见；
- **Prompt 评判模板**：固定评分 rubric（0-10 各维度 + 扣分项 + 改进建议），保证跨轮可比。

---

## 5. 模块边界（为后续实现定接口）

```
src/img_iter_agent/
├── agents/
│   ├── generator.py      # Generator: 内容采样 + prompt 构造 + 调生图
│   ├── critic.py         # Critic: 多模态打分 + 批评
│   └── summarizer.py     # Summarizer: 跨轮经验归纳
├── pipeline/
│   ├── graph.py          # LangGraph 闭环定义（节点=agents, 边=数据流）
│   └── state.py          # 跨轮 State（TypedDict）：images/scores/memory/round
├── generation/
│   ├── base.py           # ImageGenerator 抽象接口（云端/本地可换）
│   ├── gemini_backend.py # 路线1
│   └── diffusion_backend.py # 路线2: diffusers+IP-Adapter+ControlNet
├── style/
│   ├── anchor.py         # 风格条件封装（参考图→嵌入/LoRA 句柄）
│   └── metrics.py        # CLIP 风格一致性 / 内容多样性指标
├── memory/
│   ├── store.py          # 经验条目 CRUD + 检索 + 衰减
│   ├── vector.py         # ChromaDB 向量检索
│   └── schema.py         # 经验条目 Pydantic model
├── config.py             # pydantic-settings
└── cli.py                # 入口
```

**关键解耦**：`ImageGenerator` 是接口，agent 逻辑不绑死任何生图后端 →
MVP 用 Gemini，成熟后换 diffusers，agent 代码一行不改。

---

## 6. 风险与待决问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| 云端 API 不支持 IP-Adapter → G1 守不住 | 高 | 抽象 `ImageGenerator` 接口；预留 diffusers 后端 |
| Critic 自我欺骗（同模型自评自生成） | 中 | 多 judge 投票；Critic 与 Generator 用不同 provider |
| Memory 膨胀成噪声 | 中 | 衰减/去重/置信度机制（§3.3） |
| 成本失控（多轮 × 多图 × 多模态评判） | 中 | 每轮 batch 小化、judge 数量可配、加预算上限熔断 |
| 风格定义本身模糊（用户给不清"风格"） | 中 | 支持「参考图」作为风格定义的优先输入 |

---

## 7. 待用户拍板的开放问题（影响最终技术栈）

在写第一行实现代码前，建议先对齐以下决策（见 ADR 草案）：

1. **生图后端**：先云端 API（快）还是先本地 diffusers（可控）？默认建议云端起步。
2. **运行环境**：有没有本地 GPU？决定能否走路线 2。
3. **风格输入形式**：用户提供「参考图」还是「文字描述」？默认支持参考图优先。
4. **预算/规模**：单次迭代大致预算上限？决定 batch 与 judge 数量。

---

## 8. ADR（架构决策记录，草案）

### ADR-001: 生图后端用云端 API 起步，本地 diffusers 作为可替换后端
- **状态**：提案
- **背景**：G1(风格稳定) 倾向本地可控(IP-Adapter)，但 MVP 要快、要低运维。
- **决策**：抽象 `ImageGenerator` 接口；MVP 用 Gemini API（参考图编辑做风格迁移）；
  精度不足时加 diffusers 后端，agent 逻辑不变。
- **后果**：+快速验证闭环；−云 API 的风格可控性弱于本地。

### ADR-002: 风格锚定优先用 IP-Adapter，prompt 仅辅助
- **状态**：提案
- **背景**：纯 prompt 守不住跨内容风格一致性。
- **决策**：风格以图像条件(IP-Adapter/参考图/LoRA)为主锚定，prompt 的 style 段只做微调。
- **后果**：+风格稳定可复现；−需要参考图或训练 LoRA，门槛略高。

### ADR-003: 用 LangGraph 编排闭环，State 跨轮持久化
- **状态**：提案
- **决策**：闭环用 LangGraph 的 cyclic graph；State(TypedDict) 含 images/scores/memory/round，
  每轮 checkpoint，支持中断恢复。
- **后果**：+原生支持环、可视化、断点续跑；−引入 langgraph 依赖。

### ADR-004: Memory = 向量库(检索) + SQLite(结构化经验)
- **状态**：提案
- **决策**：经验条目存 SQLite(SQLModel)，向量化进 ChromaDB 做语义检索；带置信度/衰减。
- **后果**：+经验可查可改可衰减；−两套存储要保证一致（写入双写）。

### ADR-005: Critic 多 judge 投票，且与 Generator 异 provider
- **状态**：提案
- **决策**：Critic 默认 Gemini 多模态；重要场景叠加 GPT-4o 做第二 judge；Generator 生图
  尽量用不同 provider，降低自评偏差。
- **后果**：+评判更可信；−成本与延迟上升。
