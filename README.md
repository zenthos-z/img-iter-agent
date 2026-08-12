# img-iter-agent

> 把 GAN 的「生成—对抗」搬到 **Agent 层面**：不训练两个神经网络，而是让两个 LLM agent
> （生成方 vs 评判方）在一道考题上反复博弈，每轮改动由 Critic 客观判定有效/无效，沉淀为
> 可复用经验，驱动下一轮生成——直到还原度收敛。

一个**自我迭代的 AI 生图 Agent**：给定一张产品实物图 + 验收标准，自动在「**生成 → Critic 对抗评判 →
经验闭环验证 → 优化提示词**」的闭环里逐轮改进，逼近还原度最高的生图配置。首个落地场景是
**家具跨境电商白底产品图**（结构 / 材质 / 颜色最易翻车、退货主因）。

## 它解决什么

| 问题 | 解法 |
|---|---|
| 生图模型在结构 / 比例 / 材质 / 颜色上反复翻车 | Critic 对照参考图逐维度评判 → 把失败项反馈给 Generator，针对性改 prompt，逐轮收敛 |
| LLM 单轮打分有系统性偏差、不可信 | 混合评分：客观维度二分 ✓/✗（可复现）+ 渐变维度连续分；权重由人工排序校准吸收偏差 |
| 经验散落为单轮快照，无法复用 | Critic 驱动的经验闭环：每轮改动 → 前后 verdict 对比 → 判定有效/无效 → 沉淀进结构化知识库 |
| 单题经验无法跨题复用 | 独立蒸馏器跨 loop 归纳通用经验，蒸馏成 deepagents 技能包，跑哪个 benchmark 就激活哪个 |

## 核心原理

两个 agent 在「Critic 验证的经验知识库」上博弈与自我改进——Critic 是改动有效性的**客观裁判**，
其前后评分对比是经验沉淀的唯一证据。

```mermaid
flowchart LR
  G["<b>Generator</b><br/>改进 prompt → 出图"] -->|"生成图"| C["<b>Critic</b><br/>对照 target 打分"]
  C -->|"verdict<br/>分数 + 失败项 + 理由<br/>＋ 兼任经验总结"| S["经验闭环<br/>前后 verdict 对比"]
  S -->|"改动有效 → 保留约束<br/>改动无效 → 换思路"| KB[("经验知识库<br/>Critic 机器验证")]
  KB -->|"注入下一轮"| G
```

> [!NOTE]
> 还原度不是 LLM 直接给一个数，而是 **restoration = w · features**：每个维度产出 `∈[0,1]` 的特征值
> （二分维度 = 通过率，连续维度 = LLM 分），加权求和。权重 `w` 由 benchmark 先验给出，再由人工排序校准更新。

## 架构

### 1. 自我迭代闭环（LangGraph）

一题一 loop：同一道考题只有一条 loop，再次运行会在原 loop 上**续跑**（轮数叠加）。每轮经过三个节点，
在 `human_review` 处 `interrupt()` 等人工裁决（continue / stop / 调方向），不自动收敛。

```mermaid
flowchart LR
  START([start]) --> generator
  generator["<b>generator_node</b><br/>Generator deepagent<br/>读经验 → 改 prompt → 出图"]
  generator --> critic["<b>critic_node</b><br/>Critic deepagent<br/>对照 target 多维打分<br/>＋ 兼任经验总结（前后 verdict 对比）"]
  critic --> review{{"<b>human_review</b><br/>interrupt()"}}
  review -- "continue / 调方向" --> generator
  review -- stop --> END([end])
```

`SqliteSaver` 持久化每轮状态，可断点续跑；整个 loop 跑在 `run_loop_session`（`@traceable`）下，
在 LangSmith 里是**一条 trace**（多轮 invoke 按 `loop_id` 聚合）。

> [!IMPORTANT]
> loop 里只有 **两个 agent**：Generator 与 Critic。经验总结（原独立 Summarizer 节点的职责）已并入
> `critic_node`——Critic 内部用规则驱动的 `Summarizer` 类做前后 verdict 对比，判 effective / ineffective，
> 写 `conclusions.json`。Critic 既是评判方，也是经验验证方。

### 2. Agent 即引擎（deepagents）

Generator / Critic / ExperienceDistiller 不是「单次 LLM 调用 + 手抠 JSON」，而是用官方
[`deepagents`](https://github.com/langchain-ai/deepagents)（`create_deep_agent`）构建的 **tool-using agent**，
嵌入外层 LangGraph 节点当「引擎」：每轮内部跑完一个「模型想 → 调工具 → 看结果 → 再想」的 ReAct 循环，
最后用 `response_format` 结构化输出交付。**新增策略 / 能力 = 新增工具，不改外层图。**

```mermaid
flowchart TD
  subgraph node["外层 LangGraph 节点 · 每轮调用一次"]
    direction TB
    LLM["ChatOpenAI<br/>dmxapi · OpenAI 兼容端点 · 支持 tool-calling"]
    LLM --> dec{"还要调工具?"}
    dec -- 是 --> tools["工具（闭包捕获本轮上下文）<br/>generate_image / query_experience / query_general_experience …"]
    tools --> LLM
    dec -- 否 --> out["结构化输出 response_format<br/>GeneratorOutput / CriticAgentOutput / RenovationPlan"]
  end
  skill["experience/&lt;bench&gt;/&lt;slug&gt;/SKILL.md<br/>蒸馏技能 · per-benchmark 激活"] -.-> LLM
```

三个 agent 的配置（工具 / 模型 / 技能 / 输出）：

| Agent | 职责 | 模型（`IMG_ITER_*`） | 工具 | 结构化输出 | Skill |
|---|---|---|---|---|---|
| **Generator** | 构造/改进 prompt → 出图 | `generator_model`（文本即可） | `generate_image`、`query_experience`、`query_general_experience` | `GeneratorOutput{prompt, delta_note}` | **per-benchmark 蒸馏技能**（见下节）；未蒸馏则裸跑 |
| **Critic** | 对照 target 多维打分 + 兼任经验总结 | `critic_model`（**必须多模态**） | `query_rubric`（按需查判定标准） | `CriticAgentOutput{dimensions}` | 无（靠 rubric / checklist 判分） |
| **ExperienceDistiller** | 跨 loop 蒸馏通用经验 → 技能包 | `summarizer_model`（多模态） | 两阶段：①蒸馏 agent（无工具）→ ②skill-author agent（全工具，受权限沙箱约束） | `RenovationPlan` | 内置 `skill-author`（meta-skill） |

> [!IMPORTANT]
> Critic 只输出**每维度的原始评分**，不算还原度——权重不在 agent 手上。代码侧用 `recompute_restoration`
> 把评分映射成 `DimensionScore` 再加权。这样「权重变更不影响打分」，闭环 B 的校准才能安全回灌。

### 3. 技能 = 蒸馏经验（per-benchmark 激活）

这是本系统与「静态技能」最大的区别：**Generator / Critic 本身不带静态技能**。
技能是蒸馏器跨 loop 归纳出的、可移植的 deepagents 技能包，**跑哪个 benchmark 就激活哪个**——
没蒸馏过的 benchmark，Generator 裸跑（无技能）。系统提示词因此**固定**：只定义 agent 的身份 / 能力 / 流程，
与 benchmark 类目无关；考题相关的指令 / 约束 / 失败项在每轮 **user message 里动态注入**。

```mermaid
flowchart LR
  D["ExperienceDistiller<br/>(离线 · 跨 loop)"] -->|"distill() 内调 author_skill()"| PKG["experience/&lt;bench&gt;/&lt;slug&gt;/<br/>SKILL.md + references/lessons.md + assets/"]
  PKG -.->|"SkillsMiddleware 自动发现<br/>(progressive disclosure)"| G["Generator deepagent<br/>跑 &lt;bench&gt; 时激活该技能"]
  NK["未蒸馏的 bench"] -.->|"generator 裸跑"| G
```

- 落地结构：`experience/<bench>/<slug>/SKILL.md`，由 `generator_skills_source(data_root, bench_id)` 解析，
  兼容 deepagents `SkillsMiddleware`（扫 source 下子目录读 `SKILL.md`）。
- Generator 另有两层**经验注入**与技能叙事互补：system prompt 常驻精简经验索引（恒定 ~6 行）+ 每轮
  user message 按上下文 `select_lessons` 注入 ≤4 条选定详情 + `query_general_experience` 工具钻取。
- Critic **不加载任何技能**——它对照 benchmark 的 rubric / checklist 客观判分。
- 所有 agent 的系统提示词外部化到 `data/agents_config/<agent>.md`，可在 Web 台「Agent 设置」页在线编辑。

### 4. 经验闭环：两层知识

经验不是单轮事实快照，而是分两层沉淀：

```mermaid
flowchart LR
  C["Critic 兼任的经验总结<br/>（规则驱动 · 每轮）"] -->|"Critic 前后对比<br/>判 effective / ineffective"| CJ[("conclusions.json<br/>单 loop · 机器验证")]
  ED["ExperienceDistiller<br/>（离线 deepagent）"] -->|"跨 run 综合 + 蒸馏技能包"| GJ[("general.json<br/>跨 loop · 通用经验")]
  CJ --> ED
  CJ -->|"query_experience<br/>本题已验证经验"| G["Generator"]
  GJ -->|"query_general_experience<br/>跨题先验（首题也用得上）"| G
```

- **`conclusions.json`**（每 run）：Critic 每轮写。Critic 前后 verdict 对比**机器验证**的逐条结论
  （按 `(dim, change)` 的 effective / ineffective + `critic_evidence`）。
- **`general.json`** + 技能包（按 bench 共享）：独立蒸馏器写。跨 run、LLM **综合**的上层通用经验，
  每条带 `dim / insight / dos / donts / evidence / confidence`，并渲染成 deepagents 技能包被 Generator 激活。

蒸馏器以「已验证 effective/ineffective」为锚（比单看分数可靠），跨 run 找反复出现的有效/无效做法。

### 5. 评分校准（闭环 B）

另一个独立闭环：让 Critic 的还原度贴合人的判断。人只做**排序**（listwise，人擅长的），
`learning-to-rank` 拟合维度权重，天然修正 LLM 连续打分的系统性偏差。

```mermaid
flowchart LR
  feat["各 trace 的维度特征<br/>features ∈ [0,1]"] --> rank["人工 listwise 排序"]
  prior["先验权重<br/>weight_init"] --> fit["learning-to-rank<br/>SLSQP · Σw=1 · w≥0"]
  rank --> fit
  fit --> w["校准后权重 weights"]
  w -->|"回灌 Critic"| r["restoration = w · features"]
```

| | 闭环 A：生成迭代 | 闭环 B：评分校准 |
|---|---|---|
| **目的** | 提升生成还原度 | 让 Critic 评判贴合人的判断 |
| **驱动** | Critic 多维分 + 人工审批 | 人工排序 vs Critic 初评 |
| **产出** | 更好的图 + 验证过的经验 | 校准后的维度权重 |

### 评分维度（首个 benchmark）

6 个维度，混合评分（二分 ✓/✗ → 通过率；连续 → LLM 0-1 分）：

| 维度 | 类型 | 先验权重 | 评什么 |
|---|---|---:|---|
| `consistency` | 二分 | 0.25 | 三视图跨张一致性（同产品 / 同色 / 几何比例一致） |
| `product_structure` | 二分 | 0.22 | 部件数 / 位置 / 形态正确，无缺失 / 重复 / 穿模 |
| `material_texture` | 连续 | 0.18 | 材质类型与还原程度（对照参考） |
| `color_accuracy` | 连续 | 0.13 | 与产品实物的色差程度 |
| `artifact_defect` | 二分 | 0.12 | 无变形 / 失真 / 模糊 / 伪影 / 悬浮 |
| `commercial_focus` | 二分 | 0.10 | 主体突出 / 白底干净 / 构图合规 |

## 快速开始

### 前置

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip + venv
- [dmxapi](https://www.dmxapi.cn) 密钥（生图与评判的统一后端）

### 安装

```bash
git clone <repo> && cd img-iter-agent
uv sync                       # 或: python -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env          # 填入 IMG_ITER_DMXAPI_KEY 与各 model_id
```

### 跑一个迭代闭环（CLI）

```bash
# 对 s001（双人床）跑闭环 A：生成→评分→经验验证→人工审批，逐轮 continue/stop
python -m img_iter_agent.cli run --bench furniture_product_whitebg --sample s001
```

### 蒸馏跨 loop 通用经验 → 技能包

跑过多个 sample 的 loop 后，用独立蒸馏器跨 loop 归纳通用经验，并把它**装配成 deepagents 技能包**：

```bash
# 读该 bench 下所有 run 的 trajectory + 已验证结论 → 蒸馏 → general.json + experience/<bench>/<slug>/SKILL.md
python -m img_iter_agent.cli distill --bench furniture_product_whitebg
```

产物分两份：`general.json`（结构化经验）被 Generator 的 `query_general_experience` 工具读回；
`experience/<bench>/<slug>/`（技能包）被 SkillsMiddleware 发现，下次跑该 bench 的 loop 时
Generator **自动激活**。首题也能用得上跨题先验。

> [!TIP]
> 想让某个 benchmark 的 Generator 有技能？先对它跑一次 `distill`。没蒸馏过的 bench，Generator 裸跑（不报错）。

### 启动可视化打分台（Web）

```bash
img-iter-web          # 或: python -m uvicorn img_iter_agent.web.app:app --port 8765
```

打分台 5 屏：

| 屏 | 能力 |
|---|---|
| **总览** | 每个 benchmark 的 sample 卡片 + loop 状态 / 还原度徽标；一键起新 loop |
| **loop 详情** | 迭代轨迹时间线（每轮大图对照 target + Critic 明细 + prompt diff）+ 经验沉淀面板；运行中显示当前节点 / 待审批给 continue/stop |
| **经验管理**（per-bench） | 触发蒸馏、查看蒸馏状态、导出技能包 zip（`skill.zip`）、对单条 lesson 标记 refute |
| **人工排序** | 拖拽给 trace 打分 → 自动 learning-to-rank 校准维度权重，并显示每维权重变化进度条 |
| **Agent 设置** | 在线改 generator / critic 的系统提示词 + 模型 id（benchmark 下拉切换看 generator 的 per-bench 蒸馏技能）；distiller 栏只读展示 |

其它 CLI：`calibrate`（用人工排序拟合权重）、`analyze`（跨 run 汇总还原度 + 画图）。

## 配置

所有配置走环境变量（前缀 `IMG_ITER_`），见 `.env.example`：

- **dmxapi 凭证与地址**（`IMG_ITER_DMXAPI_HOST` / `IMG_ITER_DMXAPI_KEY`）
- **四个协议族的生图 model_id** —— dmxapi 聚合了四族差异，Router 按任务模式路由：

  | 族 | 协议 | 模型 | 用途 |
  |---|---|---|---|
  | A | OpenAI Images | `gpt-image-2` | 纯文生图 / 图生图（multipart 参考图） |
  | B | 豆包 Responses | `seedream-5.0-pro` | 参考图风格迁移 / 多图融合（默认优先） |
  | C | Qwen Responses | `qwen-image-2.0-pro` | 纯文生图 / 图生图（强文字渲染） |
  | D | Gemini native | `gemini-3.1-flash-image` | 多轮改图（唯一支持） |

- **三个 agent 的推理 model_id**（`generator_model` / `critic_model` / `summarizer_model`），均走 dmxapi
  的 OpenAI 兼容端点（`/v1/chat/completions`，需支持 tool-calling；Critic 必须多模态）。distiller 复用
  `summarizer_model`（蒸馏要审图，必须多模态）。
- **LangSmith 追踪**（`LANGSMITH_*`，agent 运转可视化）。

## 数据布局

```
data/
├── benchmarks/                     # 〔你准备的考题 · ✅入 git · 跨 run 复用〕
│   └── furniture_product_whitebg/
│       ├── manifest.json           # 6 维评分定义 + 权重先验
│       ├── rubric.md               # 各维度判定细则
│       └── samples/sNNN/           # target.jpg 参考锚 + content_spec.json（约束 + checklist）
├── runs/<bench>-<sample>/          # 〔系统产出 · ❌不入 git〕 一题一 loop
│   ├── trajectory.jsonl            # 完整轨迹（每轮含 prompt / verdict / delta_note）
│   ├── lessons/conclusions.json    # 单 loop 经验（Critic 机器验证）
│   ├── out/aNNN/                   # 每轮生成图
│   └── calibrated_weights.json     # 校准后权重
├── experience/<bench>/             # 〔跨 loop 通用经验 · ❌不入 git〕
│   ├── general.json                # 蒸馏出的 dos/donts（多 run 综合）
│   └── <slug>/SKILL.md             # 蒸馏技能包（Generator per-bench 激活）
└── analyses/                       # 〔离线分析 · 只读〕
```

> [!NOTE]
> 你唯一需要手动管理素材的地方是 `data/benchmarks/<bench>/samples/`——放产品实物图
> `target.jpg` + 写 `content_spec.json`（要生成什么、约束、各维度 checklist）。系统跑出来的产物自动进 `data/runs/`。

## 测试

```bash
python -m pytest          # 163 个测试（deepagent 路径、经验闭环、跨 loop 蒸馏、混合评分、权重校准）
ruff check src/ tests/    # 代码风格
```

基础层测试无需任何密钥——离线、用 `FakeToolCallingChatModel` 驱动 agent 路径。

## 项目结构

```
src/img_iter_agent/
├── agents/
│   ├── generator.py             # 生成方 deepagent：改 prompt + 经验注入（产出 delta_note）
│   ├── critic.py                # 评判方 deepagent：多模态混合评分 + 兼任经验总结（summarize_round）
│   ├── summarizer.py            # 规则驱动的前后 verdict 对比（被 critic 内嵌为工具，非独立节点）
│   ├── experience_distiller.py  # 独立蒸馏方 deepagent：跨 loop 蒸馏 → 通用经验 + 装配技能包
│   ├── skill_authoring/         # skill-author meta-skill（供 distiller 的技能包编写 agent 加载）
│   ├── _agent_output.py         # deepagent 的结构化输出 schema
│   └── tools/                   # 工具注册中心（= 策略/能力扩展点）
├── pipeline/
│   ├── graph.py                 # LangGraph 闭环：generator→critic→human_review(interrupt)
│   ├── runner.py                # build_loop_context：构造 agent + checkpointer + 标准 config
│   └── state.py                 # 跨轮 State
├── llm/chat_model.py            # build_chat_model → ChatOpenAI（指向 dmxapi，支持 tool-calling）
├── generation/
│   ├── router.py                # 按「任务模式 + model_hint」路由到协议族 dispatcher
│   └── protocols/               # 屏蔽四族差异：A(OpenAI)/B(豆包)/C(Qwen)/D(Gemini)
├── memory/
│   ├── knowledge.py             # 单 loop 经验库（conclusions.json + Critic 驱动 status 判定）
│   ├── experience.py            # 跨 loop 经验库（general.json 读写 + 技能包装配 + generator_skills_source）
│   └── schema.py                # Pydantic v2 数据契约
├── calibration/                 # 闭环 B：排序拟合权重（learning-to-rank, SLSQP）
├── data/                        # 三层数据管理 + trajectory.jsonl（可独立加载重放）
└── web/                         # FastAPI 打分台（前后端解耦，Vanilla JS 前端）
```

**关键设计**：dmxapi 聚合 API（全云端，屏蔽 4 协议族）· LangGraph 编排（闭环原生 + 断点续跑）·
deepagents 引擎（tool-using agent + 结构化输出）· **技能 = 蒸馏经验（per-benchmark 激活，非静态）** ·
系统提示词固定（考题走 user message）· 混合评分（二分可复现 + 连续保渐变）·
三层数据分离（benchmarks / runs / experience）· 图片全程用文件路径（不用 base64）。

## 文档

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — 完整架构与 10 条 ADR（关键决策记录）
- **[docs/EVALUATION.md](docs/EVALUATION.md)** — 混合评分与排序校准的方法论
- **[docs/EXPERIENCE_FLOW.md](docs/EXPERIENCE_FLOW.md)** — 两层经验闭环与蒸馏技能包的流转
