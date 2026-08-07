# img-iter-agent

> 一个自我迭代的 AI 生图 Agent：通过「**生成 → Critic 对抗评判 → 经验闭环验证 → 优化提示词**」的闭环，
> 在一道考题上持续迭代，自动收敛到还原度最高的生图配置。

核心思想是把 GAN 的「生成—对抗」搬到 **Agent 层面**：不训练两个神经网络，而是让 LLM agent
（生成方 vs 评判方）在「**Critic 驱动验证的经验知识库**」上博弈与自我改进——每轮改动都由 Critic 客观
判定有效/无效，沉淀为可复用经验，指导下一轮生成。

## 它解决什么问题

给定一道考题（一张产品实物图 + 验收标准），系统自动迭代生成图：

- **还原度持续提升**——材质、结构、颜色、比例逐轮逼近目标参考图；
- **经验闭环验证**——不是记录事实，而是「上轮改了 X → Critic 前后评分对比 → 判定有效/无效 → 沉淀」；
- **人工排序校准**——人只做擅长的排序，自动拟合维度权重，让 AI 评判越来越贴合人的判断。

首个落地场景：**家具跨境电商白底产品图**（结构/材质/颜色易翻车、退货主因）。

## 快速开始

### 前置要求

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip + venv
- dmxapi 密钥（生图与评判的统一后端）

### 安装

```bash
git clone <repo> && cd img-iter-agent
uv sync            # 或: python -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # 填入 IMG_ITER_DMXAPI_KEY 与各 model_id
```

### 跑一个迭代闭环（CLI）

```bash
# 对 s001（双人床）跑闭环 A：生成→评分→经验验证→人工审批，逐轮 continue/stop
python -m img_iter_agent.cli run --bench furniture_product_whitebg --sample s001
```

一题一 loop：同一道题只有一条 loop，再次 `run` 会**在原有 loop 上续跑**（轮数叠加），不会新建。

### 蒸馏跨 loop 通用经验（CLI）

跑过多个 sample 的 loop 后，用独立「经验蒸馏器」跨 loop 归纳通用经验（dos/donts）：

```bash
# 读该 bench 下所有 run 的 trajectory + 已验证结论 → 蒸馏通用经验 → data/experience/<bench>/general.json
python -m img_iter_agent.cli summarize --bench furniture_product_whitebg
```

蒸馏器读每 run 的 `conclusions.json`（in-loop Summarizer **机器验证**过的 effective/ineffective）+ trajectory，
跨 run 综合。产物 `general.json` 会被 Generator 的 `query_general_experience` 工具读回，作为**跨题先验**
注入下一轮生成（首题尤其有用——本题还没经验时也能借鉴别题）。

### 启动可视化打分台（Web）

```bash
img-iter-web       # 或: python -m uvicorn img_iter_agent.web.app:app --port 8765
# 浏览器打开 http://localhost:8765
```

打分台提供 5 屏：总览（每个 sample 的 loop 状态）、loop 详情（迭代轨迹+经验沉淀）、trace 详情
（大图对照 target+Critic 明细）、人工排序（拖拽打分→自动校准权重）、Agent 设置（改系统提示词+模型 id）。

## 核心机制

### Generator / Critic 是真正的 tool-using agent

Generator 和 Critic 不是「单次 LLM 调用 + 手抠 JSON」的封装，而是用官方 `deepagents`
（`create_deep_agent`）构建的会调工具的 agent，嵌入外层 LangGraph 的节点里当「引擎」：
每轮内部跑完一个「模型想 → 调工具 → 看结果 → 再想」的 ReAct 循环，最后用结构化输出
（`response_format`）交付。新增策略/能力 = 新增工具，不改外层图。

- **Generator** 工具：`generate_image`（出图）、`query_experience`（读本 loop 已验证经验）、
  `query_general_experience`（读跨 loop 通用经验）。
- **Critic** 工具：`query_rubric`（按需查判定标准）；生成图 + target 直接注入多模态消息。

### Critic 驱动的经验闭环（单 loop）

```
轮次 N:   Generator 改 prompt（产出 delta_note）→ 出图
轮次 N:   Critic 看图 → verdict（分 + 失败项 + 理由）    ← 改动有效性的客观裁判
                ↓
轮次 N+1: in-loop Summarizer 综合【上轮 delta_note + 前后 verdict】:
            对比 → 判定 status(verified_effective / ineffective) → 沉淀进 conclusions.json
          Generator 读 conclusions：有效的保留约束、无效的换思路
```

经验不再散落为单轮事实快照，而是沉淀为**结构化知识库**
（`runs/<loop>/lessons/conclusions.json`），每条结论带 Critic 验证证据（前后分数+理由）。

### 跨 loop 经验蒸馏（独立 Summarizer）

```
in-loop Summarizer（每轮）            独立蒸馏器（离线，CLI `summarize`）
   写 conclusions.json（单题机器验证）       读 N 个 run 的 trajectory + conclusions
        │                                  │ 跨 run 综合（LLM agent）
        ▼                                  ▼
  Generator.query_experience ←────    general.json（跨 loop 通用经验：dos/donts + 证据）
  Generator.query_general_experience ←──┘  （首题也用得上：跨题先验）
```

`conclusions.json` 是单 loop、Critic **机器验证**的逐条结论；`general.json` 是跨 loop、LLM **综合**
的上层通用经验（每条带 dim/insight/dos/donts/evidence/confidence）。两者分层：前者是后者的原料。
蒸馏器以「已验证 effective/ineffective」为锚（比单看分数可靠），跨 run 找反复出现的有效/无效做法。

### 两个闭环

| | 闭环 A：生成迭代 | 闭环 B：评分校准 |
|---|---|---|
| **目的** | 提升生成还原度 | 让 Critic 评判贴合人的判断 |
| **驱动** | Critic 多维分 + 人工审批 | 人工排序 vs Critic 初评 |
| **产出** | 更好的图 + 验证过的经验 | 校准后的维度权重 |

闭环 B：人工只做排序（listwise，人擅长的）→ learning-to-rank 拟合权重 → 回灌 Critic，
天然修正 LLM 连续打分的系统性偏差。

## 架构

> 完整架构与决策记录见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**（含 10 条 ADR）。

```
src/img_iter_agent/
├── agents/
│   ├── generator.py          # 生成方 deepagent：改 prompt + 读经验注入（产出 delta_note）
│   ├── critic.py             # 评判方 deepagent：多模态打分（混合评分：二分✓/✗ + 连续0-1）
│   ├── summarizer.py         # in-loop 总结方：Critic 前后 verdict 对比→判有效/无效→沉淀 conclusions
│   ├── experience_distiller.py  # 独立总结方 deepagent：跨 loop 蒸馏通用经验 → general.json
│   ├── _agent_output.py      # Generator/Critic deepagent 的结构化输出 schema
│   └── tools/                # 工具注册中心（= 策略/能力扩展点）：generator/critic/experience 工具
├── pipeline/
│   ├── graph.py              # LangGraph 闭环：generator→critic→summarizer→human_review(interrupt)
│   ├── runner.py             # build_loop_context：构造 agent + checkpointer + 标准 config
│   └── state.py              # 跨轮 State（verdicts 累加器等）
├── llm/
│   └── chat_model.py         # build_chat_model → ChatOpenAI（指向 dmxapi，支持 tool-calling）
├── generation/
│   ├── router.py             # 按"任务模式+model_hint"路由到协议族 dispatcher
│   └── protocols/            # 屏蔽 dmxapi 四族差异：A(OpenAI)/B(豆包)/C(Qwen)/D(Gemini)
├── memory/
│   ├── knowledge.py          # 单 loop 经验知识库（conclusions.json + Critic 驱动 status 判定）
│   ├── experience.py         # 跨 loop 通用经验库（general.json 读写 + schema）
│   └── schema.py             # Pydantic v2 数据契约（AttemptRecord/CriticVerdict/KnowledgeConclusion）
├── calibration/              # 闭环 B：排序拟合权重（learning-to-rank, scipy SLSQP）
├── data/                     # 三层数据管理 + trajectory.jsonl（可复用重放）
└── web/                      # FastAPI 打分台（前后端解耦，Vanilla JS 前端）
```

**关键技术决策**：dmxapi 聚合 API（全云端，屏蔽 4 协议族）· LangGraph 编排（闭环原生+断点续跑）·
混合评分（二分可复现 + 连续保渐变）· 三层数据分离（benchmarks/runs/analyses）· 图片全程用文件路径。

## 项目结构

```
data/
├── benchmarks/              # 〔你准备的考题，✅入 git，跨 run 复用〕
│   └── furniture_product_whitebg/
│       ├── manifest.json    # 6 维评分定义 + 权重先验
│       └── samples/sNNN/    # target.png(参考锚) + content_spec.json(约束)
├── runs/                    # 〔系统产出，❌不入 git〕
│   └── <bench>-<sample>/    # 一题一 loop
│       ├── trajectory.jsonl # 完整轨迹（每轮含 delta_note）
│       ├── lessons/conclusions.json   # 单 loop 经验（Critic 机器验证）
│       └── out/             # 生成图
├── experience/<bench>/      # 〔跨 loop 通用经验，❌不入 git〕
│   └── general.json         # 蒸馏出的 dos/donts（多 run 综合）
└── analyses/                # 〔离线分析，只读〕
```

> [!NOTE]
> 你唯一需要手动管理素材的地方是 `data/benchmarks/<bench>/samples/`——放产品实物图
> `target.png` + 写 `content_spec.json`（要生成什么、约束、验收 checklist）。系统跑出来的产物自动进 `data/runs/`。

## 测试

```bash
python -m pytest          # 96 个测试（含 deepagent 路径、经验闭环、跨 loop 蒸馏、自动校准）
ruff check src/ tests/    # 代码风格
```

## 配置

所有配置走环境变量（前缀 `IMG_ITER_`），见 `.env.example`：
- dmxapi 凭证与地址
- 四个协议族的生图 model_id（seedream / gpt-image / gemini / qwen）
- 三个 agent 的推理 model_id（generator / critic / summarizer，均走 dmxapi 的 OpenAI 兼容端点，
  需支持 tool-calling；Critic 必须多模态）

Agent 的系统提示词也可在 Web 台「Agent 设置」页在线编辑（外部化到 `data/agents_config/`，
不在代码里）；更长的诀窍/流程沉淀在 `skills/<agent>/SKILL.md`，由 agent 按需加载。

## 状态

- ✅ 核心闭环（生成→评判→经验验证→迭代）已跑通，含 CLI 与 Web 打分台
- ✅ Generator / Critic 改造为 deepagents（tool-using agent）；经验闭环验证（Critic 驱动 status）
- ✅ 独立经验蒸馏器：跨 loop 总结通用经验 → Generator 跨 loop 注入（首题先验）
- ✅ 一题一 loop、自动权重校准
- ✅ 首个 benchmark（家具白底产品图，6 维评分，3 个 sample）
- 🚧 多策略扩展（增参考图/尺度参考图等）——工具注册中心已留好，暂只实现 prompt 策略
