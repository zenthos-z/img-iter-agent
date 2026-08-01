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

### 启动可视化打分台（Web）

```bash
img-iter-web       # 或: python -m uvicorn img_iter_agent.web.app:app --port 8765
# 浏览器打开 http://localhost:8765
```

打分台提供 5 屏：总览（每个 sample 的 loop 状态）、loop 详情（迭代轨迹+经验沉淀）、trace 详情
（大图对照 target+Critic 明细）、人工排序（拖拽打分→自动校准权重）、Agent 设置（改系统提示词+模型 id）。

## 核心机制

### Critic 驱动的经验闭环

```
轮次 N:   Generator 改 prompt（产出 delta_note）→ 出图
轮次 N:   Critic 看图 → verdict（分 + 失败项 + 理由）    ← 改动有效性的客观裁判
                ↓
轮次 N+1: Summarizer 综合【上轮 delta_note + 前后 verdict】:
            对比 → 判定 status(verified_effective / ineffective) → 沉淀进 conclusions.json
          Generator 读 conclusions：有效的保留约束、无效的换思路
```

经验不再散落为单轮事实快照，而是沉淀为**结构化知识库**
（`runs/<loop>/lessons/conclusions.json`），每条结论带 Critic 验证证据（前后分数+理由）。

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
│   ├── generator.py     # 生成方：改 prompt（产出 delta_note）+ 读经验注入
│   ├── critic.py        # 评判方：多模态打分（混合评分：二分✓/✗ + 连续0-1）
│   └── summarizer.py    # 总结方：Critic 前后 verdict 对比→判有效/无效→沉淀
├── pipeline/
│   ├── graph.py         # LangGraph 闭环：generator→critic→summarizer→human_review(interrupt)
│   └── state.py         # 跨轮 State（verdicts 累加器等）
├── generation/
│   ├── router.py        # 按"任务模式+model_hint"路由到协议族 dispatcher
│   └── protocols/       # 屏蔽 dmxapi 四族差异：A(OpenAI)/B(豆包)/C(Qwen)/D(Gemini)
├── memory/
│   ├── knowledge.py     # 经验知识库（conclusions.json 读写 + Critic 驱动 status 判定）
│   └── schema.py        # Pydantic v2 数据契约（AttemptRecord/CriticVerdict/KnowledgeConclusion）
├── calibration/         # 闭环 B：排序拟合权重（learning-to-rank, scipy SLSQP）
├── data/                # 三层数据管理 + trajectory.jsonl（可复用重放）
└── web/                 # FastAPI 打分台（前后端解耦，Vanilla JS 前端）
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
│       ├── lessons/conclusions.json   # 经验知识库（Critic 驱动验证）
│       └── out/             # 生成图
└── analyses/                # 〔离线分析，只读〕
```

> [!NOTE]
> 你唯一需要手动管理素材的地方是 `data/benchmarks/<bench>/samples/`——放产品实物图
> `target.png` + 写 `content_spec.json`（要生成什么、约束、验收 checklist）。系统跑出来的产物自动进 `data/runs/`。

## 测试

```bash
python -m pytest          # 83 个测试（含经验闭环验证、一题一 loop、自动校准）
ruff check src/ tests/    # 代码风格
```

## 配置

所有配置走环境变量（前缀 `IMG_ITER_`），见 `.env.example`：
- dmxapi 凭证与地址
- 四个协议族的生图 model_id（seedream / gpt-image / gemini / qwen）
- Agent LLM 的 model_id + protocol（Critic 必须多模态）

Agent 的系统提示词与模型 id 也可在 Web 台「Agent 设置」页在线编辑（外部化到
`data/agents_config/`，不在代码里）。

## 状态

- ✅ 核心闭环（生成→评判→经验验证→迭代）已跑通，含 CLI 与 Web 打分台
- ✅ 经验闭环验证（Critic 驱动 status）、一题一 loop、自动权重校准
- ✅ 首个 benchmark（家具白底产品图，6 维评分，3 个 sample）
- 🚧 多策略扩展（增参考图/尺度参考图等）——基建已留好，暂只实现 prompt 策略
