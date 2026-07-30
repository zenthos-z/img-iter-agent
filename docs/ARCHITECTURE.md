# img-iter-agent 架构与技术栈分析

> 目标：建立「AI 生图自动迭代 → 风格元素相同、内容可变」的**自我迭代对抗总结** Agent 系统。
> 本文档是当前阶段的核心交付，确定系统形态、闭环机制，并对候选技术栈做选型分析与推荐。
>
> **关键决策（已定）**：生图后端 = **dmxapi 聚合 API**（实测分 4 个协议族：OpenAI Images / 豆包 Responses /
> Qwen Responses / Gemini 原生）；目标 = **还原度**（非泛泛风格一致，§2.6）；风格锚定 = **纯参考图**；
> Agent 编排 = **LangGraph**（弃 pi sdk，§ADR-003）；测试调度 = **分开独立 + 控制变量法 + 逐轮人工审批**（§2.5）；
> 评分 = **多维初评 + 人工复评 + 权重校准**双闭环（§2.6，提升 agent 判断准确率）；
> 数据管理 = **三层 benchmarks/runs/analyses + trajectory.jsonl**，产出物可复用做策略对比/校准（§3.5）；
> 记忆 = **经验 MD + JSON 索引**（不引向量库，文件链接，§3.3）；图片 = **文件路径**（不存 base64，§3.4）。
> 核心难点 = 屏蔽 dmxapi 四个协议族的接口差异（§4.2）。

---

## 0. 先把目标拆清楚

用户的原始描述里其实埋了几个互相约束的子目标，必须先分开，否则技术栈会打架：

| # | 子目标 | 含义 | 技术上对应什么 |
|---|--------|------|----------------|
| **G1** | **还原度**（核心，用户明确） | 生成图与**目标/参考**的多维吻合程度：色彩、笔触、构图、内容主体、瑕疵——**不是泛泛"风格一致"** | 参考图锚定 + **多维评分**（§2.6）+ 人工校准权重 |
| **G2** | **内容可变** | 主体、场景、主题在还原度约束下自由变化 | 由 prompt 的「内容维度」驱动，与参考图锚定解耦 |
| **G3** | **自动迭代 + 自我对抗总结** | 生成还原度随轮次提升，系统自己归纳经验并改进 | 闭环A：生成→多维评判→总结→人工审批（§2, §2.5） |
| **G4** | **评判自我校准**（用户补充） | Critic 判断准确率随人工标注积累而提升 | 闭环B：人工多维评分 vs 初评 → 回归最优权重 → 校准 Critic（§2.6） |

**已确定约束（用户拍板，2026-07-30）：**

- **全云端，无本地 GPU** → 不走 diffusers / IP-Adapter / ControlNet / LoRA 本地方案。
- **风格锚定 = 纯参考图** → 不做本地开源模型测试。
- **生图后端 = dmxapi**（多模型统一聚合 API），需屏蔽 gemini/openai(gpt-image)/seedream/qwen
  各家在「参考图上传、单/多图、提示词、尺寸、图片编辑」上的差异（详见 §4.2 dmxapi 适配层）。

关键洞察：**G1（风格稳定）本质不是"prompt 工程"能解决的，它是图像条件控制问题**；
而 G3（自我迭代）是 Agent 编排问题。两者必须用不同技术栈，文档第 4 节会把它们拼起来。

---

## 1. 核心机制：把 GAN 思想搬到 Agent 层

用户说"生成对抗总结自我迭代"，这其实是一个非常具体的范式——
**不要训练两个神经网络，而是用两个 LLM agent 在「经验」上博弈**：

```
                  ┌─────────────────────────────────────────────────┐
                  │       目标/参考（锚定还原度 G1，见 §4.2）          │
                  │  参考图 → 各协议族风格迁移（seedream/gpt/gemini）    │
                  └───────────────────────┬─────────────────────────┘
                                          │ 锚定不变
                                          ▼
   ┌──────────┐  生成图    ┌──────────┐ 多维度初评   ┌──────────┐
   │ Generator│ ────────▶ │  Image   │ ───────────▶ │  Critic  │
   │  Agent   │           │  Pool    │              │  Agent   │
   │(控制变量 │ ◀──────── │(本轮产物) │ ◀─ 经验/高分 ─┤(多模态,  │
   │ +内容采样)│  经验反馈  └─────┬────┘              │  多维分) │
   └────┬─────┘                 │                   └────┬─────┘
        │                       │                        │ 各维度分+理由
        │ prompt修改             │                        ▼
        ▼                       │              ┌───────────────────┐
   ┌──────────┐                 │              │  人工审批(逐轮)     │── continue/stop/调方向
   │ Summarizer│                 │              │  + 人工多维复评(异步)│── 校准监督信号
   │  Agent   │──写经验MD──────┼─────────────▶└───────────────────┘
   │(归纳跨轮 │                 │                        │
   │ 规律)    │                 │                        ▼ 离线校准(§2.6)
   └──────────┘                 │              ┌───────────────────┐
        │                       │              │  权重校准: 人工vs初评│
        │                       │              │  → 回归最优维度权重 │──▶ 回灌 Critic
        │                       │              └───────────────────┘
        │                       ▼
        │         ┌──────────────────────────────────────────────┐
        │         │  数据层(三层, 全文件链接, §3.5)                │
        └────────▶│  benchmarks/(定制基准) → runs/(轨迹,可复用)     │
           读经验  │    → analyses/(策略对比/校准/泛化,只读)        │
                  └──────────────────────────────────────────────┘
                     trajectory.jsonl = 完整训练轨迹(重放/分析)
```

> **两个闭环（用户设计）**：
> - **闭环A 生成迭代**（在线）：生成 → Critic 多维初评 → Summarizer 写经验 → 人工审批 → 下一轮。
> - **闭环B 评分校准**（离线/异步）：人工多维复评 vs Critic 初评 → 回归最优维度权重 → 校准 Critic，
>   让 agent 判断准确率随人工标注积累而提升（§2.6）。

> **记忆设计原则（用户明确）：不引向量库。** 用**双层载体**：
> - **经验写 Markdown**——可丰富、给人读、Summarizer 自然语言归纳；
> - **JSON 只做参数/版本的整理索引**——记录每次尝试的 model/mode/params/scores，并指向素材文件；
> - **JSON 条目和经验 MD 互相用文件链接引用**，不复制内容（图片更是只存路径，见 §3.4）。

三个 agent 的职责切分（对应"生成对抗总结"四个字）：

- **Generator（生成方）**：负责 **G2 内容可变**。按控制变量法（§2.5.2）每轮只变一个维度，
  读相关经验 MD + JSON 筛选出的高分先例，产出 prompt 并调用生图模型。
- **Critic（对抗评判方）**：多模态 LLM 看图，按 benchmark 的 `score_dimensions` **逐维度打分**
  （还原度导向，非单一总分）。这一"对抗"压力倒逼 Generator 改进；其权重可被闭环B校准。
- **Summarizer（总结方）**：跨轮归纳。把零散的"Critic 评分理由"抽象成可复用的经验
  **写进 Markdown**（"用 XX 笔触描述时还原度低，应改用 YY"），并在 JSON 索引里登记指向该 MD，
  让系统真正"越跑越好"。

> 这就是"自我对抗总结自我迭代"：对抗(Critic) → 总结(Summarizer 写 MD) → 自我迭代(下一轮 Generator 读 MD+JSON)。

---

## 2. 迭代闭环（一次 epoch）

```
for round t in 1..T:
    1. [内容采样]   Generator 从主题空间采一批新内容（保证"内容可变"）
    2. [经验召回]   按 model/mode/scores 用 JSON 筛出相关先例 → 读其指向的经验 MD
    3. [生成]       prompt = f(内容, 经验MD, 固定风格描述)
                    images  = 生图模型(prompt, 参考图)   # 风格锚定见 §4.2
    4. [评判]       Critic 对每张图打 style_score / content_score / 总分，给批评
    5. [筛选]       选 top-k 进"标杆"；选 worst-k 做"反面教材"
    6. [总结]       Summarizer 看本轮 + 历史，提炼新经验写进 MD；JSON 登记索引
    7. [收敛判定]   若平均 style_score ≥ 阈值 且 内容多样性达标 → 停；否则下一轮
```

**收敛指标（自动判定停止 / 监控）**：

- `style_consistency`：同批图两两的风格特征距离（CLIP/特征向量 cosine），越高越好；
- `content_diversity`：同批图的内容主题熵 / 聚类数，保证"内容可变"；
- `quality_trend`：`content_score` 的移动平均，看是否还在涨。

**LangGraph 落地映射（context7 官方 API 核实）**：上面 7 步映射到一个 StateGraph——

```python
# State：跨轮累积（reducer 自动合并）
class RunState(TypedDict):
    round: int
    model: str                                   # 本闭环固定测这一个模型（分开独立评测）
    images: Annotated[list[str], operator.add]   # 生成图路径累积
    scores: Annotated[list[dict], operator.add]  # 每轮分数累积
    decision: str                                # 人工审批结果: continue/stop/adjust

# 节点 = 三个 agent
builder = StateGraph(RunState)
builder.add_node("generator", generator_node)   # 步 1~3
builder.add_node("critic", critic_node)         # 步 4~5
builder.add_node("summarizer", summarizer_node) # 步 6 + 写经验
builder.add_node("human_review", human_review_node)  # 步 7：人工审批

# 闭环：summarizer 后用 interrupt 暂停等人工审批
def human_review_node(state: RunState):
    from langgraph.types import interrupt
    verdict = interrupt({                         # 把本轮结果摊给人看
        "round": state["round"], "images": ...,
        "scores": state["scores"][-1], "lesson": ...,
        "prompt": "本轮测了 X，分数 Y。回复 continue / stop / 调整方向..."})
    return {"decision": verdict}                  # Command(resume=...) 注入

def route(state: RunState) -> Literal["generator", END]:
    return END if state["decision"] == "stop" else "generator"

builder.add_edge(START, "generator")
builder.add_edge("generator", "critic")
builder.add_edge("critic", "summarizer")
builder.add_edge("summarizer", "human_review")
builder.add_conditional_edges("human_review", route)

graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string("data/runs/<run>/checkpoints.sqlite"),
)
```

- **停止交人工**：收敛不自动判定（CLIP 分数只是代理指标，自动停易误判）。每轮 `interrupt()` 暂停，
  你看图+分数+经验后回复，`graph.invoke(Command(resume=...), config)` 恢复。
- 仍设 `recursion_limit` 作硬上限兜底，防意外无限循环。
- 观测：`graph.get_graph().draw_mermaid_png()` 导流程图；checkpoint 逐轮回看 `state`。
- 恢复：`graph.invoke(None, config=config)` 用同 `thread_id` 从断点继续。

---

## 2.5 测试调度策略（Generator 作为"测试调度者"必须知道的事）

> 用户指出：Generator 的系统提示词要先回答"**怎么测**"，否则闭环无从谈起。
> 以下策略已定（用户 2026-07-30 确认），是 Generator 系统提示词的核心输入。

### 2.5.1 评测范式：分开独立（每个模型一条闭环）

- **决策**：每个生图模型**各跑一条独立自迭代闭环**，互不共享经验。
  目的是**公平横向对比各模型的天花板**——这正是"测试调度"的本意。
- **代价**：各模型从零学、成本×模型数。但换来经验干净、可归因、可对比。用户已接受此权衡。
- **实现**：一条 LangGraph 闭环绑**一个固定 model**（`RunState.model` 不变）；
  要测 N 个模型就跑 N 条独立闭环（N 个 `thread_id` / N 个 run 目录）。
- **经验归属**：因为是分开独立，经验的 `applies_to` 恒等于当前模型，无需区分通用/特异——
  经验天然只在本闭环内累积，简单干净。

### 2.5.2 控制变量法：自回归怎么跑

自回归的本质是**每轮只变一个维度、其余固定**，分数变化才能归因到那个变量。
Generator 每轮必须在 `test_variable` 字段声明"本轮我在测什么"。

五个可变维度：

| 维度 | 例子 | 说明 |
|---|---|---|
| `prompt` | 提示词措辞/风格描述写法 | 风格描述怎么写才稳 |
| `reference_images` | 参考图选择/数量/组合 | 哪张参考图、几张参考图最好 |
| `size` | 尺寸/分辨率档 | 2K vs 3K 对风格的影响 |
| `generation_mode` | 单图编辑 vs 多图融合 vs 多轮改图 | 哪种生图方式风格迁移最稳 |
| `model_params` | seed / quality / n | 模型参数（注意：model 本身固定，不在此列） |

**自回归驱动**：上一轮的 `scores` + 经验 → Generator 决定下一轮变哪个维度、变成什么，
并在新尝试的 `test_variable` 记录，`baseline_ref` 指向对照组尝试 id。
这样经验能说清"X 变化导致 style_score 从 7.2→8.5"。

### 2.5.3 停止策略：逐轮人工审批（不自动收敛）

- **决策**：收敛**不自动判定**。每轮 summarizer 后 `interrupt()` 暂停，把本轮
  （图路径 + 分数 + 经验 + 本轮测了什么）摊给人看，人回复 `continue` / `stop` / `调整方向`。
- **理由**：风格一致性 CLIP 分数只是代理指标，"分数达标但图难看"或"该停却没停"风险大，
  人工判断更可靠（用户明确）。
- **兜底**：设 `recursion_limit` / 最大轮数硬上限，防意外无限循环。

### 2.5.4 Generator 系统提示词骨架（基于上述策略）

```
你是一个「生图测试调度者」。当前闭环只测模型：{model}。

【你的目标】找到让 {model} 稳定复刻参考图风格、同时内容可变的最佳生图配置。

【测试方法：控制变量法】
每轮只改变一个维度（prompt / reference_images / size / generation_mode / model_params），
其余保持与上一轮基线一致。你必须在 test_variable 字段声明本轮变化了什么。

【可用经验】（仅来自当前闭环历史，见 index.json + lessons/）
{检索到的本模型历史高分尝试 + 经验 MD}

【你的产出】每轮输出一个 GenRequest：
- 明确 test_variable（本轮变化维度）+ baseline_ref（对照组）
- 参考风格锚定见 §4.2（reference_images 非空走风格迁移路径）

【风格锚定】风格以参考图为准（见 §3.1），不要靠 prompt 硬描述。
```

### 2.5.5 MVP 首批模型（用户指定，model id 待用户给）

纳入 4 个（每条独立闭环）：seedream-5.0-pro（B族）、gpt-image-2（A族）、
gemini-3.1-flash-image（D族）、qwen-image-2.0（C族，能力待探测）。
**具体 model id 由用户指定**（不同版本价格不同）——做成 config，不写死（见 §config）。

---

## 2.6 评分校准闭环：混合评分 + 排序校准（用户提出并修正）

> 用户明确了评分机制，并经历两轮关键修正：**重点是还原度（不是泛泛风格一致）**；
> 大模型擅长分类、不擅长量化——但不能因此把**所有**维度都二分（材质还原度、颜色准确度是**渐变**的，
> 二分会丢信息）。最终机制是**混合评分（二分 + 连续）+ 排序校准**：
> 客观维度用 ✓/✗ 二分（可复现、归因清晰）；渐变维度让 LLM 打连续分（承认有偏差）；
> 权重不靠"人工逐项打分"拟合，而靠**人工对 trace 排序**（人擅长的）→ learning-to-rank 拟合，
> **天然修正 LLM 连续打分的系统性偏差**。完整流程图见 `docs/EVALUATION.md`。
> 这是与生成迭代闭环(§2)并行的**第二个闭环**，且是离线/异步的。

### 2.6.1 混合评分（Critic 初判）

每个维度按 benchmark `manifest.json` 的 `scoring_type` 字段分派，统一产出 `∈[0,1]` 的特征值：

```
每个 trace → 特征向量 features[dim] ∈ [0,1]  (逐维度一个数)

二分型维度(scoring_type=binary): consistency/product_structure/artifact_defect/commercial_focus
   Critic 按 content_spec.json 的 checklist 逐项 ✓/✗(每项带一句理由)
   features[dim] = 通过项数 / 总项数

连续型维度(scoring_type=continuous): material_texture/color_accuracy
   Critic(多模态) 按 rubric points 整体给 0-1 分(承认有偏差)
   features[dim] = LLM 归一化分

还原度总分 = Σ(wᵢ × features[i])   # 权重 w 初始用 benchmark 先验, 后续被排序校准更新
```

- 二分型维度**可复现**（同输入同判定，不像打分会漂移），归因精准（每项 ✗ 都有理由）；
  连续型维度虽带偏差，但保留了渐变信息，其偏差由排序校准的权重吸收。
- 判定/评分写入轨迹 `trajectory.jsonl`：二分项进 `critic_checks`、连续分进 `critic_scores`（§3.5.3）。
- **维度定义的真源是 `manifest.json`**（含 `scoring_type`/`weight_init`/`check_items`）；
  三视图任务的 6 维见 §3.6.2 / `data/benchmarks/furniture_product_whitebg/manifest.json`。

### 2.6.2 人工复判（异步补）

- 人工对（部分或全部）生成 trace 按**同样的维度**复评：二分项可逐项 ✓/✗，连续维度可不必逐项打分。
- 与 Critic 初评**配对**——这是校准的监督信号。人工复评不阻塞生成闭环（异步补），攒够一批再做校准。
- 人工复评存入 `runs/<id>/human_scores/`。

### 2.6.3 校准：排序拟合权重（learning-to-rank）

当积累了一定量的 trace 后，**核心校准信号是「人工对 trace 的整体排序」**——人猜不准"材质 72 分"，
但能可靠判断"trace A 比 trace B 好"，**排序是人类的强项**。校准即：找权重 `w` 使 `w·features`
给出的排序**吻合人类排序**。

```
目标: 找权重 w(约束 Σw=1, w≥0) 使按 (w·features) 排出的 trace 顺序
      尽量吻合人工对 trace 的排序
方法: learning-to-rank(基于 pairwise 或 listwise 损失; 初始可用 rank-aware 回归)
为什么修正偏差: 若某连续维度 LLM 系统性偏高, 拟合会把该维度权重压低,
              使加权排序仍贴合人工判断 → 天然吸收 LLM 连续分偏差
产出: runs/<id>/calibrated_weights.json + analyses/weight_calibration/ 报告
```

- 这比"量化×量化回归"(量化分对量化分, 误差叠加)或"逐项打分"都更干净：
  人只做擅长的**排序**，不被迫猜绝对分。
- 校准出的权重**回灌 Critic 算还原度总分的权重**（features 与二分项判定逻辑本身不变），
  下一轮 Critic 用新权重算还原度总分
  → **agent 判断准确率随人工标注积累而提升**（这正是用户要的）。
- 二分项若发现 LLM 系统性判错，可针对性改该项的判定描述/阈值/示例（与权重校准正交，可并行做）。

### 2.6.4 两个闭环的关系

| | 闭环A：生成迭代（§2） | 闭环B：评分校准（本节） |
|---|---|---|
| **目的** | 提升生成还原度 | 提升 Critic 评判准确率 |
| **频率** | 在线、逐轮 | 离线、攒批后 |
| **驱动信号** | Critic 多维分 + 人工审批 | 人工多维分 vs Critic 多维分 |
| **产出** | 更好的图 + 经验 | 校准后的维度权重 |
| **相互关系** | 闭环A 用 闭环B 校准出的权重算总分 | 闭环B 读 闭环A 的轨迹做校准 |

> 闭环B 让系统从"Critic 固定打分"进化为"Critic 自我校准"——人工标注越多，agent 越准。

---

## 3. 三个核心难点与技术对应

### 3.1 难点 A：风格怎么稳定（G1 的命根子）—— 基于 dmxapi 实测

这是最容易踩坑的地方。**纯靠 prompt 描述风格是守不住的**——不同内容下，"水彩风"
四个字会漂移。用户已确定用**参考图**做唯一风格锚定手段，全云端无本地 GPU。

**关键约束（dmxapi 实测，见 §4.2）：模型能力不对称，且直接决定 G1 能否成立。**

| dmxapi 模型 | 文生图 | 图生图/编辑 | 多图融合 | 对 G1 的价值 | 协议族 |
|------|:---:|:---:|:---:|------|:---:|
| **seedream-5.0-pro**（豆包） | ✅ | ✅ 图生图 | ✅ 2~10张 | ✅ **风格迁移主力**（JSON、多图融合强） | B |
| **seedream-5.0-lite**（豆包） | ✅ | ✅ 编辑 | ✅ 最多14张+组图 | ✅ 风格迁移/组图 | B |
| **gpt-image-2**（OpenAI） | ✅ | ✅ `/edits` | ✅ 多个 `image` | ✅ **风格迁移主力**（高质量） | A |
| **gemini-3.1-flash-image** | ✅ | ✅ 多轮改图 | ✅ 历史多图 | ✅ **多轮迭代改图**（天然契合自迭代） | D |
| **qwen-image-2.0**（阿里） | ✅ | ?（待测） | ?（待测） | ? 文字渲染强，待测 | C |

**结论——风格锚定的可行路径（全云端，实测官方 doc.dmxapi.cn）：**

好消息：**现行主力模型 seedream 5.0 系列（pro/lite）和 gpt-image-2 都支持图生图/多图融合**，
风格锚定路径很宽（旧文档说 seedream-3.0 不支持图生图，那是旧模型，已淘汰）。

1. **参考图编辑迁移（主路径）**：把用户参考图作"底图"，通过 gpt-image-2（`/v1/images/edits`）
   或 seedream-5.0（`/v1/responses` 的 `image` 字段），用指令"保持风格，把主体换成 X"
   让模型在新内容上复刻原风格。**这是云端最强风格锚定。**
2. **多图参考融合**：seedream-5.0-pro 支持 2~10 张、5.0-lite 最多 14 张参考图融合，
   gpt-image-2 支持多个 `image`——可同时给"风格参考图 + 内容草图"，风格约束更强。
3. **多轮迭代改图（Gemini 独有）**：gemini-3.1-flash-image 支持**对话式逐轮改图**，
   每轮基于上一张图（需带 `inline_data` + `thoughtSignature`）继续调整——天然契合
   "自我迭代"闭环（见 §1），可作为单模型内的微迭代。

> ⚠️ 这意味着：风格锚定有 **3 条路径**，分别走 4 个协议族中的 A(gpt) / B(seedream) / D(gemini)。
> `ImageGenerator` 适配层必须按协议族分 dispatcher，详见 §4.2。

### 3.2 难点 B：内容怎么"可控地变"（G2）

内容采样不能随机乱来，否则 Critic 没法稳定打分。建议给内容建模一个**主题空间**：

- 显式 schema：`{主体, 动作, 场景, 视角, 光照}`，每维有受控取值池；
- Generator 每轮从池里做"受控变异"（改一两个维度，其余固定）——这样跨图既有多样性，
  又能归因"是哪个维度的变化导致了风格漂移"。

### 3.3 难点 C：Memory 记录（最简 JSON，按用户要求设计字段）

自我迭代靠 Memory 积累经验。**按用户要求：不引向量库/SQLite，用双层载体**——

- **经验写 Markdown**（`.md`）：给人读、可写丰富、Summarizer 自然语言归纳的产物；
- **JSON 只做参数/版本的整理索引**：记录每次尝试的 model/mode/params/scores，**用文件链接**
  指向素材（图片、经验 MD），不复制内容；
- **图片、经验 MD 都用文件链接**串起来，绝不在 JSON 里塞 base64 或长正文。

#### 3.3.1 目录结构（一次 run 的记忆布局）

`index.json` + `lessons/` + `out/` + `ref/` 都在 `data/runs/<run_id>/` 下，
是 §3.5.1 三层布局中「层2 runs」的组成部分（还含 trajectory/human_scores/calibrated_weights）。
完整目录见 **§3.5.1**，本节只聚焦记忆相关的两个文件：

```
data/runs/<run_id>/
├── index.json            # ← 索引：每次生图尝试一条记录（参数/版本/打分/链接）
├── lessons/              # ← 经验库：每个经验一篇 Markdown
│   ├── 001-style-drift-on-multi-ref.md
│   └── 002-seedream-vs-gpt-color-matching.md
├── out/                  # ← 生成图（文件，路径被 index.json 引用）
└── ref/                  # ← 本次用到的参考图副本（或软链到 benchmarks/samples）
```

**核心关系**：`index.json` 是枢纽——每条尝试记录都带「图路径 + 经验 MD 路径」的链接；
经验 MD 也可反向引用 `index.json` 里的尝试 id 作为证据。文件之间互相链接，内容零复制。

#### 3.3.2 index.json：尝试记录字段（只管参数/版本/索引，不存正文）

```jsonc
{
  "id": "try_00017",                       // 唯一 id（自增）
  "timestamp": "2026-07-30T15:20:33+08:00",// 尝试时间（ISO 8601）

  // —— 模型与生图方式（区分维度，索引/筛选用）——
  "model": "seedream-5.0-pro",             // 模型名（含 ssvip 等后缀）
  "protocol_family": "B",                  // 协议族 A/B/C/D（见 §4.2.1）
  "generation_mode": "multi_ref_fusion",   // 生图方式枚举：
                                           //   text_to_image / single_ref /
                                           //   image_edit / multi_ref_fusion /
                                           //   multi_turn_edit（Gemini 多轮）
  "endpoint": "/v1/responses",

  // —— 测试调度专属（控制变量法，见 §2.5.2）——
  "test_variable": "reference_images",     // 本轮变化的维度：prompt /
                                           //   reference_images / size /
                                           //   generation_mode / model_params
  "baseline_ref": "try_00016",             // 对照组尝试 id（其余维度与之一致）
  "delta_note": "参考图从 1 张→2 张",       // 本轮变化的人类可读说明

  // —— 生图参数（参数/版本的整理，JSON 的本职）——
  "params": {
    "prompt": "保持水彩风格，把主体换成猫",  // 提示词/编辑指令（短的可直存）
    "prompt_ref": "prompts/try_00017.md", // 长提示词则单独存 MD，这里只放链接
    "size": "2K",                          // 尺寸（各族格式原样记，便于归因）
    "quality": "high", "n": 1, "seed": 12345, "watermark": false,
    "raw_request_ref": "requests/try_00017.json" // 原始请求体快照(去密钥)的链接
  },

  // —— 素材链接（全是文件路径，不存 base64/正文）——
  "reference_images": [                    // 参考图路径
    "ref/style_watercolor_01.png",
    "ref/subject_cat.png"
  ],
  "outputs": ["out/try_00017_1.png"],      // 生成图路径

  // —— 评判（Critic 填，驱动迭代）——
  "scores": {
    "style_consistency": 8.5,              // G1 风格一致性
    "content_quality": 7.0,                // G2 内容质量
    "overall": 7.7
  },
  "verdict": "keep",                       // best / keep / discard / worst
  "critique_ref": "critiques/try_00017.md",// Critic 批评（较长）存 MD，这里放链接

  // —— 经验链接（指向 lessons/ 下的 MD，可多条）——
  "lesson_refs": ["lessons/001-style-drift-on-multi-ref.md"],
  "source_round": 3,
  "tags": ["multi_ref", "seedream", "style_drift"]   // 便于 JSON 筛选召回
}
```

> 设计要点：**JSON 里只有「字段值」或「文件链接」，没有任何大段正文/base64**。长 prompt、
> Critic 批评、归纳出的经验，都各自落成 MD 文件，JSON 只记路径。

#### 3.3.3 经验 Markdown（lessons/*.md，给人读，可丰富）

经验是 Summarizer 归纳的可复用知识，写成结构化 MD：

```markdown
# 001 · seedream 多图融合的风格平均化现象

## 现象
当参考图 >3 张做 multi_ref_fusion 时，输出风格倾向"平均"，
失去单张参考的鲜明特征。

## 证据（指向 trajectory.jsonl 的轮次 / index.json 尝试记录）
- try_00012（2张参考）：material_texture 8.7  ← keep
- try_00015（5张参考）：material_texture 6.1  ← worst
- try_00017（2张参考）：material_texture 8.5  ← keep

## 结论 / 建议
风格迁移用 multi_ref_fusion 时，参考图建议 ≤2 张；
需要更多内容素材时改用 image_edit 逐张改。

## 适用范围
model: seedream-5.0-* ; mode: multi_ref_fusion
```

MD 里用尝试 id 引用 `index.json` 作为证据，反向闭环；人也能直接打开图片核对。

#### 3.3.4 迭代时怎么用这套双层记忆

- **召回（筛）**：按 `model`/`generation_mode`/`scores`/`tags` 在 `index.json` 里过滤出相关先例；
- **注入（读）**：顺链接读对应 `lessons/*.md` 的经验正文 + 高分尝试的 prompt，作为 few-shot/指导注入；
- **写入**：Critic 评分 → 追加一条 index 记录（含图路径、critique MD 链接）；
  Summarizer 跨轮归纳 → 新写一篇 `lessons/*.md`，并在相关 index 记录的 `lesson_refs` 登记链接。
- **不做向量检索**：召回靠 JSON 字段过滤 + MD 链接跳转；记录量增大后再考虑升级。

### 3.4 难点 D：素材一律用文件路径，不存 base64 / 不内联正文（按用户要求）

**约定（贯穿全系统，含 JSON 和 MD 两类载体）：**

- **图片**：任何地方记录/传递图片都用**文件路径**，绝不存 base64 字符串（无论 JSON 还是 MD）。
- **长文本**：长 prompt、Critic 批评等不塞进 JSON 字段，单独存 MD，JSON 只放链接。
- **存盘**：生图后立即把返回的 `b64_json`/`url` 解码下载，存到 `data/runs/<run_id>/out/`，
  `index.json` 里只记路径（如 `outputs` 字段）。这样人能直接打开图片查看。
- **入参转换**：调用 dmxapi 时，适配层**按文档说明自动把文件路径转成各家要求的格式**：
  - 协议族 A（gpt-image-2 edits）：路径 → 读文件 → multipart 二进制上传；
  - 协议族 B/C（seedream/qwen responses）：路径 → 读文件 → `data:image/<fmt>;base64,...` 字符串；
  - 协议族 D（Gemini）：路径 → 读文件 → `inline_data.data`(base64)。
- **好处**：人类可读可查；JSON/MD 不膨胀；便于复现与对比；文件间用链接互引、内容零复制。
- **注意**：base64 转换是「调用前临时生成、用完即弃」，绝不持久化进任何 JSON/MD。

### 3.5 难点 E：数据管理体系——benchmark/轨迹/产出物的组织与复用（用户强调）

> 用户指出三个此前缺失的需求：① benchmark 标准答案要能**方便接入与定制**；
> ② 总结出的经验要**能验证泛化性**；③ 每次跑的产出物要**可重复使用**（训练轨迹分析、
> 策略效果对比）。三者本质是同一个根问题：**产出物必须是结构化、解耦、可独立加载的"数据集"**，
> 而非埋在运行态里的临时数据。以下数据层设计把这三者统一起来。

#### 3.5.1 顶层目录布局（三层分离：benchmarks / runs / analyses）

```
data/
├── benchmarks/                      # 〔层1〕基准集：你准备/定制的，跨 run 复用
│   └── <bench_id>/                  #   一个 benchmark = 一组测试样本
│       ├── manifest.json            #   基准元数据 + 评分维度定义（见 3.5.2）
│       ├── samples/                 #   样本：每个含"目标"与"内容要求"
│       │   ├── s001/
│       │   │   ├── target.png       #   目标参考图（还原度的基准）
│       │   │   ├── target.md        #   目标描述 + 验收 checklist
│       │   │   └── content_spec.json#   该样本要生成的内容主题/约束
│       │   └── ...
│       └── rubric.md                #   评分维度说明（给人读）
│
├── runs/                            # 〔层2〕运行轨迹：每次跑的完整产出，可复用
│   └── <run_id>/                    #   = <model>__<bench>__<timestamp>
│       ├── meta.json                #   run 配置快照(model/mode/参数/软/硬件版本)
│       ├── trajectory.jsonl         #   ★完整训练轨迹：每轮一行(见 3.5.3)
│       ├── checkpoints.sqlite       #   LangGraph checkpoint(可断点续跑/回放)
│       ├── index.json               #   尝试记录索引(§3.3.2, 轨迹的轻量索引)
│       ├── lessons/                 #   经验 MD(本 run 归纳, 见 §3.3.3)
│       ├── out/                     #   生成图(路径被 trajectory/index 引用)
│       ├── ref/                     #   本次用到的参考图软链/副本(指向 benchmarks/samples, 可追溯)
│       ├── human_scores/            #   人工评分(异步补, 用于校准 agent, 见 §2.6)
│       │   └── s001.json            #   人对各图的多维评分
│       └── calibrated_weights.json  #   本 run 校准出的维度权重(见 §2.6)
│
└── analyses/                        # 〔层3〕离线分析：读层1+层2产出的结果, 不污染原始数据
    ├── strategy_compare/            #   策略对比(读多个 run 的 trajectory)
    ├── weight_calibration/          #   权重校准(读 human_scores 算最优权重)
    └── generalization/              #   泛化分析(预留, 用户暂不做, 结构留好)
```

**素材归属总规则（回答"素材放哪"）：**
- **你手动准备的考题素材**（产品实物图、验收说明、生成约束）→ 放
  `data/benchmarks/<bench>/samples/`。**这是你唯一需要手动管理素材的地方**。
- **系统生成的产物**（图、轨迹、经验、checkpoint）→ 自动进 `data/runs/<run>/`，不用管。
- **没有全局素材库**——每份考题自包含在自己的 `samples/` 里；`runs/.../ref/` 只是软链/副本
  指向 benchmark 样本，便于 run 自追溯，不引入新素材。

**核心原则**：
- **benchmarks 与 runs 解耦**——一个 benchmark 可被多个 run（不同模型/策略）复用；
  一个 run 永远指向一个固定 benchmark（`meta.json` 里记 `bench_id`）。
- **轨迹是头等公民**——`trajectory.jsonl` 完整记录训练过程，能脱离运行时独立加载重放。
- **analyses 只读**——分析产物独立放一层，绝不回写 runs，保证原始轨迹不可变、可复现。

#### 3.5.2 benchmark 标准：还原度导向 + 多维评分（用户定制）

> **关键认知修正（用户 2026-07-30）**：系统目标重点是**还原度**（生成图与目标的吻合程度），
> 不是泛泛的"风格一致"。且评分是**多维 + 人机协同校准**，见 §2.6。

`benchmarks/<bench_id>/manifest.json`：

```jsonc
{
  "bench_id": "watercolor_v1",
  "description": "水彩风还原度基准",
  "score_dimensions": [               // ★多维评分因子，用户定义——这就是"标准答案"的载体
    {"dim": "color_match",       "desc": "色彩与目标一致性",  "weight_init": 0.25},
    {"dim": "brushstroke",       "desc": "笔触/纹理还原",     "weight_init": 0.20},
    {"dim": "composition",       "desc": "构图结构还原",       "weight_init": 0.20},
    {"dim": "content_fidelity",  "desc": "内容主体还原度",     "weight_init": 0.25},
    {"dim": "artifact",          "desc": "无瑕疵/伪影",        "weight_init": 0.10}
  ],
  "samples": ["s001", "s002", "s003"]
}
```

- **"标准答案"= 目标参考图 + 验收 checklist + 多维评分定义**，三者都由用户准备/定制。
- 维度的**初始权重**是用户给的先验；**最终权重**由人机校准得出（§2.6）。
- Critic 按这些维度逐项给分（不是单一总分）→ 还原度 = 加权多维分。

#### 3.5.3 完整训练轨迹 trajectory.jsonl（可复用的核心）

每轮迭代一行，自包含、可独立加载重放：

```jsonc
{
  "round": 3,
  "ts": "2026-07-30T15:20:33+08:00",
  "model": "seedream-5.0-pro-xxx",     // 用户指定的 model id
  "bench_id": "watercolor_v1", "sample": "s001",

  "gen_request": {                      // Generator 本轮决策（完整, 可重放）
    "prompt_ref": "prompts/r3.md", "reference_images": ["ref/target_s001.png"],
    "size": "2K", "generation_mode": "image_edit", "seed": 12345,
    "test_variable": "reference_images", "baseline_ref": "round2",   // 控制变量(§2.5.2)
    "raw_request_ref": "requests/r3.json"
  },
  "outputs": ["out/r3_s001_1.png"],

  "critic_scores": {                    // agent 多维初评(§2.6)
    "color_match": 8.2, "brushstroke": 7.0, "composition": 8.5,
    "content_fidelity": 7.8, "artifact": 9.0,
    "weighted": 7.96                      // 用当前权重算出的还原度总分
  },
  "critic_ref": "critiques/r3.md",

  "lesson_refs": ["lessons/003-ref-count.md"],   // 本轮归纳的经验
  "human_decision": "continue",                  // 人工审批结果(§2.5.3)
  "human_scores_ref": null                        // 人工评分异步补, 补上后指向 human_scores/
}
```

**为什么用 JSONL**：每行一轮、可流式追加、可 `grep`/`jq` 快速分析、单行损坏不毁全局。
策略对比 = 读多个 run 的 `trajectory.jsonl` 比较分数曲线、`test_variable` 效果。

#### 3.5.4 数据流可视化（三层 + 两个闭环 + 校准）

```
                         ┌─────────────────────────────────────────────┐
                         │  data/benchmarks/  〔层1 你定制, 跨run复用〕   │
                         │   manifest.json (score_dimensions/权重先验)   │
                         │   samples/sNNN/{target.png, target.md, spec}  │
                         └───────────────┬─────────────────────────────┘
                                         │ 加载样本+评分维度
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                        闭环A：生成迭代 (§2, §2.5)                       │
   │  Generator ─▶ 生图 ─▶ Critic多维度初评 ─▶ Summarizer写经验 ─▶ 人工审批  │
   │      ▲                                                          │       │
   │      │ 经验/历史轨迹注入                                          ▼       │
   │      └───────────── index.json + lessons/ ◀────────────── 每轮落盘轨迹  │
   └──────────────────────────┬───────────────────────────────────────────┘
                              │ 完整轨迹 trajectory.jsonl + 生成图 + 初评分
                              ▼
                   ┌──────────────────────────────────────────┐
                   │  data/runs/<run_id>/  〔层2 产出, 可复用〕   │
                   │   trajectory.jsonl  ★训练轨迹(重放/分析)    │
                   │   human_scores/     人工多维评分(异步补)     │
                   │   calibrated_weights.json  校准后权重       │
                   └───────────────┬──────────────────────────┘
                                   │ 人工评分 + 初评分 配对
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                 闭环B：评分校准 (§2.6, 离线/异步)                        │
   │   人工多维评分 vs Critic多维初评 ─▶ 回归求最优权重 ─▶ 校准 Critic 提示词   │
   └──────────────────────────┬───────────────────────────────────────────┘
                              │ 校准结果
                              ▼
                   ┌──────────────────────────────────────────┐
                   │  data/analyses/  〔层3 只读, 不污染原始〕   │
                   │   strategy_compare/  多run策略对比         │
                   │   weight_calibration/ 权重校准报告         │
                   │   generalization/    泛化分析(预留)        │
                   └──────────────────────────────────────────┘
```

- **闭环A**（在线）：生成→评判→总结→人工审批，产出轨迹与经验。
- **闭环B**（离线/异步）：用人工评分校准 Critic 的维度权重，提升 agent 判断准确率。
- **三层分离**保证：原始轨迹不可变 → 任何分析都能 100% 复现；benchmark 独立 → 方便定制接入。

### 3.6 首个 benchmark：家具跨境电商白底产品图（用户指定，已落地）

> 用户要求把"这个场景下需要的图片指标"作为第一个迭代目标。场景源自真实 AIGC 视觉工程师 JD
> （家具/跨境电商），其任职要求第4条"关注产品结构、比例、材质、纹理，对变形/失真/材质不准提出优化"
> 已直接点明核心指标。**已落地为 `data/benchmarks/furniture_product_whitebg/`**。

#### 3.6.1 场景为什么适合做第一个 benchmark

- **产品即主角**：家具白底图是强结构、强材质、严比例的品类，AI 翻车点最典型、最易归因；
- **参考图易得**：用户提供产品实物图即可作材质/颜色的对比锚；
- **指标收敛**：白底图去掉场景融合维度，Critic 评得更准、闭环验证更干净；
- **商业价值高**：直接对应家具电商详情页主图/SKU 图的还原度痛点（色差=退货）。

#### 3.6.2 指标体系（7 维度，已写入 manifest.json）

| 维度 | 权重 | 类型 | 评什么 | 家具场景的典型扣分 |
|------|------|------|--------|---------------------|
| `material_texture` 材质纹理 | **0.22** | 对比型 | 材质正确真实，对比实物图 | 金属画成塑料、木纹糊/重复 |
| `product_structure` 产品结构 | **0.20** | 绝对型 | 部件数/位置/形态正确 | 椅子少腿、把手位置错、穿模 |
| `proportion` 比例 | 0.15 | 绝对型 | 各部件比例协调 | 桌腿过粗、椅背过高、坐深失调 |
| `color_accuracy` 颜色一致性 | 0.15 | 对比型 | 与实物无色差（退货主因） | 偏色、饱和异常、色块不均 |
| `artifact_defect` 无瑕疵 | 0.13 | 绝对型 | 无变形/失真/伪影 | 直线变弯、不对称、悬浮 |
| `commercial_focus` 商业可用 | 0.15 | 绝对型 | 主体突出、白底干净、构图合规 | 主体偏移、留白不当 |
| `scene_integration` 场景融合 | 0.00 | — | 白底图不评测 | 场景图 benchmark 启用 |

**两类指标的关键区分**（影响 Critic 怎么评、benchmark 怎么建）：
- **对比型**（`material_texture`/`color_accuracy`）：必须有产品实物参考图作锚，Critic **对照参考图**打分；
- **绝对型**（其余）：单看生成图即可评。

> 权重是**初始先验**（材质+结构是家具命根子故权重最高），会被 §2.6 校准闭环持续优化。

#### 3.6.3 benchmark 落地结构（已建）

```
data/benchmarks/furniture_product_whitebg/
├── manifest.json              # 7 维度 + 权重先验 + 对比型标记
├── rubric.md                  # 评分细则（给人读，含扣分点 + samples 约定）
└── samples/
    └── _TEMPLATE/             # 样例模板：复制为 s001/s002/...
        └── content_spec.json  # 含 must_keep/may_change/must_avoid 约束
```

**待用户准备**（不阻塞代码骨架）：产品实物图 `samples/sNNN/target.png` +
对应 `target.md`（结构/材质/颜色说明）+ `content_spec.json`（要生成什么）。

#### 3.6.4 对系统设计的影响

- Critic 评分逻辑须区分**对比型/绝对型**：对比型维度需同时喂"生成图+参考图"给多模态 LLM；
- `GenRequest` 已有 `reference_images`，正好承载产品实物图（对比锚）——风格锚定与还原度评判
  **复用同一张参考图**，设计自洽；
- 控制变量法（§2.5.2）的 `test_variable` 现在有明确候选：prompt 措辞、视角、参考图选择、
  size、generation_mode（单图编辑/多图融合）——首轮可固定"用产品图做 image_edit 还原"。

---

## 4. 技术栈选型（已按用户决策收敛）

### 4.1 选型总表

| 层 | 决策 | 说明 |
|----|------|------|
| **语言** | **Python ≥ 3.11** | 生图、agent、向量库生态全在 Py |
| **生图后端** | **dmxapi 聚合 API（单一后端）** | 统一接入 gpt-image-2 / seedream-5.0 / qwen-image / gemini-image 等；不走本地 diffusers（无 GPU）。注意：dmxapi 按**协议族**分多套接口（§4.2），非单一端点 |
| **多模态评判(Critic)** | **dmxapi 接 Gemini 多模态**（可选叠加 GPT 多 judge） | 评分+批评质量高、价格好；评判/生成尽量异模型降偏差 |
| **Agent 编排** | **LangGraph** | 环路(闭环)是一等公民；状态可视化 + checkpoint 支持中断恢复，**便于观测自迭代的每一轮**。弃用 pi sdk（仅 Node.js 无 Python 版，跨语言成本高）；用 LangChain/LangChain-OpenAI 接 dmxapi 的 OpenAI 兼容端点 |
| **LLM 接口统一** | **dmxapi 已聚合**（文本对话走 chat completions） | 评判/总结用多模态文本模型 |
| **结构化输出** | **Pydantic v2** | Critic 评分、生图记录需严格 schema |
| **记忆存储** | **双层载体：MD(经验) + JSON(索引)（按用户要求）** | 不引向量库/SQLite；经验写 MD 给人读，JSON 只管参数/版本/索引并用文件链接串起图片与 MD（§3.3）；人类可读可手改 |
| **图像存储** | **文件路径（不用 base64）** | §3.4：生图落盘，JSON 只记路径；调用时适配层按文档自动转格式 |
| **图像/特征** | **Pillow + OpenCLIP** | 算 style_consistency / content_diversity 指标用 CLIP 特征 |
| **配置/密钥** | **pydantic-settings + .env** | 类型安全、12-factor；DMXAPI key 走环境变量 |
| **异步/并发** | **asyncio + httpx** | 批量生图、批量评判是 IO 密集；httpx 异步上传 multipart |
| **重试/限流** | **tenacity** | API 调用必须容错（生图偶发超时/失败） |
| **可观测** | **logging + run 目录** | 先做最小可观测，不引入外部依赖 |
| **数据管理** | **三层目录 benchmarks/runs/analyses + trajectory.jsonl** | §3.5：benchmark 与 run 解耦可复用；轨迹 JSONL 可独立加载重放；analyses 只读不污染原始；支撑策略对比/权重校准/泛化分析 |
| **实验追踪** | **LangGraph checkpoint + trajectory.jsonl** | checkpoint 断点续跑；trajectory.jsonl 是给人分析的训练轨迹（每轮一行） |
| **数据分析/校准** | **pandas + scikit-learn(回归) + matplotlib** | 离线读 trajectory+human_scores 做策略对比、权重回归校准（§2.6）、画图 |

### 4.2 dmxapi 适配层（核心难点：屏蔽多协议族差异）—— 关键设计

**实测发现（官方文档 doc.dmxapi.cn）：dmxapi 远不止"两个端点"——它按模型家族分成
多套协议、多个端点、多种字段命名。** 这正是用户强调"要兼容不同模型特点"的核心。
适配层的目标是把这些差异全部吞掉，对上层只暴露一个统一接口。

#### 4.2.1 dmxapi 真实的协议族（按家族划分）

| 协议族 | 端点 | 认证头 | 请求格式 | 提示词字段 | 适用模型 |
|--------|------|--------|----------|-----------|----------|
| **A. OpenAI Images** | `/v1/images/generations`（文生图）、`/v1/images/edits`（编辑） | `Authorization: Bearer <key>` | 文生图=JSON；编辑=**multipart/form-data** | `prompt` | **gpt-image-2** / gpt-image-1 |
| **B. 豆包 Responses** | `/v1/responses` | `Authorization: <key>`（无 Bearer！） | **JSON** | `input`（字符串） | **seedream-5.0-pro / 5.0-lite / 4.5 / 4.0** |
| **C. Qwen Responses** | `/v1/responses` | `Authorization: <key>` | **JSON** | `input.messages[].content[].text`（嵌套对象） | **qwen-image-2.0 / 2.0-pro** |
| **D. Gemini 原生** | `/v1beta/models/<model>:generateContent` | **`x-goog-api-key: <key>`**（不是 Bearer！） | JSON | `contents[].parts[].text` | **gemini-3.1-flash-image** |

> **协议差异极大**：连认证头、端点路径、提示词嵌套层级都不同。B/C 同端点但提示词结构不同；
> A 是 multipart；D 连域名路径都不一样。适配层必须按"协议族"分 dispatcher。

#### 4.2.2 各模型生图能力矩阵（实测，决定路由）

| 模型（dmxapi） | 协议族 | 文生图 | 图生图/编辑 | 多图参考/融合 | 多轮改图 | 参考图传法 | size 格式 | 角色 |
|------|:---:|:---:|:---:|:---:|:---:|------|------|------|
| **gpt-image-2**(-ssvip) | A | ✅ generations | ✅ edits | ✅ 多个 `image` 文件 | ❌ | multipart 文件上传 | `1024x1024`/`2K`等 | **风格迁移主力** |
| **seedream-5.0-pro** | B | ✅ | ✅ 图生图 | ✅ 2~10 张 | ❌ | `image`=URL/Base64（单图string/多图array） | `2K`或`2048x2048` | **风格迁移主力** |
| **seedream-5.0-lite** | B | ✅ | ✅ 编辑 | ✅ 最多14张+组图 | ❌ | 同上 | `2K/3K/4K`或像素 | 风格迁移/组图 |
| **qwen-image-2.0**(-pro) | C | ✅ | ?（待测） | ?（待测） | ❌ | 文生图无图输入 | `宽*高`（星号!） | 文字渲染强 |
| **gemini-3.1-flash-image** | D | ✅ | ✅ 多轮改图 | ✅ 历史多图 | ✅ | `parts[].inline_data`(base64)+`thoughtSignature` | `imageConfig.aspectRatio/imageSize` | **多轮迭代改图** |

> ⚠️ **重要修正**：旧文档(imagemodels.dmxapi.com)说 seedream-3.0 不支持图生图，那是**旧模型**。
> 官方现行主力是 **seedream 5.0 系列，全部支持图生图/多图融合**。这是好消息——风格锚定主力又多一个，
> 且豆包走 JSON（比 gpt 的 multipart 更易写异步）。

#### 4.2.3 协议族差异详情（写 adapter 必看）

**协议族 A — OpenAI Images（gpt-image-2）**
- 文生图：`POST /v1/images/generations`，JSON，字段 `model/prompt/n/size/quality/output_format`
- 编辑：`POST /v1/images/edits`，**multipart**，`files=[("image",(name,fp,mime)),...]`，
  `data={model,prompt,size,quality,n,...}`，参考图一张或多张（同名 `image` 字段）
- 返回：`data[].b64_json` 或 `data[].url`
- size：精确像素 `1024x1024`/`1536x1024`/`2048x2048`/`3840x2160`（须16倍数），或 `auto`

**协议族 B — 豆包 Responses（seedream 5.0）**
- 端点：`POST /v1/responses`，JSON，认证头 **`Authorization: <key>`（无 Bearer）**
- 字段：`model` / `input`(提示词string) / `image`(string单图 或 array多图) / `size` /
  `output_format` / `response_format` / `watermark`
- 参考图：`image` = 公网URL 或 `data:image/png;base64,...`；多图融合 2~10 张；lite 最多14张+组图
- size：分辨率档 `"1K"/"2K"/"3K"/"4K"` 或 像素 `"2048x2048"`
- 返回：`data[].url`（默认24h有效）或 `data[].b64_json`；lite 返回 `output[].image_url.url`

**协议族 C — Qwen Responses（qwen-image-2.0）**
- 端点：`POST /v1/responses`，JSON（与豆包**同端点但提示词结构完全不同**）
- 提示词嵌套：`input.messages[].content[].text`（仅单轮、仅一个 text，传多个会报错）
- **编辑图**：参考图放在同一个 `content[]` 数组里：`content[] = [{text:...}, {image:"<url或base64>"}]`，
  支持 1~3 张（有序）
- 参数：`input.parameters.{negative_prompt, size, n(1-6), prompt_extend, watermark, seed}`
- size：`"宽*高"`（**用星号**，如 `2048*2048`，与 A/B 的 `x` 不同！**无分辨率档位**）
- 返回：`output[].content[].text`（图片URL字符串）

**协议族 D — Gemini 原生（gemini-3.1-flash-image）**
- 端点：`POST /v1beta/models/gemini-3.1-flash-image:generateContent`，认证头 **`x-goog-api-key`**
- 提示词+图：`contents[].parts[]`，文本用 `{text}`，图片用 `{inline_data:{mime_type,data}}`
- **大小写差异（易错）**：请求用 snake_case（`inline_data`/`mime_type`），响应用 camelCase（`inlineData`/`mimeType`）
- **多轮改图**：把历史 user/model 轮次按序放入 `contents`；model 轮必须同时带
  `inline_data`(图片base64) **和** `thoughtSignature`(签名)，二者缺一不可
- 尺寸：`generationConfig.imageConfig.{aspectRatio:"16:9", imageSize:"2K"}`
  （aspectRatio 取值 `"1:1"/"2:3"/"3:2"/"3:4"/"4:3"/"4:5"/"5:4"/"9:16"/"16:9"/"21:9"`；
  Lite 模型 imageSize 硬限 1K）
- 返回：`candidates[].content.parts[]` → `inlineData.{mimeType,data}`(base64)。
  **注意：无 `fileData.fileUri` 路径**（官方文档只给 inlineData base64）

#### 4.2.4 `ImageGenerator` 统一接口（抹平四族差异）

```python
class GeneratedImage(BaseModel):
    image_path: Path          # ✅ 统一落盘为文件路径（见 §3.4，不用 base64）
    model: str                # 实际用的模型（含 ssvip 后缀）
    endpoint: str             # 实际端点（generations/edits/responses/generateContent）
    meta: dict                # 完整请求参数快照，便于复现与归因（写入 Memory）

class GenRequest(BaseModel):
    # —— 统一入参，上层 agent 无需知道协议族 ——
    prompt: str               # 提示词/编辑指令
    size: SizeSpec            # {ratio: "16:9"} 或 {pixels: (1024,1024)} 或 {tier: "2K"}
    reference_images: list[Path] = []     # 非空 → 风格锚定（走 edits/responses 多图）
    conversation_history: list[Path] = [] # 非空 → Gemini 多轮改图（需带历史图+签名）
    model_hint: ModelFamily | None = None # 偏好；None 时适配层自动选（见路由规则）
    quality: str = "high"
    n: int = 1

class ImageGenerator(Protocol):
    def generate(self, req: GenRequest) -> GeneratedImage: ...
```

**适配层路由规则（核心）：**

```
按 "任务模式" + model_hint 选协议族：
 ├─ 多轮迭代改图（conversation_history 非空）→ 协议族 D (Gemini)，唯一支持
 ├─ 参考图风格迁移（reference_images 非空）→ 协议族 A(gpt-image-2) 或 B(seedream)
 │       优先 B（JSON、易异步、多图融合强）；A 作高质量备选
 └─ 纯文生图（无参考图）→ 协议族 A/B/C 皆可
        按性价比/文字渲染需求选；qwen(C) 文字渲染强，seedream(B) 性价比高
```

每个协议族一个 dispatcher，内部把统一 `GenRequest` 翻译成该族的请求体：
- `size` 统一格式 → 各族格式（`x` / `*` / tier / aspectRatio）
- `reference_images`（文件路径）→ 各族要求（A 读文件做 multipart；B/D 读文件转 base64 data URI）
- `GeneratedImage.image_path` 统一落盘，各族返回的 url/b64 都先解码存盘再返回路径

#### 4.2.5 为什么这样设计支撑自我迭代

- Generator 只面对 `GenRequest`，**完全不知道四族协议差异**；
- Critic 的"风格漂移"经验可指向策略级修正（"该用 seedream 多图融合而非纯文生图"），
  而非 API 细节——经验跨模型可复用；
- 新模型加入 = 在对应协议族 dispatcher 里加一行模型名 + 一个 adapter，上层零改动。

### 4.3 评判器（Critic）方案 —— 对抗压力的来源

- **多模态打分**：经 dmxapi 调 Gemini 2.x 看图，按 §2 的三个维度评分 + 给自然语言批评；
- **多 judge 投票（可选）**：同一张图让 Gemini + GPT-4o 各打一次，取均值，降低单一模型偏见；
- **Prompt 评判模板**：固定评分 rubric（0-10 各维度 + 扣分项 + 改进建议），保证跨轮可比；
- **与生成异模型**：Critic 用 Gemini 系，生成用 gpt-image/flux/seedream，天然异构降偏差。

---

## 5. 模块边界（为后续实现定接口）

```
src/img_iter_agent/
├── agents/
│   ├── generator.py      # Generator: 内容采样 + prompt 构造 + 调生图
│   ├── critic.py         # Critic: 多模态打分 + 批评
│   └── summarizer.py     # Summarizer: 跨轮经验归纳（写 lesson 字段）
├── pipeline/
│   ├── graph.py          # LangGraph 闭环定义（节点=agents, 边=数据流，含条件边做收敛判定）
│   └── state.py          # 跨轮 State（TypedDict）：images/scores/round；checkpoint 持久化便于观测/恢复
├── generation/
│   ├── base.py           # ImageGenerator 抽象接口 + GenRequest/GeneratedImage
│   ├── client.py         # dmxapi 底层 HTTP 客户端（统一认证/超时/重试）
│   ├── router.py         # 按"任务模式+model_hint"路由到协议族 dispatcher
│   └── protocols/        # 按协议族分 dispatcher（核心：抹平四族差异）
│       ├── family_a_openai.py   # A: gpt-image-2 — generations(JSON) + edits(multipart)
│       ├── family_b_doubao.py   # B: seedream-5.0 — /v1/responses, image=URL/base64
│       ├── family_c_qwen.py     # C: qwen-image-2.0 — /v1/responses, input.messages 嵌套
│       └── family_d_gemini.py   # D: gemini-3.1-flash — generateContent, 多轮改图
├── style/
│   ├── anchor.py         # 参考图封装（风格锚定的唯一来源）
│   └── metrics.py        # CLIP 风格一致性 / 内容多样性指标
├── memory/
│   ├── index.py          # index.json 读写：追加尝试记录、按 model/mode/scores/tags 过滤召回
│   ├── schema.py         # 尝试记录 Pydantic model（§3.3.2 字段，含各种 _ref 链接）
│   ├── lessons.py        # 经验 MD 读写：Summarizer 写入、按链接读取注入
│   └── image_io.py       # §3.4：图片落盘 + 调用时按协议族要求把路径转 base64/multipart
├── data/                 # §3.5 数据层：三层 benchmarks/runs/analyses 的读写
│   ├── benchmark.py      # 加载 benchmarks/<id>/manifest.json + samples（用户定制的基准）
│   ├── trajectory.py     # trajectory.jsonl 追加/读取：完整训练轨迹（可复用重放）
│   ├── runstore.py       # runs/<id>/ 目录管理（meta/index/lessons/out/human_scores）
│   └── weights.py        # 读 benchmark 维度定义；算加权还原度总分
├── calibration/          # §2.6 评分校准闭环（离线）
│   ├── fit_weights.py    # 回归求最优维度权重（人工分 vs 初评分，sklearn）
│   └── report.py         # 生成 analyses/weight_calibration/ 校准报告
├── analysis/             # §3.5 analyses 层工具（只读）
│   └── strategy_compare.py # 读多 run 的 trajectory 做策略对比（pandas+matplotlib）
├── config.py             # pydantic-settings（DMXAPI key/host、model_id 由用户填）
└── cli.py                # 入口：run(闭环A) / calibrate(闭环B) / analyze(离线)
```

**关键解耦**：`ImageGenerator` 接口屏蔽 dmxapi 四个协议族差异；agent 只面对 `GenRequest`
（带不带参考图/历史图），不知道哪个协议族怎么传图。新增模型 = 在对应 `protocols/family_x.py`
里加一行模型名。风格锚定（G1）走 `reference_images` 非空的路径；图片全程用文件路径（§3.4）。

---

## 6. 风险与待决问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| **dmxapi 多协议族差异大**（认证头/端点/字段嵌套都不同） | 高 | 适配层按协议族分 dispatcher（§4.2），统一 `GenRequest` 入参屏蔽差异 |
| **参考图编辑可控性 < 本地 IP-Adapter** | 高 | 多图参考融合（seedream 多图/gpt 多 image）；强约束 edit_instruction；Critic 闭环纠偏 |
| **Gemini 多轮改图依赖 thoughtSignature**（易丢失） | 中 | 把签名随历史图一起落盘记录；签名失效则重置对话 |
| Critic 自我欺骗（同模型自评自生成） | 中 | 多 judge 投票；Critic(Gemini) 与生成(gpt/seedream) 异协议族天然异模型 |
| Memory JSON 膨胀 | 低 | 现阶段记录量小；图片用路径不存 base64（§3.4）；量大后再升级存储 |
| 成本失控（多轮 × 多图 × 多模态评判） | 中 | 每轮 batch 小化、judge 数量可配、加预算上限熔断；seedream 性价比优先 |
| qwen-image 等模型 edits 能力未实测 | 低 | 写 dispatcher 时先做能力探测，不确定就降级到文生图 |

---

## 7. 开放问题（待用户进一步拍板）

主要决策已定（dmxapi 后端 / 全云端 / 参考图锚定 / 双层记忆 / 图片用路径 /
**评测分开独立 / 控制变量法 / 逐轮人工审批** / **还原度+多维评分校准** / **数据三层分离**）。剩余问题：

1. **各模型具体 dmxapi model id**：用户指定（不同版本价格不同）。当前候选：
   seedream-5.0-pro / gpt-image-2 / gemini-3.1-flash-image / qwen-image-2.0。
   做成 config，用户填 `model_id` 字段。
2. **首个 benchmark 待用户填料**（§3.6，框架已建）：往 `data/benchmarks/furniture_product_whitebg/samples/`
   放产品实物图 `target.png` + 写 `target.md`（结构/材质/颜色说明）+ `content_spec.json`。
   7 维度 + 权重已定，可后续被校准。
3. **预算/规模**：单次迭代大致预算上限？决定每轮 batch 与 judge 数量。
4. **qwen-image 能力待实测**：写 dispatcher 时探测其是否支持 edits/多图，再决定是否纳入。
5. **评判 provider**：Critic 默认 Gemini（经 dmxapi）；是否叠加第二 judge？

---

## 8. ADR（架构决策记录）

### ADR-001: 生图后端 = dmxapi 聚合 API（全云端，无本地 GPU）
- **状态**：✅ 已采纳（用户 2026-07-30 确认）
- **背景**：无本地 GPU；需接入多个生图模型。
- **决策**：统一用 dmxapi 作为唯一生图后端。**实测发现 dmxapi 并非单一端点**，而是按模型
  分 4 个协议族（A:OpenAI Images / B:豆包 Responses / C:Qwen Responses / D:Gemini 原生），
  认证头、端点、字段命名、提示词嵌套各不相同（§4.2）。不走 diffusers。
- **后果**：+一 key 多模型；−适配层须按协议族分 dispatcher 抹平差异。

### ADR-002: 风格锚定 = 纯参考图；主力 seedream-5.0 + gpt-image-2，Gemini 做多轮迭代
- **状态**：✅ 已采纳（用户 2026-07-30 确认；2026-07-30 按官方文档修正）
- **修正**：旧文档称 seedream-3.0 不支持图生图，但**现行主力 seedream 5.0 系列（pro/lite）
  全部支持图生图/多图融合**。风格迁移路径更宽。
- **决策**：风格锚定走"参考图非空"路径，主力 seedream-5.0-pro（B族，多图融合2~10张、JSON易异步）
  + gpt-image-2（A族，高质量）；gemini-3.1-flash-image（D族）做多轮对话改图作单模型内微迭代。
  各协议族参考图传法由 dispatcher 抹平。
- **后果**：+风格有图像级约束且路径多；−可控性仍 < 本地 IP-Adapter，靠 Critic 闭环 + 多图参考补救。

### ADR-003: Agent 编排用 LangGraph（闭环原生 + 可观测），弃用 pi sdk
- **状态**：✅ 已采纳（用户 2026-07-30 确认；曾短暂选 PydanticAI，用户改回；context7 官方文档核实 API）
- **背景**：曾考虑 pi agent sdk，但实测 **pi 官方 SDK 仅 Node.js（npm `@earendil-works/pi-coding-agent`），
  无 Python 版**（社区 Issue #4174 仍在请求），跨语言（Py+Node 子进程 JSON-RPC）成本高，排除。
  本项目主体是 Python（dmxapi 生图、Pillow、CLIP 皆 Py 生态）。
- **决策**：用 **LangGraph** 编排自迭代闭环。经 context7 核实官方 API（/websites/langchain_oss_python_langgraph），
  以下特性原生支持、正好覆盖本项目诉求：
  - **闭环收敛**：`add_conditional_edges("summarizer", route)`，`route(state)->Literal["generator",END]`
    按 `scores` 是否达标决定继续迭代还是停；
  - **状态累积**：`Annotated[list, operator.add]` reducer 累积跨轮记忆/分数/图引用；
  - **可观测**：`graph.get_graph().draw_mermaid_png()` 一行导出流程图；checkpoint 可逐轮回看状态；
  - **持久化/恢复**：`compile(checkpointer=...)`（开发用 `InMemorySaver`，持久化用 `SqliteSaver`），
    `thread_id=run_id`，`graph.invoke(None, config)` 从断点恢复；
  - **多 agent 作节点**：generator/critic/summarizer 各为一个 node，`OverallState` 流转。
  LLM 接入经 LangChain 的 OpenAI 兼容 client 指向 dmxapi。
- **后果**：+闭环/可观测/断点续跑开箱即用、官方文档扎实；−引入 langgraph 依赖（本项目可接受）；
  注意设 `recursion_limit` 防止收敛判定失效时无限循环。

### ADR-004: Memory = 双层载体：经验 MD + JSON 索引（不引向量库/SQLite）
- **状态**：✅ 已采纳（用户 2026-07-30 确认并细化）
- **决策**：记忆用双层，全部基于文件、互相用链接引用、内容零复制：
  - **经验写 Markdown**（`lessons/*.md`）——给人读、可丰富、Summarizer 归纳的产物；
  - **JSON（`index.json`）只做参数/版本/索引整理**——记录每次尝试的 model/mode/params/scores，
    并用文件链接（`*_ref`、`outputs`、`reference_images`、`lesson_refs`）指向图片与 MD；
  - **图片、MD、长 prompt/critique 都用文件链接**串起来，JSON/MD 内不存 base64 或长正文。
- **后果**：+人类可读可手改、无外部依赖、内容不重复；−检索靠 JSON 字段过滤（无语义检索），
  记录量大后再升级存储。

### ADR-005: 图片全程用文件路径，base64 仅调用时临时转换
- **状态**：✅ 已采纳（用户 2026-07-30 确认）
- **决策**：生图结果落盘，JSON 只记路径；调用 dmxapi 时按各协议族文档要求把路径自动转成
  multipart 文件（A族）/ base64 data URI（B/C族）/ inline_data（D族），用完即弃不持久化。
- **后果**：+人类可读、JSON 不膨胀、便于复现对比；−多一层 IO 转换（成本可忽略）。

### ADR-006: Critic 多 judge 投票，且与 Generator 异协议族
- **状态**：提案（待实现时确认）
- **决策**：Critic 默认 Gemini 多模态（经 dmxapi）；重要场景叠加第二 judge；
  Generator 生图用 gpt-image/seedream，天然异协议族降偏差。
- **后果**：+评判更可信；−成本与延迟上升。

### ADR-007: 测试调度策略——分开独立评测 + 控制变量法 + 逐轮人工审批
- **状态**：✅ 已采纳（用户 2026-07-30 确认）
- **背景**：Generator 作为"测试调度者"，其系统提示词需先回答"怎么测"。多模型如何评测、
  经验如何区分、上下文如何组织、自回归怎么跑、何时停——都未定，写代码前必须厘清（§2.5）。
- **决策**（三条）：
  1. **分开独立评测**：每个模型各跑一条独立闭环，互不共享经验，公平横向对比各模型天花板。
     经验归属天然 = 当前模型，无需区分通用/特异。
  2. **控制变量法自回归**：每轮只变一个维度（prompt/reference_images/size/
     generation_mode/model_params），`test_variable`+`baseline_ref` 记录，分数变化可归因。
  3. **逐轮人工审批停止**：收敛不自动判定；每轮 `interrupt()` 暂停，人回复 continue/stop/调方向，
     `recursion_limit` 作硬上限兜底。
- **后果**：+经验干净可归因、停止可控、测试方法论清晰；−各模型从零学成本×N、人工每轮介入。
  Generator 系统提示词骨架见 §2.5.4。

### ADR-008: 目标=还原度；评分=混合评分(二分+连续)+排序校准双闭环
- **状态**：✅ 已采纳（用户 2026-07-30 确认，经两轮修正定稿）
- **背景**：用户明确系统目标是**还原度**（生成图与目标的多维吻合），非泛泛"风格一致"。
  评分机制经历两轮修正：① 全量化打分不稳定（LLM 不擅长回归、不可复现）→ 改二分；
  ② 全二分丢渐变信息（材质/颜色是连续的）→ 最终定为**混合评分**。校准信号也从"人工逐项打分回归"
  改为"**人工对 trace 排序**"——人擅长排序不擅长猜绝对分。
- **决策**：
  1. Critic 按 benchmark `score_dimensions` 的 `scoring_type` **分派评分**：客观维度逐项二分 ✓/✗→通过率；
     渐变维度(材质/颜色)给 LLM 连续 0-1 分；统一成 features∈[0,1]，还原度=w·features（§2.6.1）；
  2. 人工**异步复评**并与 Critic 初评配对作监督信号（§2.6.2）；
  3. 攒批后用**人工对 trace 的排序**做 **learning-to-rank 拟合权重 w**（约束 Σw=1,w≥0），
     天然修正连续分偏差，回灌 Critic 算总分的权重（§2.6.3）。
- **后果**：+评判从固定走向自我校准、人只做擅长的排序、连续分偏差被权重吸收；
  −需积累人工排序才能校准、离线流程需额外实现（sklearn rank fitting）。
  **注**：维度定义的真源是各 benchmark 的 `manifest.json`（含 scoring_type/weight_init/check_items）。

### ADR-009: 数据三层分离 benchmarks/runs/analyses + trajectory.jsonl 可复用
- **状态**：✅ 已采纳（用户 2026-07-30 确认）
- **背景**：用户要求——benchmark 标准答案要方便接入定制、经验要能验证泛化性、产出物要可复用
  做轨迹分析/策略对比。三者本质都是"产出物须是结构化可独立加载的数据集"。
- **决策**：
  1. **三层目录**（§3.5.1）：`benchmarks/`(用户定制基准,跨run复用) / `runs/`(完整轨迹产出) /
     `analyses/`(只读分析,不污染原始)；benchmark 与 run 解耦，一个 benchmark 可被多 run 复用。
  2. **trajectory.jsonl** 为头等公民：每轮一行、自包含、可脱离运行时独立加载重放，
     支撑策略对比/校准/泛化分析；泛化分析当前不做但结构留好。
  3. benchmark "标准答案" = 目标参考图 + 验收 checklist + 多维评分定义（用户准备/定制）。
- **后果**：+产出可复用、原始不可变可复现、定制接入简单；−目录规范需遵守、分析脚本需另写。

---

## 9. 参考资料（dmxapi 官方文档实测，doc.dmxapi.cn）

**说明**：初版曾误读第三方说明站 `imagemodels.dmxapi.com`（信息过时，如称 seedream-3.0 不支持图生图）。
以下结论均来自 **官方文档站 doc.dmxapi.cn**（经 ego 浏览器抓取，2026-07-30）。

官方文档把**每个模型的每种能力拆成独立页面**，关键页（已抓取验证）：

- gpt-image-2 文生图：`gpt-image-2-text-to-image.html` → 协议族 A，`/v1/images/generations`，JSON
- gpt-image-2 图片编辑：`qwen-image-2.0-image-editing.html`（页内实际为 gpt-image-2）→ 协议族 A，`/v1/images/edits`，multipart
- seedream-5.0-pro 图生图：`doubao-seedream-5-0-pro-260628-image-to-image.html` → 协议族 B，`/v1/responses`
- seedream-5.0-pro 多图融合：`doubao-seedream-5-0-pro-260628-multi-image-fusion.html` → 2~10 张参考图
- seedream-5.0-lite 图片编辑：`doubao-seedream-5.0-lite-img-edit.html` → 最多 14 张参考图 + 组图
- qwen-image-2.0 文生图：`qwen-image-2.0-text-to-image.html` → 协议族 C，`/v1/responses`，`input.messages` 嵌套
- gemini-3.1-flash-image 多轮改图：`gemini-3.1-flash-image-preview-duolun.html` → 协议族 D，`generateContent`

**实测要点（详见 §4.2.3）：**
- 协议族 A（gpt-image-2）：`Authorization: Bearer`；编辑用 multipart，参考图字段 `image`（多个同名）
- 协议族 B（seedream-5.0）：`Authorization: <key>`（**无 Bearer**）；JSON；`image`=URL 或 base64 data URI
- 协议族 C（qwen-image-2.0）：同 B 端点但 `input.messages[].content[].text` 嵌套；size 用**星号** `2048*2048`
- 协议族 D（gemini）：`x-goog-api-key`；`contents[].parts[]`；多轮需带 `inline_data` + `thoughtSignature`
- **现行主力 seedream 5.0 全系列支持图生图/多图融合**（旧文档的 seedream-3.0 已淘汰）

> 官方文档站存在 VitePress 客户端路由问题：跨页导航时正文会短暂显示上一页内容。
> 抓取时已通过 H1+URL 一致性校验规避；实现时建议每个 adapter 写集成测试实际打一次接口确认。
