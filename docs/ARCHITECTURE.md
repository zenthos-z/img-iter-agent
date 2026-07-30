# img-iter-agent 架构与技术栈分析

> 目标：建立「AI 生图自动迭代 → 风格元素相同、内容可变」的**自我迭代对抗总结** Agent 系统。
> 本文档是当前阶段的核心交付，确定系统形态、闭环机制，并对候选技术栈做选型分析与推荐。
>
> **关键决策（已定）**：生图后端 = **dmxapi 聚合 API**（实测分 4 个协议族：OpenAI Images / 豆包 Responses /
> Qwen Responses / Gemini 原生）；风格锚定 = **纯参考图**（主力 seedream-5.0 + gpt-image-2）；
> 记忆 = **双层载体：经验 MD + JSON 索引**（不引向量库，全用文件链接，§3.3）；
> 图片 = **文件路径**（不存 base64，§3.4）。
> 核心难点 = 屏蔽 dmxapi 四个协议族的接口差异（§4.2）。

---

## 0. 先把目标拆清楚

用户的原始描述里其实埋了三个互相约束的子目标，必须先分开，否则技术栈会打架：

| # | 子目标 | 含义 | 技术上对应什么 |
|---|--------|------|----------------|
| **G1** | **风格元素相同** | 配色、笔触、材质、构图语言、氛围在多张图之间稳定 | **以参考图为条件**，通过支持图生图的云端模型（gpt-image-1 / flux-kontext）做风格迁移锚定 |
| **G2** | **内容可变** | 主体、场景、主题在风格约束下自由变化 | 由 prompt 的「内容维度」驱动，并与风格条件解耦 |
| **G3** | **自动迭代 + 自我对抗总结** | 生成质量随轮次提升，系统自己归纳经验并改进 | Agent 闭环：生成 → 评判 → 总结 → 改进 |

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
                  │            风格条件（锚定 G1，见 §4.2）            │
                  │  参考图 → 各协议族风格迁移（seedream/gpt/gemini）    │
                  └───────────────────────┬─────────────────────────┘
                                          │ 稳定不变
                                          ▼
   ┌──────────┐   生成图    ┌──────────┐     评分+批评     ┌──────────┐
   │ Generator│ ──────────▶ │  Image   │ ───────────────▶ │  Critic  │
   │  Agent   │             │  Pool    │                  │  Agent   │
   │（改prompt│ ◀────────── │（本轮产物）│ ◀─── 取最好/最差 ───┤（多模态 │
   │+内容采样)│   经验反馈   └──────────┘                  │ 打分)    │
   └────┬─────┘             └─────▲────┘                  └────┬─────┘
        │                         │                            │ 打分维度+理由
        │   prompt 修改建议         │                           │
        ▼                         │                            ▼
   ┌──────────┐                   │            ┌───────────────────────────────┐
   │ Summarizer│                  │            │  Memory（双层载体，全用文件链接）│
   │  Agent    │──── 写经验 ──────┼───────────▶│  • 经验 = Markdown（给人读）   │
   │（归纳跨轮 │                  │            │  • 索引 = JSON（参数/版本整理） │
   │ 规律）    │                  │            │  • 素材(图/MD) 都用路径互链     │
   └──────────┘                  │            └───────────────┬───────────────┘
        │                        │                            │ 读经验MD + JSON筛选
        └────────────────────────┴────────────────────────────┘
                       每轮开始时注入 Generator/Critic
```

> **记忆设计原则（用户明确）：不引向量库。** 用**双层载体**：
> - **经验写 Markdown**——可丰富、给人读、Summarizer 自然语言归纳；
> - **JSON 只做参数/版本的整理索引**——记录每次尝试的 model/mode/params/scores，并指向素材文件；
> - **JSON 条目和经验 MD 互相用文件链接引用**，不复制内容（图片更是只存路径，见 §3.4）。

三个 agent 的职责切分（对应"生成对抗总结"四个字）：

- **Generator（生成方）**：负责 **G2 内容可变**。从内容空间采样新主题，读相关经验 MD +
  JSON 筛选出的高分先例，产出 prompt 并调用生图模型。
- **Critic（对抗评判方）**：多模态 LLM 看图打分。**关键设计：评判时同时给"风格一致性"和
  "内容质量"两个分**——前者强约束 G1，后者驱动 G2/G3。这一"对抗"压力倒逼 Generator 改进。
- **Summarizer（总结方）**：跨轮归纳。把零散的"Critic 评分理由"抽象成可复用的经验
  **写进 Markdown**（"用 XX 笔触描述时风格漂移，应改用 YY"），并在 JSON 索引里登记指向该 MD，
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

```
data/runs/<run_id>/
├── index.json            # ← 索引：每次生图尝试一条记录（参数/版本/打分/链接）
├── lessons/              # ← 经验库：每个经验一篇 Markdown
│   ├── 001-style-drift-on-multi-ref.md
│   └── 002-seedream-vs-gpt-color-matching.md
├── out/                  # ← 生成图（文件，路径被 index.json 引用）
│   ├── try_00017_1.png
│   └── ...
└── ref/                  # ← 本次用到的参考图副本（或软链到 data/reference）
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

## 证据（指向 index.json 的尝试记录）
- try_00012（2张参考）：style_consistency 8.7  ← keep
- try_00015（5张参考）：style_consistency 6.1  ← worst
- try_00017（2张参考）：style_consistency 8.5  ← keep

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

---

## 4. 技术栈选型（已按用户决策收敛）

### 4.1 选型总表

| 层 | 决策 | 说明 |
|----|------|------|
| **语言** | **Python ≥ 3.11** | 生图、agent、向量库生态全在 Py |
| **生图后端** | **dmxapi 聚合 API（单一后端）** | 统一接入 gpt-image-2 / seedream-5.0 / qwen-image / gemini-image 等；不走本地 diffusers（无 GPU）。注意：dmxapi 按**协议族**分多套接口（§4.2），非单一端点 |
| **多模态评判(Critic)** | **dmxapi 接 Gemini 多模态**（可选叠加 GPT 多 judge） | 评分+批评质量高、价格好；评判/生成尽量异模型降偏差 |
| **Agent 编排** | **LangGraph** | 环路(闭环)是一等公民，状态/检查点/可视化原生支持 |
| **LLM 接口统一** | **dmxapi 已聚合**（文本对话走 chat completions） | 评判/总结用多模态文本模型 |
| **结构化输出** | **Pydantic v2** | Critic 评分、生图记录需严格 schema |
| **记忆存储** | **双层载体：MD(经验) + JSON(索引)（按用户要求）** | 不引向量库/SQLite；经验写 MD 给人读，JSON 只管参数/版本/索引并用文件链接串起图片与 MD（§3.3）；人类可读可手改 |
| **图像存储** | **文件路径（不用 base64）** | §3.4：生图落盘，JSON 只记路径；调用时适配层按文档自动转格式 |
| **图像/特征** | **Pillow + OpenCLIP** | 算 style_consistency / content_diversity 指标用 CLIP 特征 |
| **配置/密钥** | **pydantic-settings + .env** | 类型安全、12-factor；DMXAPI key 走环境变量 |
| **异步/并发** | **asyncio + httpx** | 批量生图、批量评判是 IO 密集；httpx 异步上传 multipart |
| **重试/限流** | **tenacity** | API 调用必须容错（生图偶发超时/失败） |
| **可观测** | **logging + run 目录** | 先做最小可观测，不引入外部依赖 |
| **实验追踪(可选)** | **run 目录 + index.json 起步** | 自迭代本身就有完整 run 记录（§3.3.1 布局） |

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
- 提示词嵌套：`input.messages[].content[].text`（仅单轮、仅一个 text）
- 参数：`input.parameters.{negative_prompt, size, n(1-6), prompt_extend, watermark, seed}`
- size：`"宽*高"`（**用星号**，如 `2048*2048`，与 A/B 的 `x` 不同！）
- 返回：`output[].content[].text`（图片URL字符串）

**协议族 D — Gemini 原生（gemini-3.1-flash-image）**
- 端点：`POST /v1beta/models/gemini-3.1-flash-image:generateContent`，认证头 **`x-goog-api-key`**
- 提示词+图：`contents[].parts[]`，文本用 `{text}`，图片用 `{inline_data:{mime_type,data}}`
- **多轮改图**：把历史 user/model 轮次按序放入 `contents`；model 轮必须同时带
  `inline_data`(图片base64) **和** `thoughtSignature`(签名)，二者缺一不可
- 尺寸：`generationConfig.imageConfig.{aspectRatio:"16:9", imageSize:"2K"}`
- 返回：`candidates[].content.parts[]`：`inlineData.{mimeType,data}`(base64) 或 `fileData.fileUri`(url)

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
│   ├── graph.py          # LangGraph 闭环定义（节点=agents, 边=数据流）
│   └── state.py          # 跨轮 State（TypedDict）：images/scores/round
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
├── config.py             # pydantic-settings（含 DMXAPI key/host）
└── cli.py                # 入口
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

主要决策已定（dmxapi 后端 / 全云端 / 参考图锚定 / JSON 记忆 / 图片用路径）。剩余问题：

1. **风格迁移主力模型**：seedream-5.0-pro（JSON、多图融合强、性价比）还是 gpt-image-2（高质量）？
   建议 seedream 为主、gpt 为高质量备选。Gemini 多轮改图作为"单模型内微迭代"补充。
2. **预算/规模**：单次迭代大致预算上限？决定每轮 batch 与 judge 数量、主模型选择。
3. **qwen-image 能力待实测**：写 dispatcher 时探测其是否支持 edits/多图。
4. **评判 provider**：Critic 默认 Gemini（经 dmxapi）；是否叠加第二 judge？

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

### ADR-003: 用 LangGraph 编排闭环，State 跨轮持久化
- **状态**：提案（待实现时确认）
- **决策**：闭环用 LangGraph 的 cyclic graph；State(TypedDict) 含 images/scores/round，
  每轮 checkpoint，支持中断恢复。
- **后果**：+原生支持环、可视化、断点续跑；−引入 langgraph 依赖。

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
