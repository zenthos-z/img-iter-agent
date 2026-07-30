# img-iter-agent

> 一个自我迭代的 AI 生图 Agent：通过「生成 → 对抗评判 → 总结 → 优化」的闭环，
> 自动产出**风格元素一致、内容可变**的图像集合。

## 它解决什么问题

给定一种目标风格（几张参考图 / 一段风格描述），系统自动生成一组新图：

- **风格元素保持相同**——配色、笔触、构图语言、材质、氛围稳定可复现；
- **内容可以不一样**——主体、场景、主题在风格约束下自由变化；
- **过程自我进化**——每一轮由「评判器」打分、「总结器」归纳经验、「提示词工程师」据此改进下一轮，
  生成质量随迭代单调上升（理想情况下）。

核心思想是把 GAN 的「生成—对抗」搬到了 **Agent 层面**：不是训练两个神经网络，
而是让两个 LLM agent（生成方 vs 评判方）在「总结出的经验库」上博弈与自我改进。

## 架构与技术栈

完整分析见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

一句话概览：Python · dmxapi 多模型生图（实测 4 协议族，按族分 dispatcher）· 目标=还原度（多维评分）·
LangGraph 自迭代闭环（控制变量法+逐轮人工审批）· 评分校准闭环（人工复评→回归最优权重→校准 agent）·
数据三层管理（benchmarks/runs/analyses + trajectory.jsonl，产出可复用）· 双层记忆（经验 MD+JSON 索引）· 图片全程用文件路径。

## 快速开始（待实现）

```bash
# 待技术栈定稿、代码落地后补全
uv sync            # 或 pip install -e .
cp .env.example .env  # 填入 API key
python -m img_iter_agent  # 跑一个迭代
```

## 状态

✅ 架构与技术栈已定稿（dmxapi 后端 / 全云端 / 参考图锚定风格）。见 `docs/ARCHITECTURE.md`。
🚧 代码骨架与实现待推进。

## 目录与素材管理

**核心规则：素材按"谁创建"分三类，各归其位，绝不混放。**

```
img-iter-agent/
├── docs/ARCHITECTURE.md       # 架构与技术栈分析（核心文档）
├── src/img_iter_agent/        # 源码（待实现）
├── tests/
└── data/
    │
    ├── benchmarks/            # 〔你准备的考题素材，✅入git，跨run复用〕
    │   └── <bench_id>/        #   一个 benchmark = 一组考题
    │       ├── manifest.json  #   评分维度+权重（如家具7维度）
    │       ├── rubric.md      #   评分细则
    │       └── samples/       #   ★你的考题素材放这里
    │           └── sNNN/
    │               ├── target.png       # 产品实物参考图（对比锚）
    │               ├── target.md        # 结构/材质/颜色说明+验收点
    │               └── content_spec.json# 要生成什么（视角/背景/约束）
    │
    ├── runs/                  # 〔系统生成的产物，❌不入git，每次重生成〕
    │   └── <run_id>/          #   = <model>__<bench>__<时间>
    │       ├── trajectory.jsonl  # 完整训练轨迹（重放/分析的依据）
    │       ├── out/             # 生成的图
    │       ├── lessons/         # 归纳的经验MD
    │       ├── human_scores/    # 人工评分（异步补）
    │       └── ...
    │
    └── analyses/              # 〔离线分析产物，❌不入git，可重算〕
        └── strategy_compare/    # 策略对比报告等
```

**我（用户）要往哪放素材？**
- 准备/修改考题 → `data/benchmarks/<bench>/samples/`（这是你唯一需要手动管理素材的地方）
- 系统跑出来的东西 → 自动进 `data/runs/`，不用管
- 没有全局素材库——每份考题自包含在自己的 `samples/` 里

**首个 benchmark 已建**：`data/benchmarks/furniture_product_whitebg/`（家具跨境电商白底产品图），
待往 `samples/` 放产品实物图即可。详见该目录下 `rubric.md`。

## License

MIT
