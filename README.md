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

## 目录

```
img-iter-agent/
├── docs/ARCHITECTURE.md   # 架构与技术栈分析（当前核心交付）
├── src/img_iter_agent/    # 源码（待实现）
├── data/
│   ├── reference/         # 风格参考图（输入）
│   ├── outputs/           # 生成结果（不入 git）
│   └── runs/              # 迭代运行记录（不入 git）
└── tests/
```

## License

MIT
