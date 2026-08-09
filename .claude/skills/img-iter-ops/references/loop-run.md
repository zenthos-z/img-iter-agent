# 启动 loop：路径选择 + loop_id 规范 + 续跑

## 三种启动路径，何时用哪个

| 路径 | 命令 | 何时用 |
|---|---|---|
| **自动脚本（推荐）** | `run_loop_auto.py` | 测试/批量：无人值守跑 N 轮，全新 loop_id 不污染，一条 trace |
| **CLI 交互** | `python -m img_iter_agent.cli run --bench B --sample sNNN` | 想逐轮人工审批（每轮 stdin 输 continue/stop/调方向） |
| **web UI** | 端口 8765，「启动 loop」按钮 + 自动连跑 rounds | 想边看 trace 边点审批；续跑已有 loop 最方便 |

## 推荐路径：run_loop_auto.py（本 skill 核心）

```bash
.venv/bin/python .claude/skills/img-iter-ops/scripts/run_loop_auto.py \
    --bench furniture_product_whitebg --sample s003 --rounds 6 --tag exp6
```

**它做的事**：
- 用全新 loop_id `<bench>-<sample>-<tag>` 新起 run（**不污染**默认 `<bench>-<sample>` 正式数据）
- 检测到 loop_id 已存在会**直接拒绝**（提示换 tag），绝不偷偷续跑
- `prompt_decision=None` 全自动 continue，无人值守跑满 N 轮
- 内部把「想要 N 轮」换算成底层 resume 次数（屏蔽 `run_loop_session(rounds=R)` 实际生成 R+1 轮的瑕疵），跑完校验 trajectory 行数 == N
- 作为一条 LangSmith trace（project=`img-iter-agent`）

**参数**：`--rounds`（想要几轮，默认 6）/ `--tag`（loop_id 后缀）/ `--loop-id`（显式覆盖）/ `--note`。

## loop_id 三条规范（踩坑总结）
1. **测试永远用全新 id**：`<bench>-<sample>-<tag>`（exp6/batch1/run2…），别用默认 `<bench>-<sample>`——那是「一题一 loop」的正式数据，续跑会叠加轮数、污染对比。
2. **正式迭代才用默认 id**：当确认要在这道题上长期迭代、想要累积轮次时，才用 `<bench>-<sample>`（或 web 一键启动，它就用默认 id）。
3. **一次测试一个 tag**：同 tag 重跑会被脚本拒绝（防覆盖）；想重跑换 tag（exp6→exp6b）。

## 续跑已有 loop
`run_loop_auto.py` 暂不支持续跑（底层 `run_loop_session` 续跑会重置轮次计数，有坑）。续跑走：
- **web UI**（最方便）：端口 8765，loop 详情页「继续下一轮」/「停止」；web 的 LoopRunner 正确处理了 END 态追加新轮。
- **CLI**：`python -m img_iter_agent.cli run --bench B --sample sNNN`（同 sample = 续跑），但每轮阻塞等 stdin。

## 监控运行
- **LangSmith**：每 loop 一条 trace，按 `loop_id` 聚合多轮；project=`img-iter-agent`。轮次在 config metadata 的 `round`/`phase`。
- **web UI**：loop 详情页看每轮 prompt/生成图/Critic 明细/经验沉淀面板。
- **跑完**：脚本末尾打印完整还原度曲线 + 提示用 `diagnose_loop.py` 诊断。

## 环境前提（跑前确认）
- `.venv/bin/python` 存在；`.env` 配好 `IMG_ITER_DMXAPI_KEY` + 生图/agent model_id
- 缺 key 会在首轮 LLM/出图时报错（SSL EOF / 401 等）；先确认 `.env` 完整
- 生图模型由 `store.meta.model` 决定（`--model` 覆盖；默认 `IMG_ITER_MODEL_SEEDREAM_PRO`），反查 family 路由
