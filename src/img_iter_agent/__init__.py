"""img-iter-agent: 自我迭代的 AI 生图 agent 系统。

双层闭环（见 docs/ARCHITECTURE.md）：
  - 闭环 A：生成迭代（在线，逐轮人工审批）
  - 闭环 B：评分校准（离线，排序拟合权重）

评分机制：混合评分（二分 + 连续）+ 排序校准（learning-to-rank）。
"""

__version__ = "0.0.1"
