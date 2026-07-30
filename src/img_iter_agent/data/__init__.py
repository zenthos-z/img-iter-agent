"""数据层：三层数据（benchmarks/runs/analyses）的加载与轨迹读写。

  - benchmark.py   加载 benchmark manifest + 各考题 content_spec
  - weights.py     features 向量 + 加权还原度 + 校准权重加载（纯函数）
  - trajectory.py  trajectory.jsonl 读写（头等公民，可重放）
  - runstore.py    runs/<id>/ 目录管理
"""
