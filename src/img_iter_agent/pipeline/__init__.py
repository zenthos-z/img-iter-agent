"""pipeline：LangGraph 闭环 A 的编排。

  state.py  跨轮 State（TypedDict，含 operator.add 累加器）
  graph.py  4 节点（generator/critic/summarizer/human_review）+ interrupt + 条件边
"""
