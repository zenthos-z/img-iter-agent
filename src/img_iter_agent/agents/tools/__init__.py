"""deepagent 工具注册中心（策略扩展点）。

Generator/Critic 的 deepagent 工具按每轮上下文构建（闭包捕获 sample/run_dir/router 等）。
新增策略/能力 = 在对应模块加一个 `make_*_tool` 工厂并在 generate_round/evaluate 注册。
"""

from .critic_tools import make_critic_tools, make_query_rubric_tool
from .generator_tools import (
    make_generate_image_tool,
    make_generator_tools,
    make_query_experience_tool,
)

__all__ = [
    "make_critic_tools",
    "make_generate_image_tool",
    "make_generator_tools",
    "make_query_experience_tool",
    "make_query_rubric_tool",
]
