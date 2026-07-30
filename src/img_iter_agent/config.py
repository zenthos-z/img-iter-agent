"""全局配置：dmxapi 凭证/地址、生图 model_id、agent LLM model_id、数据根目录。

用 pydantic-settings 从环境变量 / `.env` 读取。所有字段都给默认值或允许为空——
这样**基础层（无网络、无 key）的测试与脚本可在不配置任何密钥的情况下运行**。
DMXAPI key 仅在 Step 2（Critic 真正调 LLM）及之后才需要。

目录约定见 `docs/ARCHITECTURE.md §3.5`：
  <data_root>/benchmarks/<bench_id>/   — 用户准备的基准（提交进 git）
  <data_root>/runs/<run_id>/           — 某次运行的完整产出（git 忽略，仅留 .gitkeep）
  <data_root>/analyses/                — 只读分析产出（git 忽略）
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Agent LLM（及生图）走 dmxapi 的协议族。openai=OpenAI 兼容端点（默认，最通用）；
# gemini/doubao/qwen=对应原生族；claude=Anthropic 兼容端点。各族的端点/认证/请求体差异
# 由后续的 LLM client（Step 2 Critic 起）按此字段分派。
AgentProtocol = Literal["openai", "gemini", "doubao", "qwen", "claude"]

# 默认数据根目录 = 项目下的 data/
# 本文件位于 <project>/src/img_iter_agent/config.py，向上 parents[2] 即项目根。
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


class Settings(BaseSettings):
    """运行配置。

    所有字段都有默认值或可空，故 `Settings()` 在无 `.env` 时也能构造——
    这是基础层测试不依赖任何密钥的前提。
    """

    model_config = SettingsConfigDict(
        env_prefix="IMG_ITER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 数据目录 ---
    data_root: Path = Field(default=_DEFAULT_DATA_ROOT, description="三层数据的根目录")

    # --- dmxapi 凭证 / 地址（Step 3 真正生图时才用到） ---
    dmxapi_host: str = Field(default="https://api.dmxapi.cn", description="dmxapi 网关地址")
    dmxapi_key: str = Field(default="", description="dmxapi 密钥；为空表示未配置（基础层测试无需）")

    # --- 四个协议族的「生图」model_id（用户后填；不同模型价格不同，做成配置而非写死） ---
    # 见 docs/ARCHITECTURE.md §4.2。family A=OpenAI Images, B=豆包 Responses,
    # C=Qwen Responses, D=Gemini native。
    model_seedream_pro: str = Field(default="", description="Family B: seedream-5.0-pro 的 dmxapi model_id")
    model_gpt_image: str = Field(default="", description="Family A: gpt-image-2 的 dmxapi model_id")
    model_gemini_image: str = Field(default="", description="Family D: gemini-3.1-flash-image 的 dmxapi model_id")
    model_qwen_image: str = Field(default="", description="Family C: qwen-image-2.0 的 dmxapi model_id")

    # --- Agent LLM（用户决定每个 agent 用哪个模型，见 ADR-006）---
    # Critic 看图打分（必须多模态）；Generator 构造 prompt、Summarizer 写经验（文本即可）。
    # 三者各自独立配置 model_id + protocol，因为不同模型系列走 dmxapi 的不同端点/认证/请求体
    # （生图四协议族同样适用于 agent LLM 的 chat/多模态端点）。
    # 默认 protocol=openai：dmxapi 对 OpenAI 兼容端点支持最广（/v1/chat/completions, Bearer 鉴权）。
    # 若用 Gemini/Claude/豆包 系列模型，把对应 protocol 改成对应族。
    critic_model: str = Field(default="", description="Critic 的 dmxapi model_id（必须多模态）")
    critic_protocol: AgentProtocol = Field(
        default="openai", description="Critic LLM 走的协议族（openai/gemini/doubao/claude）"
    )
    generator_model: str = Field(default="", description="Generator 的 dmxapi model_id（文本即可）")
    generator_protocol: AgentProtocol = Field(
        default="openai", description="Generator LLM 走的协议族"
    )
    summarizer_model: str = Field(default="", description="Summarizer 的 dmxapi model_id（文本即可）")
    summarizer_protocol: AgentProtocol = Field(
        default="openai", description="Summarizer LLM 走的协议族"
    )

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_root / "benchmarks"

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def analyses_dir(self) -> Path:
        return self.data_root / "analyses"

    def benchmark_dir(self, bench_id: str) -> Path:
        return self.benchmarks_dir / bench_id

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id


# 单例：模块级共享一份配置。测试若要隔离可自行构造 Settings(...)。
def get_settings() -> Settings:
    """返回全局 Settings 单例。"""
    return Settings()
