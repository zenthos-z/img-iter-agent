"""全局配置：dmxapi 凭证/地址、各协议族 model_id、数据根目录。

用 pydantic-settings 从环境变量 / `.env` 读取。所有字段都给默认值或允许为空——
这样**基础层（无网络、无 key）的测试与脚本可在不配置任何密钥的情况下运行**。
DMXAPI key 仅在 Step 3（真正调用生成模型）时才需要。

目录约定见 `docs/ARCHITECTURE.md §3.5`：
  <data_root>/benchmarks/<bench_id>/   — 用户准备的基准（提交进 git）
  <data_root>/runs/<run_id>/           — 某次运行的完整产出（git 忽略，仅留 .gitkeep）
  <data_root>/analyses/                — 只读分析产出（git 忽略）
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # --- 四个协议族的 model_id（用户后填；不同模型价格不同，做成配置而非写死） ---
    # 见 docs/ARCHITECTURE.md §4.2。family A=OpenAI Images, B=豆包 Responses,
    # C=Qwen Responses, D=Gemini native。
    model_seedream_pro: str = Field(default="", description="Family B: seedream-5.0-pro 的 dmxapi model_id")
    model_gpt_image: str = Field(default="", description="Family A: gpt-image-2 的 dmxapi model_id")
    model_gemini_image: str = Field(default="", description="Family D: gemini-3.1-flash-image 的 dmxapi model_id")
    model_qwen_image: str = Field(default="", description="Family C: qwen-image-2.0 的 dmxapi model_id")

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
