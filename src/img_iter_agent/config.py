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
    # 注意：网关域名是 www.dmxapi.cn（api.dmxapi.cn 的证书已失效/主机名不匹配，勿用）。
    dmxapi_host: str = Field(default="https://www.dmxapi.cn", description="dmxapi 网关地址")
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
    # Generator/Critic 改造为 deepagent 后，由 ChatOpenAI 指向 dmxapi 的 OpenAI 兼容端点
    # （/v1/chat/completions, Bearer 鉴权，支持 tool-calling）；模型 id 在此配置。
    critic_model: str = Field(default="", description="Critic 的 dmxapi model_id（必须多模态）")
    generator_model: str = Field(default="", description="Generator 的 dmxapi model_id（文本即可）")
    summarizer_model: str = Field(default="", description="Summarizer 的 dmxapi model_id（文本即可）")

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
_settings: Settings | None = None


def get_settings() -> Settings:
    """返回全局 Settings 单例。"""
    global _settings
    if _settings is None:
        _settings = Settings()
        # LangSmith SDK 读 os.environ（非 pydantic 字段），此处从 .env 同步过去。
        # 首次构造 settings 时注入一次，确保 tracing 在任何 LLM 调用前生效。
        _sync_langsmith_env(_settings)
        # 预热 langsmith cached client 挂图片压缩（必须在任何 tracing 前调一次：
        # get_cached_client 是单例，仅首次 init_kwargs 生效）。
        _warm_langsmith_client()
    return _settings


def _warm_langsmith_client() -> None:
    """预热 langsmith cached client，挂上 compress_images_in_trace。

    所有 trace（含 langchain 自动 trace 的带图 LLM 调用）上传前会过这个压缩函数，把大 data-URI 图
    缩成小图——**模型仍看 critic 消息里的高清原图**，只有上报到 LangSmith 的副本被压（避免带图 trace
    撑到十几 MB 上传超时）。tracing 未开启则跳过。预热失败不阻断（退回无压缩 trace）。
    """
    import os

    tracing_on = (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    if not tracing_on:
        return
    try:
        from langsmith.run_trees import get_cached_client

        from .generation.image_io import compress_images_in_trace
        get_cached_client(
            hide_inputs=compress_images_in_trace,
            hide_outputs=compress_images_in_trace,
        )
    except Exception:  # noqa: BLE001  预热失败不阻断
        pass


def _sync_langsmith_env(_unused: Settings = None) -> None:
    """把 LangSmith 配置写入 os.environ（SDK 直读环境变量，不走 pydantic）。

    用 python-dotenv 读 .env 里 LANGSMITH_* / LANGCHAIN_TRACING_* 系列变量，
    仅在尚未设置时注入（不覆盖调用方显式传入的环境变量）。
    """
    import os

    keys = (
        "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT",
    )
    # 先尝试从 .env 读取（uvicorn 进程未必加载了 .env 到 os.environ）
    env_vals: dict[str, str] = {}
    try:
        from pathlib import Path

        env_path = Path(_DEFAULT_DATA_ROOT).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k in keys:
                    env_vals[k] = v.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001  .env 读取失败不阻断启动
        pass
    for k in keys:
        if not os.environ.get(k) and env_vals.get(k):
            os.environ[k] = env_vals[k]
