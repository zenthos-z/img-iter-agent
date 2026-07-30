"""加载 benchmark 与单道考题。

benchmark 目录结构（ARCH §3.5）：
  benchmarks/<bench_id>/manifest.json          # Benchmark
  benchmarks/<bench_id>/samples/<sample_id>/
      target.jpg          # 对比型维度的参考锚图
      target.md           # 产品说明 + 还原要点
      content_spec.json   # 任务 + 约束 + 各维度 checklist

加载后的 sample 同时带「相对 bench 目录的 target 路径」和「绝对磁盘路径」，方便后续喂图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, get_settings
from ..memory.schema import Benchmark, ContentSpec


@dataclass(frozen=True)
class Sample:
    """一道加载好的考题：spec + target 的绝对路径。"""

    spec: ContentSpec
    target_path: Path  # 磁盘绝对路径（对比型维度的参考锚图）
    target_md_path: Path  # 产品说明 MD（可选，可能不存在）
    sample_dir: Path  # samples/<sample_id>/ 绝对路径


@dataclass(frozen=True)
class LoadedBenchmark:
    """一个加载好的 benchmark：manifest + 目录根 + 各样例。"""

    bench: Benchmark
    bench_dir: Path
    samples: dict[str, Sample]

    def sample(self, sample_id: str) -> Sample:
        if sample_id not in self.samples:
            raise KeyError(f"sample {sample_id!r} not in benchmark {self.bench.bench_id!r}")
        return self.samples[sample_id]


def _bench_dir_from(settings: Settings | None, bench_id: str) -> Path:
    s = settings or get_settings()
    return s.benchmark_dir(bench_id)


def load_benchmark(
    bench_id: str, *, settings: Settings | None = None
) -> LoadedBenchmark:
    """加载整个 benchmark：读 manifest.json，再逐个读 samples。"""
    bench_dir = _bench_dir_from(settings, bench_id)
    manifest_path = bench_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")

    bench = Benchmark.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    samples: dict[str, Sample] = {}
    for ref in bench.samples:
        samples[ref.sample_id] = _load_sample(bench_dir, ref.sample_id)

    return LoadedBenchmark(bench=bench, bench_dir=bench_dir, samples=samples)


def _load_sample(bench_dir: Path, sample_id: str) -> Sample:
    sample_dir = bench_dir / "samples" / sample_id
    spec_path = sample_dir / "content_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"content_spec not found: {spec_path}")
    spec = ContentSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))

    # target 图扩展名不确定（target.jpg / target.png），按 manifest 里 sample.target 解析。
    return Sample(
        spec=spec,
        target_path=_resolve_target(bench_dir, sample_dir),
        target_md_path=sample_dir / "target.md",
        sample_dir=sample_dir,
    )


def _resolve_target(bench_dir: Path, sample_dir: Path) -> Path:
    """定位 target 图：优先 sample_dir 下常见名，找不到则按 manifest 相对路径。"""
    for name in ("target.jpg", "target.png", "target.jpeg"):
        p = sample_dir / name
        if p.exists():
            return p
    # 兜底：manifest 里写的是相对 bench 目录的路径，如 samples/s001/target.jpg
    # （此处不读 manifest，直接在 sample_dir 下 glob target.*）
    candidates = sorted(sample_dir.glob("target.*"))
    if candidates:
        return candidates[0]
    # 最后回退到一个默认路径（即使不存在，也返回一个可预测的占位，便于上层报错）
    return sample_dir / "target.jpg"


def load_sample(
    bench_id: str, sample_id: str, *, settings: Settings | None = None
) -> tuple[Benchmark, Sample]:
    """便捷：只加载单道考题（仍需读 manifest 拿到维度定义）。"""
    loaded = load_benchmark(bench_id, settings=settings)
    return loaded.bench, loaded.sample(sample_id)


__all__ = ["LoadedBenchmark", "Sample", "load_benchmark", "load_sample"]
