"""里程碑 1 演示：无网络、无 key 的端到端路径。

加载真实家具 benchmark → 注入一份 mock Critic 判定（混合：二分通过率 + 连续 LLM 分）
→ 用 benchmark 先验权重算还原度 → 写一条 trajectory → 读回验证。

运行：.venv/bin/python scripts/milestone1_demo.py
（run 数据写到临时目录，绝不触碰真实 data/runs）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 让脚本在未安装时也能 import（src layout）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_iter_agent.config import Settings
from img_iter_agent.data.benchmark import load_benchmark
from img_iter_agent.data.runstore import RunStore
from img_iter_agent.data.trajectory import TrajectoryReader, TrajectoryWriter
from img_iter_agent.data.weights import compute_features, init_weights, weighted_restoration
from img_iter_agent.memory.schema import (
    CriticItemJudgment,
    CriticVerdict,
    DimensionScore,
)


def main() -> None:
    bench_id = "furniture_product_whitebg"

    # 1) 加载真实 benchmark（不依赖 run 目录）
    lb = load_benchmark(bench_id)
    bench = lb.bench
    print(f"[1] benchmark: {bench.bench_id} | scoring={bench.scoring_method}")
    for d in bench.score_dimensions:
        print(f"    - {d.dim:<18} {d.scoring_type:<10} w={d.weight_init}")

    # 2) 取 s001 + 先验权重
    sample = lb.sample("s001")
    weights = init_weights(bench)
    print(f"\n[2] sample={sample.spec.sample_id} target={sample.target_path.name}")
    print(f"    weights(归一化)={ {k: round(v,3) for k,v in weights.items()} }")

    # 3) mock 一份混合 Critic 判定（演示两种维度类型）
    #    二分维度：逐项 ✓/✗ → 通过率；连续维度：LLM 0-1 分
    feats: dict[str, float] = {
        "consistency": 3 / 4,        # C1-C4 通过 3 项
        "product_structure": 3 / 4,
        "material_texture": 0.72,    # LLM 连续分（带偏差，由排序校准吸收）
        "color_accuracy": 0.88,
        "artifact_defect": 1.0,
        "commercial_focus": 2 / 3,
    }
    binary_items = {
        "consistency": [("C1", True), ("C2", True), ("C3", False), ("C4", True)],
        "product_structure": [("S1", True), ("S2", False), ("S3", True), ("S4", True)],
        "artifact_defect": [("A1", True), ("A2", True), ("A3", True), ("A4", True)],
        "commercial_focus": [("B1", True), ("B2", True), ("B3", False)],
    }
    dims = []
    for dim, val in feats.items():
        if dim in binary_items:
            items = [CriticItemJudgment(id=i, passed=p, reason="mock") for i, p in binary_items[dim]]
            dims.append(DimensionScore(dim=dim, scoring_type="binary", value=val, items=items))
        else:
            dims.append(DimensionScore(dim=dim, scoring_type="continuous", value=val, raw="LLM mock"))

    verdict = CriticVerdict(
        sample_id="s001",
        dimensions=dims,
        weights_used=weights,
        restoration=weighted_restoration(feats, weights),
    )
    print(f"\n[3] mock Critic verdict:")
    print(f"    features={ {k: round(v,3) for k,v in compute_features(verdict).items()} }")
    print(f"    restoration={verdict.restoration:.4f}")

    # 4) 写 trajectory（run 写到临时目录，绝不污染真实 data/runs）
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # 把 benchmark 软链进临时 data_root，使 Settings 能找到真实 benchmark
        (tdp / "benchmarks").mkdir()
        (tdp / "benchmarks" / bench_id).symlink_to(lb.bench_dir)
        settings = Settings(data_root=tdp)

        store = RunStore.create(
            "demo-run", bench_id, "seedream-5.0-pro", note="milestone1 demo", settings=settings
        )
        writer = TrajectoryWriter(store.trajectory_path)
        from img_iter_agent.memory.schema import AttemptRecord

        rec = AttemptRecord(
            attempt_id="a001", run_id="demo-run", round=1, sample_id="s001",
            bench_id=bench_id, model="seedream-5.0-pro",
            test_variable="prompt", baseline_ref=None,
            gen_mode="image_edit", prompt="生成三视图白底素材图...",
            reference_image_refs=["samples/s001/target.jpg"],
            size="2K",
            output_image_refs=["out/a001/front.jpg", "out/a001/side.jpg", "out/a001/perspective.jpg"],
            verdict=verdict,
        )
        writer.append(rec)
        print(f"\n[4] 写 trajectory -> {store.trajectory_path.relative_to(tdp)}")

        # 5) 读回验证
        back = TrajectoryReader(store.trajectory_path).read_all()
        assert len(back) == 1
        r0 = back[0]
        print(f"\n[5] 读回 trajectory: attempt={r0.attempt_id} round={r0.round}")
        print(f"    restoration={r0.verdict.restoration:.4f} (与写入一致: {r0.verdict.restoration == verdict.restoration})")
        print(f"    二分项判定数(consistency)={len(r0.verdict.item_judgments('consistency'))}")
        store.finish(note="demo done")
        print(f"\n[✓] 里程碑1 通过：benchmark加载→混合评分→还原度→轨迹读写 全部跑通（无网络/无key）。")


if __name__ == "__main__":
    main()
