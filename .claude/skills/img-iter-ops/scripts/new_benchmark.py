#!/usr/bin/env python3
"""生成新 benchmark 考题包骨架（目录 + manifest + rubric + content_spec 模板）。

职责：搭标准目录结构和模板文件，保证结构完整（6 维度/权重/checklist id 齐全）。
**不替代 LLM 起草内容**——content_spec 的具体 checklist 由 Claude（读
references/benchmark-create.md）用 LLM 根据产品图+口述起草后填入；本脚本只搭骨架 +
复制 content_spec 模板 + 生成 manifest/rubric 框架 + 校验。

默认沿用「家具白底三视图」场景的 6 维度评分体系（项目当前唯一标准场景）。全新场景
（如服装/食品）需手动改 manifest 的 score_dimensions——见 references/benchmark-create.md。

用法（项目根下）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/new_benchmark.py \\
      --bench my_bench --samples s001 s002 [--scene "场景描述"]

产物：
  data/benchmarks/<bench>/manifest.json          （6 维度 + samples 骨架，待填 product）
  data/benchmarks/<bench>/rubric.md              （人类可读说明骨架）
  data/benchmarks/<bench>/samples/<s>/content_spec.json  （从模板复制，待填 checklist）
  ⚠️ 还需手动：放 samples/<s>/target.jpg + 填 content_spec + 改 manifest samples 描述
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "assets" / "content_spec.template.json"


def project_root() -> Path:
    p = HERE
    for _ in range(8):
        if (p / "data").exists():
            return p
        p = p.parent
    return Path.cwd()


def build_manifest(bench_id: str, scene: str, samples: list[str]) -> dict:
    """6 维度评分体系（家具白底三视图标准场景）。score_dimensions 复用项目既定定义。"""
    return {
        "bench_id": bench_id,
        "version": "1.0.0",
        "scene": scene or f"{bench_id} · 白底产品图（产品还原度导向）",
        "description": "TODO: 描述本 benchmark 聚焦的还原度难点；参考锚=用户提供的产品实物图。",
        "scene_type": "product_whitebg",
        "scoring_method": "hybrid_with_rank_calibration",
        "scoring_note": "混合评分(二分+连续) + 排序校准。详见 docs/EVALUATION.md。",
        "score_dimensions": [
            {"dim": "consistency", "desc": "三视图跨张一致性：同产品/同色/几何比例一致",
             "weight_init": 0.25, "ref_needed": True, "scoring_type": "binary",
             "check_items": ["C1 三视图是同一产品", "C2 三视图颜色一致",
                             "C3 侧视宽=正视宽(几何一致)", "C4 各视图高度比例一致"]},
            {"dim": "product_structure", "desc": "产品结构：部件数/位置/形态正确，无缺失/重复/穿模",
             "weight_init": 0.22, "ref_needed": True, "scoring_type": "binary",
             "check_items": ["S1 部件数量与参考一致", "S2 无部件穿模/重叠",
                             "S3 无部件缺失", "S4 关键形态正确(对照参考)"]},
            {"dim": "material_texture", "desc": "材质还原度(对照参考)，连续分",
             "weight_init": 0.18, "ref_needed": True, "scoring_type": "continuous",
             "rubric_ref": "rubric.md#材质纹理"},
            {"dim": "color_accuracy", "desc": "颜色准确度(对照参考)，连续分",
             "weight_init": 0.13, "ref_needed": True, "scoring_type": "continuous",
             "rubric_ref": "rubric.md#颜色一致"},
            {"dim": "artifact_defect", "desc": "无瑕疵：无变形/失真/模糊/伪影/悬浮",
             "weight_init": 0.12, "ref_needed": False, "scoring_type": "binary",
             "check_items": ["A1 直线无弯曲", "A2 对称结构对称",
                             "A3 无模糊/拼接痕", "A4 家具接地有阴影(不悬浮)"]},
            {"dim": "commercial_focus", "desc": "商业可用：主体突出/白底干净/构图合规",
             "weight_init": 0.10, "ref_needed": False, "scoring_type": "binary",
             "check_items": ["B1 主体居中突出", "B2 背景纯白干净", "B3 留白构图符合平台规范"]},
        ],
        "comparative_dims": ["consistency", "product_structure", "material_texture",
                             "color_accuracy", "artifact_defect", "commercial_focus"],
        "task": {"type": "three_view_whitebg_single_image", "layout": "three_view_single_image",
                 "views": ["front", "side", "perspective"]},
        "samples": [{"sample_id": s, "product": "TODO: 填产品名", "category": "TODO",
                     "target": f"samples/{s}/target.jpg",
                     "difficulty_note": "TODO: 这题难在哪（结构/材质/颜色）"} for s in samples],
    }


RUBRIC_TMPL = """# Benchmark: {scene}

> `bench_id`: {bench_id}
> 评分维度真源 = `manifest.json`（本文件仅人类可读说明，二者不一致以 manifest 为准）。

## 评分维度（6 项，混合二分 + 连续）

还原度 = Σ(wᵢ × features[i])，各 feature ∈[0,1]。权重 w 初始用 manifest 的 weight_init，
后续由排序校准闭环更新。

### 二分型（逐项 ✓/✗ → 通过率）
- `consistency`(0.25) 三视图跨张一致性 — C1~C4
- `product_structure`(0.22) 产品结构 — S1~S4
- `artifact_defect`(0.12) 无瑕疵 — A1~A4
- `commercial_focus`(0.10) 商业可用 — B1~B3

### 连续型（LLM 给 0-1 分，偏差由排序校准吸收）
- `material_texture`(0.18) 材质纹理
- `color_accuracy`(0.13) 颜色准确度

### 对比型 vs 绝对型
- 对比型（consistency/product_structure/material_texture/color_accuracy）：Critic 同时喂参考图+生成图。
- 绝对型（artifact_defect/commercial_focus）：单看生成图。

## samples/ 目录约定
```
samples/<sNNN>/
├── target.jpg          # 产品实物参考图（对比型维度的评判锚）★必放
├── target.md           # 该产品结构/材质/颜色说明 + 还原要点（可选但推荐）
└── content_spec.json   # 任务 + 约束 + 各维度 checklist/rubric points
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成新 benchmark 考题包骨架")
    ap.add_argument("--bench", required=True, help="bench_id（目录名）")
    ap.add_argument("--samples", nargs="+", default=["s001"], help="sample id 列表（默认 s001）")
    ap.add_argument("--scene", default=None, help="场景描述（默认用 bench_id）")
    args = ap.parse_args()

    root = project_root()
    bench_dir = root / "data" / "benchmarks" / args.bench
    if bench_dir.exists():
        print(f"[FATAL] benchmark 已存在: {bench_dir}", file=sys.stderr)
        return 2

    bench_dir.mkdir(parents=True)
    scene = args.scene or args.bench
    (bench_dir / "manifest.json").write_text(
        json.dumps(build_manifest(args.bench, scene, args.samples), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (bench_dir / "rubric.md").write_text(
        RUBRIC_TMPL.format(bench_id=args.bench, scene=scene), encoding="utf-8")

    if not TEMPLATE.exists():
        print(f"[WARN] content_spec 模板缺失: {TEMPLATE}", file=sys.stderr)
        tmpl = "{}"
    else:
        tmpl = TEMPLATE.read_text(encoding="utf-8")

    for s in args.samples:
        sdir = bench_dir / "samples" / s
        sdir.mkdir(parents=True)
        spec = tmpl.replace("<sNNN>", s)
        (sdir / "content_spec.json").write_text(spec, encoding="utf-8")

    print(f"[new] 已生成 benchmark 骨架: {bench_dir}")
    print(f"[new] samples: {args.samples}")
    print("\n[next] 还需手动完成（按顺序）：")
    print(f"  1. 每个 sample 放产品实物图: {bench_dir}/samples/<s>/target.jpg")
    print(f"  2. 填 content_spec.json 的 checklist（参考 references/benchmark-create.md 用 LLM 起草）")
    print(f"  3. 改 manifest.json 的 samples[].product/category/difficulty_note（去掉 TODO）")
    print(f"  4. （可选）写 samples/<s>/target.md 补充结构/材质/颜色要点")
    print(f"\n[next] 跑起来: .venv/bin/python {HERE}/run_loop_auto.py --bench {args.bench} --sample {args.samples[0]} --rounds 6 --tag exp1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
