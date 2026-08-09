#!/usr/bin/env python3
"""诊断一个 loop 的经验闭环效果：还原度曲线 + A/B/C 三环判定 + 关键点高亮。

读 trajectory.jsonl + lessons/conclusions.json，输出结构化诊断报告：
  - 还原度逐轮曲线 + 各维度失败项
  - 关键点高亮：突崩(>0.1 跌幅) / 未收敛(峰值<0.7) / 末轮回落
  - B 环：fail_streaks + escalated dims（连续失败≥2 轮升级）
  - A 环：lesson 富化质量（含具体建议词=富化；干瘪模板=退化）
  - C 环：generator 是否换了 test_variable（还是一直改 prompt）
  - 总结：A/B/C 各环 ✅/⚠️ 一目了然

用法（项目根下）：
  .venv/bin/python .claude/skills/img-iter-ops/scripts/diagnose_loop.py <loop_id>
  .venv/bin/python .../diagnose_loop.py furniture_product_whitebg-s003-exp6
  .venv/bin/python .../diagnose_loop.py data/runs/furniture_product_whitebg-s003-exp6   # 也支持路径
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DIMS = ["consistency", "product_structure", "material_texture",
        "color_accuracy", "artifact_defect", "commercial_focus"]

# A 环「已富化」的标志词（LLM 富化后 lesson 里常出现的具体建议）
ENRICHED_MARKERS = ["ControlNet", "reference", "test_variable", "seed", "Inpaint",
                    "图生图", "上报人工", "上限", "瓶颈", "negative", "重画", "局部"]
# A 环「干瘪回退」的标志（纯规则模板，LLM 富化失败）
STALE_MARKERS = ["建议保持该方向"]


def project_root() -> Path:
    """从脚本位置向上找含 data/runs 的目录（健壮定位项目根，不依赖 cwd）。"""
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "data" / "runs").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd()


def find_run_dir(loop_id: str) -> Path:
    p = Path(loop_id)
    if p.is_dir():
        return p
    root = project_root()
    cand = root / "data" / "runs" / loop_id
    if cand.is_dir():
        return cand
    runs = root / "data" / "runs"
    matches = [d for d in runs.iterdir() if d.is_dir() and loop_id in d.name] if runs.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[ERR] 多个 loop 匹配 '{loop_id}': {[m.name for m in matches]}", file=sys.stderr)
        sys.exit(2)
    print(f"[ERR] 找不到 loop: {loop_id}（在 {runs}）", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: diagnose_loop.py <loop_id 或 run_dir>", file=sys.stderr)
        return 2
    run_dir = find_run_dir(sys.argv[1])
    traj = run_dir / "trajectory.jsonl"
    conc = run_dir / "lessons" / "conclusions.json"
    if not traj.exists():
        print(f"[ERR] 无 trajectory: {traj}", file=sys.stderr)
        return 2

    recs = [json.loads(l) for l in traj.read_text(encoding="utf-8").splitlines() if l.strip()]
    kb = json.loads(conc.read_text(encoding="utf-8")) if conc.exists() else {"conclusions": [], "fail_streaks": {}}

    print(f"=== {run_dir.name} | {len(recs)} 轮 ===\n")

    # --- 还原度曲线 + 各维度失败项 ---
    print("【还原度曲线】")
    rest_series: list[float | None] = []
    for r in recs:
        v = r.get("verdict") or {}
        rest = v.get("restoration")
        rest_series.append(rest)
        dims = {d["dim"]: d for d in v.get("dimensions", [])}
        fails = []
        for dn in DIMS:
            d = dims.get(dn, {})
            if d.get("scoring_type") == "binary":
                fl = [it["id"] for it in (d.get("items") or []) if not it.get("passed")]
                if fl:
                    fails.append(f"{dn}{fl}")
            elif d.get("value", 1.0) < 0.7:
                fails.append(f"{dn}={d.get('value'):.2f}")
        rest_s = f"{rest:.4f}" if rest is not None else "N/A"
        print(f"  R{r['round']}: {rest_s}" + (f"  ❌ {' '.join(fails)}" if fails else "  ✓"))
    print()

    # --- 关键点高亮 ---
    print("【关键点高亮】")
    flags = []
    for i in range(1, len(rest_series)):
        a, b = rest_series[i - 1], rest_series[i]
        if a is not None and b is not None and (a - b) > 0.1:
            flags.append(f"  ⚠️ R{i}→R{i+1} 还原度突崩 {a:.4f}→{b:.4f}（跌 {a-b:.3f}）")
    valid = [x for x in rest_series if x is not None]
    if len(valid) >= 2:
        if max(valid) < 0.7:
            flags.append(f"  ⚠️ 全程未达 0.7（峰值仅 {max(valid):.4f}），未收敛")
        if valid[-1] < max(valid) - 0.05:
            flags.append(f"  ⚠️ 末轮 {valid[-1]:.4f} 低于峰值 {max(valid):.4f}，回落未恢复")
    print("\n".join(flags) if flags else "  无突崩/收敛异常")
    print()

    # --- B 环 ---
    fs = kb.get("fail_streaks", {}) or {}
    print("【B 环 fail_streaks / escalated】")
    if fs:
        for dim, streak in sorted(fs.items(), key=lambda x: -x[1]):
            mark = " 🔴升级(≥2)" if streak >= 2 else ""
            print(f"  {dim}: 连续失败 {streak} 轮{mark}")
    else:
        print("  （无 fail_streaks）")
    concs = kb.get("conclusions", []) or []
    escalated = [c for c in concs if c.get("escalated")]
    print(f"  escalated 结论: {len(escalated)} / {len(concs)} 条")
    print()

    # --- A 环 ---
    print("【A 环 lesson 富化质量】")
    enriched = stale = total = 0
    for c in concs:
        ls = c.get("lesson") or ""
        if not ls:
            continue
        total += 1
        if any(k in ls for k in ENRICHED_MARKERS):
            enriched += 1
        elif any(m in ls for m in STALE_MARKERS) and len(ls) < 60:
            stale += 1
    print(f"  富化(含具体建议): {enriched}/{total} | 干瘪模板: {stale}/{total}")
    ineff = next((c for c in concs if c.get("status") == "ineffective" and c.get("lesson")), None)
    if ineff:
        print(f"  [ineffective 样本·{ineff['dim']}] {ineff['lesson'][:220]}")
    print()

    # --- C 环 ---
    print("【C 环 generator 是否换 test_variable】")
    tvars = [r.get("test_variable") for r in recs]
    uniq = {t for t in tvars if t}
    print(f"  各轮 test_variable: {tvars}")
    if uniq <= {"prompt"}:
        print("  ⚠️ 全程只改 prompt，从未换 test_variable（reference_images/size/seed）"
              "——升级建议送达但未被执行（执行缺口）")
    else:
        print(f"  换过 test_variable: {uniq} ✓")
    print()

    # --- 总结 ---
    print("【总结】")
    first = rest_series[0] if rest_series else None
    last = rest_series[-1] if rest_series else None
    peak = max(valid) if valid else None
    fs_first = f"{first:.4f}" if first is not None else "?"
    fs_last = f"{last:.4f}" if last is not None else "?"
    fs_peak = f"{peak:.4f}" if peak is not None else "?"
    print(f"  还原度: {fs_first} → {fs_last}（峰值 {fs_peak}）")
    print(f"  A 环: {'✅ lesson 已富化' if enriched > 0 and enriched >= stale else '⚠️ lesson 偏干瘪/未富化'}")
    esc_dims = sorted(d for d, s in fs.items() if s >= 2)
    print(f"  B 环: {'✅ 升级命中 ' + str(esc_dims) if esc_dims else '⚠️ 无升级（可能没触发连续失败）'}")
    print(f"  C 环: {'⚠️ 未换 test_variable（执行缺口）' if uniq <= {'prompt'} else '✅ 有换思路'}")
    if esc_dims or (valid and max(valid) < 0.7):
        print("  → 建议人工介入：escalated 维度已撞模型上限 / 还原度未收敛，"
              "继续 prompt 微调收益有限，考虑换 test_variable 或上报。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
