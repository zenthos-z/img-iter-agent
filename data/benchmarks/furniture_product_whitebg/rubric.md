# Benchmark: 家具跨境电商白底产品图 · 评分细则

> `bench_id`: furniture_product_whitebg
> 场景：家具跨境电商**三视图白底产品图**，产品还原度导向。首个 benchmark。
> 任务：从参考图生成家具**三视图（正/侧/立体）白底素材图**。

## ⚠️ 评分维度真源 = `manifest.json`

> **本文件仅作人类可读说明。评分维度的权威定义（维度名、`scoring_type`、`weight_init`、
> `check_items`、`comparative_dims`）一律以 `manifest.json` 为准。** 若两者不一致，以
> `manifest.json` 为准并修正本文件。

`manifest.json` 的 `scoring_method = "hybrid_with_rank_calibration"`：混合评分（二分 + 连续）
+ 排序校准。完整机制见 `docs/ARCHITECTURE.md §2.6` 与 `docs/EVALUATION.md §4`。

## 评分维度（6 项，混合二分 + 连续）

每个维度统一产出 `∈[0,1]` 的特征值，还原度总分 = `Σ(wᵢ × features[i])`。
权重 `w` 初始用 `manifest.json` 的 `weight_init`，后续由排序校准闭环更新
（`runs/<id>/calibrated_weights.json`）。

### 二分型维度（逐项 ✓/✗ → 通过率；可复现、归因清晰）

每项 LLM 给 ✓/✗ + 一句理由，`features[dim] = 通过项数 / 总项数`。

| 维度 | 初始权重 | 评什么 | checklist 项（见各 `content_spec.json`） |
|------|------|--------|--------|
| `consistency` 三视图跨张一致性 | 0.25 | 三张是否同一产品/同色/几何比例一致（三视图核心质量） | C1~C4 |
| `product_structure` 产品结构 | 0.22 | 每张视图各自评：部件数/位置/形态正确，无缺失/重复/穿模 | S1~S4 |
| `artifact_defect` 无瑕疵 | 0.12 | 每张各自评：无变形/失真/伪影/模糊/悬浮 | A1~A4 |
| `commercial_focus` 商业可用 | 0.10 | 每张各自评：主体突出/白底干净/构图合规 | B1~B3 |

### 连续型维度（LLM 给 0-1 分；承认有偏差，由排序校准的权重吸收）

按各 `content_spec.json` 的 rubric `points` 整体评分，`features[dim] = LLM 归一化分`。

| 维度 | 初始权重 | 评什么 |
|------|------|--------|
| `material_texture` 材质纹理 | 0.18 | 材质类型正确且有真实纹理（非塑料感）、纹理还原程度 |
| `color_accuracy` 颜色准确度 | 0.13 | 与产品实物参考图无色差（电商退货主因）、色块均匀 |

### 对比型 vs 绝对型（决定 Critic 喂图）

- **对比型维度**（`comparative_dims`：consistency/product_structure/material_texture/color_accuracy）：
  必须同时喂**参考图 + 生成图**作锚评。
- **绝对型维度**（artifact_defect/commercial_focus）：单看生成图即可评。

## 还原度总分

```
features[dim] = 二分型→通过率 ; 连续型→LLM 0-1 分      # 各 ∈[0,1]
还原度 = Σ(wᵢ × features[i])                          # w 初始用 weight_init, 校准后更新
```

## 校准闭环（离线闭环 B）

人**不**被要求对渐变维度猜绝对分；而是对一组 trace 做**整体排序**（人擅长的），
系统用 learning-to-rank 拟合权重 `w`（约束 Σw=1, w≥0），让 `w·features` 的排序吻合人工排序，
天然修正 LLM 连续打分的系统性偏差。详见 `docs/ARCHITECTURE.md §2.6.3`。

## samples/ 样例目录约定

每个样例一个子目录，如 `s001/`：

```
samples/s001/
├── target.jpg          # ★产品实物参考图（对比型维度的评判锚）
├── target.md           # 该产品的结构/材质/颜色说明 + 还原要点
└── content_spec.json   # 任务(三视图白底) + 约束 + 各维度 checklist/rubric points
```

- `target.jpg` 是对比型维度的评判基准；
- 三视图任务还需**跨张同评一致性**（C1~C4）。
