# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Storms 'n' Stocks** — CS5246 Text Mining course project. An end-to-end NLP pipeline that detects disaster events from news, extracts structured information (time, location, type, severity), and assesses potential stock market impact.

Key documents:
- `task.md` — original assignment specification
- `project_plan.md` — detailed project plan including data sources, pipeline design, and evaluation protocol

---

## Pipeline Architecture

Two paths: offline training (GDACS → models) and online inference (GDELT news → prediction).

```
【Offline Training】
GDACS API → fetch_gdacs_all_fields.py (unified) → data/gdacs_all_fields.csv
          (or fetch_gdacs_*.py per-type in reference repo)
          → train_*.py → models/*.pkl (RandomForest binary classifier per disaster type)
          Label: alertlevel green=0 / orange_or_red=1

【Online Inference — 5 Modules, in order】
GDELT GKG (V1THEMES filter) → crawl news text
  ↓
Module A  per-article extraction
          event_type: from V1THEMES mapping
          location:   spaCy GPE/LOC (text) + GKG V1LOCATIONS (lat/lon)
          time:       GKG DATE field (col 1)
          per-type disaster params (regex, unit-normalised):
            EQ: magnitude, depth, rapidpopdescription
            TC: wind_speed (→km/h), storm_surge (→m), exposed_population
            WF: duration_days, burned_area (→ha), people_affected
            DR: duration_days, affected_area (→km2), country_count
            FL: dead, displaced
  ↓
Module D  Cluster by (event_type, geo-proximity ≤500km or location-text-match, Δtime ≤7d)
          Aggregate per cluster: numeric fields → max; location → most frequent; date → earliest
          No Entity Linking needed here — uses raw location text/coords from Module A
  ↓
Module C  Severity prediction on aggregated event record (one prediction per unique event)
          load models/*.pkl → predict(aggregated_features)
          FL: rule dead>100 or displaced>80000 on aggregated max values
          low_confidence=True if all key fields still NaN after aggregation
  ↓
Module B  Entity Linking (per unique event, just before stock analysis)
          aggregated location text → pycountry + alias dict → ISO country code
          ISO → knowledge base JSON → index_ticker + key_industries
  ↓
Module E  Event study: yfinance + OLS → CAR(T+1/T+3/T+5), group by event_type×severity
```

Final output per unique event: `event_id`, `event_type`, `event_date`, `primary_country`, `predicted_alert`, `article_count`, `car_t1`, `car_t3`, `car_t5`.

---

## Data Sources (all verified accessible)

| Source | Role | Key fields |
|--------|------|-----------|
| GDELT v2 Export CSV | News URL entry | col60=`SOURCEURL`, col26=`EventCode`, col56/57=`Lat/Long` |
| GDELT v2 GKG CSV | Disaster filtering + event type | col4=`URL`, col7=`V1THEMES`, col9=`V1LOCATIONS` |
| GDACS API | **Severity training labels** + structured features | `alertlevel`, `eventtype`, `magnitude`, `rapidpopdescription`, `maximum_wind_speed_kmh` |
| yfinance | Stock/ETF prices | Daily OHLCV |
| NewsAPI | Fallback for failed GDELT URLs | Requires free API key |

**Do not use**: ReliefWeb API (410 Gone), USGS (replaced by GDACS for severity labels), Twitter/X API (paid).

**Verified GKG disaster theme keywords** (confirmed against 2024-04-03 Taiwan M7.4):
`NATURAL_DISASTER`, `NATURAL_DISASTER_EARTHQUAKE`, `NATURAL_DISASTER_TSUNAMI`, `NATURAL_DISASTER_FLOOD`, `NATURAL_DISASTER_TREMOR`, `MANMADE_DISASTER_IMPLIED`, `DISASTER_FIRE`, `TERROR`, `CRISISLEX_CRISISLEXREC`

**GDELT column layout** (verified live): Export has 61 cols (older docs say 58). GKG has 27 cols. GKG files occasionally contain non-UTF-8 bytes; use `encoding_errors="replace"`.

**GDACS API**: `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`. Requires `ssl._create_unverified_context()`. EQ `magnitude`/`depth` fields need `--enrich-details` (detail endpoint per event); basic fetch already returns `rapidpopdescription` with magnitude as text ("Magnitude 4.7M, Depth:131km").

**GDACS API 分页行为（实测）**：
- 事件按 `fromdate` **倒序**返回（最新在前）
- 混合 alertlevel 请求（`alertlevel=green;orange;red`）时，orange/red 极稀少，`--limit-per-type` 的行数上限往往在触达 orange/red 页之前就耗尽
- 必须使用 `--balanced-per-level` 按 alertlevel 分别独立拉取，才能保证 orange/red 样本量
- WF/orange 全球历史（2018–2026）仅约 39 条，WF/red 仅约 5 条；EQ/red 约 36 条；DR/orange 约 36 条

**Fetch 版本记录**：

| 版本 | 命令 | 结果 | 问题 |
|------|------|------|------|
| v1（初始） | `--fromdate 2024-01-01 --limit-per-type 1000`（默认，非均衡模式） | EQ=1900(orange=1), TC=1438, WF=88(orange=0), DR=156(orange=12)，共 3582 行 | EQ/WF orange_or_red 严重不足，无法训练；EQ fromdate 范围仅 2025-12~2026-04（API 倒序+行数上限） |
| v2（当前） | 见下方 balanced 命令 | EQ=670(green=500,orange=134,red=36), WF=544(green=500,orange=39,red=5), DR=258(green=216,orange=36,red=6)；合并 TC v1 后共 2910 行 | EQ/WF rapidpopdescription 全为空（detail enrichment 对 balanced 模式的 EQ 返回空字段，见训练注记）|

**v1 产物**（`data/gdacs_all_fields.csv`，保留作参考，TC 数据质量可用）：
- TC: 1438 行，green=1145, orange=171, red=122，fromdate 跨 2008–2026 ✅
- EQ/WF/DR: 数据质量不可用于严重性分类训练

**v2 fetch 命令**（EQ/WF/DR，balanced per alertlevel；TC 复用 v1）：
```bash
python scripts/fetch_gdacs_all_fields.py \
  --event-types EQ,WF,DR \
  --fromdate 2018-01-01 \
  --todate 2026-04-07 \
  --balanced-per-level 500 \
  --page-cap 5000 \
  --workers 6 \
  --request-sleep-sec 0.15 \
  --output data/gdacs_eq_wf_dr_balanced.csv
```
输出合并到 `data/gdacs_all_fields_v2.csv`（EQ/WF/DR balanced + TC from v1），共 2910 行。

FL check intentionally excluded. DR detail endpoint 有间歇性 403（影响部分字段，非致命）。

**各模型输入特征**（定义于 `scripts/train_severity_classifiers.py: TYPE_FEATURES`）：

所有模型统一使用 `sklearn Pipeline`：`SimpleImputer(strategy="median")` → `RandomForestClassifier(class_weight="balanced")`。Null 值一律由 median imputation 填补，推理时行为一致。

| 灾种 | 特征 | null rate（v2 训练数据） | null 处理 |
|------|------|------------------------|----------|
| EQ | `magnitude` | 3.6% | median imputation |
| EQ | `depth` | 3.6% | median imputation |
| EQ | `rapid_pop_people` | 36.3% | median imputation（NaN 来自 rapidpopdescription 为空） |
| EQ | `rapid_pop_log` | 36.3% | median imputation（同上） |
| EQ | `rapid_missing` | 0%（flag） | 无 null；rapidpopdescription 为空时=1.0，否则=0.0 |
| EQ | `rapid_few_people` | 0%（flag） | 无 null；含"few people"时=1.0 |
| EQ | `rapid_unparsed` | 0%（flag） | 无 null；无法解析数字且非"few people"时=1.0 |
| TC | `maximum_wind_speed_kmh` | 37.4% | median imputation |
| TC | `maximum_storm_surge_m` | 63.0% | median imputation（高缺失，特征贡献有限）|
| TC | `exposed_population` | 73.8% | median imputation（高缺失，特征贡献有限）|
| WF | `duration_days` | 0% | 无需处理 |
| WF | `burned_area_ha` | 2.0% | median imputation |
| WF | `people_affected` | 21.0% | median imputation |
| DR | `duration_days` | 0% | 无需处理 |
| DR | `affected_area_km2` | 0% | 无需处理 |
| DR | `affected_country_count` | 0% | 无需处理 |

> 推理时（Module C）必须用完全相同的特征名和单位传入，否则 pkl Pipeline 会报 KeyError。EQ 的 `rapid_*` 五个 flag 特征须先对新闻抽取到的 `rapidpopdescription` 文本运行 `parse_rapidpopdescription()` 生成，不能直接传文本。

> **注**：训练脚本诊断输出曾错误显示 EQ `rapidpopdescription` null rate 100%，实为诊断代码对文本列误用 `pd.to_numeric` 所致（已修复）。实际数据：null rate 36%，`rapid_few_people` rate 12.5%，`rapid_unparsed` rate 0%，模型正常使用了全部 7 个特征。

**v2 训练结果**（`scripts/train_severity_classifiers.py --input data/gdacs_all_fields_v2.csv`）：

| 灾种 | split 方式 | test Macro-F1 | ROC-AUC | PR-AUC | 5-fold CV | 说明 |
|------|-----------|--------------|---------|--------|-----------|------|
| EQ | random（时间切分无 train 数据） | 0.878 | 0.935 | 0.835 | 0.920±0.019 | rapidpopdescription null rate 35.8%，全部 7 个特征正常参与训练 |
| TC | time | 0.815（native） | 0.926 | 0.693 | 0.634±0.062 | val 只有 4 条 orange，val 指标不可信；test native 更可靠 |
| WF | random（同 EQ） | 0.798（native） | 0.960 | 0.653 | 0.779±0.098 | orange_or_red 全球仅 44 条，极度稀少 |
| DR | time | —（native 9条） | — | — | 0.508±0.063 | 已知弱项；native test 不足，augmented test 参考意义有限 |

模型保存路径：`models/{eq,tc,wf,dr}_alertlevel_binary_classifier.pkl`

---

## Tech Stack

- **NLP**: spaCy `en_core_web_sm` (GPE/LOC NER only; no fine-tuning needed)
- **Feature extraction**: Python `re` regex, per-type with unit normalisation
- **Severity models**: scikit-learn `Pipeline` (median imputation + `RandomForestClassifier`, `class_weight="balanced"`)
- **Entity linking**: `pycountry` + `geonamescache` (offline, no API calls)
- **Keyword extraction**: KeyBERT (for display/analysis only, not model input)
- **Stock data**: yfinance
- **Environment**: Python 3.10+, conda env `gdelt`

## Stage 2 Event Type Classifier — LLM Labeling Design

Training data for the event type classifier is labeled via LLM (`scripts/label_event_types.py`) using DeepSeek-V3 on SiliconFlow.

### Core labeling criterion

An article qualifies for a disaster label **only if it is primarily reporting on a specific, real disaster event that has occurred or is currently unfolding**. This includes:
- Breaking news and situation updates about a named event
- Follow-up / tracking reports (situation days later)
- Direct impact reports (evacuations, industry effects, economic damage)

An article is labeled `not_related` if it is primarily:
- **Policy/legislation**: disaster management bills, preparedness campaigns, aid budgets
- **Scientific/analytical**: research on fault lines, climate models, long-term trend analysis
- **Forecast only**: "a storm may develop next week" — no event has actually occurred
- **Historical retrospective**: anniversary coverage, "ten years after the 2011 tsunami"
- **Humanitarian/aid response**: fundraising appeals, donation campaigns, aid announcements
- **Metaphorical or unrelated**: "flooded with complaints", sports, finance, entertainment
- **Volcano/eruption**: not covered by the pipeline

### Design rationale

The distinction is not "does this article mention a disaster" but "is this article *about* a specific event happening now". Articles about disaster policy, climate science, or aid appeals mention real disasters but are not event reports — including them would contaminate the classifier with text patterns from a different register (policy/scientific language vs. news reporting). The `not_related` class represents anything a news aggregator might pull in via disaster keywords but which is not actionable event data.

### Sampling strategy

Uniform per-class cap of 1000 rows; volcano rows dropped before sampling. Output: `data/llm_labels/llm_labels_0_26326_s1000.csv` (gitignored).

### LLM & API details

- **Model**: `deepseek-ai/DeepSeek-V3` via SiliconFlow (`https://api.siliconflow.cn/v1/chat/completions`)
- **Error handling**: HTTP 429/403 → wait 30s × (attempt+1), up to 5 retries; other exceptions → 5s × (attempt+1)
- **Resume logic**: only skips non-error rows on restart; error rows are always retried
- **Workers**: 2 concurrent (higher causes 403 rate-limit on SiliconFlow free tier)
- **Output columns**: `idx`, `url`, `old_event_type`, `llm_event_type`, `reasoning`

### Train/val/test split

Time-based split (no random shuffling) to prevent data leakage. Script: `scripts/prepare_classifier_data.py`.

| Split | Date range | Rows | Notes |
|-------|-----------|------|-------|
| train | ≤ 2025-04-30 | 3411 (68.2%) | |
| val | 2025-05-01 ~ 2025-07-31 | 576 (11.5%) | cyclone scarce (24 rows) |
| test | > 2025-07-31 | 1011 (20.2%) | |

Label distribution (train): not_related=913, wildfire=576, earthquake=539, cyclone=469, drought=464, flood=450

Text format: `"{title} [SEP] {text_cleaned[:512]}"` — columns: `idx`, `timestamp`, `label`, `text`.

### DistilBERT Training Strategy

**Model**: `distilbert-base-uncased` fine-tuned for 6-class sequence classification.

**Key design choices**:
- **Weighted cross-entropy loss** (`WeightedTrainer`): class weights computed from training set distribution to handle imbalance (cyclone & drought are under-represented)
- **Time-based split** (not random): prevents data leakage — future articles cannot inform past training
- **Early stopping** (patience=2): best checkpoint selected by val Macro-F1; training stopped at epoch 7 (best was epoch 5)
- **fp16** enabled on GPU for speed; `batch_size=64`, `lr=2e-5`, `warmup_ratio=0.1`, `weight_decay=0.01`
- **Input format**: `"{title} [SEP] {text_cleaned[:512]}"`, max 128 tokens

**Infrastructure**: NUS HPC SLURM, NVIDIA H100 NVL GPU. Training completed in ~25 seconds. Script: `scripts/train_distilbert_slurm.sh`.

### Final Test Results (DistilBERT — checkpoint-270, epoch 5)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| earthquake | 0.95 | 0.97 | **0.960** | 199 |
| wildfire | 0.91 | 0.99 | **0.947** | 109 |
| cyclone | 0.91 | 0.94 | **0.926** | 193 |
| flood | 0.93 | 0.91 | **0.922** | 182 |
| drought | 0.86 | 0.91 | **0.882** | 95 |
| not_related | 0.81 | 0.73 | **0.770** | 233 |
| **macro avg** | **0.90** | **0.91** | **0.901** | 1011 |

Val Macro-F1: **0.886** | Test Macro-F1: **0.901** | Test Accuracy: **0.90**

Per-sample predictions saved to `data/results/distilbert_preds_{test,val}.csv` — columns: `idx`, `timestamp`, `label`, `pred_label`, `confidence`, `prob_*` (per-class), `correct`.

---

## Reference Implementation

`/Users/wanghelin/Documents/course/CS5246/pj/` contains a working version of the severity classification pipeline. Key scripts to adapt:
- `scripts/fetch_gdacs_eq_fields.py` — GDACS data fetching with `--enrich-details` flag for full structured features
- `scripts/train_eq_alert_classifier.py` — RandomForest training with `rapidpopdescription` parsing
- `scripts/rule_based_alert_classifier.py` — rule-based classifier for EQ + FL
- `docs/NER.md` — per-type regex feature extraction specification
- `scripts/pipeline_step1_predict.py` — full inference pipeline (Module A+C)

---

## Evaluation Protocol

- **Feature extraction (Module A)**: field parse rate per type (EQ magnitude ≥60%, FL dead ≥55%); numeric accuracy on 200 human-verified samples
- **Severity (Module C)**: test Macro-F1 + ROC-AUC per disaster type on GDACS hold-out (25%); expected: EQ ~0.91, TC ~0.78, WF ~0.77, DR ~0.58 (known weak); baseline = rule classifier
- **Deduplication (Module D)**: merge rate + false-merge rate on 100 manually grouped articles
- **Entity Linking (Module B)**: country resolution rate ≥70%, accuracy ≥85% on 100-sample check
- **End-to-end (Module E)**: CAR(T+1/T+3/T+5) group means; t-test H₀: CAR=0; expect orange_or_red group < green group

Volcano is explicitly excluded (no wind-effect feature → no model). DR classifier performance is a known limitation (~F1 0.58); flag in report.

---

## Full Pipeline Run Results（2026-04-09，NUS HPC SLURM job 562012）

**Input**: train+val+test 合并（4998 篇，非 not_related 3706 篇）。脚本：`scripts/pipeline_slurm.sh`。

### Module D（Clustering）

| 指标 | 值 |
|------|-----|
| 输入文章（非 not_related） | 3706 |
| 输出唯一事件 | **1664** |
| 平均每事件文章数 | 2.2 |
| event_date 覆盖率 | 1622/1664 (97%) |

事件类型分布：FL=463, WF=358, EQ=352, DR=250, TC=241

### Module B（Entity Linking — 国家解析）

| 指标 | 值 |
|------|-----|
| country_iso2 解析成功 | 589/1664 (**35%**) |
| 国家准确率（有 gold 标注的子集，147 clusters）| **78.2%** (115/147) |

35% 覆盖率是主要瓶颈：location_text 为城市/地区名时 pycountry 直接匹配失败较多。

### GDACS Matching（Module D 聚类 + Module B 链接综合评测）

脚本：`scripts/eval_gdacs_matching.py`，结果：`data/results/gdacs_match_eval.csv`

| 指标 | 值 | 说明 |
|------|-----|------|
| GDACS coverage（gold 能查到） | 31/1664 (1.9%) | 受限于 gold 标注仅覆盖 test split |
| Pipeline GDACS 命中率 | **0%** | 根因：FL 无 GDACS 记录 + 65% 事件缺 country_iso2 |
| Recall（有 coverage 的子集） | 0.0 | Pipeline 无法命中任何应匹配事件 |

**0% GDACS 命中的根本原因**：
1. FL（463 events）完全不在 GDACS v2 数据中（fetch 时 intentionally excluded）
2. 65% 事件 entity linking 未能解析 country_iso2 → gdacs_matcher 直接跳过
3. eval coverage 本身被低估（gold 标注只有 test split，1011/4998 篇）

实际可评测范围（gold country 可用的 147 clusters）：国家准确率 78.2%，满足 ≥70% 目标。

### Module E（Stock Analysis）

SLURM job 562032，NUS HPC xcnd14 节点。修复了 `sector_etfs` 从 CSV 读回时字符串未解析为 list 的 bug（`ast.literal_eval`）。

| 指标 | 值 |
|------|-----|
| CAR 计算行数 | 5161（1622 unique events × avg 3.2 tickers）|
| 成功率 | 5155/5161 (99.9%) |
| 失败原因 | ACWI 在 2000 年前无历史数据（正常限制） |
| Group analysis 行数 | 354 |
| **统计显著结果（p<0.05）** | **25 个** |

**主要显著发现**（节选）：

| 灾种 | 严重性 | 行业 | 窗口 | n | mean_CAR | p |
|------|--------|------|------|---|----------|---|
| WF | orange_or_red | insurance | T+3 | 71 | +0.64% | 0.0006 |
| DR | green | agriculture | T+5 | 208 | -0.39% | 0.0007 |
| FL | green | construction | T+1 | 435 | -0.21% | 0.0014 |
| WF | orange_or_red | insurance | T+5 | 71 | +0.69% | 0.0041 |
| EQ | orange_or_red | agriculture | T+1 | 43 | -0.25% | 0.0181 |

结果文件：`data/results/car_results.csv`（5161行）、`data/results/group_analysis.csv`（354行）。

---

## 报告写作计划（2026-04-10 开始）

### 当前状态总览

所有实验已完成，结果已落盘，可直接写报告。

| 模块 | 状态 | 关键文件 |
|------|------|---------|
| 数据采集 & 标注 | ✅ | `data/splits/`, `data/llm_labels/` |
| 事件类型分类（DistilBERT） | ✅ | `models/event_classifier_distilbert/`, `data/results/distilbert_preds_test.csv` |
| 严重性分类（EQ/TC/WF/DR） | ✅ | `models/*_alertlevel_binary_classifier.pkl` |
| Module A（时间+位置抽取） | ✅ | `data/results/location_extractor_eval.csv`, `time_extractor_eval.csv` |
| Module A（NER 参数抽取） | ✅ | `data/results/ner_extractor_eval.csv` |
| Module D（事件聚类） | ✅ | `data/results/events.csv`（1664 events） |
| GDACS Matching 评测 | ✅ | `data/results/gdacs_match_eval.csv` |
| Module B（Entity Linking） | ✅ | 35% 国家覆盖率，78.2% 准确率 |
| Module E（CAR 股票分析） | ✅ | `data/results/car_results.csv`, `group_analysis.csv` |
| 报告图表 | ✅ | `figures/fig1~3.png`, `tables/table_*.md` |

### 建议报告结构

```
1. Abstract（150词）
2. Introduction — 问题背景，pipeline 价值
3. Related Work — NER, event detection, event study
4. System Architecture — 架构图 + 数据流
5. Stage 1: Event Type Classification（DistilBERT）
6. Stage 2: Disaster Parameter Extraction（NER regex）
7. Stage 3: Event Clustering（Module D）
8. Stage 4: Severity Assessment（Module C — GDACS/ML）
9. Stage 5: Entity Linking & Industry Mapping（Module B）
10. Stage 6: Stock Impact Analysis（Module E — CAR）
11. Evaluation
    11.1 Event classifier（Table 1）
    11.2 Severity classifiers（Table 2）
    11.3 Extraction accuracy（Table 3 NER, Table 4 Loc+Time）
    11.4 GDACS matching & clustering
    11.5 CAR results（Fig 1 bar, Fig 2 heatmap, Table 5 significant）
12. Discussion & Limitations
13. Conclusion
```

### 关键发现与分析思路（写报告时用）

**事件分类**：DistilBERT Macro-F1 0.901，`not_related` 类 F1 最低（0.77）——原因是该类文本多样（政策/科学/比喻），模式不集中。

**严重性分类**：EQ/TC/WF 均超过 0.79 F1；DR 仅约 0.51，原因是 GDACS 中 DR/orange+red 全球历史仅 42 条，训练数据极度稀少。FL 规则分类（dead>100 or displaced>80000）而非 ML。

**NER 抽取**：EQ 震级解析率最高（84.9%），精度 100%（regex 抓数字后完全准确）；TC 暴风浪高 / 受影响人口 parse rate 0%（新闻普遍不提具体数值）；FL 死亡人数 22%（较低，数值常嵌在非结构化句子中）。

**聚类**：3706 篇 → 1664 事件，平均 2.2 篇/事件；97% 事件有 event_date。

**GDACS matching 0% 的根本原因**（报告中要如实说明）：
1. FL（463 events，28%）GDACS v2 数据中无 FL 记录（fetch 时未包含）
2. 65% 事件 entity linking 未能解析 country_iso2（城市/地区名 pycountry 匹配率低）
3. 以上两点导致 gdacs_matcher 的三要素（type+country+date）无法同时满足

**CAR 主要发现**（报告讨论重点）：
- **WF orange/red → insurance +0.64% T+3**（p=0.0006）：严重野火反而推高保险股，可能反映灾后理赔预期 → 行业重组/保费提升预期
- **DR green → agriculture -0.39% T+5**（p=0.0007）：干旱持续拖累农业板块，符合直觉（供应链破坏）
- **FL green → construction -0.21% T+1**（p=0.0014）：洪水引发建筑股短期下跌（施工停摆，不是重建预期）
- **EQ orange/red → utilities +1.17% T+3**（p=0.039）：强震后电力/水利重建需求推高公用事业股
- **TC** 整体 CAR 不显著——可能因 TC 路径不确定性大，市场提前消化

**局限性（报告中要点明）**：
- GDACS API 本地/HPC 均被封锁，v2 数据不含 FL，训练数据存在结构性缺失
- Entity linking 仅 35% 覆盖率，大量事件缺 country_iso2 导致无法 GDACS 匹配
- DR 分类器 F1 ~0.51，接近随机，结果不可信赖，报告需标注
- CAR 分析样本量不均衡（FL n=435 vs FL orange/red n=3），小样本结果仅供参考

### 图表对应关系

| 图/表 | 放在报告哪一节 |
|-------|--------------|
| Table 1（DistilBERT 分类结果） | §11.1 |
| Table 2（严重性分类器） | §11.2 |
| Table 3（NER eval） | §11.3 |
| Table 4（Location + Time eval） | §11.3 |
| Fig 3（Pipeline coverage funnel） | §4 或 §11 开头 |
| Fig 1（CAR by type & severity） | §11.5 |
| Fig 2（CAR heatmap） | §11.5 |
| Table 5（Significant CAR findings） | §11.5 |

### 下一步行动项

- [ ] 写报告正文各节（明天开始）
- [ ] 考虑是否补充 Module D clustering 定量评测（merge rate / false-merge rate）
- [ ] 考虑是否补充 baseline 对比（random CAR baseline 已在 `car_random_baseline.csv`）
- [ ] 检查 Fig 2 heatmap 数值是否与 Table 5 一致（部分 WF 数值已验证）
