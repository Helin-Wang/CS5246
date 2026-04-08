# CS5246 Project Plan: Storms 'n' Stocks
# 灾难事件检测与股票市场影响评估系统

**课程**：CS5246 Text Mining  
**日期**：2026-04-05

---

## 1. 项目概述

自然灾难（地震、洪水、风暴）和人为灾难会对股票市场产生显著影响：供应链中断、旅行限制、救援行动等都会影响相关行业股价。本项目构建一个端到端 NLP pipeline，从新闻中自动检测灾难事件，评估严重程度，并分析对股票的潜在影响。

**支持灾种**：EQ（地震）、TC（热带气旋）、WF（野火）、DR（干旱）、FL（洪水）。Volcano 明确排除（缺 wind effect 特征）。

---

## 2. 系统架构

```
【训练路径（离线）】
GDACS API → alertlevel (green/orange/red) + 结构化特征
          → per-type 二分类模型训练
            EQ/TC/WF/DR → RandomForestClassifier
            FL          → 规则（dead>100 or displaced>80000）
          → models/*.pkl

【推断路径（在线）】
GDELT v2 GKG
    ↓
Stage 1  粗过滤：V1THEMES 含 NATURAL_DISASTER_* / DISASTER_FIRE 等
         → 过滤约 95%+ 无关条目，仅保留疑似灾难文章
    ↓
全文抓取（对过滤后 URL 发起 HTTP 请求）
    ↓
Stage 2  事件类型分类器（6类文本分类）
         → earthquake / flood / cyclone / wildfire / drought / not_related
         → not_related 丢弃
    ↓
Module A  per-article 信息抽取
          event_type  ← Stage 2 分类器输出
          location    ← spaCy GPE/LOC + GKG V1LOCATIONS (lat/lon)
          time        ← GKG DATE 字段
          per-type 参数 ← regex + 单位归一化
    ↓
Module D  事件聚类去重（★先于预测）
          规则聚合：event_type + geo-proximity ≤500km + Δtime ≤7d
          特征合并：数值字段取 max，date 取最早，location 取众数
    ↓
Module C  严重性预测（per-event，基于聚合特征）
          EQ/TC/WF/DR → models/*.pkl
          FL          → 规则
    ↓
Module B  Entity Linking（per-event）
          地名文本 → pycountry + 别名字典 → ISO 国家代码
          国家 → 知识库 JSON → index_ticker + key_industries
    ↓
Module E  股票影响分析
          yfinance → OLS 市场模型 → CAR(T+1/T+3/T+5)
          按 event_type × severity 分组统计
```

**最终输出**（per unique event）：`event_id`, `event_type`, `event_date`, `primary_country`, `predicted_alert`, `article_count`, `car_t1`, `car_t3`, `car_t5`

---

## 3. 数据采集

### 3.1 数据来源

| 来源 | 用途 | 状态 |
|------|------|------|
| GDELT v2 GKG | 新闻 URL 入口 + V1THEMES 过滤 + 地点坐标 | ✅ 已验证 |
| GDACS API | 严重性训练标签（alertlevel）+ 结构化特征 | ✅ 已验证 |
| yfinance | 股票/ETF 历史价格 | ✅ 已验证 |
| NewsAPI | GDELT URL 失效时补充全文（100次/天） | ⚠️ 需注册 Key |

**不使用**：GDELT Export（GKG 已足够，Export 与 GKG 的 URL 重叠率仅 25.4%）、ReliefWeb（410 Gone）、Twitter API（付费）。

### 3.2 关键字段

**GDELT v2 GKG**（27列，flat file CSV，每行一篇文章）

| 字段 | 列索引 | 用途 |
|------|--------|------|
| DATE | col 0 | 事件时间戳（精确到15分钟） |
| URL | col 4 | 文章链接，用于全文抓取 |
| V1THEMES | col 7 | 主题标签，用于粗过滤 |
| V1LOCATIONS | col 9 | 结构化地名（含 lat/lon），用于聚类 proximity |

> 注：GKG 文件偶含非 UTF-8 字节，读取需 `encoding_errors="replace"`。

**GDACS API**

| 字段 | 用途 |
|------|------|
| `alertlevel` | green/orange/red → 二分类训练标签 |
| `magnitude`, `depth`, `rapidpopdescription` | EQ 特征 |
| `maximum_wind_speed_kmh`, `maximum_storm_surge_m`, `exposed_population` | TC 特征 |
| `duration_days`, `burned_area_ha`, `people_affected` | WF 特征 |
| `duration_days`, `affected_area_km2`, `affected_country_count` | DR 特征 |
| `dead`, `displaced` | FL 推理特征（**注：不来自 GDACS，由 Module A regex 从新闻文章抽取；FL 用规则分类，不从 GDACS 拉取训练数据**）|

### 3.3 标注策略

- **事件类型**（两阶段）：
  - Stage 1 粗过滤：用 V1THEMES 中的灾难主题标签（如 `NATURAL_DISASTER_EARTHQUAKE`、`DISASTER_FIRE`）筛选文章，无需人工标注。
  - Stage 2 分类器训练标签：使用 SiliconFlow **DeepSeek-V3** 对 `training_events_gdelt.xlsx` 中每类各最多 1000 条文章（共 5000 条）进行 LLM 标注，去除 error（2 条）后得到 **4998 条有效标签**。
  - **标注核心判准**：文章必须是对某个正在发生或刚发生的具体灾难事件的报道——含突发新闻、追踪报道、直接影响报道。以下内容标为 `not_related`：灾害政策/立法、气候科学研究、纯预测（事件未发生）、历史周年回顾、人道主义援助呼吁、隐喻用法、火山/其他未支持灾种。
  - V1THEMES 不直接作为 event_type：同一类型主题碎片化（earthquake 分散为 EARTHQUAKE/TEMBLOR/TREMOR/AFTERSHOCKS 等多个标签），TC/DR 极度稀疏，分类器从全文统一解决。

- **严重性**：GDACS `alertlevel` → 二分类（`green` vs `orange_or_red`）。

  **为什么做成二分类、为什么合并 orange 和 red**：
  1. **Red 样本量极少，无法可靠训练三分类**。全局历史数据中 red 事件极少（EQ=36，TC=122，WF=5，DR=6），直接做三分类时 red 类的召回率几乎为零，模型退化为二分类。
  2. **Orange 和 red 在下游任务中的影响方向相同**。本项目关注股票异常收益（CAR）——无论是 orange 还是 red 级别的灾难，对相关行业的冲击方向一致（负向）；green 级别事件通常无显著影响。将 orange/red 合并为一个"高严重性"类符合 Module E 的分析粒度。
  3. **GDACS 自身定义支持合并**。GDACS 的 orange 和 red 均触发国际人道主义响应，区别在于规模和强度，而非性质；green 则代表无需协调响应的局部小事件。orange/red 之间的语义距离远小于 green 与 orange 之间的距离。
  4. **合并后仍是不平衡问题，进一步分裂会加剧**。合并后 orange_or_red 占比约 8%（WF）~20%（TC），已需要 `class_weight="balanced"` 处理；若保留三类，red 占比将降至 1%~8%，训练集中 red 样本可能不足 30 条。

- **股票影响**：事件后 CAR(T+1/T+3/T+5)，无需人工标注。

### 3.4 数据集与切分

**事件类型分类器数据集**（`data/splits/`）

切分策略：**按时间线切分**，防止同一灾难事件的多篇报道跨 train/test 造成数据泄露。

| 集合 | 时间范围 | 行数 | 占比 |
|------|---------|------|------|
| train | ≤ 2025-04-30 | 3411 | 68.2% |
| val   | 2025-05-01 ~ 2025-07-31 | 576 | 11.5% |
| test  | > 2025-07-31 | 1011 | 20.2% |

各集合标签分布（注：val 中 cyclone 仅 24 条，训练时需 `class_weight`）：

| 标签 | train | val | test |
|------|-------|-----|------|
| not_related | 913 | 146 | 233 |
| wildfire | 576 | 116 | 109 |
| earthquake | 539 | 95 | 199 |
| cyclone | 469 | 24 | 193 |
| drought | 464 | 54 | 95 |
| flood | 450 | 141 | 182 |

**严重性训练数据**（`data/gdacs_all_fields_v2.csv`，共 2910 行）：

| 灾种 | 总行数 | green | orange | red | orange_or_red合计 | 来源 |
|------|--------|-------|--------|-----|------------------|------|
| EQ | 670 | 500 | 134 | 36 | 170 | v2 balanced fetch，fromdate 2018–2026 |
| TC | 1438 | 1145 | 171 | 122 | 293 | v1 fetch，fromdate 2008–2026（已够用） |
| WF | 544 | 500 | 39 | 5 | 44 | v2 balanced fetch；全球历史 orange/red 仅此数量 |
| DR | 258 | 216 | 36 | 6 | 42 | v2 balanced fetch，fromdate 2018–2026 |

FL 规则无需训练数据。

**Fetch 命令（v1 → v2 升级原因）**：v1 使用非均衡模式（`--limit-per-type 1000`），API 倒序返回导致只拿到最新绿色事件，EQ 仅 1 条 orange、WF 0 条 orange。v2 改用 `--balanced-per-level 500` 分 alertlevel 独立拉取：
```bash
# EQ/WF/DR balanced（v2 新增）
python scripts/fetch_gdacs_all_fields.py \
  --event-types EQ,WF,DR --fromdate 2018-01-01 \
  --balanced-per-level 500 --page-cap 5000 \
  --output data/gdacs_eq_wf_dr_balanced.csv
# 合并：gdacs_eq_wf_dr_balanced.csv + TC rows from v1 → gdacs_all_fields_v2.csv
```

### 3.5 GDELT 采集实现说明（组员代码 `src/data_collection/gdelt.py`）

组员采用 **BigQuery**（`gdelt-bq.gdeltv2.gkg_partitioned`）替代 flat file 下载，逻辑等价。已发现以下问题需修正：

| 严重程度 | 问题 | 修正方案 |
|---------|------|---------|
| 🔴 High | `NATURAL_DISASTER_WILDFIRE` 不是有效 GKG 主题，BigQuery 返回 0 条 | 改为 `DISASTER_FIRE` |
| 🔴 High | `NATURAL_DISASTER_DROUGHT` 不是有效 GKG 主题，返回 0 条 | 改为 `ENV_DROUGHT` |
| 🔴 High | `process_row` 要求 GKG 主题 AND 标题关键词同时匹配，实测丢失 **70.5%** 有效文章（台湾地震时段实测）。很多合法灾难报道标题不含灾种关键词（如"死亡人数上升"、"数千人疏散"） | 去掉标题关键词硬门槛；GKG 主题匹配即可确定 event_type，关键词仅用于 blacklist 去噪 |
| 🟡 Medium | `NATURAL_DISASTER_TROPICAL_CYCLONE` 在 GKG 中覆盖率极低 | 改为 `NATURAL_DISASTER_HURRICANE` 或 `NATURAL_DISASTER_STORM` |
| 🟡 Medium | Tone ≥ 1.0 过滤丢失 **12.6%** 文章，含部分合法灾难报道（实测）。灾难文章整体 tone 分布均值 −2.76，大多数本已为负面 | 提高阈值至 ≥ 3.0 或直接移除，保留 blacklist 去噪即可 |
| 🟡 Medium | `pd.read_csv` 缺少 `encoding_errors="replace"`，遇非 UTF-8 字节崩溃 | 加参数 |
| 🟠 Low | 包含 volcano 类型，pipeline 不支持 | 从 DISASTER_MAPPING 移除 |
| 🟠 Low | 无 train/dev/test split 代码 | 需单独实现 |

---

## 4. 系统各模块详细设计

### 4.1 预处理

- HTML/噪声去除、仅保留英文
- URL + 标题 MinHash 去重

### 4.2 事件类型分类（两阶段）

**Stage 1 — V1THEMES 粗过滤**

读取 GKG V1THEMES，检测是否含以下主题标签之一，过滤无关文章：

```
NATURAL_DISASTER, NATURAL_DISASTER_EARTHQUAKE, NATURAL_DISASTER_FLOOD,
NATURAL_DISASTER_TSUNAMI, NATURAL_DISASTER_TREMOR, DISASTER_FIRE,
NATURAL_DISASTER_HURRICANE, NATURAL_DISASTER_STORM, CRISISLEX_CRISISLEXREC
```

此阶段不输出 event_type，仅做量的筛选（实测过滤约 89% 无关条目）。

**Stage 2 — 文本分类器**

- 6 类：earthquake / flood / cyclone / wildfire / drought / not_related
- 主模型：distilbert-base-uncased 微调，输入截断至 max_length=128（实现见 `src/train_event_classifier_distilbert.py`）
- 训练数据：LLM 标注标签 + `data/splits/` 时间切分（见 §3.4）
- **强基线（对比用）**：TF-IDF + 多项逻辑回归，实现见 `src/train_event_classifier.py`，输出 `models/event_classifier_tfidf_lr.pkl`；设计思想、测试结果与误差分析见 **§5.1**

### 4.3 NER 与参数抽取（Module A，per-article）

**地点**（两路互补）：spaCy `en_core_web_sm` GPE/LOC 得地名文本；GKG V1LOCATIONS 得 lat/lon 坐标。

**时间**：直接用 GKG DATE 字段（col 0），精确到15分钟。

**per-type 参数 regex 抽取**（与 GDACS 训练特征严格对齐）：

| 类型 | 抽取字段 | 示例 |
|------|---------|------|
| EQ | `magnitude`, `depth`, `rapidpopdescription` | `magnitude\s+(\d+\.?\d*)` |
| TC | `wind_speed`(→km/h), `storm_surge`(→m), `exposed_population` | `(\d+)\s*(knots\|mph\|km/h)` |
| WF | `duration_days`, `burned_area`(→ha), `people_affected` | `(\d+[\.,]?\d*)\s*(ha\|acres)` |
| DR | `duration_days`, `affected_area`(→km²), `country_count` | `(\d+)\s*countries` |
| FL | `dead`, `displaced` | `(\d+[\.,]?\d*)\s*(?:people\s+)?(?:killed\|dead)` |

关键字段全为 NaN → 标记 `low_confidence=True`。

### 4.4 事件聚类去重（Module D）

规则聚合（不用向量聚类）——同时满足以下 3 条 → 合并为同一事件：
1. `event_type` 相同
2. 地理接近：坐标距离 ≤ 500km，或坐标缺失时地名精确匹配
3. 时间差 ≤ 7 天

特征合并：数值字段取 max，`event_date` 取最早，`primary_country` 取众数，`article_count` 计数。

> 先聚类再预测：单篇文章往往只包含灾害部分信息，聚合后特征更完整，避免 low_confidence 文章污染预测结果。

### 4.5 严重性评估（Module C）

所有 ML 模型统一使用 `sklearn Pipeline(SimpleImputer(strategy="median") → RandomForestClassifier(class_weight="balanced"))`，null 值一律 median imputation。

| 灾种 | 特征 | null rate | null 处理 | 算法 | 实测 Macro-F1 |
|------|------|-----------|----------|------|--------------|
| EQ | `magnitude`, `depth` | 3.6% | median imputation | RandomForest | 0.878（random split）|
| EQ | `rapid_pop_people`, `rapid_pop_log` | 36.3% | median imputation（rapidpopdescription 为空时 NaN）| — | — |
| EQ | `rapid_missing`, `rapid_few_people`, `rapid_unparsed` | 0%（flag，永不为 null）| 无需处理 | — | — |
| TC | `maximum_wind_speed_kmh` | 37.4% | median imputation | RandomForest | 0.815（time split，native test）|
| TC | `maximum_storm_surge_m` | 63.0% | median imputation | — | — |
| TC | `exposed_population` | 73.8% | median imputation | — | — |
| WF | `duration_days` | 0% | 无需处理 | RandomForest | 0.798（random split，native test）|
| WF | `burned_area_ha` | 2.0% | median imputation | — | — |
| WF | `people_affected` | 21.0% | median imputation | — | — |
| DR | `duration_days`, `affected_area_km2`, `affected_country_count` | 0% | 无需处理 | RandomForest | CV=0.508，已知弱项（native test 仅 9 条）|
| FL | `dead`, `displaced`（Module A 从新闻抽取） | 推理时取决于抽取结果 | 缺失视为 0 | 规则：dead>100 or displaced>80000 | — |

模型：sklearn `Pipeline`（SimpleImputer median + RandomForestClassifier class_weight="balanced"），保存为 `.pkl`。

#### 4.5.1 Severity 训练数据协议（按灾种独立 + 时间防泄露）

为避免时间穿越与分布污染，Severity 训练采用以下强制流程：

1. **先按灾种拆分**（`eventtype`）：
   - `EQ` / `TC` / `WF` / `DR` 各自独立建模与评估；
   - `FL` 不训练模型，继续规则：`dead > 100 or displaced > 80000`。

2. **每个灾种内部按时间切分**（基于 `fromdate`）：
   - 设 `T_val = 2025-04-30 06:45:00`
   - 设 `T_test = 2025-07-31 00:15:00`
   - `train`: `fromdate < T_val`
   - `val`: `T_val <= fromdate < T_test`
   - `test`: `fromdate >= T_test`

3. **防泄露硬校验（不通过即中止训练）**：
   - `max(train.time) < min(val.time)`
   - `max(val.time) < min(test.time)`
   - 三集合无重叠（按 `eventtype + eventid` 唯一键）
   - 阈值调参与模型选择仅使用 `train + val`，禁止使用 `test`

   > **WF / DR 小样本补充方案**：防泄露的核心目标是 GDACS 训练事件不与 GDELT 推理文章重叠，GDACS 内部不需要严格时间隔离。因此若某灾种 test 集行数不足（如 < 30 条），可直接从 train 集中补采：先保留 test 集已有样本，再按 green / orange_or_red 比例从 train 集中随机采样缺少的类型，直至 test 集达到目标规模。补采样本在报告中注明"augmented from train pool"，并单独报告原生 test 集上的指标作对照。

4. **每灾种训练前统计（必须输出）**：
   - `n_train / n_val / n_test`
   - `green` vs `orange_or_red` 标签分布
   - 关键字段缺失率与 `coverage_score` 分层
   - 这些统计将用于解释 low_confidence 与最终 CAR 分组稳定性

### 4.6 Entity Linking（Module B）

时机：Module C 之后、Module E 之前，per unique event 执行一次。

方案：聚合地名 → `pycountry` + 别名字典 → ISO 代码 → 静态知识库 JSON（~30国）→ `index_ticker` + `key_industries`。城市坐标用 `geonamescache` 离线库（无需 API）。

### 4.7 关键词抽取

KeyBERT 提取每篇文章核心关键词，供展示和分析用；不作为严重性模型输入。TF-IDF 作对比 baseline。

### 4.8 股票影响分析（Module E）

事件研究法（OLS 市场模型）：

1. yfinance 取 `[T-45, T+10]` 日收益率
2. 估计窗口 `[T-30, T-5]`（25个交易日）拟合：`R_i = α + β×R_m + ε`
3. 事件窗口 `[T-1, T+5]` 计算异常收益 `AR_t = R_i,t − (α̂ + β̂×R_m,t)`
4. `CAR(T+1/T+3/T+5)` = 累计 AR
5. 按 `event_type × severity` 分组：均值 + 单样本 t 检验（H₀: CAR=0）+ 箱线图

---

## 5. 评估方案

### 5.1 事件类型分类器（Stage 2）

| 指标 | 说明 |
|------|------|
| 6类 Macro-F1 | 主要指标，含 not_related 类 |
| Per-class F1 | 重点关注 cyclone（训练集最稀缺，仅 642 条） |
| Not_related 精确率 | 误保留率（噪音进入后续 pipeline 的比例） |

**数据**：时间切分（见 §3.4），train=3411，val=576，test=1011；验证/测试集时间在训练集之后，避免同一事件跨集合泄露。

**上游规则对照（非文本基线）**：V1THEMES 映射的 `old_event_type` 无法输出 `not_related`，与 LLM 金标准可比性有限，仅作粗对照。

#### 5.1.1 TF-IDF + 逻辑回归强基线：设计思想

- **目标**：在相同 6 类标签与时间切分下，提供可复现、低开销的线性基线，便于与 Transformer 对比。
- **特征**：`TfidfVectorizer` 词级 analyzer，**ngram (1,2)**，`max_features=100_000`，`sublinear_tf=True`，`min_df=2`，控制稀疏高维词袋。
- **分类器**：`LogisticRegression` **multinomial** L-BFGS，`C=5.0`，`class_weight=balanced`（缓解 cyclone 等类稀疏）。
- **输入**：`"{title} [SEP] {text_cleaned[:512]}"` 统一文本格式，max 128 tokens。

#### 5.1.2 DistilBERT 微调：设计思想

- **模型**：`distilbert-base-uncased`，6 类序列分类头，max_length=128。
- **加权损失**（`WeightedTrainer`）：由训练集分布计算 class weights，缓解 not_related 与 cyclone 的不平衡。
- **早停**（patience=2）：按 val Macro-F1 选最优 checkpoint，避免在 val 小类上过拟合。
- **时间切分防泄露**：与 TF-IDF 使用完全相同的 split，确保对比公平。
- **训练参数**：`batch_size=64`，`lr=2e-5`，`warmup_ratio=0.1`，`weight_decay=0.01`，`fp16`（GPU），最多 10 epochs。
- **基础设施**：NUS HPC SLURM，NVIDIA H100 NVL，约 25 秒完成训练（脚本：`scripts/train_distilbert_slurm.sh`）。

#### 5.1.3 实测结果对比

| 模型 | Val Macro-F1 | Test Macro-F1 | Test Accuracy |
|------|-------------|--------------|---------------|
| TF-IDF + LR（baseline） | 0.831 | 0.867 | 0.86 |
| **DistilBERT**（主模型） | **0.886** | **0.901** | **0.90** |

**Per-class F1（test set）**

| 模型 | earthquake | flood | cyclone | wildfire | drought | not_related |
|------|-----------|-------|---------|----------|---------|-------------|
| TF-IDF + LR | 0.946 | 0.871 | 0.872 | 0.946 | 0.831 | 0.733 |
| DistilBERT | **0.960** | **0.922** | **0.926** | **0.947** | **0.882** | **0.770** |

**结论**：DistilBERT 在所有类别均优于 TF-IDF+LR，提升最大的是 cyclone（+0.054）、not_related（+0.037）、drought（+0.051）。not_related 在两个模型上都是最弱类（F1 0.733/0.770），原因见下。

#### 5.1.4 效果分析（侧重 `not_related`）

- **`not_related` 的标注含义**：文章主要是对某个正在发生的灾难事件的报道，才归入五类之一；若主要是政策讨论、气候研究、纯预测、历史回顾或援助呼吁，则标为 `not_related`。标准更严格（相比原始版本），使 `not_related` 类更有语义一致性。
- **为何 `not_related` 仍是最弱类**：
  - 这类文章本身含有大量灾难词汇（flood、earthquake、drought），与五类的词面高度重叠，两个模型均难以仅凭词频区分。
  - `not_related` 内部异质性高（政策文章、援助新闻、科学论文共处一类），缺乏统一的词汇特征。
  - DistilBERT 相比 TF-IDF 改善明显（+0.037），说明上下文编码对区分"报道某事件"与"讨论灾难话题"有效，但上限受标注边界模糊限制。
- **对 pipeline 的影响**：not_related recall=0.73 意味着约 27% 的非事件文章会漏进后续模块，作为低置信度噪声参与聚类与严重性预测；not_related precision=0.81 意味着约 19% 被标为 not_related 的文章实为真实事件报道（漏召）。后续 Module D 聚类的 article_count 阈值可部分吸收前者噪声。

### 5.2 NER / 参数抽取

| 指标 | 目标 |
|------|------|
| 字段解析率 | EQ magnitude ≥ 60%，FL dead ≥ 55% |
| 数值准确率 | 200条人工核验样本，±10% 内视为正确 |

**Baseline**：第一个数字 regex（不做单位换算）。

### 5.3 严重性分类

| 指标 | 说明 |
|------|------|
| Test Macro-F1 | 主要指标（各灾种独立评估）；EQ~0.91、TC~0.78、WF~0.77、DR~0.58（已知弱项）|
| ROC-AUC | 对 orange_or_red 类的区分能力 |
| PR-AUC | 类别不平衡下的辅助指标（与 ROC-AUC 并列报告） |
| Minority Recall | `orange_or_red` 召回，防止漏报高严重性 |
| 5-fold CV | 稳定性验证（仅在训练时间窗内）|

**数据**：`data/gdacs_all_fields_v2.csv`（EQ=670，TC=1438，WF=544，DR=258，共 2910 行）。按灾种拆分，时间切分失败时 fallback 随机切分；test 不足时从 train 池补采（见 §4.5.1）。`FL` 仅规则，不进入模型训练。**Baseline**：规则分类器（震级/死亡人数阈值）。

**实测结果**（`scripts/train_severity_classifiers.py`）：

| 灾种 | split | test Macro-F1 | ROC-AUC | PR-AUC | 5-fold CV | 注 |
|------|-------|--------------|---------|--------|-----------|---|
| EQ | random | 0.878 | 0.935 | 0.835 | 0.920±0.019 | rapidpopdescription null rate 35.8%，全部 7 个特征正常参与训练 |
| TC | time | 0.815（native） | 0.926 | 0.693 | 0.634±0.062 | val 4 条 orange，val 指标不可信 |
| WF | random | 0.798（native） | 0.960 | 0.653 | 0.779±0.098 | 全球历史 orange_or_red 仅 44 条 |
| DR | time | —（native 9条） | — | — | 0.508±0.063 | 已知弱项；native test 不足，augmented test 参考意义有限 |

### 5.4 聚类去重

| 指标 | 说明 |
|------|------|
| 去重率 | 同一真实事件正确合并比例 |
| 误合并率 | 不同事件被错误合并比例 |

**Gold standard**：100篇文章人工分组标注。**Baseline**：不去重（每篇独立）。

### 5.5 Entity Linking

- 国家识别率 ≥ 70%，准确率 ≥ 85%（100条样本核验）

### 5.6 端到端（股票影响）

- 有效 CAR 事件数 ≥ 500
- orange_or_red 组 CAR 显著低于 green 组（t 检验 p < 0.05）

### 5.7 基线汇总

| 模块 | Baseline | 改进方法 |
|------|---------|---------|
| 事件类型 | V1THEMES 映射（无 not_related）；文本强基线见 §5.1 TF-IDF+LR | DistilBERT 6 类微调 |
| 参数抽取 | 第一个数字 regex | per-type 完整 regex + 单位换算 |
| 严重性 | 规则阈值分类器 | per-type RandomForest（GDACS 训练）|
| 聚类 | 不去重 | 规则聚合（type + geo + 7天）|
| Entity Linking | 精确字符串匹配 | pycountry + 别名字典 |
| 关键词 | TF-IDF Top-K | KeyBERT |

---

## 6. 时间线

```
Week 1–2（4月）：数据采集
  - GDACS 拉取历史事件（EQ/TC/WF/DR/FL）→ data/gdacs_*_fields.csv
  - GDELT GKG 历史下载（2024–2025）→ Stage 1 过滤 → 全文抓取
  - 修正组员代码中的 GKG 主题名错误 + 双重过滤逻辑

Week 3（4月底）：分类器训练 + 严重性模型
  - Stage 2 事件类型分类器训练（silver + gold labels）
  - 复用参考实现 train_*.py，验证各灾种 Macro-F1

Week 4–5（5月）：Module A/B/C 实现
  - per-type regex 抽取 + 单位换算
  - Entity Linking（pycountry + 知识库 JSON）
  - 严重性预测调用 models/*.pkl
  - 各模块 baseline 对比实验

Week 6（5月）：Module D/E + 评估
  - 规则聚合去重
  - 事件研究法 OLS + CAR 可视化
  - 200条人工核验样本

Week 7（6月）：报告 + Demo
```

---

## 7. 技术栈

| 类别 | 工具 |
|------|------|
| NLP | spaCy `en_core_web_sm`（GPE/LOC NER）|
| 事件类型分类 | distilbert-base-uncased（微调）或 fastText |
| 参数抽取 | Python `re` |
| 关键词 | KeyBERT |
| 严重性模型 | scikit-learn（RandomForest + Pipeline）|
| Entity Linking | `pycountry`，`geonamescache` |
| 股票数据 | yfinance |
| 环境 | Python 3.10+，conda env `gdelt` |

---

## 8. 风险与应对

| 风险 | 应对 |
|------|------|
| GDACS API SSL 证书 | `ssl._create_unverified_context()`（已验证）|
| GDACS 训练数据类别不平衡 | `class_weight="balanced"`；DR F1~0.58 为已知弱项 |
| WF/DR test 集过小（<30条），Macro-F1 置信区间极宽 | 从 train 池补采缺少的类型至目标规模（见 §4.5.1）；防泄露核心目标是 GDACS 不与 GDELT 推理数据重叠，GDACS 内部无需严格时间隔离 |
| V1THEMES 碎片化 / TC·DR 稀疏 | Stage 2 文本分类器从全文判断，不依赖 V1THEMES 作 event_type |
| 事件类型分类器 silver label 噪音 | GDACS 匹配设严格时间（±3天）+ 地理（≤200km）窗口；gold 验证集 500条 |
| regex 解析率低 | 先统计解析率；对 <40% 字段补充 regex 模式 |
| GDELT URL 失效率高 | 超时+重试；失败时保留 GKG 摘要；NewsAPI 补充 |
| Entity Linking 覆盖率不足 | pycountry + 人工别名字典（省份/缩写/常见别名）|
| 小国股票数据稀疏 | 回退到地区 ETF（如 EEM 新兴市场）|
