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
| `dead`, `displaced` | FL 规则特征 |

### 3.3 标注策略

- **事件类型**（两阶段）：
  - Stage 1 粗过滤：用 V1THEMES 中的灾难主题标签（如 `NATURAL_DISASTER_EARTHQUAKE`、`DISASTER_FIRE`）筛选文章，无需人工标注。
  - Stage 2 分类器训练标签：使用 SiliconFlow DeepSeek-R1-Distill-Qwen-7B 对 `training_events_gdelt.xlsx` 中每类各 2000 条文章（共 8481 条）进行 LLM 标注，去除 error（86 条）后得到 **8395 条有效标签**。
  - V1THEMES 不直接作为 event_type：同一类型主题碎片化（earthquake 分散为 EARTHQUAKE/TEMBLOR/TREMOR/AFTERSHOCKS 等多个标签），TC/DR 极度稀疏，分类器从全文统一解决。
  - LLM 标注与原始粗筛标签的同意率：wildfire 81%、drought 75%、earthquake 79%、flood 71%、**cyclone 52%**（最低，31% 被标为 not_related）。

- **严重性**：GDACS `alertlevel` → 二分类（`green` vs `orange_or_red`）。

- **股票影响**：事件后 CAR(T+1/T+3/T+5)，无需人工标注。

### 3.4 数据集与切分

**事件类型分类器数据集**（`data/splits/`）

切分策略：**按时间线切分**，防止同一灾难事件的多篇报道跨 train/test 造成数据泄露。

| 集合 | 时间范围 | 行数 | 占比 |
|------|---------|------|------|
| train | 2024-01 ≤ t ≤ 2025-04 | 5680 | 67.7% |
| val   | 2025-05 ≤ t ≤ 2025-07 | 1037 | 12.4% |
| test  | t > 2025-07           | 1678 | 20.0% |

各集合标签分布（注：cyclone 是最稀缺类，训练时需 `class_weight`）：

| 标签 | train | val | test |
|------|-------|-----|------|
| wildfire | 1289 | 267 | 247 |
| earthquake | 1133 | 190 | 366 |
| flood | 1054 | 254 | 394 |
| drought | 832 | 113 | 196 |
| not_related | 730 | 139 | 260 |
| cyclone | 642 | 74 | 215 |

**严重性训练数据**：GDACS API 历史事件，EQ ~1000–2000 条，TC ~300–500，WF ~100–200，DR ~150–300，FL 规则无需训练数据。

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
- 模型：distilbert-base-uncased 微调（或 fastText），输入前 512 token
- 训练数据：GDACS 时间 + 地理匹配的 silver labels + LLM 核验的 gold 验证集

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

| 灾种 | 特征 | 算法 | 参考 Macro-F1 |
|------|------|------|--------------|
| EQ | magnitude, depth, rapid_pop_people, rapid_pop_log, ... | RandomForest | ~0.91 |
| TC | wind_speed, storm_surge, exposed_population | RandomForest | ~0.78 |
| WF | duration_days, burned_area_ha, people_affected | RandomForest | ~0.77 |
| DR | duration_days, affected_area_km2, country_count | RandomForest | ~0.58（已知弱项）|
| FL | dead, displaced | 规则 | — |

模型：sklearn `Pipeline`（SimpleImputer median + RandomForestClassifier class_weight="balanced"），保存为 `.pkl`。

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

**数据**：时间切分（见 §3.4），val=1037 条，test=1678 条，均为 2025 年数据（时间上晚于训练集）。  
**Baseline**：V1THEMES 规则映射（无 not_related 类）。

### 5.2 NER / 参数抽取

| 指标 | 目标 |
|------|------|
| 字段解析率 | EQ magnitude ≥ 60%，FL dead ≥ 55% |
| 数值准确率 | 200条人工核验样本，±10% 内视为正确 |

**Baseline**：第一个数字 regex（不做单位换算）。

### 5.3 严重性分类

| 指标 | 说明 |
|------|------|
| Test Macro-F1 | 主要指标（各灾种独立评估） |
| ROC-AUC | 对 orange_or_red 类的区分能力 |
| 5-fold CV | 稳定性验证 |

**数据**：GDACS 历史数据，25% 测试集。**Baseline**：规则分类器（震级/死亡人数阈值）。

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
| 事件类型 | V1THEMES 规则映射 | distilbert 6类分类器 |
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
| V1THEMES 碎片化 / TC·DR 稀疏 | Stage 2 文本分类器从全文判断，不依赖 V1THEMES 作 event_type |
| 事件类型分类器 silver label 噪音 | GDACS 匹配设严格时间（±3天）+ 地理（≤200km）窗口；gold 验证集 500条 |
| regex 解析率低 | 先统计解析率；对 <40% 字段补充 regex 模式 |
| GDELT URL 失效率高 | 超时+重试；失败时保留 GKG 摘要；NewsAPI 补充 |
| Entity Linking 覆盖率不足 | pycountry + 人工别名字典（省份/缩写/常见别名）|
| 小国股票数据稀疏 | 回退到地区 ETF（如 EEM 新兴市场）|
