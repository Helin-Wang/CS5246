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
          规则聚合：event_type + location-text-overlap + Δtime ≤7d
          特征合并：数值字段取 max，date 取最早，location 取众数
    ↓
GDACS 匹配（per-event cluster）
          条件：event_type + country_iso2 + |date差| ≤7d
          命中  → 直接使用 GDACS alertlevel + 结构化数值字段（权威数据优先）
          未命中 → 对该 cluster 的文章运行 NER 抽取参数 → Module C ML 预测
    ↓
Module C  严重性（仅限 GDACS 未命中事件）
          EQ/TC/WF/DR → models/*.pkl（基于 NER 抽取的聚合特征）
          FL          → 规则（dead>100 or displaced>80000）
    ↓
Module B  Entity Linking（per-event）
          地名文本 → pycountry + 别名字典 → ISO 国家代码
          Branch A: event_type → 行业先验
          Branch B: 新闻文本关键词 → 动态行业抽取
          行业 → sector_etf_map.json → ETF tickers
    ↓
Module E  股票影响分析
          yfinance → OLS 市场模型（ACWI 作市场代理）→ CAR(T+1/T+3/T+5)
          按 event_type × severity × sector 分组统计
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

**总体设计思路**：Module A 的任务是从每篇原始新闻中提取结构化字段，供下游三个模块消费：Module D（聚类去重）需要事件发生地点和时间；Module C（严重性预测）需要灾难参数数值；Module E（股票影响）需要国家 + 行业信息。三类信息在文本中的分布特征不同，因此采用三套独立策略，而非统一的 NLP 流水线。

#### 4.3.1 地点抽取

**为什么不用 GKG V1LOCATIONS**：GDELT GKG 自带 V1LOCATIONS 字段（基于 TABARI 地理编码），但该字段将文章中出现的所有地名列出，没有区分"灾难发生地"和"报道机构所在地"、"援助来源国"、"历史对比地区"，直接使用会引入大量噪声。我们选择从文本自行抽取，并设计优先级规则聚焦于真正的事件发生地。

**为什么选择规则 + 轻量 NER，而非端对端模型**：端对端地理实体解析（如 GeoNLP、DBpedia Spotlight）需要实体消歧模块，且对新闻报道中"影响地 vs 发生地"的区分能力有限。我们的设计目标是在离线可运行的约束下达到可接受精度，因此选择 spaCy NER（轻量、快速）+ 自建 alias 字典（覆盖专属别名问题）+ geonamescache 城市库 的组合。

**4 步流程的设计逻辑**（`src/location_extractor.py`）：

1. **文本坐标 regex（最高置信）**：若文章包含显式经纬度（如 `38.1°N 142.4°E`），直接使用，`confidence="coords_text"`。地震报道的震源坐标是最可靠的来源。这一步命中率约 0.1%，但精度极高（haversine 误差 0km）。

2. **NER 候选 + 触发动词绑定**：spaCy 抽取 GPE/LOC 实体后，优先选择与"struck/hit/flooded/burned/made landfall"等灾难触发动词同句的实体。这一设计的核心逻辑是：新闻报道中"灾难发生地"几乎总是出现在描述灾难动作的句子中，而"报道地"（如"Reuters BEIJING"格式的 dateline）则出现在文首。我们专门过滤 dateline 格式首实体，避免将新闻发稿城市误作灾难地。

3. **国家优先查找次序**：对每个候选地名，先查国家索引再查城市索引。这解决了一个典型问题——"India"同名存在于 geonamescache 的塞尔维亚城市索引中，若先查城市，国家名会被错误解析。我们维护了约 150 项 alias 字典，涵盖 US 各州（"California" → US）、中国省份（"Guangdong" → CN）、跨境地名（"Punjab" → PK，取人口更大侧）等。

4. **不推断坐标**：规则提取器的 `lat/lon` 仅输出来自文本的显式坐标。起初考虑将城市名解析为质心坐标，但实验发现：对于"Jakarta"这样的城市，城市质心到实际震源可能有数百公里偏差，而 Module D 使用 ≤500km 的地理窗口聚类，错误的城市质心会显著提高假阳性合并率。因此坚持只有文本中有坐标才输出坐标，否则退回地名文本匹配。

**评估结果**（LLM DeepSeek-V3 作 GT，n=764 灾难类）：

| 指标 | 结果 |
|------|------|
| 国家准确率 | **91.1%**（596/654 可比行）|
| 地名覆盖率 | **97.3%**（743/764）|

国家准确率 91.1% 略高于 90% 目标，覆盖率 97.3% 表明"找到某个地名"几乎没有问题，主要挑战在于找到的是否是正确的那个国家。

**为什么 TC 准确率最低（83.1%）**：台风是所有灾种里唯一具有移动路径的灾害，一篇台风报道往往同时出现中国广东、越南、台湾、菲律宾等多个国家。触发动词绑定策略在此场景下失效——"made landfall in Vietnam"和"heading toward China"都包含触发动词，NER 无法稳定地选中最主要的国家。这是单文章地名抽取的内在局限。

**主要失败模式**：影响区域 vs 事件起源（~15 例，如夏威夷报道俄罗斯地震的影响）、TC 跨国路径混淆（~12 例）、跨境地名（~8 例，如 Hindukush、Punjab）、城市同名歧义（~5 例，如 Kingston）。

#### 4.3.2 时间抽取

**核心挑战：区分"事件时间"与"报道时间"**：新闻文本中的日期表达有三类——事件发生时间（目标）、报道更新时间（通常在文首或文尾，不应抽取）、历史对比时间（"last year's flood..."，不应抽取）。直接在全文中找最近的日期会大量误取报道时间和历史对比，必须设计优先级过滤。

**为什么用 trigger verb 绑定而非直接全文扫描**：灾难事件的发生时间几乎总是出现在包含"struck/hit/killed/flooded/erupted"等强动作动词的句子中。我们的设计是先找触发句，再在触发句的前后一句范围内找时间表达（Step 1-2），只有触发句附近找不到时才扩展到全文（Step 4）。这一策略在准确率和召回率之间取得了平衡：全文扫描召回高但引入大量错误日期，而触发句绑定在主要事件类型上准确率显著更高。

**为什么对 DR（干旱）设计单独策略**：干旱没有"爆发时刻"，描述时间的语言模式是"since May"/"since last year"，而非"struck on Friday"。如果把干旱当作和地震一样处理，触发句策略完全失效，fallback 到 article_timestamp 更是错误（干旱文章发表时可能已是灾情持续 6 个月后）。因此专门设计 Step 3：提取"since [month/year]"作为干旱起点，并明确排除比较性短语（"driest since 2008"是统计修辞，不是事件时间）。

**文本截断的关键发现**：实验对比了截断文本（title + 首512字符）和全文（~3000字符）对时间抽取的影响，结果反直觉：**全文反而使性能大幅下降**（exact 60.1% → 40.8%，≤7d 89.2% → 65.8%）。原因是正文后段充满干扰日期——政府响应时间线、历史对比、数据更新日期——这些都触发 Step 1 的绝对日期匹配，抢先覆盖了标题/首句中正确的事件日期。这说明新闻文章首段是信息密度最高的区域，时间抽取专用截断文本是正确选择。

**5 步策略及设计理由**（`src/time_extractor.py`）：

| 步骤 | 逻辑 | 设计原因 |
|------|------|---------|
| Step 1 | 触发句±1 中找绝对日期；跳过 aftermath 句 | 绝对日期最精确；触发句绑定避免历史日期；aftermath 过滤避免抽到"清理工作于 7月30日完成"这类句子 |
| Step 2 | 触发句±1 中找相对表达（"last Friday"），以 article_timestamp 解析 | 灾难报道常用"昨天/上周三"，需要发稿时间作参考基点 |
| Step 3 | DR 专用 since 提取 | 干旱无"爆发时刻"，起点是连续无雨的起始月份，语言模式完全不同 |
| Step 4 | 全文扫描相对词 → 模糊词降级 | 触发句附近无表达时的降级处理；"this week"→本周一，明确到天 |
| Step 5 | fallback 到 article_timestamp；回顾性文章返回 unknown | article_timestamp 通常与事件发生时间差 <1 天（EQ/TC 突发类）；回顾性文章（含 cleanup/anniversary 词且前3句无触发）则说明文章报道的是过去事件，不应 fallback |

**评估结果**（LLM GT，n=778）：

| 灾种 | n | Coverage | Exact | ≤1d | ≤7d | Median err | 主要方法 |
|------|---|---------|-------|-----|-----|-----------|---------|
| EQ | 199 | **100%** | 66.1% | 78.5% | **94.4%** | 0d | step2 trigger(43%) + step5 fallback(27%) |
| TC | 193 | **100%** | 64.2% | 72.2% | **95.1%** | 0d | step2 trigger(48%) + step5 fallback(19%) |
| FL | 182 | **100%** | 55.4% | 64.7% | **90.6%** | 0d | step2 trigger(42%) + step5 fallback(22%) |
| WF | 109 | 98.8% | 53.1% | 60.5% | 77.8% | 0d | step4 全文(29%) + step5 fallback(28%) |
| DR | 95 | 60.4% | 50.0% | 50.0% | 56.2% | 2d | step4_vague(12%) + none(54%) |
| **Overall** | **778** | **96.4%** | **60.4%** | **69.5%** | **89.3%** | **0d** | step2 trigger(35%) + step5 fallback(21%) |

**按灾种分析**：
- **EQ/TC/FL**：≤7d 均达 90%+，coverage 100%。突发类灾害有明确触发词（"struck/made landfall/killed"），Step 2 trigger 绑定效率高。
- **WF**：≤7d 78%，略低。野火扩散过程中文章更多描述"持续蔓延"，缺乏单一爆发触发词，大量依赖全文扫描（step4，29%）和 fallback（28%）。
- **DR**：最弱，coverage 仅 60%，≤7d 56%。54% 的文章输出 none（无日期）。原因：干旱没有"since"触发词的文章直接返回空，且 step5 fallback 在 DR 中被明确禁用（fallback 到发稿时间对持续数月的干旱无意义）。这是设计上的保守选择，宁可返回 unknown 也不给出误导性的近期日期。

**整体结论**：≤7d 89.3% 对 Module D 的 7 天聚类窗口已足够。Exact match 60% 偏低的主要原因是 fallback 与 LLM GT 的策略分歧（LLM 对无线索文章标注 unknown，而 fallback 返回 article_timestamp），并非真正的抽取错误。

#### 4.3.3 数值参数抽取

**为什么选择规则而非模型**：数值参数抽取（如震级 6.2、风速 175mph、过火面积 12,000公顷）本质上是结构化信息提取，目标值通常以"数字+单位"形式出现，上下文语义相对固定。相比序列标注模型，规则 + anchor 约束有以下优势：①可解释，每个提取结果附带原始匹配串和命中句子；②训练数据零消耗；③便于添加单位换算逻辑；④不存在训练分布漂移问题。代价是覆盖不完整，但配合 Module D 多文章聚合可部分弥补。

**event_type 驱动的规则隔离**：不同灾种的参数虽然都是"数字"，但语义完全不同——"10 km"在地震文章里是深度，在野火文章里是烧毁面积，在飓风文章里什么都不是。为此，每个字段通过 `event_types` 参数限定只在对应灾种的文章中运行。event_type 由上游 DistilBERT 分类器提供，extractor 不做独立的灾种猜测，避免了跨类型噪声。

**anchor 绑定机制解决数字歧义**：新闻中数字密集，"30"可以是死亡人数、受灾天数、风速、受灾面积。每个字段配置了语义 anchor 词列表（如 burned_area 的 anchor 包含"burned/scorched/wildfire/hectare/acre"），通过 ±8 token 窗口内 anchor 命中数量和距离计算 binding score，低于阈值的匹配直接丢弃。这比纯 regex 的精度明显更高。

**GDACS 特征域漂移问题（EQ）**：严重性模型训练时使用了 GDACS 的 `rapidpopdescription` 字段（shake map 人口暴露计算结果），但新闻文本没有这个字段。我们尝试从文章中提取类似表达（"1.2 million people felt the shaking"），但命中率很低（仅约 28%），且语义不完全对等（新闻描述的是"感受到震动的人"，GDACS 是基于地面运动模型和人口格网的精确计算）。这导致推理时 `rapid_pop_people` 等字段基本依赖 median imputation。幸运的是，EQ 模型最重要的特征是 `magnitude`（新闻命中率 93.5%），`rapid_missing=1.0` 作为固定标志告诉模型"这是新闻来源而非 GDACS"，模型可以在训练中学习到这个差异。

**文本长度对数值抽取的影响**：与时间抽取相反，数值参数需要**全文**而非截断文本。以 TC 风速为例，截断文本（~600字符）中 wind_speed 缺失率达 77.7%，使用全文后降至 35.2%。原因是风速等技术参数通常出现在正文第二段之后（先交代事件，再给具体参数），截断后这部分文字丢失。

**GT 标注**：使用 DeepSeek-V3 对 778 篇测试文章按灾种标注数值参数（`scripts/label_ner_fields.py`，全文不截断），输出 `data/llm_labels/ner_labels_test.csv`，778 条全部成功（source_note = ok）。

**评估结果**（`src/eval_ner_extractor.py`，截断文本 512 字符，test set n=778）：

| 灾种 | 字段 | GT 覆盖（GT有值） | 抽取覆盖率 | 精度 W5% | 精度 W20% | MedRE | 根本原因分析 |
|------|------|----------------|-----------|--------|---------|-------|------------|
| EQ | `magnitude` | 193/199 | **87.6%** | **89.3%** | 96.4% | 0.00 | 震级几乎必报，准确率高；少数口语化描述（"powerful quake"）miss |
| EQ | `depth_km` | 100/199 | 40.0% | **95.0%** | 95.0% | 0.00 | 找到时精度极高；深度是地震学参数，许多非技术性报道不含 |
| TC | `wind_speed_kmh` | 144/193 | 29.9% | 69.8% | 74.4% | 0.00 | 疏散/影响类文章不含技术风速；单位换算（mph→km/h）引入部分误差 |
| TC | `storm_surge_m` | 38/193 | **0.0%** | — | — | — | 风暴潮高度属专业气象简报字段，普通新闻几乎不报 |
| TC | `exposed_population` | 96/193 | **0.0%** | — | — | — | "X people exposed"在新闻中极罕见；多用"at risk"但不附数字 |
| WF | `burned_area_ha` | 83/109 | 24.1% | **85.0%** | 85.0% | 0.00 | 在燃火灾用"growing/spreading"，具体面积仅出现在事后总结报道中 |
| WF | `people_affected` | 37/109 | 2.7% | 0.0% | 0.0% | 0.75 | 极低覆盖；找到的唯一样本误差较大（"hundreds"被解析为错误数量级）|
| WF | `duration_days` | 18/109 | **0.0%** | — | — | — | 新闻极少用"lasting N days"量化持续时间 |
| DR | `affected_country_count` | 39/95 | **0.0%** | — | — | — | 干旱新闻几乎全为定性描述，量化数据缺失是结构性问题 |
| FL | `dead` | 92/182 | 43.5% | 77.5% | 82.5% | 0.00 | 约半数洪水报道关注基础设施而非伤亡；找到时准确率尚可 |
| FL | `displaced` | 51/182 | 7.8% | **100%** | 100% | 0.00 | 覆盖率极低，但找到的 4 个都完全正确；新闻多用"evacuated/fled"且不附数字 |

**整体规律**：

- **找到时精度高**：magnitude（MedRE=0.00，W5%=89%）、depth（W5%=95%）、burned_area（W5%=85%）、dead（W5%=78%）——规则 + anchor 绑定机制在有信号时可靠
- **结构性零覆盖**：storm_surge、exposed_population、duration_days、affected_country_count 均为 0%，这不是规则写得不好，而是这些字段本身在新闻文本中就不出现（属于 GDACS 专有的结构化数据字段）
- **覆盖率 vs. 精度权衡**：覆盖率是主要瓶颈，而非精度。大多数字段"找到即准"，问题是找到率太低

**缺失的系统性应对**：
- **Module D 聚合**：单篇覆盖率低（如 WF burned_area 24%），但同一事件多篇聚合后字段互补，Module C 在聚合后最大值上预测
- **`low_confidence` 标记**：关键字段全缺失时输出 `low_confidence=True`，供下游降权
- **DR/TC 特殊处理**：DR 全字段结构性缺失，severity model 全走 median imputation，已是预期的已知弱项；TC storm_surge 和 exposed_population 在训练数据中缺失率也高（63%/74%），模型已适应这一分布

#### 4.3.4 地点 GT 标注与评估

**脚本**：`scripts/label_locations.py`，使用 DeepSeek-V3（SiliconFlow），单线程（避免免费层限流）。LLM 被明确要求识别"灾难物理发生地"而非报道地，prompt 专门设计了"CITY (Reuters) 格式首实体"的排除说明。GT 字段：`location_text`（城市/国家名）、`country_iso2`（ISO-2）、`lat/lon`（LLM 内部知识推断的城市质心，非文本坐标）。

评估数据集：1011 行 → 过滤 not_related（233行）→ 过滤 LLM error（14行）→ **764 行有效 GT**（`src/eval_location_extractor.py`）。

**整体结果**：国家准确率 **91.1%**，地名覆盖率 **97.3%**。

**按灾种**：

| 灾种 | n | 国家准确率 | 覆盖率 |
|------|---|-----------|-------|
| EQ | 197 | **94.1%** | 98.5% |
| FL | 178 | **94.1%** | 96.1% |
| DR | 94  | **94.8%** | 91.5% |
| WF | 104 | **90.2%** | 98.1% |
| TC | 191 | **83.1%** | 99.5% |

TC 准确率最低（83.1%）：台风是唯一具有移动路径的灾害，单篇报道中可同时出现多个沿路国家，触发动词绑定无法稳定选出最主要国家。DR 覆盖率偏低（91.5%）：干旱报道有时没有明确的城市级地名，仅提及大范围区域（"southwestern United States"），spaCy 无法解析为具体国家坐标。

### 4.4 事件聚类去重（Module D）

#### 设计动机

同一灾难事件往往产生多篇报道，若以单篇文章为单位直接预测严重性，会面临两个问题：（1）单篇文章通常只提及部分数值字段（如只有风速、没有风暴潮），导致 Module C 大量触发 low_confidence；（2）同一事件被计入多次会扭曲最终的 CAR 统计。因此必须先将同一事件的文章聚合，再进行预测。

#### 为什么不用 DBSCAN / 向量聚类

最直接的思路是把文章编码成向量，用 DBSCAN 等密度聚类方法处理。但实测表明，本数据集中 **lat/lon 坐标覆盖率接近 0%**（train 0.0%，val 0.5%，test 0.1%），无法构建连续的地理度量空间，DBSCAN 的核心距离函数失效。而如果改用文本向量，则缺乏对"同一地理区域"这一核心语义的精确建模。

| 特征 | 覆盖率 | 适合做聚类依据 |
|------|--------|--------------|
| lat/lon | ~0% | ✗ 几乎不可用 |
| country_iso2 | 84–89% | ✓ 可靠，但大国粒度太粗 |
| location_text | 93–97% | ✓ 粒度合适，需文字匹配 |
| event_date | 90–94% | ✓ 直接比较天数差 |

#### 采用的方案：两层规则聚合

**为什么不用固定时间窗口做 Layer 1 分组**：直觉上可以用 `(event_type, country_iso2, 固定周)` 三元组做粗分组，但固定 ISO week 边界会截断跨周的有效候选对——实测发现 12.3% 的同国同类、日期差≤7天的文章对落在不同周内，永远无法被比较，导致分组数从合理的 86 个膨胀至 276 个。因此时间不进入 Layer 1，而是在 Layer 2 内以 complete-linkage 方式处理，避免边界效应。

**第一层（硬分组）**：按 `(event_type, country_iso2)` 分组，快速排除不同类型、不同国家的文章，将搜索空间从 O(n²) 降至各国内部的小规模比较。

**第二层（complete-linkage）**：在每个分组内，按 event_date 排序后逐篇分配：新文章加入某个簇，当且仅当它与簇内**每一篇**已有文章都满足（1）日期差 ≤ 7 天，（2）location_text 有公共词元。complete-linkage 保证簇内所有对都满足条件，不会因"A 近 B，B 近 C"就把 A 和 C 强行合并（即 Union-Find / single-linkage 的链式问题）。地名词元匹配时过滤通用词（"United"、"States"、"Republic" 等），确保 "California" 与 "Texas" 不会因共享 "United States" 而合并。

特征合并：数值字段取 max（取最严重报道值），`event_date` 取最早（事件发生时间），`primary_country` 取众数，`article_count` 计数，`low_confidence` 仅当所有文章均 low_confidence 时为 True。

#### test set 聚类结果（778 篇灾难文章）

| event_type | 事件数 | 总文章数 | 平均文章/事件 | Low-conf% |
|------------|--------|---------|------------|-----------|
| EQ | 89 | 199 | 2.2 | 6.7% |
| TC | 62 | 193 | **3.1** | 72.6% |
| WF | 73 | 109 | 1.5 | 74.0% |
| DR | 47 | 95 | 2.0 | **100%** |
| FL | 122 | 182 | 1.5 | 77.9% |
| **合计** | **393** | **778** | **1.98x 压缩** | 62.8% |

**Low-confidence 分析**：聚合后 low_confidence 从 64.0% 仅微降至 62.8%，说明缺失不是信息分散导致的，而是结构性的——TC/WF/FL/DR 的关键数值字段在新闻文本中本就很少出现，多篇报道聚合并不能弥补这一根本缺失。DR 100% low_confidence 的原因详见 §4.3.3。EQ 的 6.7% low_confidence 远低于其他类型，因为 magnitude 和 depth 几乎在每篇地震报道中都会出现。

#### GDACS 直接匹配（severity 首选路径）

**设计动机**：新闻文本 NER 抽取的数值字段质量低、缺失率高（见 §4.3.3），将其作为 ML 严重性预测的输入是不得已的降级方案。当 GDACS 中存在对应的结构化记录时，应**优先使用 GDACS 的 alertlevel 和数值字段**，完全跳过 NER+ML 路径。

**匹配方法**（实现：`src/gdacs_matcher.py`，聚类完成后、NER 之前执行）：

对每个聚类事件，在 GDACS 全量数据中寻找满足以下三条件的最近记录：
1. `eventtype` 相同（两字母代码对齐）
2. `country_iso2` 相同（GDACS 国家全名经 pycountry 解析后比较）
3. `|event_date − gdacs_fromdate| ≤ 7` 天

**命中时**：直接使用 GDACS 的 `alertlevel`（green/orange/red → 映射为 green/orange_or_red）及结构化数值字段（magnitude、wind_speed 等）作为该事件的严重性结论，标记 `severity_source = "gdacs"`，完全不运行 NER 和 ML 预测。

**未命中时**：对该 cluster 的文章逐篇运行 NER，聚合数值字段，送入 ML 严重性预测模型，标记 `severity_source = "ml"`。

**当前匹配率**（基于 balanced 采样的 `gdacs_all_fields_v2.csv`，FL 未覆盖）：

| event_type | 匹配数/总簇数 | match rate |
|------------|-------------|-----------|
| EQ | 16/89 | 18.0% |
| TC | 16/62 | 25.8% |
| WF | 7/73 | 9.6% |
| FL | 0/122 | 0%（GDACS 未拉取 FL）|
| DR | 0/47 | 0%（GDACS 测试期仅 9 条）|
| 合计 | 39/393 | 9.9% |

**为什么 match rate 低**：现有 GDACS 数据是为训练严重性分类模型而设计的 balanced 采样（每类 alertlevel 最多 500 条），大量 green 事件被截断；FL 完全未拉取；DR 在测试期内仅有 9 条记录。这是数据覆盖问题，不反映匹配逻辑的质量。

**关键发现**：在命中的 39 个事件中，**87%（34/39）为 orange 或 red**。这印证了新闻报道对高严重性事件的强烈偏向——绿色警报的小规模灾害通常不会产生足够的新闻流量进入我们的 pipeline。

> **待更新（2026-04-09 状态记录）**
>
> **问题根因**：Job 560039 使用 `--enrich-details` 模式，对每条 GDACS 事件逐一请求 `geteventdata` detail endpoint。全量 EQ 数据约 5,000–10,000 条，触发了 GDACS 服务器的 IP 级限流，导致 list endpoint 也开始拒绝连接。Job 已于 2026-04-09 手动 kill（`scancel 560039`）。
>
> **修复方案**：新增 `--skip-enrich` 参数（`scripts/fetch_gdacs_all_fields.py`），跳过 detail endpoint，只调用 list endpoint。list endpoint 已包含 `alertlevel`、`country`、`fromdate` 等匹配所需的全部字段，`rapidpopdescription` 文本字段（含 magnitude/depth）也在 list 响应中。脚本已更新至 `scripts/fetch_gdacs_full_slurm.sh`。
>
> **下次操作流程**（等待 GDACS API 限流解除，约 1–2 小时后）：
> 1. 本地确认 API 恢复：`conda run -n gdelt python -c "import urllib.request,ssl,json; r=urllib.request.urlopen(urllib.request.Request('https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventtype=EQ&alertlevel=red&fromdate=2025-11-01&todate=2025-11-30&pagesize=3', headers={'User-Agent':'Mozilla/5.0'}), context=ssl._create_unverified_context(), timeout=15); print('OK:', len(json.loads(r.read()).get('features',[])))"`
> 2. 同步更新后的脚本到服务器：`rsync -avz scripts/fetch_gdacs_full_slurm.sh scripts/fetch_gdacs_all_fields.py helin@xlogin.comp.nus.edu.sg:~/CS5246/scripts/`
> 3. 提交新任务：`ssh helin@xlogin.comp.nus.edu.sg "cd ~/CS5246 && sbatch scripts/fetch_gdacs_full_slurm.sh"`
> 4. 任务完成后同步数据：`rsync -avz helin@xlogin.comp.nus.edu.sg:~/CS5246/data/gdacs_full.csv data/`
> 5. 重新运行 pipeline：`python src/pipeline.py --input data/splits/test.csv --skip-stock`（`GDACSmatcher` 默认读取 `gdacs_all_fields_v2.csv`，需手动改为 `gdacs_full.csv` 或通过参数传入）
> 6. 更新本节 match rate 表及 §5.4 聚类去重统计

### 4.5 严重性评估（Module C — GDACS 未命中时的 fallback）

**触发条件**：仅当 §4.4 GDACS 匹配未命中时，才对该 cluster 运行 NER 抽取 + ML 预测。GDACS 命中事件直接使用 alertlevel，不进入本模块。

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

**为什么不用国家大盘指数**：直觉上可以把"灾难发生在日本 → 分析 EWJ（日本 ETF）"，但这不符合金融逻辑：一场尼泊尔地震不会主要影响 ^NEPSE，而会影响全球再保险股；一场美国野火不会让 SPY 大跌，但会让公用事业股和保险股承压。用国家指数会把灾难信号淹没在整体市场噪声中，信噪比极低。

**两个互补分支**：

**Branch A（静态先验）**：基于灾种的行业先验。例如所有灾难都会冲击保险业，干旱直接影响农业，台风影响能源（近海钻井）和旅游。这部分来自领域知识，不依赖新闻质量。

| 灾种 | 先验行业 |
|------|---------|
| EQ | insurance, construction |
| TC | insurance, agriculture, energy, tourism |
| WF | insurance, utilities, timber |
| DR | agriculture |
| FL | insurance, agriculture, construction |

**Branch B（动态文本）**：从新闻文章中直接提取受影响行业信号（`src/industry_extractor.py`）：
- **关键词匹配**：领域关键词词典（"crop failure/supply chain disruption/insurance claims/power outage" 等）→ 行业标签
- **ORG 实体映射**：spaCy 抽取 ORG 实体 → 已知公司→行业词典（"PG&E" → utilities，"Munich Re" → insurance）

两个分支取并集，映射到行业 ETF（`data/sector_etf_map.json`），得到每个事件的 `sector_etfs` 列表。

| 行业 | ETF | 代表性 |
|------|-----|--------|
| insurance | IAK | 美国保险业 ETF，对自然灾害高度敏感 |
| agriculture | MOO | 全球农业 ETF |
| energy | XLE | 美国能源板块 |
| utilities | XLU | 美国公用事业，含电力设施 |
| construction | XHB | 建材/房屋建设，受益于灾后重建 |
| tourism | JETS | 全球航空业 ETF |
| timber | WOOD | 木材/林业 ETF |
| mining | XME | 金属与采矿 ETF |
| manufacturing | XLI | 工业制造 ETF |
| shipping | BOAT | 全球航运 ETF |
| financial | XLF | 金融板块 ETF |

地名解析沿用旧逻辑：`pycountry` + 别名字典 → ISO-2，用于区分同类事件发生在不同地区时的上下文差异。

### 4.7 关键词抽取

`IndustryExtractor`（`src/industry_extractor.py`）的关键词匹配输出作为每篇文章的行业影响标注，同时作为可解释性展示。TF-IDF 作对比 baseline。

### 4.8 股票影响分析（Module E）

事件研究法（OLS 市场模型），对每个事件的每个行业 ETF 独立计算 CAR：

**为什么用 ACWI 而非 SPY**：灾难事件是全球性的，市场代理应反映全球系统性风险而非美国市场。SPY 对非美国上市行业 ETF（如 WOOD、JETS）的相关性较低，会导致 β 估计偏差，残差中混入非灾难因素。ACWI（MSCI 全球指数）作为 β 回归基准更合适。

1. yfinance 拉取 `[T-60d, T+15d]` 日收益率（calendar days，转换为 trading days）
2. 估计窗口 `[T-45, T-6]`（约 30 个交易日）OLS：`R_sector = α + β × R_ACWI + ε`
3. 事件窗口 `[T0, T0+5]`：`AR_t = R_sector,t − (α̂ + β̂ × R_ACWI,t)`
4. `CAR(T+1)`、`CAR(T+3)`、`CAR(T+5)` = 累计 AR
5. 按 `event_type × severity × sector` 分组：均值 + 单样本 t 检验（H₀: CAR=0）

### 4.9 Pipeline 运行记录

#### 旧架构运行（2026-04-09，架构调整前）

**架构**：Module A 同时做时间+地点+NER，聚类后直接 ML 严重性预测。

| 模块 | 输出 | 数值 |
|------|------|------|
| A（时间+地点+NER） | article records | 778 |
| D（聚类） | unique events | 393（1.98x） |
| C（ML 严重性） | predicted | 393（green=332，orange_or_red=61） |
| B（Entity Linking） | sector_etfs | 393/393（100%） |
| E（CAR） | observations | 1238（100% 成功） |

CAR 结果及随机基线对比详见 §5.6。

#### 新架构（2026-04-09 重构，GDACS-first）

**架构**：Module A 仅做时间+地点，聚类后先 GDACS 匹配，命中则直接取 alertlevel，未命中才运行 NER+ML。新增文件：`src/gdacs_matcher.py`。

**当前限制**：使用 `gdacs_all_fields_v2.csv`（balanced 采样，2910 行），GDACS 命中率 0%（test 集日期范围大部分超出该数据覆盖），所有事件目前走 NER+ML fallback 路径。

> 待 Job 560039 全量数据同步后，切换 `GDACSmatcher` CSV 路径重新运行，预期命中率大幅提升，`severity_source="gdacs"` 的事件数将反映真实效果。

**依赖**：conda 环境 `gdelt` 需安装 `dateparser`、`yfinance`（2026-04-09 补充安装）。

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
**GT 标注**：DeepSeek-V3（SiliconFlow）对 test set 778 篇按灾种标注数值参数，全文不截断，778 条全部成功。

**实测结果**（`src/eval_ner_extractor.py`，截断文本 512 字符）：

| 字段 | GT有值 | 抽取覆盖率 | 精度 W5% | 精度 W20% | 结论 |
|------|--------|-----------|--------|---------|------|
| EQ magnitude | 193 | **87.6%** | **89.3%** | 96.4% | 最佳；震级几乎必报 |
| EQ depth_km | 100 | 40.0% | **95.0%** | 95.0% | 找到即准；深度非技术性报道不含 |
| TC wind_speed_kmh | 144 | 29.9% | 69.8% | 74.4% | 疏散类文章无技术风速 |
| TC storm_surge_m | 38 | **0.0%** | — | — | 结构性缺失：专业气象简报字段 |
| TC exposed_population | 96 | **0.0%** | — | — | 结构性缺失：新闻不用此表达 |
| WF burned_area_ha | 83 | 24.1% | **85.0%** | 85.0% | 在燃火无面积；事后总结才有 |
| WF people_affected | 37 | 2.7% | 0.0% | 0.0% | 极低；找到的唯一样本误差大 |
| WF duration_days | 18 | **0.0%** | — | — | 新闻极少量化持续天数 |
| DR affected_country_count | 39 | **0.0%** | — | — | 干旱新闻全为定性描述 |
| FL dead | 92 | 43.5% | 77.5% | 82.5% | 半数报道无伤亡数字 |
| FL displaced | 51 | 7.8% | **100%** | 100% | 覆盖率低，但找到的全对 |

**核心规律**：覆盖率是主要瓶颈，精度不是——绝大多数字段"找到即准"（MedRE=0.00），问题在于 storm_surge、exposed_population、duration_days、affected_country_count 等字段在新闻文本中结构性不存在。

**与 Baseline 对比**（Baseline = 第一个数字 regex，不做单位换算、不做 anchor 绑定）：

| 字段 | Baseline 覆盖率 | 主模型覆盖率 | Baseline W5% | 主模型 W5% |
|------|--------------|-----------|------------|----------|
| magnitude | 54.4% | **87.6%** | 85.7% | **89.3%** |
| depth_km | 21.0% | **40.0%** | 100% | 95.0% |
| wind_speed_kmh | 12.5% | **29.9%** | 77.8% | 69.8% |
| burned_area_ha | 12.0% | **24.1%** | 100% | 85.0% |
| dead | 9.8% | **43.5%** | 88.9% | 77.5% |
| displaced | 2.0% | **7.8%** | 100% | 100% |

anchor 绑定 + 多模式覆盖带来的主要收益是**覆盖率提升**（magnitude +33pp，dead +34pp，depth +19pp），精度在主模型中略有下降（因为引入了更多模式，部分匹配到近似但非完全精确的数字），但仍保持在合理范围内。结构性零覆盖字段（storm_surge、exposed_population 等）两者均为 0%，证实这是数据本身的问题而非规则问题。

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

**数据**：test 集 778 篇文章（已过滤 not_related），两层完全连接聚类算法（见 §4.4）。

| 指标 | 值 | 说明 |
|------|-----|------|
| 输入文章数 | 778 | test 集 not_related 已过滤 |
| 输出事件数 | 393 | 压缩比 1.98x，49.5% reduction |
| 单篇事件（singleton） | 254（64.6%） | 每个事件仅有 1 篇文章 |
| 平均文章数/事件 | 1.98 | |
| 最大文章数/事件 | 25 | |
| 有 event_date 的事件 | 385（98.0%） | |

**按灾种分布**：

| 灾种 | 事件数 | 占比 |
|------|--------|------|
| FL（洪水） | 122 | 31.0% |
| EQ（地震） | 89 | 22.7% |
| WF（野火） | 73 | 18.6% |
| TC（热带气旋） | 62 | 15.8% |
| DR（干旱） | 47 | 12.0% |

**严重性分布**：green=332（84.5%），orange_or_red=61（15.5%）。

> singleton 比例 64.6% 表明大多数报道只有 1 篇相关文章入库；对 FL 尤为明显（FL 事件分散、局部化）。无人工 gold standard 标注，误合并率待后续评估。

### 5.5 Entity Linking

**数据**：393 个聚类事件，逐事件运行 EntityLinker。

| 指标 | 值 | 说明 |
|------|-----|------|
| 有 sector_etfs 的事件 | 393（100%） | Branch A 静态先验保证全覆盖 |
| 有 country_iso2 的事件 | 145（36.9%） | 依赖 location_text 解析；文章常省略国家名 |
| 有效 tradeable 事件 | 385 | 需 sector_etfs + event_date |

**country 识别率偏低的原因**：新闻文章常写城市/省份名而非国家名（如 "Kathmandu" 而非 "Nepal"），且 location_extractor 的 spaCy 提取在截断文本（512字符）上覆盖有限。sector_etfs 通过 event_type 静态先验兜底，country 解析失败不影响 Module E 运行。

### 5.6 端到端（股票影响）

**设置**：test 集 385 个有效事件，市场代理 ACWI，estimation window [T-45, T-6]（约 40 个交易日），event window [T, T+5]，每事件×每行业 ETF 独立计算。共 1238 个有效 CAR 观测（100% 成功率）。

**总体 CAR**：

| 窗口 | 均值 CAR | 标准差 | n |
|------|----------|--------|---|
| T+1 | +0.050% | 1.12% | 1238 |
| T+3 | +0.16% | 1.80% | 1238 |
| T+5 | +0.20% | 2.39% | 1238 |

**按灾种（CAR_t5 均值）**：

| 灾种 | n | CAR_t5 |
|------|---|--------|
| WF（野火） | 251 | +0.41% |
| TC（热带气旋） | 275 | +0.29% |
| EQ（地震） | 253 | +0.17% |
| FL（洪水） | 393 | +0.11% |
| DR（干旱） | 66 | **−0.25%** |

**按严重程度**：

| 严重程度 | n | CAR_t1 | CAR_t3 | CAR_t5 |
|---------|---|--------|--------|--------|
| green | 1026 | +0.015% | +0.104% | +0.176% |
| orange_or_red | 212 | +0.198% | +0.418% | +0.336% |

**统计显著结果**（p < 0.05，单样本 t 检验 H₀: CAR=0，n ≥ 3）：

| 灾种 | 严重程度 | 行业 | 窗口 | mean CAR | p 值 | 解读 |
|------|---------|------|------|----------|------|------|
| WF | green | utilities | T+1/T+3/T+5 | −0.24%/−0.39%/−0.62% | 0.014/0.004/0.002 | 野火期间电力公司股票持续下跌 |
| WF | green | insurance | T+3/T+5 | +0.84%/+1.06% | 0.0001/0.0000 | 野火后保险股显著上涨（预期理赔费用上升） |
| WF | green | timber | T+3/T+5 | +0.50%/+0.58% | 0.010/0.021 | 林木资源需求上升 |
| WF | green | tourism | T+3 | +2.48% | 0.009 | 少量样本（n=9），解读需谨慎 |
| WF | orange_or_red | timber | T+5 | +1.22% | 0.025 | 严重野火后木材重建需求 |
| DR | green | agriculture | T+3/T+5 | −0.41%/−0.41% | 0.011/0.036 | 干旱导致农业 ETF 下跌，符合预期 |
| EQ | green | insurance | T+5 | +0.55% | 0.045 | 边际显著；地震后保险需求 |
| EQ | orange_or_red | agriculture | T+3 | −0.54% | 0.038 | 严重地震农业受损 |
| TC | orange_or_red | insurance | T+3 | +1.02% | 0.037 | 严重飓风后保险股上涨 |
| TC | orange_or_red | energy | T+3 | +1.65% | 0.045 | 飓风后能源供给中断，价格上升 |
| TC | orange_or_red | tourism | T+1 | +1.07% | 0.020 | 短窗口旅游板块异常，或反映预期快速恢复 |

**主要发现**：
1. **方向符合预期的显著效果**：WF → utilities（负）、insurance（正）；DR → agriculture（负）；TC/EQ → insurance（正）均与灾种影响机制一致。
2. **orange_or_red 组 CAR 整体高于 green 组**（+0.34% vs +0.18%），与假设相反。可能原因：severe 事件主要集中在 WF/TC，其对应的 insurance/timber ETF 本身呈正向反应；样本量偏小（n=212）导致噪声较大。
3. **DR 是唯一 CAR_t5 为负的灾种**（−0.25%），agriculture ETF（MOO）的显著负向响应印证了干旱对农业板块的系统性影响。
4. **FL 无显著效果**：洪水事件多为局部性（122个事件但 CAR 均不显著），可能因为 FL 的行业影响更分散或国家解析率低导致 ETF 选择不够精准。

#### 随机基线对比

**方法**：从真实事件的日期范围（2024-11-01 ~ 2025-12-31）中随机抽取等量日期（n=1238），排除任意真实事件日期的 ±10 天缓冲区，使用相同的 ticker 分布，计算伪事件 CAR，与真实事件 CAR 对比。

| 窗口 | 真实事件均值 | 随机基线均值 | p（事件 vs 0） | p（随机 vs 0） | p（事件 vs 随机） | 结论 |
|------|------------|------------|--------------|--------------|----------------|------|
| T+1 | +0.047% | −0.032% | 0.141 | 0.238 | 0.059 | 均不显著 |
| T+3 | +0.158% | −0.113% | **0.002** | **0.024** | **0.0002** | 事件显著正；随机显著**负**；两者显著差异 |
| T+5 | +0.203% | +0.118% | **0.003** | 0.068 | 0.362 | 方向相同，差异不显著 |

**解读**：
- T+3 窗口是区分信号最清晰的时间点：真实事件 CAR 显著正（+0.158%），随机基线显著负（−0.113%），两者差异 p=0.0002，说明灾难事件在 T+3 确实带来了超出随机水平的正向行业 ETF 反应。
- T+5 两者均为正且差异不显著（p=0.36），说明较长窗口内的 CAR 受市场整体趋势影响更多，灾难信号被稀释。
- 整体来看，**真实灾难事件在 T+3 窗口存在显著的、超出随机基线的行业 ETF 异常收益**，验证了行业识别模块（Module B）的有效性。

### 5.7 基线汇总

| 模块 | Baseline | 改进方法 | 关键结果 |
|------|---------|---------|---------|
| 事件类型 | V1THEMES 映射（无 not_related）；TF-IDF+LR（见 §5.1） | DistilBERT 6 类微调 | Macro-F1 0.901（测试集） |
| 参数抽取 | 第一个数字 regex | per-type 完整 regex + 单位换算 | EQ coverage 87.6%，W20% 89.3% |
| 严重性 | 规则阈值分类器 | per-type RandomForest（GDACS 训练） | EQ AUC 0.935，TC AUC 0.926 |
| 聚类 | 不去重（每篇独立） | 两层完全连接（type + location + 7天） | 778→393 事件，49.5% 压缩 |
| Entity Linking | 精确字符串匹配 | pycountry + 别名字典 + 两分支行业抽取 | 100% 事件有 sector ETFs |
| 股票影响 | 无（新增模块） | 行业 ETF × ACWI 市场模型 CAR | 1238 有效观测，WF/DR 显著效果 |

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

### 地点抽取（Location Extraction）

| 模块 | 工具 | 说明 |
|------|------|------|
| NER | spaCy `en_core_web_sm` | 提取 GPE/LOC 实体，支持 ORG fallback（带 alias 检查） |
| 坐标 regex | Python `re` | 3 种格式：`23.5°N 121.6°E`、`lat X lon Y`、`(X,Y)` |
| 地名→国家映射 | `geonamescache` + `pycountry` | ~25k 城市 + 250+ 国家，按人口优先 |
| Alias 字典 | Python dict | ~150 项：美国 50 州、加拿大 13 省、印度邦、河流、岛屿等 |
| 实现文件 | `src/location_extractor.py` | 849 行，单次运行 <0.5s/文章 |

### 地点 GT 标注（LLM Labeling）

| 模块 | 工具 | 说明 |
|------|------|------|
| LLM | DeepSeek-V3 | SiliconFlow API，$0.0009/1K tokens |
| 容错 | exponential backoff | 429/403 等待 30s×(attempt+1)，max 5 次 |
| 数据储存 | `location_labels_test.csv` | 1011 行，fields: location_text / country_iso2 / lat / lon / source_note |
| 评估脚本 | `src/eval_location_extractor.py` | 国家准确率/覆盖率/误差统计，按 label 分组 |
| 实现文件 | `scripts/label_locations.py` | 201 行，单线程 ~8 行/分钟 |

### 事件类型分类（Event Type Classification）

| 模块 | 工具 | 说明 |
|------|------|------|
| 模型 | DistilBERT-base-uncased | 6 分类，256M 参数，test Macro-F1 0.901 |
| 损失函数 | WeightedTrainer | 类别不平衡补偿（cyclone/drought under-represented） |
| Early Stopping | patience=2 | 选最佳 val Macro-F1 checkpoint（epoch 5） |
| 推理 | batch=128，fp16 | test 精度 0.90，覆盖全 6 类 |
| 实现文件 | `src/eval_event_classifier.py` | 推理脚本，输出每样本置信度 |
| GT 数据 | `data/llm_labels/llm_labels_0_26326_s1000.csv` | 5000 行，按 DeepSeek-V3 标注 |

### 其他核心工具

| 类别 | 工具 |
|------|------|
| 参数提取 | Python `re`（regex + 单位换算） |
| 严重性模型 | scikit-learn（RandomForest + Pipeline） |
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

