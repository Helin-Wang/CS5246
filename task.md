# Storms 'n' Stocks

## Background

Sudden natural disasters (e.g., earthquakes, floods, storms) and man-made disasters (e.g., terrorist attacks, major accidents) can have a significant effect on the stock market and the global economy in general: crucial supply chains might be interrupted, travel restrictions, relief efforts, etc. Quickly detecting and assessing such events can therefore be very useful when dealing with stocks. On the Web, information about such events can be available very early.

## Task scope

1. **Monitor** news or social media sources (e.g., Twitter accounts of governmental agencies).
2. **Extract** time, location, and type of event (e.g., earthquake).
3. **Estimate / assess** severity of event (e.g., strength of earthquake).
4. **Assess** potential impacts (e.g., affected industries near the event).

## Technical directions

A core component to address this task is certainly **Named Entity Recognition** as well as **keyword extraction**. **Clustering** might help to avoid overcounting multiple reports about the same event. Another related task is **entity linking**, i.e., to connect information (e.g., the name of a city or region) to a knowledge base to extract useful information.

## Evaluation (required)

The deliverable must include an **evaluation** section that makes results verifiable and comparable. At minimum, specify:

- **What** you measure (e.g., NER F1, clustering purity / event deduplication quality, severity calibration, downstream proxy for “usefulness” to stocks if applicable).
- **Data** used for evaluation (sources, train/dev/test split or cross-validation, and how labels or gold standards are obtained).
- **Baselines** (simple rules, off-the-shelf models, or ablations) so improvements are interpretable.
- **Results** in a clear form (tables or figures) and a short **analysis** of errors/limitations.

Adjust metrics and protocols to match your exact pipeline; the requirement is that evaluation is explicit, reproducible in description, and tied to your claims.
