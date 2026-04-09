"""
Generate all tables and figures for the Storms 'n' Stocks report.

Outputs:
  figures/fig1_car_by_type_severity.png   — CAR bar chart by event type & severity
  figures/fig2_car_heatmap_sector.png     — Heatmap: significant CAR by sector
  figures/fig3_pipeline_coverage.png      — Pipeline coverage funnel
  tables/table_event_classifier.md        — DistilBERT per-class results
  tables/table_severity_classifiers.md    — EQ/TC/WF/DR classifier results
  tables/table_ner_eval.md               — NER field parse rate & accuracy
  tables/table_location_time_eval.md     — Location + time extractor eval
  tables/table_car_significant.md        — Significant CAR findings

Usage:
    python scripts/generate_report_figures.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
BLUE   = "#2196F3"
RED    = "#F44336"
GRAY   = "#9E9E9E"

ETYPE_LABELS = {"EQ": "Earthquake", "TC": "Cyclone", "WF": "Wildfire",
                "DR": "Drought", "FL": "Flood"}
SEV_LABELS   = {"green": "Green (minor)", "orange_or_red": "Orange/Red (severe)"}


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Event Type Classifier (DistilBERT)
# ══════════════════════════════════════════════════════════════════════════════
def table_event_classifier():
    data = {
        "Class":      ["Earthquake", "Wildfire", "Cyclone", "Flood", "Drought", "Not Related", "**Macro avg**"],
        "Precision":  [0.95, 0.91, 0.91, 0.93, 0.86, 0.81, "**0.90**"],
        "Recall":     [0.97, 0.99, 0.94, 0.91, 0.91, 0.73, "**0.91**"],
        "F1":         ["**0.960**", "**0.947**", "**0.926**", "**0.922**",
                       "**0.882**", "**0.770**", "**0.901**"],
        "Support":    [199, 109, 193, 182, 95, 233, 1011],
    }
    df = pd.DataFrame(data)
    md = df.to_markdown(index=False)
    (TAB_DIR / "table_event_classifier.md").write_text(md)
    print("✓ table_event_classifier.md")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Severity Classifiers
# ══════════════════════════════════════════════════════════════════════════════
def table_severity_classifiers():
    data = {
        "Type": ["EQ", "TC", "WF", "DR"],
        "Split":       ["random", "time", "random", "time"],
        "Test Macro-F1": [0.878, 0.815, 0.798, "~0.51 (weak)"],
        "ROC-AUC":     [0.935, 0.926, 0.960, "—"],
        "5-fold CV":   ["0.920±0.019", "0.634±0.062", "0.779±0.098", "0.508±0.063"],
        "Train rows":  [670, 1438, 544, 258],
        "Notes": [
            "rapidpopdescription null 36%",
            "val only 4 orange rows",
            "orange_or_red globally only 44",
            "Known weak; test set too small",
        ],
    }
    df = pd.DataFrame(data)
    md = df.to_markdown(index=False)
    (TAB_DIR / "table_severity_classifiers.md").write_text(md)
    print("✓ table_severity_classifiers.md")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — NER field parse rate & accuracy
# ══════════════════════════════════════════════════════════════════════════════
def table_ner_eval():
    ner = pd.read_csv(ROOT / "data/results/ner_extractor_eval.csv")

    FIELD_MAP = {
        "magnitude":          ("EQ",  "Magnitude"),
        "depth_km":           ("EQ",  "Depth (km)"),
        "wind_speed_kmh":     ("TC",  "Max wind speed (km/h)"),
        "storm_surge_m":      ("TC",  "Storm surge (m)"),
        "exposed_population": ("TC",  "Exposed population"),
        "burned_area_ha":     ("WF",  "Burned area (ha)"),
        "people_affected":    ("WF",  "People affected"),
        "duration_days":      ("DR",  "Duration (days)"),
        "affected_area_km2":  ("DR",  "Affected area (km²)"),
        "dead":               ("FL",  "Deaths"),
        "displaced":          ("FL",  "Displaced"),
    }

    rows = []
    for field, (etype, label) in FIELD_MAP.items():
        gt_col   = f"gt_{field}"
        pred_col = f"pred_{field}"
        err_col  = f"relerr_{field}"
        if gt_col not in ner.columns:
            continue
        sub = ner[ner["label"] == ETYPE_LABELS.get(etype, etype).lower()]
        if sub.empty:
            sub = ner  # fall back to all rows
        has_gt   = sub[gt_col].notna().sum()
        has_pred = sub[pred_col].notna().sum()
        both     = sub[sub[gt_col].notna() & sub[pred_col].notna()]
        parse_rate = has_pred / len(sub) * 100 if len(sub) > 0 else 0
        if len(both) > 0 and err_col in ner.columns:
            median_relerr = both[err_col].abs().median()
            accuracy = f"{100*(1-median_relerr):.1f}%" if median_relerr <= 1 else "<0%"
        else:
            accuracy = "—"
        rows.append({
            "Type":  etype,
            "Field": label,
            "GT samples": has_gt,
            "Parse rate": f"{parse_rate:.1f}%",
            "Median accuracy": accuracy,
        })

    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False)
    (TAB_DIR / "table_ner_eval.md").write_text(md)
    print("✓ table_ner_eval.md")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Location + Time extractor eval
# ══════════════════════════════════════════════════════════════════════════════
def table_location_time_eval():
    loc  = pd.read_csv(ROOT / "data/results/location_extractor_eval.csv")
    time = pd.read_csv(ROOT / "data/results/time_extractor_eval.csv")

    # Location: per event type
    loc_rows = []
    for label, grp in loc.groupby("label"):
        total      = len(grp)
        country_ok = grp["country_match"].sum()
        loc_rows.append({
            "Type":            ETYPE_LABELS.get(label.upper(), label.title()),
            "N":               total,
            "Country accuracy":f"{100*country_ok/total:.1f}%",
        })
    loc_df  = pd.DataFrame(loc_rows)
    overall = loc["country_match"].mean()
    loc_df  = pd.concat([loc_df,
        pd.DataFrame([{"Type": "**Overall**", "N": len(loc),
                       "Country accuracy": f"**{100*overall:.1f}%**"}])],
        ignore_index=True)

    # Time: overall
    has_gt    = time["gt_date"].notna().sum()
    exact     = time["exact"].sum()
    within_7d = time["within_7d"].sum()
    time_summary = pd.DataFrame([
        {"Metric": "Samples with GT date",   "Value": f"{has_gt}/{len(time)} ({100*has_gt/len(time):.0f}%)"},
        {"Metric": "Exact date match",        "Value": f"{exact}/{has_gt} ({100*exact/has_gt:.1f}%)" if has_gt else "—"},
        {"Metric": "Within 7 days",           "Value": f"{within_7d}/{has_gt} ({100*within_7d/has_gt:.1f}%)" if has_gt else "—"},
    ])

    md = "### Location Extractor\n\n" + loc_df.to_markdown(index=False)
    md += "\n\n### Time Extractor\n\n" + time_summary.to_markdown(index=False)
    (TAB_DIR / "table_location_time_eval.md").write_text(md)
    print("✓ table_location_time_eval.md")
    return loc_df, time_summary


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — Significant CAR findings
# ══════════════════════════════════════════════════════════════════════════════
def table_car_significant():
    grp = pd.read_csv(ROOT / "data/results/group_analysis.csv")
    sig = grp[grp["significant_5pct"] == True].copy()
    sig["event_type"] = sig["event_type"].map(ETYPE_LABELS).fillna(sig["event_type"])
    sig["severity"]   = sig["severity"].map(SEV_LABELS).fillna(sig["severity"])
    sig["mean_car_pct"] = (sig["mean_car"] * 100).round(3)
    sig["p_value"]      = sig["p_value"].round(4)
    out = sig[["event_type", "severity", "sector", "window", "n",
               "mean_car_pct", "p_value"]].sort_values("p_value")
    out.columns = ["Event Type", "Severity", "Sector", "Window", "N",
                   "Mean CAR (%)", "p-value"]
    md = out.to_markdown(index=False)
    (TAB_DIR / "table_car_significant.md").write_text(md)
    print("✓ table_car_significant.md")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CAR by event type & severity (T+1/T+3/T+5 grouped bars)
# ══════════════════════════════════════════════════════════════════════════════
def fig_car_by_type_severity():
    grp = pd.read_csv(ROOT / "data/results/group_analysis.csv")

    # Aggregate: mean CAR across all sectors, weighted by n
    agg = (grp.groupby(["event_type", "severity", "window"])
             .apply(lambda x: np.average(x["mean_car"], weights=x["n"]))
             .reset_index(name="weighted_car"))

    etypes   = ["EQ", "TC", "WF", "DR", "FL"]
    windows  = ["car_t1", "car_t3", "car_t5"]
    win_labels = {"car_t1": "T+1", "car_t3": "T+3", "car_t5": "T+5"}
    severities = ["green", "orange_or_red"]
    sev_colors = {"green": GREEN, "orange_or_red": ORANGE}
    sev_hatches= {"green": "", "orange_or_red": "///"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Weighted Mean CAR by Event Type and Severity", fontsize=14, fontweight="bold")

    x      = np.arange(len(etypes))
    width  = 0.35

    for ax, win in zip(axes, windows):
        for i, sev in enumerate(severities):
            sub  = agg[(agg["window"] == win) & (agg["severity"] == sev)]
            vals = [sub[sub["event_type"] == et]["weighted_car"].values[0] * 100
                    if len(sub[sub["event_type"] == et]) else 0.0
                    for et in etypes]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=sev_colors[sev],
                          hatch=sev_hatches[sev],
                          alpha=0.85,
                          label=SEV_LABELS[sev] if win == "car_t1" else "")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(win_labels[win], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([ETYPE_LABELS[e] for e in etypes], rotation=20, ha="right")
        ax.set_ylabel("CAR (%)" if win == "car_t1" else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig1_car_by_type_severity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Heatmap: CAR(T+3) by event_type × sector (significant only)
# ══════════════════════════════════════════════════════════════════════════════
def fig_car_heatmap():
    grp = pd.read_csv(ROOT / "data/results/group_analysis.csv")
    t3  = grp[grp["window"] == "car_t3"].copy()

    # Keep sectors that appear in at least one significant result
    sig_sectors = set(t3[t3["significant_5pct"] == True]["sector"].unique())
    t3_sub = t3[t3["sector"].isin(sig_sectors)].copy()

    # Pivot: rows = sector, cols = event_type
    pivot = t3_sub.groupby(["sector", "event_type"]).apply(
        lambda x: np.average(x["mean_car"], weights=x["n"])
    ).unstack(fill_value=np.nan) * 100

    # Reorder columns
    col_order = [e for e in ["EQ","TC","WF","DR","FL"] if e in pivot.columns]
    pivot = pivot[col_order]

    # Significance mask
    sig_pivot = t3_sub.groupby(["sector","event_type"])["significant_5pct"].any().unstack(fill_value=False)
    sig_pivot = sig_pivot.reindex(columns=col_order, fill_value=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="CAR T+3 (%)", shrink=0.8)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([ETYPE_LABELS[e] for e in col_order], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in pivot.index], fontsize=10)
    ax.set_title("Mean CAR (T+3) by Sector × Event Type\n(★ = p < 0.05)", fontsize=13, fontweight="bold")

    # Annotate cells
    for r, sector in enumerate(pivot.index):
        for c, etype in enumerate(col_order):
            val = pivot.iloc[r, c]
            if np.isnan(val):
                continue
            is_sig = sig_pivot.loc[sector, etype] if sector in sig_pivot.index and etype in sig_pivot.columns else False
            label  = f"{'★' if is_sig else ''}{val:+.2f}%"
            color  = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(c, r, label, ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    out = FIG_DIR / "fig2_car_heatmap_sector.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Pipeline coverage funnel
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline_funnel():
    stages = [
        ("Raw articles\n(all splits)", 4998),
        ("Disaster articles\n(not_related removed)", 3706),
        ("Unique events\n(after clustering)", 1664),
        ("Events with\ncountry resolved", 589),
        ("Events with\nCAR computed", 1622),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [BLUE, BLUE, GREEN, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(stages)), values, color=colors, alpha=0.85, edgecolor="white")

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 30, i, f"{val:,}", va="center", fontsize=11, fontweight="bold")
        if i > 0:
            pct = 100 * val / values[0]
            ax.text(val / 2, i, f"{pct:.0f}%", va="center", ha="center",
                    color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Pipeline Coverage Funnel", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig3_pipeline_coverage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating tables...")
    table_event_classifier()
    table_severity_classifiers()
    table_ner_eval()
    table_location_time_eval()
    table_car_significant()

    print("\nGenerating figures...")
    fig_car_by_type_severity()
    fig_car_heatmap()
    fig_pipeline_funnel()

    print(f"\nAll outputs saved to:\n  {FIG_DIR}\n  {TAB_DIR}")
