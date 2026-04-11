"""
Storms 'n' Stocks – system architecture figure.
Clean two-column layout: offline (left) / online (right).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ────────────────────────────────────────────────────────────────
C_SRC  = "#D6EAF8"   # blue  – data source
C_TRN  = "#D5F5E3"   # green – training / model
C_MOD  = "#FEF9E7"   # yellow – pipeline module
C_OUT  = "#FDEDEC"   # red   – output
EC     = "#555555"

FIG_W, FIG_H = 11.0, 7.2

# ── helpers ────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, lines, fc, bold_first=False, fs=7.4, ec=EC):
    """Rounded rectangle with multi-line text (first line optionally bold)."""
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.03",
                           fc=fc, ec=ec, lw=0.9, zorder=3)
    ax.add_patch(patch)
    n = len(lines)
    dy = h * 0.28 if n > 1 else 0
    for i, line in enumerate(lines):
        y = cy + dy*(n-1)/2 - dy*i
        fw = "bold" if (bold_first and i == 0) else "normal"
        fsi = fs if i == 0 else fs - 1.2
        col = "#1a1a1a" if i == 0 else "#555555"
        ax.text(cx, y, line, ha="center", va="center",
                fontsize=fsi, fontweight=fw, color=col, zorder=4)

def arr(ax, x0, y0, x1, y1, rad=0.0, lw=1.1, col="#444444"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                connectionstyle=f"arc3,rad={rad}",
                                lw=lw, mutation_scale=9),
                zorder=5)

def darrrow(ax, x0, y0, x1, y1, rad=0.0):
    """Dashed arrow (cross-column dependency)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                connectionstyle=f"arc3,rad={rad}",
                                lw=1.0, mutation_scale=8,
                                linestyle="dashed"),
                zorder=5)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ── column headers ─────────────────────────────────────────────────────────
ax.text(0.24, 0.975, "Offline Path (Training)",
        ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1A5276")
ax.text(0.70, 0.975, "Online Path (Inference)",
        ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1E8449")
ax.axvline(0.48, 0, 1, color="#CCCCCC", lw=0.8, ls="-", zorder=0)

# ══════════════════════════════════════════════════════════════════════════
# LEFT  –  offline training
# ══════════════════════════════════════════════════════════════════════════
LX = 0.24
BW = 0.40; BH = 0.085

# GDACS API
rbox(ax, LX, 0.905, BW, BH,
     ["GDACS API", "3,248 records  (EQ / TC / WF / DR / FL)"],
     C_SRC, bold_first=True)

# two fetch strategy boxes
SW = 0.185; SH = 0.09
rbox(ax, LX-0.107, 0.785, SW, SH,
     ["Balanced fetch", "EQ/TC/WF/DR", "≤ 500 per alert level"],
     C_TRN, bold_first=True)
rbox(ax, LX+0.107, 0.785, SW, SH,
     ["Chronological fetch", "FL  (test window)", "338 records"],
     C_TRN, bold_first=True)

arr(ax, LX-0.107, 0.862, LX-0.107, 0.830)
arr(ax, LX+0.107, 0.862, LX+0.107, 0.830)

# Train classifiers
rbox(ax, LX-0.107, 0.665, SW, SH,
     ["Train classifiers", "RandomForest × 4", "EQ / TC / WF / DR"],
     C_TRN, bold_first=True)
arr(ax, LX-0.107, 0.740, LX-0.107, 0.710)

# Matching DB
rbox(ax, LX+0.107, 0.665, SW, SH,
     ["Matching DB", "gdacs_all_fields.csv", "(event validation)"],
     C_SRC, bold_first=True)
arr(ax, LX+0.107, 0.740, LX+0.107, 0.710)

# Severity models (merge)
rbox(ax, LX, 0.545, BW, BH,
     ["Severity models + Reference DB",
      "models/*.pkl   |   gdacs_all_fields.csv"],
     C_TRN, bold_first=True)
arr(ax, LX-0.107, 0.620, LX-0.04, 0.588)
arr(ax, LX+0.107, 0.620, LX+0.04, 0.588)

# dashed cross-column arrow  →  Module C
darrrow(ax, LX+0.20, 0.545, 0.522, 0.385, rad=-0.18)
ax.text(0.395, 0.46, "severity\nmodels", fontsize=6, color="#888888",
        ha="center", va="center", style="italic")

# ══════════════════════════════════════════════════════════════════════════
# RIGHT  –  online inference
# ══════════════════════════════════════════════════════════════════════════
RX = 0.715
RW = 0.48; RH = 0.082

# data sources row
DSW = 0.215
rbox(ax, RX-0.135, 0.905, DSW, BH,
     ["GDELT GKG", "9 disaster themes → 26,326 articles"],
     C_SRC, bold_first=True)
rbox(ax, RX+0.150, 0.905, DSW, BH,
     ["Yahoo Finance", "Sector ETF daily OHLCV"],
     C_SRC, bold_first=True)

# LLM labeling + DistilBERT
rbox(ax, RX-0.135, 0.790, DSW, SH,
     ["LLM labeling + DistilBERT", "4,998 labeled articles",
      "6-class fine-tuned model"],
     C_TRN, bold_first=True)
arr(ax, RX-0.135, 0.862, RX-0.135, 0.836)

# Stage 1
rbox(ax, RX, 0.680, RW, RH,
     ["Stage 1 · Event Type Classifier",
      "earthquake / cyclone / flood / wildfire / drought / not_related"],
     C_MOD, bold_first=True)
arr(ax, RX-0.135, 0.744, RX-0.07, 0.721)   # LLM → Stage1

# Yahoo Finance long arrow (right edge → Module E)
arr(ax, RX+0.150, 0.862, RX+0.240, 0.175, rad=-0.28)
ax.text(0.978, 0.52, "stock\nprices", fontsize=6, color="#888888",
        ha="center", va="center", style="italic")

# Module A
rbox(ax, RX, 0.570, RW, RH,
     ["Module A · Information Extraction",
      "time (GKG date + regex)  ·  location (spaCy NER)  ·  type-specific params (regex)"],
     C_MOD, bold_first=True)
arr(ax, RX, 0.639, RX, 0.611)

# Module D
rbox(ax, RX, 0.460, RW, RH,
     ["Module D · Event Clustering",
      "geo-proximity ≤ 500 km  ·  Δt ≤ 7 days  →  1,664 unique events"],
     C_MOD, bold_first=True)
arr(ax, RX, 0.529, RX, 0.501)

# Module C
rbox(ax, RX, 0.350, RW, RH,
     ["Module C · Severity Assessment",
      "RF classifiers (EQ/TC/WF/DR)  ·  rules for FL  ·  GDACS validation"],
     C_MOD, bold_first=True)
arr(ax, RX, 0.419, RX, 0.391)

# Module B
rbox(ax, RX, 0.240, RW, RH,
     ["Module B · Entity Linking",
      "pycountry → ISO country code  ·  KB: country → sector ETF tickers"],
     C_MOD, bold_first=True)
arr(ax, RX, 0.309, RX, 0.281)

# Module E
rbox(ax, RX, 0.130, RW, RH,
     ["Module E · Stock Impact Analysis",
      "OLS market model  ·  CAR(T+1 / T+3 / T+5)  ·  group t-test"],
     C_MOD, bold_first=True)
arr(ax, RX, 0.199, RX, 0.171)

# Output
rbox(ax, RX, 0.048, RW, 0.062,
     ["Output: event_id  ·  event_type  ·  severity  ·  country  ·  CAR(T+1/T+3/T+5)"],
     C_OUT, bold_first=False, fs=7.0)
arr(ax, RX, 0.089, RX, 0.079)

# ── legend ─────────────────────────────────────────────────────────────────
items = [
    mpatches.Patch(fc=C_SRC, ec=EC, label="Data source"),
    mpatches.Patch(fc=C_TRN, ec=EC, label="Training / model"),
    mpatches.Patch(fc=C_MOD, ec=EC, label="Pipeline module"),
    mpatches.Patch(fc=C_OUT, ec=EC, label="Output"),
]
ax.legend(handles=items, loc="lower left", fontsize=7,
          framealpha=0.95, edgecolor="#BBBBBB",
          bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.2)
plt.savefig("figures/fig_architecture.png", dpi=200, bbox_inches="tight")
print("Saved → figures/fig_architecture.png")
