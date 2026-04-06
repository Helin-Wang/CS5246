"""
Quick test: double-filter (GKG theme AND title keyword) and tone filter necessity.
Uses real GKG flat files from the Taiwan M7.4 earthquake window (2024-04-03).
"""
import urllib.request, zipfile, io, pandas as pd

URLS = [
    "http://data.gdeltproject.org/gdeltv2/20240403120000.gkg.csv.zip",
    "http://data.gdeltproject.org/gdeltv2/20240403121500.gkg.csv.zip",
    "http://data.gdeltproject.org/gdeltv2/20240403123000.gkg.csv.zip",
    "http://data.gdeltproject.org/gdeltv2/20240403124500.gkg.csv.zip",
]

DISASTER_THEMES = [
    "NATURAL_DISASTER", "NATURAL_DISASTER_EARTHQUAKE", "NATURAL_DISASTER_FLOOD",
    "NATURAL_DISASTER_TSUNAMI", "NATURAL_DISASTER_TREMOR", "DISASTER_FIRE",
    "NATURAL_DISASTER_HURRICANE", "NATURAL_DISASTER_STORM",
]

KEYWORDS = {
    "earthquake": ["earthquake", "quake", "tsunami", "seismic", "tremor"],
    "cyclone":    ["cyclone", "hurricane", "typhoon", "tropical storm", "storm"],
    "flood":      ["flood", "flooding", "inundation", "flash flood"],
    "volcano":    ["volcano", "eruption", "lava"],
    "wildfire":   ["wildfire", "bushfire", "forest fire", "wild fire"],
    "drought":    ["drought", "water shortage"],
}
ALL_KEYWORDS = [kw for kws in KEYWORDS.values() for kw in kws]

dfs = []
for url in URLS:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
            fname = zf.namelist()[0]
            df = pd.read_csv(zf.open(fname), sep="\t", header=None,
                             encoding_errors="replace", on_bad_lines="skip")
            dfs.append(df)
        print(f"  fetched {url.split('/')[-1]}: {len(df)} rows")
    except Exception as e:
        print(f"  SKIP {url.split('/')[-1]}: {e}")

if not dfs:
    print("No data fetched")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(df)}")

url_col   = df.iloc[:, 4].fillna("").astype(str)
theme_col = df.iloc[:, 7].fillna("").astype(str)

# Stage 1: GKG theme coarse filter
def has_disaster_theme(t):
    t = t.upper()
    return any(d in t for d in DISASTER_THEMES)

mask_theme = theme_col.apply(has_disaster_theme)
df_disaster = df[mask_theme].copy()
print(f"\nAfter GKG theme filter: {len(df_disaster)} / {len(df)} rows ({100*len(df_disaster)/len(df):.1f}%)")

# Proxy for title keyword check: use URL path
def url_has_keyword(u):
    u = u.lower()
    return any(kw in u for kw in ALL_KEYWORDS)

urls_disaster = url_col[mask_theme]
url_pass = urls_disaster.apply(url_has_keyword)
print(f"\n--- Double-filter test (keyword in URL as title proxy) ---")
print(f"  Pass : {url_pass.sum():4d} / {len(df_disaster)} ({100*url_pass.mean():.1f}%)")
print(f"  DROP : {(~url_pass).sum():4d} / {len(df_disaster)} ({100*(~url_pass).mean():.1f}%)")

print("\n  Sample DROPPED (disaster theme, no keyword in URL):")
for u in urls_disaster[~url_pass].head(12).tolist():
    print(f"    {u[:100]}")

# Tone filter test (col 15)
tone_raw = df_disaster.iloc[:, 15].fillna("").astype(str)

def parse_tone(t):
    try:
        return float(str(t).split(",")[0].strip())
    except Exception:
        return None

tones = tone_raw.apply(parse_tone).dropna()
print(f"\n--- Tone filter test (col 15, {len(tones)} parseable / {len(df_disaster)}) ---")
print(f"  Mean={tones.mean():.2f}  Median={tones.median():.2f}  Min={tones.min():.2f}  Max={tones.max():.2f}")

drop_tone = (tones >= 1.0).sum()
print(f"\n  Tone >= 1.0 → DROP: {drop_tone} / {len(tones)} ({100*drop_tone/len(tones):.1f}%)")
print(f"  Tone <  1.0 → KEEP: {(tones < 1.0).sum()} / {len(tones)} ({100*(tones<1.0).mean():.1f}%)")

print("\n  Distribution:")
for lo, hi in [(-100,-5),(-5,-2),(-2,0),(0,1),(1,5),(5,100)]:
    n = int(((tones >= lo) & (tones < hi)).sum())
    bar = "#" * (n // 5)
    print(f"    [{lo:5}, {hi:4}): {n:4d} ({100*n/len(tones):5.1f}%)  {bar}")

print("\n  Sample DROPPED by tone>=1.0:")
drop_idx = tones[tones >= 1.0].index.tolist()[:8]
for i in drop_idx:
    loc = df_disaster.index.get_loc(i)
    u = df_disaster.iloc[loc, 4]
    t = tones.loc[i]
    print(f"    tone={t:+.2f}  {str(u)[:90]}")
