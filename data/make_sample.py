"""
make_sample.py — generate a synthetic ODK-style export with planted errors.

The sample data ships with the repository so the pipeline runs the moment you
clone it, and so every check has something to find on a first run. Nothing
here is real: no household, enumerator or GPS point corresponds to anyone.

Errors planted, one per check the pipeline implements:

  Structural   duplicate submissions, whitespace-padded categories,
               mixed-case yes/no, multi-select fields as space-separated
               strings
  Logical      out-of-range household sizes and plot areas, yields
               impossible for the stated area, dates in the future,
               skip-logic violations
  Enumerator   one enumerator with implausibly short interviews, one who
               straight-lines the satisfaction block, one with heavy digit
               preference, one with inflated missingness
  Spatial      GPS points outside the survey districts

Regenerate with:  python data/make_sample.py
"""

import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)

N = 420

ENUMERATORS = [
    "A. Tumusiime",   # clean
    "B. Nakato",      # clean
    "C. Kyomuhendo",  # rushes interviews
    "D. Mugisha",     # straight-lines the satisfaction block
    "E. Ainembabazi", # rounds every yield to the nearest 50
    "F. Byaruhanga",  # leaves fields blank
]

DISTRICTS = {
    "Ibanda":   (-0.13, 30.50),
    "Kiruhura": (-0.19, 30.83),
    "Rubirizi": (-0.26, 30.11),
}

VARIETIES = ["KR1", "KR2", "KR3", "KR5", "KR6", "KR7", "KR9", "KR10"]
TOPICS = ["pruning", "stumping", "shade_mgmt", "pest_scouting", "post_harvest"]


def multiselect(options, k_lo=1, k_hi=4):
    """ODK exports multi-select answers as a space-separated string."""
    k = rng.integers(k_lo, k_hi + 1)
    return " ".join(sorted(rng.choice(options, size=k, replace=False)))


rows = []
base_date = pd.Timestamp("2026-05-04")

for i in range(N):
    enum = ENUMERATORS[i % len(ENUMERATORS)]
    district = list(DISTRICTS)[i % 3]
    lat0, lon0 = DISTRICTS[district]

    day = base_date + pd.Timedelta(days=int(rng.integers(0, 24)))
    start_hour = int(rng.integers(8, 17))
    start = day + pd.Timedelta(hours=start_hour, minutes=int(rng.integers(0, 60)))

    # Interview duration. C. Kyomuhendo is far too fast to have asked the
    # questions — the single most common fabrication signal in field surveys.
    if enum == "C. Kyomuhendo":
        minutes = int(rng.normal(7, 2))
    else:
        minutes = int(rng.normal(38, 8))
    minutes = max(minutes, 2)

    area = float(np.round(rng.gamma(2.2, 1.1) + 0.3, 2))
    trees = int(rng.normal(480, 90))
    yield_kg = float(np.round(area * rng.normal(620, 160), 1))
    yield_kg = max(yield_kg, 15.0)

    # E. Ainembabazi rounds everything to the nearest 50 — digit preference,
    # a sign of estimating rather than measuring or asking.
    if enum == "E. Ainembabazi":
        yield_kg = float(np.round(yield_kg / 50) * 50)

    uses_fert = rng.random() < 0.55
    trained = rng.random() < 0.4

    # Satisfaction block, used to detect straight-lining.
    if enum == "D. Mugisha" and rng.random() < 0.75:
        v = int(rng.integers(3, 6))
        sat = [v] * 5
    else:
        sat = list(rng.integers(1, 6, size=5))

    row = {
        "submission_id": f"uuid:{i:05d}",
        "household_id": f"HH-{district[:3].upper()}-{i:04d}",
        "enum_name": enum,
        "start_time": start.isoformat(),
        "end_time": (start + pd.Timedelta(minutes=minutes)).isoformat(),
        "submission_date": day.date().isoformat(),
        "district": district,
        "subcounty": rng.choice(["Kicuzi", "Nyamarebe", "Rukiri", "Bisheshe"]),
        "gps_lat": round(lat0 + rng.normal(0, 0.09), 5),
        "gps_lon": round(lon0 + rng.normal(0, 0.09), 5),
        "hh_size": int(np.clip(rng.normal(6.2, 2.4), 1, 16)),
        "gender_hh_head": rng.choice(["male", "female"], p=[0.72, 0.28]),
        "age_hh_head": int(np.clip(rng.normal(46, 13), 19, 88)),
        "plot_area_acres": area,
        "trees_per_acre": trees,
        "coffee_varieties": multiselect(VARIETIES),
        "yield_kg": yield_kg,
        "price_per_kg": int(np.clip(rng.normal(5200, 700), 3000, 9000)),
        "uses_fertilizer": "yes" if uses_fert else "no",
        "fertilizer_type": (rng.choice(["NPK", "DAP", "organic"])
                            if uses_fert else ""),
        "training_attended": "yes" if trained else "no",
        "training_topics": multiselect(TOPICS) if trained else "",
    }
    for j, s in enumerate(sat, start=1):
        row[f"satisfaction_{j}"] = int(s)

    rows.append(row)

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Planted errors
# ---------------------------------------------------------------------------

# F. Byaruhanga leaves fields blank.
mask_f = df["enum_name"] == "F. Byaruhanga"
for col in ["trees_per_acre", "price_per_kg", "age_hh_head"]:
    idx = df[mask_f].sample(frac=0.35, random_state=SEED).index
    df.loc[idx, col] = np.nan

# Duplicate submissions — a form re-sent after a sync failure.
dupes = df.sample(6, random_state=SEED).copy()
dupes["submission_id"] = [f"uuid:9{i:04d}" for i in range(len(dupes))]
df = pd.concat([df, dupes], ignore_index=True)

# Out-of-range values.
df.loc[df.sample(3, random_state=1).index, "hh_size"] = [45, 0, 61]
df.loc[df.sample(3, random_state=2).index, "plot_area_acres"] = [0.02, 94.0, 71.5]
df.loc[df.sample(2, random_state=3).index, "trees_per_acre"] = [12, 4200]

# Yields impossible for the stated area.
impossible = df.sample(4, random_state=4).index
df.loc[impossible, "yield_kg"] = df.loc[impossible, "plot_area_acres"] * 5200

# Skip-logic violations: fertiliser type recorded for households that said no.
viol = df[df["uses_fertilizer"] == "no"].sample(5, random_state=5).index
df.loc[viol, "fertilizer_type"] = "NPK"

viol2 = df[df["training_attended"] == "no"].sample(4, random_state=6).index
df.loc[viol2, "training_topics"] = "pruning shade_mgmt"

# Dates in the future.
future = df.sample(3, random_state=8).index
df.loc[future, "submission_date"] = "2027-03-14"

# GPS points well outside the survey area.
off = df.sample(4, random_state=9).index
df.loc[off, "gps_lat"] = [1.94, -1.62, 0.88, 2.31]
df.loc[off, "gps_lon"] = [33.61, 29.10, 34.02, 31.40]

# Dirty categories — the state real exports arrive in.
dirty = df.sample(25, random_state=10).index
df.loc[dirty, "gender_hh_head"] = df.loc[dirty, "gender_hh_head"].str.upper()
dirty2 = df.sample(18, random_state=11).index
df.loc[dirty2, "district"] = " " + df.loc[dirty2, "district"] + " "
dirty3 = df.sample(20, random_state=12).index
df.loc[dirty3, "uses_fertilizer"] = df.loc[dirty3, "uses_fertilizer"].str.title()

df = df.sample(frac=1, random_state=13).reset_index(drop=True)
df.to_csv("data/sample_export.csv", index=False)

print(f"Wrote data/sample_export.csv — {len(df)} rows, {df.shape[1]} columns")
print(f"Enumerators: {df['enum_name'].nunique()}, districts: {df['district'].nunique()}")
