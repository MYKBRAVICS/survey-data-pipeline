"""
indicators.py — compute the indicators the report needs, disaggregated.

Indicators are defined in the config as pandas expressions, so adding one to a
survey means editing YAML rather than editing code. Each is computed on the
cleaned data, summarised overall and broken down by whatever the config asks
for.

Every table carries an n. An indicator mean with no denominator next to it is
how a report ends up quoting a figure computed from four households.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _summarise(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan,
                "sd": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "sd": float(s.std(ddof=1)) if s.size > 1 else np.nan,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def compute(df: pd.DataFrame, cfg: dict,
            flagged_ids: set | None = None) -> dict[str, pd.DataFrame]:
    """Return one table per indicator, overall plus each disaggregation.

    `flagged_ids` are submissions carrying a critical flag that the pipeline
    deliberately did NOT auto-correct — a yield that looks impossible for the
    stated area, for instance, where either number could be the wrong one.

    Those records stay in the dataset, so they are in the headline figure. But
    every indicator also gets a second overall row computed without them. The
    gap between the two rows is the sensitivity of the number to records under
    dispute, and it belongs in front of the reader rather than in a footnote.
    If excluding four households moves a district mean by 15%, nobody should
    be quoting that mean to two decimal places.
    """
    tables: dict[str, pd.DataFrame] = {}
    flagged_ids = flagged_ids or set()

    keep = pd.Series(True, index=df.index)
    if flagged_ids and "submission_id" in df.columns:
        keep = ~df["submission_id"].isin(flagged_ids)

    for spec in cfg.get("indicators", []):
        name = spec["name"]
        expr = spec["expression"]
        min_n = spec.get("min_n", 5)

        try:
            values = df.eval(expr)
        except Exception as exc:                        # noqa: BLE001
            tables[name] = pd.DataFrame([{
                "group": "ERROR", "level": str(exc), "n": 0}])
            continue

        rows = [{"group": "Overall", "level": "All records",
                 **_summarise(values)}]

        excl = _summarise(values.loc[keep])
        rows.append({"group": "Overall",
                     "level": "Excluding disputed records", **excl})

        for by in spec.get("disaggregate_by", []):
            if by not in df.columns:
                rows.append({"group": by, "level": "COLUMN NOT FOUND", "n": 0})
                continue
            for level, idx in df.groupby(by, dropna=True).groups.items():
                summary = _summarise(values.loc[idx])
                summary["suppressed"] = summary["n"] < min_n
                rows.append({"group": by, "level": str(level), **summary})

        tbl = pd.DataFrame(rows)

        # Small cells are reported as counts but their statistics are removed.
        # Publishing a mean from three households is disclosure risk as well as
        # bad statistics.
        if "suppressed" in tbl.columns:
            small = tbl["suppressed"].fillna(False)
            tbl.loc[small, ["mean", "median", "sd", "min", "max"]] = np.nan

        tables[name] = tbl

    return tables


def enumerator_table(df: pd.DataFrame, flags: pd.DataFrame,
                     cfg: dict) -> pd.DataFrame:
    """One row per enumerator: workload, pace, and flags raised.

    This is the table a field manager acts on. Everything else in the report
    describes the data; this describes the people collecting it.
    """
    enum_col = cfg["survey"].get("enumerator_column", "enum_name")
    if enum_col not in df.columns:
        return pd.DataFrame()

    rows = []
    subs = [c for c in df.columns if not c.startswith("_") and "__" not in c]

    for enum, grp in df.groupby(enum_col):
        f = flags[flags["enum_name"] == enum] if not flags.empty else flags
        rows.append({
            "enumerator": enum,
            "submissions": len(grp),
            "median_duration_min": (
                round(float(grp["_duration_min"].median()), 1)
                if "_duration_min" in grp.columns
                and grp["_duration_min"].notna().any() else np.nan),
            "missing_rate": round(float(grp[subs].isna().mean().mean()), 3),
            "flags_total": 0 if f.empty else len(f),
            "flags_critical": 0 if f.empty else int((f["severity"] == "critical").sum()),
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    tbl["flags_per_submission"] = (tbl["flags_total"] / tbl["submissions"]).round(2)
    return tbl.sort_values("flags_per_submission", ascending=False)


def to_excel(tables: dict[str, pd.DataFrame], enum_tbl: pd.DataFrame,
             path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if not enum_tbl.empty:
            enum_tbl.to_excel(writer, sheet_name="Enumerators", index=False)
        for name, tbl in tables.items():
            # Excel sheet names cap at 31 characters and reject several
            # punctuation marks, so they are sanitised rather than left to fail
            # halfway through writing the file.
            sheet = "".join(c for c in name if c not in "[]:*?/\\")[:31]
            tbl.to_excel(writer, sheet_name=sheet or "indicator", index=False)
