"""
checks.py — everything that looks for a problem.

Four families, in the order they are worth running:

  structural  Is the dataset the shape it claims to be? Duplicates, missing
              identifiers, empty required fields.
  logical     Does each record make sense on its own terms? Ranges, skip
              logic, internal contradictions, impossible dates.
  enumerator  Does the pattern of answers suggest how they were collected?
              Duration, straight-lining, digit preference, missingness.
  spatial     Is the record where it says it is?

The enumerator family is the one most surveys skip and the one that most often
finds something. Enumerator effects are usually the single largest source of
error in a field survey, and none of them are visible one record at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .util import flag


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    id_col = cfg["survey"]["id_column"]
    if id_col not in df.columns:
        return out

    dup_mask = df.duplicated(subset=[id_col], keep=False) & df[id_col].notna()
    for idx in df.index[dup_mask]:
        out.append(flag(df, idx, "duplicate_id", "structural", "critical",
                        id_col, df.at[idx, id_col],
                        "another submission carries this identifier — likely a "
                        "form re-sent after a sync failure"))

    # Near-duplicates: same identifier AND same answers to the substantive
    # questions. These are the ones that are safe to drop; a repeated ID with
    # different answers is a data problem that needs a human.
    subs = [c for c in df.columns
            if not c.startswith("_") and c not in {
                "submission_id", "start_time", "end_time"}]
    exact = df.duplicated(subset=subs, keep="first")
    for idx in df.index[exact]:
        out.append(flag(df, idx, "exact_duplicate_row", "structural", "critical",
                        "*", None,
                        "every substantive field matches an earlier submission"))
    return out


def check_required(df: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    for col in cfg.get("required", []):
        if col not in df.columns:
            out.append(flag(df, df.index[0] if len(df) else 0,
                            "missing_column", "structural", "critical",
                            col, None, "required column is absent from the export"))
            continue
        for idx in df.index[df[col].isna()]:
            out.append(flag(df, idx, "missing_required", "structural", "critical",
                            col, None, "required field is empty"))
    return out


# --------------------------------------------------------------------------
# Logical
# --------------------------------------------------------------------------

def check_ranges(df: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    for col, (lo, hi) in cfg.get("ranges", {}).items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad = series.notna() & ((series < lo) | (series > hi))
        for idx in df.index[bad]:
            out.append(flag(df, idx, "out_of_range", "logical", "critical",
                            col, series[idx],
                            f"outside the plausible range {lo} to {hi}"))
    return out


def check_consistency(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Cross-field rules, written as pandas expressions in the config.

    Each rule states what SHOULD be true. Rows where it evaluates False are
    flagged. Rows where it cannot be evaluated (missing inputs) are not — a
    missing value is a separate finding, not a contradiction.
    """
    out = []
    for spec in cfg.get("consistency", []):
        rule, message = spec["rule"], spec["message"]
        severity = spec.get("severity", "critical")
        try:
            holds = df.eval(rule)
        except Exception as exc:                        # noqa: BLE001
            out.append(flag(df, df.index[0] if len(df) else 0,
                            "rule_error", "logical", "warning", "*", rule,
                            f"rule could not be evaluated: {exc}"))
            continue
        violated = (holds == False)                     # noqa: E712
        for idx in df.index[violated]:
            out.append(flag(df, idx, "inconsistent", "logical", severity,
                            "*", None, message))
    return out


def check_skip_logic(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Fields answered that the form should never have shown.

    A value here means either the form logic was wrong or the record was
    edited after collection. Both matter, and neither shows up in a range
    check or a missingness table.
    """
    out = []
    for spec in cfg.get("skip_logic", []):
        gate, target = spec["when"], spec["then_empty"]
        if gate not in df.columns or target not in df.columns:
            continue
        gate_false = df[gate] == False                  # noqa: E712
        filled = df[target].notna() & (df[target].astype(str).str.strip() != "")
        for idx in df.index[gate_false & filled]:
            out.append(flag(df, idx, "skip_logic_violation", "logical", "critical",
                            target, df.at[idx, target],
                            f"answered although {gate} is no — the form should "
                            f"not have shown this question"))
    return out


def check_dates(df: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    date_col = cfg["survey"].get("date_column", "submission_date")
    if date_col not in df.columns:
        return out

    window = cfg.get("fieldwork_window", {})
    start = pd.Timestamp(window["start"]) if window.get("start") else None
    end = pd.Timestamp(window["end"]) if window.get("end") else None

    dates = pd.to_datetime(df[date_col], errors="coerce")
    today = pd.Timestamp.today().normalize()

    for idx in df.index[dates.notna() & (dates > today)]:
        out.append(flag(df, idx, "future_date", "logical", "critical",
                        date_col, dates[idx].date(),
                        "submission dated in the future"))

    if start is not None:
        for idx in df.index[dates.notna() & (dates < start)]:
            out.append(flag(df, idx, "before_fieldwork", "logical", "warning",
                            date_col, dates[idx].date(),
                            f"dated before fieldwork opened on {start.date()}"))
    if end is not None:
        for idx in df.index[dates.notna() & (dates > end) & (dates <= today)]:
            out.append(flag(df, idx, "after_fieldwork", "logical", "warning",
                            date_col, dates[idx].date(),
                            f"dated after fieldwork closed on {end.date()}"))
    return out


# --------------------------------------------------------------------------
# Enumerator behaviour
# --------------------------------------------------------------------------

def check_duration(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Interviews too short to have contained the questionnaire.

    The threshold is absolute, not relative: an interview that takes a
    quarter of the median is suspicious, but an interview that takes six
    minutes for a forty-minute instrument is not possible.
    """
    out = []
    if "_duration_min" not in df.columns:
        return out

    floor = cfg.get("enumerator", {}).get("min_duration_minutes", 10)
    d = df["_duration_min"]
    for idx in df.index[d.notna() & (d < floor)]:
        out.append(flag(df, idx, "short_interview", "enumerator", "critical",
                        "_duration_min", round(float(d[idx]), 1),
                        f"interview lasted under the {floor}-minute floor"))

    # Also flag the long tail — usually a form left open rather than a real
    # interview, which distorts any duration analysis built on top.
    ceiling = cfg.get("enumerator", {}).get("max_duration_minutes", 180)
    for idx in df.index[d.notna() & (d > ceiling)]:
        out.append(flag(df, idx, "long_interview", "enumerator", "info",
                        "_duration_min", round(float(d[idx]), 1),
                        "form was probably left open rather than in use"))
    return out


def check_straightlining(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Identical answers across a block of scale questions.

    One respondent genuinely answering 4 to everything is unremarkable. An
    enumerator doing it on a quarter of their interviews is not.
    """
    out = []
    block = [c for c in cfg.get("enumerator", {}).get("scale_block", [])
             if c in df.columns]
    if len(block) < 3:
        return out

    vals = df[block].apply(pd.to_numeric, errors="coerce")
    identical = vals.nunique(axis=1, dropna=True) == 1
    complete = vals.notna().all(axis=1)

    for idx in df.index[identical & complete]:
        out.append(flag(df, idx, "straight_lining", "enumerator", "warning",
                        ", ".join(block), int(vals.loc[idx, block[0]]),
                        "the entire scale block carries one value"))
    return out


def check_digit_preference(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Numeric answers rounded far more often than chance allows.

    A farmer estimating their harvest will round. Every single answer landing
    on a multiple of 50 means the number was produced rather than collected.
    This is an enumerator-level test, not a record-level one — it flags the
    person, not the row.
    """
    out = []
    conf = cfg.get("enumerator", {})
    enum_col = cfg["survey"].get("enumerator_column", "enum_name")
    if enum_col not in df.columns:
        return out

    for col in conf.get("digit_preference_columns", []):
        if col not in df.columns:
            continue
        base = conf.get("digit_preference_base", 50)
        threshold = conf.get("digit_preference_threshold", 0.6)

        series = pd.to_numeric(df[col], errors="coerce")
        rounded = (series % base == 0)

        for enum, grp in df.groupby(enum_col):
            vals = series.loc[grp.index].dropna()
            if len(vals) < 15:
                continue
            share = float(rounded.loc[vals.index].mean())
            if share >= threshold:
                out.append(flag(df, grp.index[0], "digit_preference",
                                "enumerator", "warning", col, round(share, 2),
                                f"{enum}: {share:.0%} of values are multiples of "
                                f"{base} across {len(vals)} records"))
    return out


def check_enumerator_missingness(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Enumerators whose records are emptier than everyone else's.

    Compared leave-one-out: each enumerator is measured against the rest of
    the survey, not against an average that includes themselves. With six
    enumerators, one bad performer drags the survey average up by enough to
    hide inside it — the very person the check exists to find is the one
    inflating their own benchmark. Leave-one-out removes that.

    A two-proportion z-test is reported alongside the ratio so the finding
    carries a size as well as a verdict. A 30% missingness rate on 15 records
    is noise; the same rate on 70 is a conversation with the field manager.
    """
    out = []
    enum_col = cfg["survey"].get("enumerator_column", "enum_name")
    if enum_col not in df.columns:
        return out

    conf = cfg.get("enumerator", {})
    tol = conf.get("missingness_multiple", 1.8)
    min_records = conf.get("missingness_min_records", 10)

    subs = [c for c in df.columns if not c.startswith("_")
            and "__" not in c and c != enum_col]
    if not subs:
        return out

    n_fields = len(subs)
    missing_per_row = df[subs].isna().sum(axis=1)

    for enum, grp in df.groupby(enum_col):
        if len(grp) < min_records:
            continue

        own_missing = int(missing_per_row.loc[grp.index].sum())
        own_total = len(grp) * n_fields
        own_rate = own_missing / own_total if own_total else 0.0

        rest_idx = df.index.difference(grp.index)
        rest_missing = int(missing_per_row.loc[rest_idx].sum())
        rest_total = len(rest_idx) * n_fields
        rest_rate = rest_missing / rest_total if rest_total else 0.0

        if rest_rate <= 0 or own_rate <= rest_rate * tol:
            continue

        # Two-proportion z-test against the rest of the survey.
        pooled = (own_missing + rest_missing) / (own_total + rest_total)
        se = np.sqrt(pooled * (1 - pooled) * (1 / own_total + 1 / rest_total))
        z = (own_rate - rest_rate) / se if se > 0 else np.inf

        out.append(flag(df, grp.index[0], "high_missingness", "enumerator",
                        "warning", "*", round(own_rate, 3),
                        f"{enum}: {own_rate:.1%} of fields empty across "
                        f"{len(grp)} records, against {rest_rate:.1%} for "
                        f"everyone else (z = {z:.1f})"))
    return out


# --------------------------------------------------------------------------
# Spatial
# --------------------------------------------------------------------------

def check_gps(df: pd.DataFrame, cfg: dict) -> list[dict]:
    out = []
    geo = cfg.get("geo", {})
    lat_col = geo.get("lat_column", "gps_lat")
    lon_col = geo.get("lon_column", "gps_lon")
    if lat_col not in df.columns or lon_col not in df.columns:
        return out

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    bbox = geo.get("bounding_box")
    if bbox:
        outside = (
            lat.notna() & lon.notna() &
            ((lat < bbox["lat_min"]) | (lat > bbox["lat_max"]) |
             (lon < bbox["lon_min"]) | (lon > bbox["lon_max"]))
        )
        for idx in df.index[outside]:
            out.append(flag(df, idx, "gps_outside_area", "spatial", "critical",
                            f"{lat_col}/{lon_col}",
                            f"{lat[idx]}, {lon[idx]}",
                            "point falls outside the survey bounding box"))

    # Exact coordinate repeats across different households: either a GPS that
    # never acquired a fix, or a point copied between records.
    key = lat.round(5).astype(str) + "," + lon.round(5).astype(str)
    valid = lat.notna() & lon.notna()
    counts = key[valid].value_counts()
    repeated = set(counts[counts > 1].index)
    for idx in df.index[valid & key.isin(repeated)]:
        out.append(flag(df, idx, "gps_repeated_point", "spatial", "warning",
                        f"{lat_col}/{lon_col}", key[idx],
                        "identical coordinates recorded for more than one household"))
    return out


# --------------------------------------------------------------------------

ALL_CHECKS = [
    check_duplicates, check_required,
    check_ranges, check_consistency, check_skip_logic, check_dates,
    check_duration, check_straightlining, check_digit_preference,
    check_enumerator_missingness,
    check_gps,
]


def run_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    flags: list[dict] = []
    for fn in ALL_CHECKS:
        flags.extend(fn(df, cfg))

    if not flags:
        return pd.DataFrame(columns=[
            "submission_id", "household_id", "enum_name", "check",
            "category", "severity", "column", "value", "message"])

    out = pd.DataFrame(flags)
    order = {"critical": 0, "warning": 1, "info": 2}
    return out.sort_values(
        ["severity", "category", "check"],
        key=lambda s: s.map(order) if s.name == "severity" else s,
    ).reset_index(drop=True)
