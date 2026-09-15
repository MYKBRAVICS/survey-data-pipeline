"""
clean.py — apply the fixes that are safe to automate, and only those.

The governing rule: a value is only changed automatically when there is one
defensible thing to do with it. Everything else stays in the dataset, flagged,
for a human to decide.

Deleting an implausible household size of 61 is defensible — nobody has 61
household members and the value cannot be used. Deleting a yield that looks
high for the stated area is NOT, because the area might be the wrong number,
not the yield. Pipelines that quietly "fix" the second kind produce clean
datasets that are wrong, which is worse than a messy dataset that is honest.
"""

from __future__ import annotations

import pandas as pd

from .util import log_entry


def drop_exact_duplicates(df: pd.DataFrame, flags: pd.DataFrame,
                          log: list) -> pd.DataFrame:
    """Remove submissions where every substantive field repeats an earlier row.

    Safe because there is no information in the second copy. A repeated
    identifier with DIFFERENT answers is left in place — that one needs a
    human, and it stays flagged as critical.
    """
    ids = set(flags.loc[flags["check"] == "exact_duplicate_row",
                        "submission_id"].dropna())
    if not ids or "submission_id" not in df.columns:
        return df

    mask = df["submission_id"].isin(ids)
    for idx in df.index[mask]:
        log.append(log_entry(df, idx, "drop_row", "*",
                             df.at[idx, "submission_id"], None,
                             "exact duplicate of an earlier submission"))
    return df[~mask].copy()


def blank_out_of_range(df: pd.DataFrame, cfg: dict, log: list) -> pd.DataFrame:
    """Set impossible numeric values to missing.

    Missing is honest. An impossible number is not, and it will silently move
    every mean computed from that column.
    """
    df = df.copy()
    for col, (lo, hi) in cfg.get("ranges", {}).items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad = series.notna() & ((series < lo) | (series > hi))
        for idx in df.index[bad]:
            log.append(log_entry(df, idx, "blank_out_of_range", col,
                                 series[idx], None,
                                 f"outside plausible range {lo}-{hi}"))
        df.loc[bad, col] = pd.NA
    return df


def blank_skip_logic_violations(df: pd.DataFrame, cfg: dict,
                                log: list) -> pd.DataFrame:
    """Clear answers to questions the form should not have shown.

    Safe in one direction only. If the gate says no, the dependent answer
    should not exist, so it is cleared. The reverse — a gate of yes with an
    empty dependent field — is left alone and flagged, because that is a
    genuine missing answer, not a contradiction to be tidied away.
    """
    df = df.copy()
    for spec in cfg.get("skip_logic", []):
        gate, target = spec["when"], spec["then_empty"]
        if gate not in df.columns or target not in df.columns:
            continue
        gate_false = df[gate] == False                  # noqa: E712
        filled = df[target].notna() & (df[target].astype(str).str.strip() != "")
        hit = gate_false & filled
        for idx in df.index[hit]:
            log.append(log_entry(df, idx, "blank_skip_violation", target,
                                 df.at[idx, target], None,
                                 f"{gate} is no, so this question was never asked"))
        df.loc[hit, target] = pd.NA
    return df


def blank_future_dates(df: pd.DataFrame, cfg: dict, log: list) -> pd.DataFrame:
    df = df.copy()
    date_col = cfg["survey"].get("date_column", "submission_date")
    if date_col not in df.columns:
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    today = pd.Timestamp.today().normalize()
    bad = dates.notna() & (dates > today)
    for idx in df.index[bad]:
        log.append(log_entry(df, idx, "blank_future_date", date_col,
                             dates[idx].date(), None,
                             "date falls in the future — device clock or typo"))
    df.loc[bad, date_col] = pd.NaT
    return df


def clean(df: pd.DataFrame, flags: pd.DataFrame, cfg: dict,
          log: list) -> pd.DataFrame:
    df = drop_exact_duplicates(df, flags, log)
    df = blank_out_of_range(df, cfg, log)
    df = blank_skip_logic_violations(df, cfg, log)
    df = blank_future_dates(df, cfg, log)
    return df


def untouched_summary(flags: pd.DataFrame) -> pd.DataFrame:
    """What was flagged but deliberately NOT changed.

    This table is the honest half of the report. It is what a reviewer should
    read first, because it is the list of decisions still owed to a human.
    """
    automated = {"out_of_range", "skip_logic_violation", "future_date",
                 "exact_duplicate_row"}
    left = flags[~flags["check"].isin(automated)]
    if left.empty:
        return pd.DataFrame(columns=["check", "category", "severity", "n"])
    return (left.groupby(["check", "category", "severity"])
                .size().reset_index(name="n")
                .sort_values("n", ascending=False))
