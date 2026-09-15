"""
ingest.py — turn a raw ODK/KoBo export into a typed, predictable table.

Nothing in here makes a judgement about the data. It parses, it normalises
formatting, and it records what it did. All the judgement lives in checks.py,
so that the question "was this value changed, or was it always like this?"
always has an answer.
"""

from __future__ import annotations

import pandas as pd

from .util import log_entry


def read_export(path: str) -> pd.DataFrame:
    """Read the export with everything as text first.

    Reading as text is deliberate. Letting pandas infer types on a raw export
    silently turns '0123' household IDs into 123, turns mixed columns into
    objects with no warning, and coerces the string 'n/a' into a category
    rather than a missing value. Types are applied explicitly below, once the
    config has said what each column is supposed to be.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[
        "", "NA", "na", "N/A", "n/a", "null", "NULL", "None", "-", "--", "999",
    ])


def normalise_categoricals(df: pd.DataFrame, columns: list[str], log: list) -> pd.DataFrame:
    """Strip whitespace and standardise case on categorical columns.

    Real exports arrive with ' Ibanda ', 'Ibanda' and 'IBANDA' as three
    distinct values, which then become three rows in every table you produce.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        before = df[col].copy()
        cleaned = df[col].astype("string").str.strip().str.lower()
        changed = (before.fillna("\x00") != cleaned.fillna("\x00"))
        for idx in df.index[changed]:
            log.append(log_entry(df, idx, "normalise_categorical", col,
                                 before[idx], cleaned[idx],
                                 "whitespace or case standardised"))
        df[col] = cleaned
    return df


def normalise_yes_no(df: pd.DataFrame, columns: list[str], log: list) -> pd.DataFrame:
    """Map the many spellings of yes and no onto a boolean."""
    true_set = {"yes", "y", "true", "1"}
    false_set = {"no", "n", "false", "0"}

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        src = df[col].astype("string").str.strip().str.lower()
        out = src.map(lambda v: True if v in true_set
                      else (False if v in false_set else pd.NA))
        unmapped = src.notna() & out.isna()
        for idx in df.index[unmapped]:
            log.append(log_entry(df, idx, "unmapped_boolean", col,
                                 src[idx], None,
                                 "value is neither yes nor no — set missing"))
        df[col] = out.astype("boolean")
    return df


def coerce_numeric(df: pd.DataFrame, columns: list[str], log: list) -> pd.DataFrame:
    """Convert to numeric, recording every value that fails to convert."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        src = df[col]
        out = pd.to_numeric(src, errors="coerce")
        failed = src.notna() & out.isna()
        for idx in df.index[failed]:
            log.append(log_entry(df, idx, "non_numeric", col, src[idx], None,
                                 "could not be read as a number — set missing"))
        df[col] = out
    return df


def expand_multiselect(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Expand space-separated multi-select answers into one column per option.

    ODK exports 'KR1 KR3 KR7' as a single string. Left that way, it is
    useless: you cannot count how many farmers grow KR3 without parsing the
    string on every query, and any equality filter silently misses most of
    the matches. One binary column per option is what makes it analysable.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).str.split()
        options = sorted({v for lst in values for v in lst})
        for opt in options:
            df[f"{col}__{opt}"] = values.apply(lambda lst, o=opt: int(o in lst))
        df[f"{col}__count"] = values.apply(len)
    return df


def parse_times(df: pd.DataFrame, start: str, end: str, date: str,
                log: list) -> pd.DataFrame:
    """Parse the timestamp columns and derive interview duration."""
    df = df.copy()
    for col in (start, end):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    if date in df.columns:
        df[date] = pd.to_datetime(df[date], errors="coerce", format="mixed")

    if start in df.columns and end in df.columns:
        df["_duration_min"] = (df[end] - df[start]).dt.total_seconds() / 60
        negative = df["_duration_min"] < 0
        for idx in df.index[negative]:
            log.append(log_entry(df, idx, "negative_duration", "_duration_min",
                                 df.at[idx, "_duration_min"], None,
                                 "end time precedes start time — set missing"))
        df.loc[negative, "_duration_min"] = pd.NA

    if start in df.columns:
        df["_start_hour"] = df[start].dt.hour
    return df


def ingest(path: str, cfg: dict) -> tuple[pd.DataFrame, list]:
    """Run the whole ingest stage. Returns the table and the cleaning log."""
    log: list = []
    types = cfg.get("types", {})

    df = read_export(path)
    df = normalise_categoricals(df, types.get("categorical", []), log)
    df = normalise_yes_no(df, types.get("boolean", []), log)
    df = coerce_numeric(df, types.get("numeric", []), log)
    df = expand_multiselect(df, types.get("multiselect", []))
    df = parse_times(df,
                     cfg["survey"].get("start_time_column", "start_time"),
                     cfg["survey"].get("end_time_column", "end_time"),
                     cfg["survey"].get("date_column", "submission_date"),
                     log)
    return df, log
