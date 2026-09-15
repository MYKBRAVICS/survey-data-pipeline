"""
util.py — shared record shapes.

Every flag and every cleaning action is a dict with the same keys, so they can
be concatenated into a DataFrame without special-casing, and so nothing gets
recorded in a shape the report cannot render.
"""

from __future__ import annotations

import pandas as pd

SEVERITIES = ["critical", "warning", "info"]


def _ident(df: pd.DataFrame, idx, col: str):
    return df.at[idx, col] if col in df.columns else None


def flag(df: pd.DataFrame, idx, check: str, category: str, severity: str,
         column: str, value, message: str) -> dict:
    """One data-quality flag on one row."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    return {
        "submission_id": _ident(df, idx, "submission_id"),
        "household_id": _ident(df, idx, "household_id"),
        "enum_name": _ident(df, idx, "enum_name"),
        "check": check,
        "category": category,
        "severity": severity,
        "column": column,
        "value": value,
        "message": message,
    }


def log_entry(df: pd.DataFrame, idx, action: str, column: str,
              before, after, reason: str) -> dict:
    """One change made to the data, with its before and after values.

    This is the audit trail. A donor or a reviewer who wants to know why a
    number differs from the raw export gets their answer from this file, which
    is the difference between a cleaned dataset and an unexplained one.
    """
    return {
        "submission_id": _ident(df, idx, "submission_id"),
        "household_id": _ident(df, idx, "household_id"),
        "action": action,
        "column": column,
        "before": before,
        "after": after,
        "reason": reason,
    }
