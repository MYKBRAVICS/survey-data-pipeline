"""
report.py — a single self-contained HTML data-quality report.

Everything is inlined: CSS in the head, charts as base64 PNGs. The file can be
emailed, opened from a flash drive, or read on a laptop with no internet,
which is the situation most field teams are actually in.

The report leads with what was NOT fixed. A quality report that opens with a
list of successful automatic corrections invites the reader to assume the data
is now clean. It is not; it is cleaner, and the difference is the part they
need to act on.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

CSS = """
:root { --ink:#1a1a1a; --mute:#6b6b6b; --line:#e2e2e2; --crit:#b3261e;
        --warn:#b8770f; --info:#2a6f9e; --bg:#fbfbfa; }
* { box-sizing:border-box; }
body { font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       color:var(--ink); background:var(--bg); margin:0; padding:40px 24px; }
.wrap { max-width:980px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:18px; margin:40px 0 12px; padding-bottom:6px;
     border-bottom:1px solid var(--line); }
h3 { font-size:15px; margin:24px 0 8px; }
.sub { color:var(--mute); font-size:14px; margin-bottom:28px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 8px; }
.card { flex:1 1 150px; background:#fff; border:1px solid var(--line);
        border-radius:8px; padding:14px 16px; }
.card .n { font-size:24px; font-weight:650; letter-spacing:-0.02em; }
.card .l { font-size:12px; color:var(--mute); text-transform:uppercase;
           letter-spacing:0.05em; margin-top:2px; }
.crit .n { color:var(--crit); } .warn .n { color:var(--warn); }
.info .n { color:var(--info); }
table { border-collapse:collapse; width:100%; background:#fff; font-size:13.5px;
        border:1px solid var(--line); border-radius:8px; overflow:hidden; }
th { text-align:left; background:#f2f2f0; font-weight:600; font-size:12px;
     text-transform:uppercase; letter-spacing:0.04em; color:#444; }
th,td { padding:8px 11px; border-bottom:1px solid var(--line);
        vertical-align:top; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:1px 7px; border-radius:99px;
        font-size:11px; font-weight:600; text-transform:uppercase;
        letter-spacing:0.04em; }
.pill.critical { background:#fdecea; color:var(--crit); }
.pill.warning  { background:#fdf3e2; color:var(--warn); }
.pill.info     { background:#eaf2f8; color:var(--info); }
.note { background:#fff; border-left:3px solid var(--mute); padding:12px 16px;
        margin:14px 0; color:#333; font-size:14px; }
.note.alert { border-left-color:var(--crit); }
img { max-width:100%; border:1px solid var(--line); border-radius:8px;
      background:#fff; }
footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--line);
         color:var(--mute); font-size:12.5px; }
code { background:#efefed; padding:1px 5px; border-radius:4px; font-size:12.5px; }
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_by_category(flags: pd.DataFrame) -> str | None:
    if flags.empty:
        return None
    counts = flags.groupby(["category", "severity"]).size().unstack(fill_value=0)
    for col in ("critical", "warning", "info"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["critical", "warning", "info"]]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    counts.plot(kind="barh", stacked=True, ax=ax,
                color=["#b3261e", "#d99b2b", "#4a90b8"], width=0.65)
    ax.set_xlabel("Flags"); ax.set_ylabel("")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    return _fig_to_b64(fig)


def _chart_by_enumerator(enum_tbl: pd.DataFrame) -> str | None:
    if enum_tbl.empty or "flags_per_submission" not in enum_tbl.columns:
        return None
    t = enum_tbl.sort_values("flags_per_submission")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.4))
    ax.barh(t["enumerator"], t["flags_per_submission"], color="#4a6b82",
            height=0.6)
    ax.set_xlabel("Flags per submission"); ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    return _fig_to_b64(fig)


def _table(df: pd.DataFrame, numeric_cols: set[str] | None = None,
           limit: int | None = None) -> str:
    if df is None or df.empty:
        return "<p class='sub'>Nothing to report.</p>"
    numeric_cols = numeric_cols or set()
    shown = df.head(limit) if limit else df

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = []
    for _, r in shown.iterrows():
        cells = []
        for c in shown.columns:
            v = r[c]
            if c == "severity" and isinstance(v, str):
                cells.append(f"<td><span class='pill {html.escape(v)}'>"
                             f"{html.escape(v)}</span></td>")
            elif isinstance(v, float):
                cells.append(f"<td class='num'>{'' if pd.isna(v) else f'{v:,.2f}'}</td>")
            elif c in numeric_cols:
                cells.append(f"<td class='num'>{html.escape(str(v))}</td>")
            else:
                text = "" if pd.isna(v) else str(v)
                cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    out = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    if limit and len(df) > limit:
        out += (f"<p class='sub'>Showing {limit} of {len(df):,} rows. "
                f"The full list is in <code>outputs/quality_flags.csv</code>.</p>")
    return out


def build(cfg: dict, raw_n: int, clean_df: pd.DataFrame, flags: pd.DataFrame,
          log_df: pd.DataFrame, enum_tbl: pd.DataFrame,
          untouched: pd.DataFrame, path: str) -> None:
    name = cfg["survey"].get("name", "Survey")
    counts = flags["severity"].value_counts() if not flags.empty else pd.Series(dtype=int)

    n_crit = int(counts.get("critical", 0))
    n_warn = int(counts.get("warning", 0))
    n_info = int(counts.get("info", 0))

    affected = flags["submission_id"].nunique() if not flags.empty else 0
    share = affected / raw_n if raw_n else 0

    c1 = _chart_by_category(flags)
    c2 = _chart_by_enumerator(enum_tbl)

    by_check = (flags.groupby(["check", "category", "severity"]).size()
                .reset_index(name="n").sort_values("n", ascending=False)
                if not flags.empty else pd.DataFrame())

    parts = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data quality — {html.escape(name)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Data quality report</h1>
<div class="sub">{html.escape(name)} &middot; generated
{datetime.now():%d %B %Y, %H:%M}</div>

<div class="cards">
  <div class="card"><div class="n">{raw_n:,}</div><div class="l">Submissions received</div></div>
  <div class="card"><div class="n">{len(clean_df):,}</div><div class="l">Rows after cleaning</div></div>
  <div class="card crit"><div class="n">{n_crit:,}</div><div class="l">Critical</div></div>
  <div class="card warn"><div class="n">{n_warn:,}</div><div class="l">Warning</div></div>
  <div class="card info"><div class="n">{n_info:,}</div><div class="l">Info</div></div>
</div>

<div class="note {'alert' if share > 0.1 else ''}">
<strong>{affected:,} of {raw_n:,} submissions ({share:.1%}) raised at least one flag.</strong><br>
{len(log_df):,} automatic changes were made, every one recorded in
<code>outputs/cleaning_log.csv</code>. Automatic changes are limited to cases
with exactly one defensible action: impossible values blanked, answers to
questions the form never showed cleared, exact duplicate rows dropped.
Everything else is flagged and left in place for a human.
</div>

<h2>What was flagged but not changed</h2>
<p class="sub">Read this section first. These are the decisions still owed to
a person — the pipeline has deliberately not made them.</p>
{_table(untouched, numeric_cols={"n"})}
"""]

    parts.append("<h2>Flags by check</h2>")
    if c1:
        parts.append(f'<img src="data:image/png;base64,{c1}" alt="Flags by category">')
    parts.append(_table(by_check, numeric_cols={"n"}))

    parts.append("<h2>Enumerators</h2>")
    parts.append("<p class='sub'>Enumerator effects are usually the largest "
                 "single source of error in a field survey, and none of them "
                 "are visible one record at a time.</p>")
    if c2:
        parts.append(f'<img src="data:image/png;base64,{c2}" alt="Flags per enumerator">')
    parts.append(_table(enum_tbl, numeric_cols={"submissions", "flags_total",
                                                "flags_critical"}))

    parts.append("<h2>Critical flags</h2>")
    crit = flags[flags["severity"] == "critical"] if not flags.empty else pd.DataFrame()
    parts.append(_table(
        crit[["household_id", "enum_name", "check", "column", "value", "message"]]
        if not crit.empty else crit, limit=40))

    parts.append("<h2>Missingness by column</h2>")
    subs = [c for c in clean_df.columns if not c.startswith("_") and "__" not in c]
    miss = (clean_df[subs].isna().mean().sort_values(ascending=False)
            .reset_index())
    miss.columns = ["column", "missing_share"]
    miss = miss[miss["missing_share"] > 0]
    parts.append(_table(miss))

    parts.append(f"""
<footer>
Generated by the agricultural survey data pipeline. Cleaned dataset:
<code>outputs/clean.csv</code> &middot; audit trail:
<code>outputs/cleaning_log.csv</code> &middot; all flags:
<code>outputs/quality_flags.csv</code> &middot; indicators:
<code>outputs/indicators.xlsx</code>
</footer>
</div></body></html>""")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
