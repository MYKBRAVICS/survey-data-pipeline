#!/usr/bin/env python3
"""
run.py — run the whole pipeline on one survey.

    python run.py --config configs/sample.yaml --input data/sample_export.csv

Outputs land in outputs/:

    clean.csv                 analysis-ready dataset
    quality_report.html       self-contained report, opens in any browser
    quality_flags.csv         every flag raised, one row each
    cleaning_log.csv          every change made, with before and after
    indicators.xlsx           indicator tables, disaggregated

Exit codes are meaningful, so this can sit in a scheduled job:
    0  ran, nothing critical
    1  ran, critical flags present
    2  could not run
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

from src import checks, clean as cleaning, indicators, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Agricultural survey data pipeline")
    ap.add_argument("--config", required=True, help="survey config YAML")
    ap.add_argument("--input", required=True, help="raw ODK/KoBo export CSV")
    ap.add_argument("--outdir", default="outputs", help="output directory")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any critical flag is raised")
    args = ap.parse_args()

    for path in (args.config, args.input):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 2

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    print(f"Survey: {cfg['survey'].get('name', '(unnamed)')}")
    print(f"Config: {args.config}")
    print(f"Input:  {args.input}\n")

    # 1. Ingest -------------------------------------------------------------
    from src.ingest import ingest
    df, log = ingest(args.input, cfg)
    raw_n = len(df)
    print(f"[1/5] Ingested {raw_n:,} rows, {df.shape[1]} columns "
          f"({len(log):,} formatting fixes)")

    # 2. Check --------------------------------------------------------------
    flags = checks.run_all(df, cfg)
    by_sev = flags["severity"].value_counts().to_dict() if not flags.empty else {}
    print(f"[2/5] {len(flags):,} flags — "
          f"{by_sev.get('critical', 0)} critical, "
          f"{by_sev.get('warning', 0)} warning, "
          f"{by_sev.get('info', 0)} info")

    # 3. Clean --------------------------------------------------------------
    df_clean = cleaning.clean(df, flags, cfg, log)
    print(f"[3/5] Cleaned to {len(df_clean):,} rows "
          f"({len(log):,} total changes logged)")

    # 4. Indicators ---------------------------------------------------------
    disputed = set()
    if not flags.empty:
        disputed = set(flags.loc[
            (flags["severity"] == "critical") & (flags["check"] == "inconsistent"),
            "submission_id"].dropna())
    tables = indicators.compute(df_clean, cfg, flagged_ids=disputed)
    enum_tbl = indicators.enumerator_table(df_clean, flags, cfg)
    indicators.to_excel(tables, enum_tbl, os.path.join(args.outdir, "indicators.xlsx"))
    print(f"[4/5] Computed {len(tables)} indicators "
          f"({len(disputed)} disputed records shown as a sensitivity row)")

    # 5. Report -------------------------------------------------------------
    log_df = pd.DataFrame(log)
    untouched = cleaning.untouched_summary(flags)

    df_clean.drop(columns=[c for c in df_clean.columns if c.startswith("_")],
                  errors="ignore").to_csv(
        os.path.join(args.outdir, "clean.csv"), index=False)
    flags.to_csv(os.path.join(args.outdir, "quality_flags.csv"), index=False)
    log_df.to_csv(os.path.join(args.outdir, "cleaning_log.csv"), index=False)

    report.build(cfg, raw_n, df_clean, flags, log_df, enum_tbl, untouched,
                 os.path.join(args.outdir, "quality_report.html"))
    print(f"[5/5] Wrote report to {args.outdir}/quality_report.html\n")

    n_crit = int((flags["severity"] == "critical").sum()) if not flags.empty else 0
    if n_crit:
        print(f"{n_crit} critical flags need a human decision. "
              f"Start with the first section of the report.")
    else:
        print("No critical flags.")

    return 1 if (args.strict and n_crit) else 0


if __name__ == "__main__":
    sys.exit(main())
