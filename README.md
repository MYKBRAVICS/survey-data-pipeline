# Agricultural survey data pipeline

Takes a raw ODK or KoBoToolbox export and produces a cleaned dataset, a
data-quality report, indicator tables, and an audit trail of every change
made.

Config-driven: a new survey means editing a YAML file, not the code. Ships
with synthetic sample data containing planted errors, so it runs the moment
you clone it and every check has something to find.

```bash
pip install -r requirements.txt
python run.py --config configs/sample.yaml --input data/sample_export.csv
open outputs/quality_report.html
```

---

## The problem it solves

Field survey exports arrive with the same faults regardless of who collected
them. Multi-select answers flattened into space-separated strings. Skip-logic
gaps that look identical to genuine missing values. `Ibanda`, `ibanda` and
` Ibanda ` as three separate districts. Duplicate submissions from re-sent
forms. GPS points in the wrong region. And, underneath all of it, enumerator
effects that no amount of staring at individual records will reveal.

Cleaning this by hand in Excel takes days, cannot be audited, and has to be
done again from scratch when the endline data arrives.

---

## What it produces

| Output | Contents |
|---|---|
| `outputs/clean.csv` | Analysis-ready dataset — typed, normalised, deduplicated |
| `outputs/quality_report.html` | Self-contained report with embedded charts. No internet needed |
| `outputs/quality_flags.csv` | Every flag raised, one row each |
| `outputs/cleaning_log.csv` | Every change made, with before and after values |
| `outputs/indicators.xlsx` | Indicator tables, disaggregated, one sheet each |

The cleaning log is the output clients care about most. It is what lets a
donor audit a number that differs from the raw export, and it is the
difference between a cleaned dataset and an unexplained one.

---

## The design rule

**A value is changed automatically only when there is exactly one defensible
thing to do with it. Everything else is flagged and left alone.**

Blanking a household size of 61 is defensible: nobody has 61 household
members and the value cannot be used for anything. Correcting a yield that
looks too high for the stated plot area is **not** — the area might be the
wrong number, not the yield. A pipeline that quietly fixes the second kind
produces a clean dataset that is wrong, which is worse than a messy dataset
that is honest.

So the report opens with **what was flagged and not changed**. That section is
the list of decisions still owed to a human, and putting it first stops the
reader assuming the data is now clean. It is cleaner. That is a different
thing.

---

## Checks

**Structural** — duplicate identifiers, exact duplicate rows, missing required
fields, absent columns.

**Logical** — values outside configured plausible ranges, cross-field
contradictions written as expressions in the config, skip-logic violations
(questions answered that the form should never have shown), dates in the
future or outside the fieldwork window.

**Enumerator** — interviews too short to have contained the questionnaire,
straight-lining through scale blocks, digit preference, and missingness
compared leave-one-out against the rest of the team.

**Spatial** — GPS points outside the survey bounding box, and identical
coordinates recorded for more than one household (a device that never got a
fix, or a point copied between records).

The enumerator family is the one most surveys skip and the one that most often
finds something. None of those patterns are visible one record at a time.

---

## Verified against planted errors

The sample dataset contains 426 submissions from six enumerators, with one
deliberately planted fault for each check. A clean run finds all twelve:

| Check | Found | What was planted |
|---|---|---|
| `duplicate_id` | 12 | 6 re-sent forms |
| `exact_duplicate_row` | 6 | the same 6, matching on every field |
| `out_of_range` | 8 | household sizes of 45, 0, 61; plot areas of 0.02 and 94 acres |
| `inconsistent` | 7 | yields of 5,200 kg/acre against a 2,000 kg ceiling |
| `skip_logic_violation` | 9 | fertiliser type recorded for households that said no |
| `future_date` | 3 | submissions dated March 2027 |
| `gps_outside_area` | 4 | points dropped in eastern Uganda |
| `gps_repeated_point` | 12 | coordinates shared between households |
| `short_interview` | 71 | one enumerator averaging 7 minutes on a 38-minute instrument |
| `straight_lining` | 51 | one enumerator flat-lining the satisfaction block |
| `digit_preference` | 1 | one enumerator rounding 100% of yields to a multiple of 50 |
| `high_missingness` | 1 | one enumerator leaving 8.1% of fields blank against 3.9% (z = 7.9) |

That last one needed the test rewritten. A plain ratio-against-the-survey-
average missed it, because with six enumerators the bad performer inflates the
very average they are compared against — the person the check exists to find
hides inside their own benchmark. Comparing leave-one-out fixes it, and a
two-proportion z-test gives the finding a size rather than just a verdict.

---

## Indicator sensitivity

Records flagged as internally contradictory are **not** silently dropped from
the indicators, because the pipeline does not know which of the two numbers is
wrong. They stay in the headline figure, and every indicator also gets a
second row computed without them.

On the sample data that gap is the whole story:

| Yield per acre | n | Mean | SD | Max |
|---|---|---|---|---|
| All records | 417 | 663 | 474 | 5,200 |
| Excluding disputed records | 413 | 619 | 159 | 1,131 |

Four records out of 417 move the mean by 7% and **triple the standard
deviation**. Anyone quoting the first row without knowing that is reporting an
artefact. Putting both rows side by side takes one line of config and stops it
happening.

Small cells are also suppressed: group statistics are blanked below a
configurable `min_n`, while the count itself is still shown. Publishing a mean
computed from three households is a disclosure risk as well as bad statistics.

---

## Configure

One YAML file per survey.

```yaml
survey:
  name: "Coffee baseline 2026"
  id_column: household_id
  enumerator_column: enum_name

types:
  categorical: [district, gender_hh_head]
  boolean: [uses_fertilizer, training_attended]
  numeric: [hh_size, plot_area_acres, yield_kg]
  multiselect: [coffee_varieties]

ranges:
  hh_size: [1, 30]
  plot_area_acres: [0.1, 50]

consistency:
  - rule: "yield_kg <= plot_area_acres * 2000"
    message: "Reported yield exceeds 2,000 kg per acre"
    severity: critical

skip_logic:
  - when: uses_fertilizer
    then_empty: fertilizer_type

indicators:
  - name: "Yield per acre"
    expression: "yield_kg / plot_area_acres"
    disaggregate_by: [district, gender_hh_head]
    min_n: 5
```

---

## Structure

```
configs/sample.yaml     survey definition
data/make_sample.py     synthetic data generator with planted errors
data/sample_export.csv  generated sample, committed so it runs on clone
src/
  ingest.py             parsing, type coercion, multi-select expansion
  checks.py             all twelve checks
  clean.py              the safe automatic fixes, and only those
  indicators.py         indicator computation, disaggregation, Excel output
  report.py             self-contained HTML report
  util.py               flag and log record shapes
run.py                  CLI
```

Exit codes are meaningful, so it can sit in a scheduled job: `0` ran with
nothing critical, `1` critical flags present (with `--strict`), `2` could not
run.

---

## Adapting it

Rename your columns in the config and the pipeline runs unchanged. Four things
to set deliberately:

1. **`ranges`** — what is impossible, not what is unusual. These values get
   blanked.
2. **`consistency`** — cross-field rules, stated as what should be true.
3. **`skip_logic`** — which gate controls which dependent question.
4. **`enumerator.min_duration_minutes`** — time the questionnaire yourself
   rather than guessing. This threshold is absolute, not relative.

---

## Contact

Available for survey cleaning, analysis and indicator reporting on
agricultural baselines, endlines and impact evaluations.

braveatwemeriireho@gmail.com · https://github.com/MYKBRAVICS
