# Data

## `raw/ibm_hr_attrition.csv` — base dataset

The public **IBM HR Analytics Employee Attrition & Performance** dataset
(1,470 rows, 35 columns). Originally IBM Watson Analytics sample data; widely
redistributed for teaching and benchmarking. Fetched by
`scripts/fetch_data.py`.

- Target column in the raw file: `Attrition` (`Yes` / `No`), base rate ~16.1%.
- Constant columns (`EmployeeCount`, `StandardHours`, `Over18`) and the
  `EmployeeNumber` identifier carry no signal and are dropped on load.
- It is small, US-centric, and synthetic-feeling in places. Adequate for a
  portfolio demo of the pipeline, fairness audit, and explanation layer — not a
  basis for production claims. This is called out in the discovery brief.

## Engineered engagement-survey overlay

Quantum Workplace's product surface is engagement-survey data, so on top of the
real IBM rows the pipeline derives a set of **engagement-survey-style features**
(`retention_risk/data.py :: engineer_engagement_features`). These are:

- **Composites of real IBM columns** — e.g. `engagement_score` blends
  `JobSatisfaction`, `JobInvolvement`, `EnvironmentSatisfaction`. No new signal
  is invented; the raw Likert items are recombined into survey-shaped scales.
- **Deterministically seeded** — the same `seed` reproduces the same overlay,
  so CI and the demo are stable.
- **Documented as synthetic** — every engineered column is listed in
  `docs/data-framing.md` with the real columns it is derived from.

The **label stays real**: `Attrition == "Yes"`. A seeded synthetic
`months_to_departure` is attached to leavers only, to support the "surface risk
at least one quarter ahead" success metric and to let a small share of leavers
fall outside the prediction horizon (censored to a negative label). See
`docs/data-framing.md` for the exact target definition.

## Not committed

Model artifacts (`*.pkl`, `*.joblib`) and any additional `data/*.csv` drops are
gitignored. The raw IBM CSV lives under `data/raw/` and *is* committed (~230 KB)
so the pipeline is reproducible without a network fetch or a Kaggle account.
