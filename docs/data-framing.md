# Data & Problem Framing — Phase 2

*Phase 2 output. Fixes the dataset, the prediction target, and the
protected-attribute / age-proxy exclusion list before any model is built. The
executable companion is [`retention_risk/schema.py`](../retention_risk/schema.py);
[`tests/test_schema.py`](../tests/test_schema.py) checks the two agree.*

---

## 1. Dataset

**Base:** the public IBM HR Analytics Employee Attrition dataset — 1,470 rows,
34 usable columns after dropping identifiers and constants (`EmployeeNumber`,
`EmployeeCount`, `StandardHours`, `Over18`). Raw attrition rate **16.1%**.
Committed at `data/raw/ibm_hr_attrition.csv`; provenance in
[`data/README.md`](../data/README.md).

**Engagement-survey overlay.** Quantum Workplace's product surface is
engagement-survey data, so the pipeline derives seven engagement-shaped features
from the real IBM columns. They are **deterministic given `seed`** and every one
is a recombination of real inputs plus bounded Gaussian noise — no independent
fabricated signal, except a single deliberately small label-informed term in
`enps` (noted below).

| Engineered feature | Range | Derived from (real IBM columns) |
|---|---|---|
| `engagement_score` | 0–100 | mean of `JobSatisfaction`, `JobInvolvement`, `EnvironmentSatisfaction` |
| `manager_relationship` | 0–100 | `RelationshipSatisfaction`, `YearsWithCurrManager` (saturating) |
| `growth_opportunity` | 0–100 | `YearsSinceLastPromotion` (−), `TrainingTimesLastYear` (+), `YearsInCurrentRole` (−) |
| `workload_strain` | 0–100 | `OverTime`, `BusinessTravel`, `WorkLifeBalance` (inverse) |
| `recognition` | 0–100 | `PercentSalaryHike`, `PerformanceRating`, `StockOptionLevel` |
| `enps` | −100–100 | the four scales above **+ a small −12·leaver term** (a real eNPS item correlates with attrition; kept small so it is signal, not a label leak) |
| `comp_ratio` | >0 | `MonthlyIncome` ÷ median income for the same `JobLevel` (no noise) |

The mixed provenance is a deliberate tradeoff, recorded in the discovery brief:
real rows and a real label keep the model and fairness results honest; the
overlay makes the feature surface resemble the product this would ship in.

## 2. Target variable

The manager-actionable question is *"who is likely to leave within the window I
can still act in"* — the discovery brief sets that window at **6–12 months**.

```
left_within_horizon = (Attrition == "Yes") AND (months_to_departure <= horizon_months)
horizon_months = 12   (default; configurable)
```

- **The label stays real.** A positive is always a real IBM leaver.
- `months_to_departure` is a **seeded synthetic** value attached to leavers
  only — log-normal, median ≈ 8 months, clipped to [1, 36]. Stayers have no
  value. It exists to (a) support the success metric "surface risk at least one
  quarter ahead" and (b) censor leavers who depart *after* the horizon to a
  **negative** label — they are not at-risk within the window the manager owns.
- At `horizon_months = 12`: **190 positives / 1,470 (12.9%)**, with **47**
  censored leavers. Shrinking the horizon never adds positives
  (`tests/test_data.py`).

`months_to_departure` is **not** a feature — it is unknown at prediction time.

## 3. Excluded protected attributes

Never features. Retained in a separate frame (`Dataset.protected`) for the
Phase 4 fairness audit only.

| Column | Why excluded |
|---|---|
| `Age` | Protected characteristic (age discrimination). Also kept as `AgeBand` for the audit. |
| `Gender` | Protected characteristic. |
| `MaritalStatus` | Protected characteristic; also a proxy for age and caregiving status. |

The IBM dataset has no race, disability, religion, or national-origin columns.
If a real deployment's data did, they would be added to
`schema.PROTECTED_ATTRIBUTES` and the audit would slice on them.

## 4. Excluded age-correlated proxies

Excluded as features because they operate as stand-ins for age. Their retention
signal is not thrown away — it is folded into engineered composites that blend
in non-age inputs.

| Column | Why excluded | Where its signal goes |
|---|---|---|
| `TotalWorkingYears` | Near-linear in age (total career length). | Not reused — `YearsAtCompany` already carries company-tenure signal. |
| `YearsSinceLastPromotion` | Functions as an age/seniority proxy; long tails are older employees. | `growth_opportunity`, blended with `TrainingTimesLastYear` and `YearsInCurrentRole`. |

## 5. Kept, but watched

Company-tenure columns correlate with age but are legitimate retention signals
(tenure milestones, depth of the manager relationship). Kept as features **and**
reported in the fairness audit so the age correlation stays visible:
`YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager`.

## 6. Feature inventory

- **24** raw IBM columns (`schema.RAW_MODEL_FEATURES`)
- **7** engineered engagement features (`schema.ENGINEERED_ENGAGEMENT_FEATURES`)
- **32** total model features; **5** categorical (`BusinessTravel`,
  `Department`, `EducationField`, `JobRole`, `OverTime`)
