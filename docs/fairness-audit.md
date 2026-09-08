# Fairness Audit — Phase 4

*Phase 4 output. Method + findings for the disparate-impact audit of the
High-risk flag. Code: [`retention_risk/fairness.py`](../retention_risk/fairness.py).
Regenerate: `python scripts/audit_fairness.py`. Characterisation tests in
[`tests/test_fairness.py`](../tests/test_fairness.py) fail if these numbers move,
forcing this document to be revisited.*

---

## 1. What is audited

For each protected attribute — `Gender`, `MaritalStatus`, `AgeBand` (from
`Dataset.protected`, never model features) — the audit measures:

- **`flag_rate`** — the share of the group placed in the **High** risk tier.
- **`mean_score`** — the group's mean predicted risk (continuous, tier-independent).

The High flag is **not an adverse action**: the discovery brief is explicit that
the score is a manager conversation prompt and never feeds pay, promotion, or
performance. It is still held to the **four-fifths rule** because uneven flagging
steers manager attention unevenly and would erode trust in the tool.

## 2. Method

- Groups smaller than **`min_group_size = 30`** are reported but **excluded from
  the pass/fail decision** — a thin slice is "insufficient data", not "fair".
- **Reference group** = the sufficient group with the *lowest* flag rate.
- Per-group **`di_ratio`** = group flag rate ÷ reference flag rate (≥ 1).
- Attribute **`di_ratio`** = reference rate ÷ highest group rate (= min-rate /
  max-rate over sufficient groups, ≤ 1). **Passes when ≥ 0.80**, i.e. every
  group's flag rate within [0.8, 1.25] of every other.
- The audit runs on the **full dataset** — the population a manager rollout
  would flag across — with the shipping model trained on the standard split.

## 3. Findings (seed 42, XGBoost shipping model)

| Attribute | Groups (flag rate) | DI ratio | Four-fifths |
|---|---|---|---|
| **Gender** | Female 0.10 · Male 0.12 | **0.87** | ✅ pass |
| **MaritalStatus** | Divorced 0.07 · Married 0.08 · **Single 0.19** | **0.38** | ❌ fail |
| **AgeBand** | 40-49 0.07 · 50-60 0.10 · 30-39 0.09 · **18-29 0.19** | **0.40** | ❌ fail |

All groups exceed `min_group_size`; no slice was withheld.

## 4. Interpretation — the disparity is in the data, not the ranking

The `MaritalStatus` and `AgeBand` flag rates fail the four-fifths rule by a wide
margin. Three lines of evidence say this reflects a real difference in attrition
between cohorts in this dataset, not a model that mis-ranks a subgroup:

**a. The flag DI ratio ≈ the true attrition base-rate ratio.**

| Attribute | Lowest-rate vs highest-rate group | True attrition base rate | Flag-rate DI |
|---|---|---|---|
| MaritalStatus | Divorced (8.0%) vs Single (20.2%) | ratio **0.40** | 0.38 |
| AgeBand | 40-49 (7.7%) vs 18-29 (22.1%) | ratio **0.35** | 0.40 |
| Gender | Female (12.2%) vs Male (13.4%) | ratio **0.91** | 0.87 |

The model flags each cohort at very nearly the rate that cohort actually leaves.

**b. `mean_score` tracks `flag_rate`.** The continuous risk score shows the same
ordering and spacing as the tier flag, so the disparity is not an artefact of
where the tier cut-points fall.

**c. Removing suspected proxy features barely moves it.** Ablations
(`StockOptionLevel`, `NumCompaniesWorked`, the tenure columns, `enps`, alone and
in combination) shift `MaritalStatus` DI only from 0.38 to ~0.41 and `AgeBand`
from 0.40 to ~0.43, while costing 0.5–3.5 points of AUC. The signal is not
riding on one or two removable proxies — younger and single employees simply
leave more often across this data.

## 5. Why not "mitigate" it into compliance

The lever that would force the flag rates equal is **group-aware thresholds** —
a different High cut-point per `MaritalStatus` / `AgeBand` value. That means
**using a protected attribute at inference time** to decide who gets flagged,
which is a worse ethical and legal position than the disparity it fixes, and it
would hide a real retention-risk difference that managers arguably should see.
For a conversation-prompt tool, suppressing the signal is not obviously the
right call.

## 6. Mitigation & governance (v1)

Documented reason accepted per the discovery brief ("...or a documented reason
and mitigation"). The mitigations are process, not a model patch:

1. **The audit ships with the model.** This document and
   `scripts/audit_fairness.py` are part of the deliverable; the numbers are
   regenerated and reviewed whenever the model changes (enforced by
   `tests/test_fairness.py`).
2. **The HRBP rollup surfaces the disparity.** The secondary (HR business
   partner) view must show flag rate and group size per protected group on
   every team/org rollup, so disparate flagging is visible at the point of use,
   not buried in a report.
3. **High-tier lists are reviewed for cohort concentration** before being
   surfaced to a manager — if a manager's flagged set is entirely their
   youngest reports, that is a prompt to check the input data and the
   conversation framing, not to action the list.
4. **Manager-facing framing stays a conversation starter.** Every narrative
   (Phase 5) is worded as "worth a check-in", never "likely to quit", and the
   suggested action is always a discussion, never a retention offer or an
   escalation.
5. **Re-audit on real data.** These base rates come from a small 2017-era public
   dataset. A real deployment re-runs this audit on its own population before
   the tool is trusted, and treats a *widening* gap between flag DI and true
   base-rate ratio as the signal that the model — not the world — has become
   the problem.

## 7. Fields in the report

`FairnessReport.to_dict()` →
`{tier, min_group_size, n_rows, passes, attributes: {name: {reference_group,
di_ratio, passes, insufficient_groups, groups: {value: {n, flag_rate,
mean_score, di_ratio, sufficient}}}}}`. `passes` at the top level is true only
if every attribute the audit could decide passes the four-fifths rule.
