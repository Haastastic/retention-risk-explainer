# Model Card — Retention Risk Explainer

*Phase 3 output. Regenerate the numbers with `python scripts/train.py --no-save`.
Seed 42, held-out test size 25% (stratified), 5-fold stratified CV on the
training split.*

---

## Overview

| | |
|---|---|
| **Task** | Binary classification — will an employee leave within 12 months (`left_within_horizon`, see `docs/data-framing.md`) |
| **Models** | Baseline: logistic regression (`class_weight="balanced"`, standardised inputs). Production: XGBoost (300 trees, depth 3, lr 0.05, `scale_pos_weight` = neg/pos) |
| **Shipping model** | XGBoost — selected by `TrainingResult.best` (XGBoost unless the baseline beats it on held-out AUC by > 0.02; here the gap is 0.009 the other way) |
| **Features** | 32 — 24 raw IBM columns + 7 engineered engagement-survey features, 5 categorical (one-hot). Protected attributes and age proxies excluded by `retention_risk/schema.py` |
| **Not in the model** | Age, Gender, MaritalStatus, TotalWorkingYears, YearsSinceLastPromotion, identifiers, the label, `months_to_departure` |

## Metrics (held-out test set, n = 368, 48 positives)

| Metric | Baseline | XGBoost | Brief target |
|---|---|---|---|
| AUC-ROC | 0.834 | **0.826** | ≥ 0.75 ✅ |
| CV AUC (train, 5-fold) | 0.812 ± 0.021 | 0.797 ± 0.022 | — |
| Brier score | 0.157 | **0.105** | lower is better; XGBoost is better calibrated |
| High-tier precision | 0.50 | **0.57** | ≥ 0.30 ✅ |
| High-tier share | 11.4% | **8.2%** | ≤ 15% ✅ |
| Recall, High + Medium | 0.79 | **0.81** | ≥ 0.50 ✅ |

Both models clear every discovery-brief threshold. The baseline is within noise
on AUC (~370 test rows); XGBoost ships because it is materially better
calibrated (Brier 0.105 vs 0.157), which matters for a score managers read as a
probability.

## Risk tiers

Assigned from quantiles of the **training-set** score distribution:

| Tier | Cut-point | ~Share | Manager meaning |
|---|---|---|---|
| High | ≥ 88th pct | ~8–12% | Have a retention conversation this cycle |
| Medium | 65th–88th pct | ~25–30% | Watch; check in |
| Low | < 65th pct | ~60–65% | No action indicated |

Recall is measured across High + Medium combined, per the brief — the tool is
meant to concentrate attention, not to catch every leaver in the top tier.

## Known limitations

- **Small, dated, semi-synthetic data.** 1,470 rows, 2017-era IBM sample, plus a
  seeded engagement overlay. Demonstrates the pipeline; not a basis for
  production performance claims.
- **`enps` carries a mild label-informed term** by construction (see
  data-framing.md §1). It inflates measured performance somewhat versus a real
  deployment where eNPS is noisier.
- **Tier cut-points are global**, learned once on the training split. A real
  deployment would recalibrate per population and revisit on drift.
- **No calibration layer** beyond what XGBoost produces natively. Phase 4 checks
  fairness of the High-tier flag rate; systematic miscalibration by subgroup is
  not yet audited.
