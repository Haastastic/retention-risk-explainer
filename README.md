# Retention Risk Explainer

An explainable, fairness-audited attrition-risk model — a plausible feature for an
engagement product — built as a stage-by-stage demonstration of AI-native
engineering practice.

Two deliverables in one repo:

1. **The product** — a tabular classifier that gives a people-manager an early,
   explainable read on which of their reports are at elevated risk of leaving in
   the next 6–12 months, plus a plain-language narrative and a suggested
   conversation. A fairness audit ships alongside it.
2. **The practice** — [`PLAYBOOK.md`](PLAYBOOK.md) logs, phase by phase, where and
   how AI was embedded across discovery, framing, build, audit, explanation,
   test/review, interface, and deployment — written as the work happened, and
   framed as a template for running AI-native engineering at team scale.

---

## Architecture

```
data.py ─▶ schema.py        column policy: protected attrs & age proxies excluded, as code
   │
   ▼
model.py                    preprocessing + estimator + risk-tier thresholds  ← the decision
   │                        (baseline: logistic regression · production: XGBoost)
   ├─▶ evaluate.py           AUC / Brier / per-tier precision-recall, checked vs the brief
   ├─▶ fairness.py           four-fifths disparate-impact audit on the High flag
   │
   ▼
explain.py                  SHAP contributions per prediction (probability space, additive)
   │
   ▼
narrative.py                LLM turns the SHAP output into a manager-facing narrative
                            ── the LLM never sees or alters the score (enforced by test) ──
   │
   ▼
app.py / app_data.py        Streamlit: manager view + HR business partner rollup
```

**Principle:** the LLM lives only in the explanation layer. The classifier makes
the prediction; Claude narrates it. `tests/test_narrative.py` fails the build if
that boundary is crossed.

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements-dev.txt

pytest                          # 102 tests, ~100% coverage
python scripts/train.py --no-save        # train + evaluate both models
python scripts/audit_fairness.py         # run the disparate-impact audit
streamlit run app.py                     # the dashboard
```

The dashboard works with no configuration — narratives use a deterministic
template. To enable live Claude narration, put `ANTHROPIC_API_KEY` in a `.env`
file (local) or Streamlit secrets (deployed). See [`.env.example`](.env.example).

## Results

Held-out test set (n = 368, seed 42):

| Metric | Baseline | XGBoost | Brief target |
|---|---|---|---|
| AUC-ROC | 0.83 | 0.83 | ≥ 0.75 ✅ |
| Brier | 0.16 | 0.11 | lower is better |
| High-tier precision | 0.50 | 0.57 | ≥ 0.30 ✅ |
| High-tier share | 11% | 8% | ≤ 15% ✅ |
| Recall (High + Medium) | 0.79 | 0.81 | ≥ 0.50 ✅ |

Fairness audit: **Gender** passes the four-fifths rule; **MaritalStatus** and
**AgeBand** fail — shown to track real cohort attrition differences in this
dataset rather than a biased ranking, and handled with a documented governance
mitigation. Full method and findings in [`docs/fairness-audit.md`](docs/fairness-audit.md).

## Data

Public **IBM HR Analytics Attrition** dataset (1,470 rows), with a seeded
synthetic engagement-survey feature overlay so the surface resembles an
engagement product. The label stays real. Small, dated, and semi-synthetic —
adequate for demonstrating the pipeline, not for production claims. Provenance in
[`data/README.md`](data/README.md).

## Documentation

| Doc | Phase |
|---|---|
| [`docs/discovery-brief.md`](docs/discovery-brief.md) | 1 — problem, users, metrics, constraints |
| [`docs/data-framing.md`](docs/data-framing.md) | 2 — dataset, target, exclusion policy |
| [`docs/model-card.md`](docs/model-card.md) | 3 — models, metrics, limitations |
| [`docs/fairness-audit.md`](docs/fairness-audit.md) | 4 — disparate-impact method & findings |
| [`docs/explainability.md`](docs/explainability.md) | 5 — SHAP + narration, the LLM boundary |
| [`docs/test-strategy.md`](docs/test-strategy.md) | 6 — the six kinds of test, the CI gate |
| [`DEPLOY.md`](DEPLOY.md) | 8 — Streamlit Community Cloud deployment |
| [`PLAYBOOK.md`](PLAYBOOK.md) | all — the AI-native practice log |

## CI

Every PR runs three gates (`.github/workflows/`): **Ruff** (lint + format),
**pytest** (with a 95% coverage floor), and an automated **Claude review** that
posts inline comments. `@claude` in a PR comment pulls Claude back into the
thread.
