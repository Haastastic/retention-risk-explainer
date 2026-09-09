# retention-risk-explainer

## Purpose

A combined portfolio project for the Quantum Workplace VP Engineering search, demonstrating two things at once:

1. **Product capability**: an explainable, fairness-audited attrition risk model — a plausible feature for Quantum's Engagement module.
2. **AI-native development practice**: the build itself is documented stage by stage as evidence of embedding AI across discovery, build, test, review, and deployment — directly answering the CPO's mandate.

This mirrors the architecture and workflow proven in the `ai-underwriter` project (github.com/Haastastic/ai-underwriter), reapplied to a talent-management context.

## Architecture Principles

- LLMs live in the explanation layer, never the risk-decision layer. The classifier makes the prediction; Claude only narrates it.
- Layer isolation and testability are prioritized over speed of a first working demo.
- Every SDLC stage that used an AI-native practice gets a corresponding entry in `PLAYBOOK.md`, written as the work happens, not reconstructed afterward.
- UI is deferred until the core pipeline (model, fairness audit, explainability) is validated end-to-end.

## Tech Stack

- **ML**: Python, pandas, scikit-learn, XGBoost, SHAP
- **LLM layer**: Claude API (explanation narration only)
- **UI**: Streamlit (fastest path to a live, shareable demo)
- **Data**: IBM HR Analytics Attrition dataset (public) — evaluate a synthetic engagement-survey dataset as an alternative if closer alignment to Quantum's actual product surface is worth the extra prep time
- **Dev workflow**: Claude Code (local), GitHub Actions with Claude PR review automation, same OAuth setup as ai-underwriter

## Build Phases

### Phase 1 — Discovery
- [x] Turn the rough ask ("help managers spot flight risk before it's too late") into a one-page structured brief: problem statement, target user, success metrics, constraints → `docs/discovery-brief.md`
- [x] Log this phase in PLAYBOOK.md as the discovery-stage AI practice

### Phase 2 — Data & Problem Framing
- [x] Select and load dataset — IBM HR Analytics base + engineered engagement-survey overlay; `data/`, `retention_risk/data.py`
- [x] Define target variable (attrition within N months) — `left_within_horizon`, real leaver + seeded horizon-censored `months_to_departure`; `docs/data-framing.md`
- [x] Explicitly list excluded protected attributes and age-correlated proxies (tenure-past-threshold, time-since-last-promotion where it functions as an age proxy) — `retention_risk/schema.py`, enforced by `tests/test_schema.py`
- [x] Log this phase in PLAYBOOK.md as the data-framing AI practice

### Phase 3 — Core ML Pipeline
- [x] Baseline model — logistic regression, `retention_risk/model.py :: train_baseline`
- [x] XGBoost classifier — `train_xgboost`; ships unless baseline beats it on held-out AUC by > 0.02
- [x] Evaluate on AUC-ROC and precision/recall at manager-actionable risk tiers — `retention_risk/evaluate.py`, brief thresholds as pass/fail checks; `docs/model-card.md`
- [x] Build with Claude Code, PR review automation catching issues as they land
- [x] Log this phase in PLAYBOOK.md as the build-stage AI practice

### Phase 4 — Fairness Audit
- [x] Disparate-impact ratio calculation across protected classes — `retention_risk/fairness.py`, four-fifths rule on High-tier flag rate, thin-slice guard
- [x] Document methodology — `docs/fairness-audit.md`: Gender passes; MaritalStatus & AgeBand fail, shown to track true attrition base rates (documented reason + governance mitigation per brief)
- [x] Log this phase in PLAYBOOK.md as the fairness-audit AI practice

### Phase 5 — Explainability Layer
- [x] SHAP values per prediction — `retention_risk/explain.py`, model-agnostic explainer over `predict_proba` (probability space, additive) + one-hot folding → `Explanation`
- [x] Claude API converts SHAP output into manager-facing plain-language narrative — `retention_risk/narrative.py` (`ClaudeNarrator`), with a deterministic `TemplateNarrator` fallback; `docs/explainability.md`
- [x] Confirm layer isolation holds: LLM never touches the risk score itself — 4 boundary tests in `tests/test_narrative.py` (no score field, score not in prompt, no model callback, immutability)
- [x] Log this phase in PLAYBOOK.md as the explainability-stage AI practice

### Phase 6 — Test Generation & Code Review
- [ ] Use Claude to generate the test suite, not just review it
- [ ] Reuse GitHub Actions + Claude PR review setup from ai-underwriter
- [ ] Log this phase in PLAYBOOK.md as the test/review-stage AI practice

### Phase 7 — Interface
- [ ] Streamlit dashboard: risk score, plain-language explanation, suggested manager action
- [ ] Keep it manager-facing and actionable, not a raw model output

### Phase 8 — Deployment & Documentation
- [ ] CI/CD via GitHub Actions
- [ ] Deploy to Streamlit Community Cloud (or equivalent free tier) for a live link
- [ ] Finalize PLAYBOOK.md as a stage-by-stage account of where AI was embedded, framed as a template for org-wide rollout — this is the artifact that speaks directly to the CPO's "embed AI across the entire development lifecycle" language

## PLAYBOOK.md (create alongside this file)

A running log, one entry per phase, of:
- What AI-native practice was used
- Why it was used at that stage
- What it would look like to operationalize at team scale (not just solo-project scale)

This is the piece that turns the repo from "a demo model" into "evidence of how I'd run AI-native engineering for you."
