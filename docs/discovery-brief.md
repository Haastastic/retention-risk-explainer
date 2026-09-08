# Discovery Brief — Retention Risk Explainer

*Phase 1 output. One page. Written before any modeling decisions so that scope, users, and ethical guardrails are fixed first.*

---

## Problem statement

Managers usually learn that a valued team member is leaving when the resignation
is already on the table — too late to act. The signals that precede a departure
(declining engagement, stalled growth, workload strain, comp drift) exist in data
the organization already collects, but they are scattered and none of them is
individually alarming. We want to give a manager an **early, explainable read on
which of their people are at elevated risk of leaving in the next 6–12 months**,
plus enough context to have a useful retention conversation — not a black-box
score they can neither trust nor act on.

## Target user

**Primary:** a people-manager with 3–12 direct reports, non-technical, reviewing
their team once or twice a quarter. They need: who to pay attention to, why, and
a suggested next step. They will not read SHAP plots.

**Secondary:** an HR business partner rolling the team-level view up across an org,
and watching for fairness / adverse-impact problems before the tool is trusted.

**Not a user:** the model does not feed compensation, performance ratings,
promotion decisions, or any adverse action. It is a conversation prompt for
managers, nothing else.

## Success metrics

**Product**
- A manager can name, for each flagged report, the top 2–3 drivers in plain
  language and a concrete next action — measured by HRBP review, not self-report.
- Time-to-awareness: risk is surfaced at least one quarter before a
  regretted-departure notice in retrospective testing.

**Model**
- AUC-ROC ≥ 0.75 on a held-out set (baseline: logistic regression).
- Precision ≥ 0.30 at the "high risk" tier while that tier stays ≤ 15% of any
  team — the flag has to be rare enough to act on and right often enough to trust.
- Recall on actual leavers ≥ 0.50 across the top two risk tiers combined.

**Fairness**
- Disparate-impact ratio within [0.8, 1.25] for every protected group on the
  "high risk" flag rate, or a documented reason and mitigation.

**AI-native practice (the second deliverable)**
- Every phase has a PLAYBOOK.md entry written as the work happens.

## Scope

**In:** tabular attrition classifier; fairness audit; SHAP-based per-prediction
explanation; Claude turning that into a manager-facing narrative + suggested
action; Streamlit dashboard; CI/CD to a live link.

**Out (v1):** real-time data integration, org-hierarchy rollups beyond a single
team, intervention tracking / outcome measurement, any write-back to HR systems,
multi-tenant auth.

## Constraints

- **Ethical / legal:** protected attributes (age, gender, race, marital status,
  disability, etc.) are excluded as features. Age-correlated proxies
  (total tenure past a threshold, years-since-last-promotion where it stands in
  for age) are excluded or explicitly justified. The exclusion list is a Phase 2
  deliverable and is version-controlled.
- **Architectural:** the LLM lives only in the explanation layer. The classifier
  produces the score; Claude never sees or alters it. This isolation is asserted
  by test in Phase 5.
- **Data:** public IBM HR Analytics Attrition dataset for v1. It is small
  (~1,470 rows), US-centric, and synthetic-feeling in places — adequate for a
  portfolio demo, not for production claims. A synthetic engagement-survey
  dataset is a possible swap in Phase 2 if closer alignment to an
  engagement-product surface is worth the prep cost. **Open decision.**
- **Build:** solo build via Claude Code; GitHub Actions + Claude PR review;
  free-tier hosting (Streamlit Community Cloud).

## Key risks & assumptions

- *Assumption:* patterns in a public 2017-era dataset transfer well enough to
  demonstrate the approach. Acceptable for a portfolio; called out explicitly in
  the writeup.
- *Risk:* the "suggested action" reads as prescriptive or creepy. Mitigation:
  frame every output as a conversation starter, keep the manager in the loop,
  no individual-level output leaves the manager view.
- *Risk:* small dataset makes fairness slices tiny and noisy. Mitigation: report
  group sizes alongside every disparate-impact number; treat thin slices as
  "insufficient data," not "fair."

## Definition of done (project)

Live Streamlit link + public repo + finished PLAYBOOK.md that reads as a
template for embedding AI across the SDLC, not just a changelog.
