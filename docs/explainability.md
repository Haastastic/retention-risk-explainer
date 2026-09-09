# Explainability Layer — Phase 5

*Phase 5 output. How a risk score becomes a manager-facing explanation, and where
the layer boundary sits. Code: [`retention_risk/explain.py`](../retention_risk/explain.py),
[`retention_risk/narrative.py`](../retention_risk/narrative.py). Demo:
`python scripts/explain_demo.py [--live]`.*

---

## Pipeline

```
RiskModel.predict_proba(row)  ->  score, tier          (decision layer)
        |
explain_prediction(model, row)  ->  Explanation         (SHAP, structured)
        |
Narrator.narrate(explanation)   ->  Narrative           (plain language)
```

## 1. SHAP explanation (`explain.py`)

`explain_prediction(model, row)` returns an `Explanation`:

| Field | Meaning |
|---|---|
| `risk_score`, `risk_tier` | **copied from the model output** — nothing here recomputes them |
| `base_value` | SHAP expected value, **probability space** |
| `top_contributions` | top-k `FeatureContribution(feature, value, shap, direction)` by \|shap\| |
| `all_contributions` | every source feature → summed SHAP contribution |

Details:

- **One explainer for every estimator kind.** A model-agnostic explainer
  (`shap.Explainer` over the fitted estimator's `predict_proba`), run in the
  preprocessed numeric space against a synthetic 100-row background reference
  built at `fit` time. `TrainingResult.best` can ship either the XGBoost model
  or the logistic baseline, and both go through the identical path (~0.04s per
  row). We don't use `shap.TreeExplainer` for the XGBoost case: its default
  output is log-odds, not probability, and its interventional/probability mode
  rejects the one-hot-encoded input here — a consistent, additive probability
  space across both model kinds is worth more than the microseconds saved.
- **The background carries no PII.** `synthesize_background` resamples each
  feature column independently from its training values — every marginal is
  preserved, but no output row is a real employee. This matters because the
  background is persisted inside the model artifact by `RiskModel.save` and
  ships on deploy (Phase 8).
- **Probability space, additive.** `base_value + Σ all_contributions ≈ risk_score`
  to 1e-6, regardless of model kind — so a dashboard bar chart's segments sum to
  the score. Asserted in `tests/test_explain.py`.
- **One-hot folding.** `Department_Sales`, `Department_Research`, … are summed
  back onto `Department`, so a manager sees the feature, not the dummy column.
  `all_contributions` keys are always a subset of `schema.MODEL_FEATURES`.
- The employee's actual feature `value` is attached to each contribution for the
  narrative to reference ("workload_strain = 82").

## 2. Narrative (`narrative.py`)

`Narrator.narrate(explanation, employee_ref=…)` returns a `Narrative`:
`summary`, `drivers` (2–4 bullets), `suggested_action`, `source`.

Two implementations:

| | `TemplateNarrator` | `ClaudeNarrator` |
|---|---|---|
| Network | none | Claude API |
| Determinism | fully deterministic | model-dependent |
| How | feature→phrase lookup + a suggested action keyed to the top driver | sends the `Explanation` facts (tier + drivers, **no score**) to Claude, parses a JSON reply |
| When | default; always the CI path | when `ANTHROPIC_API_KEY` is set |

`get_narrator()` picks: Claude if `ANTHROPIC_API_KEY` is present (read from `.env`
via `python-dotenv`), else the template. Model defaults to
`claude-haiku-4-5-20251001`, overridable with `RETENTION_RISK_CLAUDE_MODEL`.

Fallback is two-layered: `get_narrator()` returns a `TemplateNarrator` if the
Claude client can't be constructed (missing key, import error), and
`ClaudeNarrator.narrate()` itself catches any per-call failure — network error,
rate limit, non-JSON or missing-key reply, or a reply whose `drivers` isn't a
list — and returns the template narrative instead (`source` then reads
`"template"`). A flaky Claude response never reaches the manager as an
exception, and never as a garbled shape.

The narrative is always framed as a **conversation prompt** — never "will quit",
never a pay/promotion/performance recommendation. The system prompt says so
explicitly and the template phrasings are written that way.

## 3. Layer isolation — the LLM never touches the score

This is the architectural constraint from the discovery brief and `CLAUDE.md`.
It holds four ways, each covered by a test in `tests/test_narrative.py`:

1. **Structural.** `Narrative` has no `risk_score` / `risk_tier` field. There is
   no slot for the LLM to return a score in.
2. **Input.** `Explanation.as_prompt_facts()` — the only thing sent to Claude —
   contains the tier and the driver directions, **not** the numeric score.
   `test_prompt_sent_to_claude_contains_no_score` asserts the score string in
   any format is absent from the request payload.
3. **No callback.** `narrate()` never imports or calls `RiskModel`.
   `test_narrate_does_not_call_the_model` monkeypatches `predict_proba` /
   `assign_tier` to raise and runs both narrators.
4. **Immutability.** The narrator receives the `Explanation` but the pipeline's
   decision (`score`, `tier`) is read from the model output upstream, never from
   anything the narrator returns.

The LLM's entire job is wording. If the Claude call fails or returns nonsense,
the risk score and tier a manager sees are unaffected.
