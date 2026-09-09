# Test Strategy — Phase 6

*Phase 6 output. How the test suite is built, what each kind of test is for, and
what the CI gate enforces. The suite was generated with Claude Code alongside the
code it covers, not written after the fact.*

---

## The gate

Every PR runs three checks (`.github/workflows/`):

| Check | Enforces |
|---|---|
| `ruff` | lint + format (`E/F/I/B/UP/SIM/PD`, 100 col) |
| `pytest` | the suite below, **≥ 95% line coverage** (`--cov-fail-under=95`; actual ≈ 100%) |
| `claude-review` | judgement review — logic, layer boundary, fairness, PII, doc drift |

The coverage floor has headroom on purpose: it's there to fail a PR that adds an
untested code path, not to make people chase the last line.

## Kinds of test, and why

The suite (~94 tests, `tests/`) is deliberately layered:

### 1. Property / invariant tests
Assert a relationship that must hold for *any* input, not a fixed expected value.
- `test_data.py` — a positive label is always a real leaver; a shorter horizon
  never adds positives; the build is deterministic for a fixed seed.
- `test_explain.py` — `base_value + Σ contributions ≈ risk_score` (probability
  space, both model kinds).
- `test_evaluate.py` — tier recall partitions the positives; tier shares sum to 1.

### 2. Characterisation tests
Pin the *current* behaviour of something data-dependent so a change is forced to
be deliberate.
- `test_fairness.py` — Gender passes four-fifths; MaritalStatus and AgeBand fail
  wide; flag DI tracks the true base-rate ratio. If the model moves these, CI
  fails and `docs/fairness-audit.md` must be revisited.
- Written loose enough to survive library-build float drift (ranges, not
  equalities) — see the comment block in that file.

### 3. Boundary tests (the architectural constraint)
`test_narrative.py` proves the LLM can't reach the risk decision, four ways:
`Narrative` has no score field; the score never enters the prompt; `narrate()`
runs with `predict_proba`/`assign_tier` monkeypatched to raise; the explanation's
score is unchanged afterward.

### 4. Contract / schema tests
`test_schema.py` — the exclusion policy in `schema.py` stays internally
consistent and matches `docs/data-framing.md` (a doc-drift test).

### 5. Error-path tests
`test_edge_cases.py` — every `raise` in the package has a test: the `Dataset`
leakage guard, `RiskModel.load` type check, the missing-background guard, the
`ClaudeNarrator` degradation paths, the "no dominant factor" narrative branches.

### 6. Mocked-integration tests
The Claude API is never called in CI. `test_narrative.py` uses a fake client
(`_FakeClient` / `_RaisingClient`) to exercise JSON parsing, code-fence
tolerance, schema validation, and fallback-to-template on any failure.

## Fixtures

`tests/conftest.py` builds the dataset once per session. Model-bearing tests
build their model in a `scope="session"` fixture too — training is ~2s and
several suites share it. Hand-built `Explanation` / `FairnessReport` objects are
used where a specific shape is needed that real data doesn't reliably produce
(e.g. an elevated tier whose top contributions are all risk-decreasing).

## How this was generated

Claude Code wrote the tests in the same PR as the code under test, then the
`claude-review` gate reviewed both. Several review findings were themselves
turned into regression tests (the non-tree explainer crash, the Low-tier
contradiction, the uniform-sign narrative edge case). The loop — generate,
review, turn the finding into a test — is the Phase 6 practice.
