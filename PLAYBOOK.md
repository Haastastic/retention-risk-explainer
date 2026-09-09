# PLAYBOOK.md

A stage-by-stage account of where AI was embedded in building the Retention Risk
Explainer, and what it would take to run each practice at team scale rather than
solo-project scale.

One entry per SDLC phase, written as the work happens.

---

## Phase 1 — Discovery

**AI-native practice used**

Used Claude Code as a discovery partner to convert a vague stakeholder ask —
*"help managers spot flight risk before it's too late"* — into a structured
one-page brief (`docs/discovery-brief.md`): problem statement, primary/secondary
users, layered success metrics (product, model, fairness), explicit in/out scope,
and a constraints section that fixes the ethical guardrails **before any modeling
choice is made**. The protected-attribute exclusion policy and the
LLM-out-of-the-decision-layer boundary were both written down here, at stage zero,
rather than discovered later.

**Why at this stage**

Discovery is the cheapest place to catch a scope or ethics problem — a bad
assumption caught here costs a paragraph; caught in Phase 4 it costs a rebuild.
Having the model articulate users, metrics, and constraints forced decisions that
are easy to defer: *who is explicitly not a user, what adverse actions the score
must never feed, which age proxies are off the table.* The brief also gives every
later phase a definition of done to check against.

**What operationalizing this at team scale looks like**

- A shared **discovery-brief template** and a Claude prompt that runs the intake
  interview — same seven headings every time, so briefs are comparable across
  teams and a reviewer knows where to look.
- The brief is a **PR**, reviewed by an eng lead and an HRBP / domain owner
  before any ticket is cut. Claude PR review checks it for the required sections
  and flags a missing constraints or fairness section.
- Briefs link to their tracking epic; the "definition of done" block becomes the
  epic's acceptance criteria, so scope drift is visible.
- A standing rule: any project touching people-data or automated decisions must
  have the protected-attribute and human-in-the-loop sections filled in at
  discovery, not retrofitted.

---

## Build Infrastructure — Automated PR Review & Linting

Stood up before Phase 3 so the first line of pipeline code is reviewed and linted
on the way in, not audited afterward. Maps to the Phase 3 and Phase 6 checklist
items in `CLAUDE.md`.

**AI-native practice used**

Two complementary gates on every pull request:

- **Claude Code review** (`.github/workflows/claude-code-review.yml`) — the
  `code-review` plugin runs on `anthropics/claude-code-action@v1` and posts
  inline comments on the diff. A second workflow (`claude.yml`) lets a maintainer
  pull Claude back into a thread with an `@claude` mention for follow-up fixes or
  questions. Auth is the same `CLAUDE_CODE_OAUTH_TOKEN` OAuth setup as
  `ai-underwriter`, so there is no new secret-management story.
- **Ruff** (`.github/workflows/lint.yml`) — one tool covering pyflakes,
  pycodestyle, import order, bugbear, pyupgrade, simplify, and pandas-vet, plus a
  format check. Config lives in `pyproject.toml`; `.pre-commit-config.yaml` runs
  the identical checks locally so most lint failures never reach CI.

The division of labor is deliberate: Ruff owns everything mechanical and
deterministic, which keeps the Claude review focused on judgment work — logic
errors, layer-isolation violations (LLM leaking into the decision path), missing
fairness-audit coverage, thin tests.

**Why at this stage**

Review automation only pays off if it exists before the code does. Landing it now
means Phase 3's baseline model and Phase 5's explainability layer each get a
reviewed, linted first commit, and the PLAYBOOK can point to real PR threads as
evidence rather than a described intention. Setting it up mid-build would mean the
riskiest code — the initial pipeline scaffolding — is the one part that never
went through the gate.

**What operationalizing this at team scale looks like**

- Both workflows ship in a **template repo** (or an `actions/` shared repo), so a
  new project inherits the review + lint gate on day one with no copy-paste.
- Ruff config is **centralized and versioned** — one `pyproject.toml` fragment
  or a shared `ruff.toml` extended by each repo — so "our lint rules" is a single
  reviewable artifact, not per-project drift.
- Branch protection requires both checks green before merge; the Claude review is
  advisory (comments), Ruff is blocking.
- `pre-commit` install is part of repo onboarding / the devcontainer, so the
  local and CI feedback loops match and CI mostly confirms rather than discovers.
- A quarterly `pre-commit autoupdate` + Ruff-rule review, run as its own PR, so
  tooling upgrades are visible and revertable.
- Track review signal over time: what fraction of Claude's comments lead to a
  change, which categories recur. That tells you where to add a lint rule, a test
  template, or a design guardrail so the same issue stops reaching review.

---

## Phase 2 — Data & Problem Framing

**AI-native practice used**

Used Claude Code to turn the discovery brief's prose constraints into
**executable policy**. The protected-attribute and age-proxy exclusion lists
from the brief became `retention_risk/schema.py` — typed column groups with a
`withheld_columns()` function — and `tests/test_schema.py` now fails the build
if a withheld column reaches the feature matrix, if the schema and
`docs/data-framing.md` drift apart, or if the engineered-feature list changes
without the doc changing. The target definition ("left within 6–12 months") was
similarly pinned: a seeded synthetic `months_to_departure` on real leavers,
horizon-censored, with property tests asserting a positive is always a real
leaver and a shorter horizon never adds positives.

The engagement-survey overlay — seven features derived from real IBM columns to
make the surface resemble Quantum's product — was designed with Claude by
working backwards from "what would an engagement survey actually measure," then
each engineered column was documented against the real inputs it blends.

**Why at this stage**

Framing is where fairness is won or lost. A rule that lives only in a brief gets
skipped under deadline; a rule that fails CI does not. Encoding the exclusion
list now means every later phase — model, audit, explanation, UI — inherits a
feature matrix that is provably free of the attributes we said we would not use,
and the reviewer checks a diff against `schema.py` rather than re-reading a doc.

**What operationalizing this at team scale looks like**

- A **schema-policy module is a required artifact** for any people-data model:
  protected attributes, proxies, and the "kept but watched" list as code, with a
  leakage test wired into CI. Templated so every project's looks the same.
- The brief → schema translation is a **standard Claude task** with a fixed
  prompt, run at the start of framing; its output is a PR reviewed by an eng
  lead and a domain owner together.
- Synthetic or derived features carry a **provenance table** (feature → real
  inputs → is-it-label-informed) that ships in the repo and is checked by test,
  so "what did you make up" has a documented answer before anyone asks.
- Doc-drift tests (schema vs. prose) become a house pattern — the cheapest way
  to keep design docs honest as code moves under them.

---

## Phase 3 — Core ML Pipeline

**AI-native practice used**

Built the pipeline (`pipeline.py`, `model.py`, `evaluate.py`, `training.py`)
with Claude Code. Two things carried the AI-native weight here:

*Review caught fairness defects, not style.* The automated PR review on the
data-framing PR flagged two real issues — both about the fairness premise:

1. `growth_opportunity` was engineered partly from `YearsSinceLastPromotion`,
   an *excluded* age proxy. Because two of the composite's other inputs ship as
   raw features, the proxy was ~90% recoverable by algebra — the exclusion was
   nominal. Rebuilt from allowed inputs only.
2. The `Dataset` leakage guard checked column *names* against
   `withheld_columns()`, which omitted the label columns. Fixed the set; the
   guard now rejects `Attrition` regardless of how `X` is assembled.

Both were fixed before merge, so the model in this phase was built on a feature
matrix that had already survived that scrutiny.

*The brief's metrics are executable.* The discovery-brief success thresholds are
coded as **pass/fail checks inside the evaluation** (`EvalReport.checks`,
`passes_brief`), and a test asserts a full training run clears all of them. The
brief stops being a document you remember to check and becomes a build gate.

**Why at this stage**

The build phase is where "we'll audit fairness later" quietly becomes "we
shipped a proxy." Having review run on every PR — with a reviewer briefed on the
exclusion policy as *code* it can diff against — is what turned a subtle
leakage bug into a same-day fix instead of a Phase 4 finding or a production
incident. Encoding the brief's thresholds as checks now means Phases 4–7 inherit
a model that is known to clear the bar, and any regression fails CI.

**What operationalizing this at team scale looks like**

- **Success metrics as executable checks** ship with every model repo — a
  `passes_brief`-style object plus a test that runs the full pipeline and
  asserts it. Model quality gates live in CI next to lint and unit tests.
- The **model card is generated from the eval run**, not hand-written — one
  command re-emits the numbers, so a stale card is a diff, not a surprise.
- Review of people-data model PRs is **staffed deliberately**: the reviewer (AI
  or human) is given the schema-policy module and the brief, and told the two
  failure modes to look for — proxy leakage and metric gaming — not just "review
  this."
- A standing check: any engineered feature is diffed against the exclusion list
  for *recoverability*, not just for whether it names a forbidden column.

---

## Phase 4 — Fairness Audit

**AI-native practice used**

Built the disparate-impact audit (`retention_risk/fairness.py`,
`docs/fairness-audit.md`) with Claude Code, and used it to *investigate* a
failure rather than to certify a pass. The audit found `MaritalStatus` and
`AgeBand` High-flag rates failing the four-fifths rule badly (DI ≈ 0.38 / 0.40).
Instead of reaching for a mitigation, the next step was three quick analyses —
comparing flag DI to true attrition base-rate ratios, checking `mean_score`
against `flag_rate`, and ablating suspected proxy features — that together showed
the disparity is a real cohort difference in this data, not a biased ranking.
That evidence, and the decision *not* to use group-aware thresholds, is written
up in the audit doc as the "documented reason and mitigation" the discovery
brief allows.

The audit result is pinned by **characterisation tests**: Gender passes,
MaritalStatus and AgeBand fail with DI in a stated band, and flag DI tracks the
true base-rate ratio. If the model changes and these move, CI fails and the doc
must be revisited — the audit can't silently drift.

**Why at this stage**

Running the audit right after the model is built — not just before launch — is
what made the disparity a research question with time to answer it, rather than
a launch blocker to paper over. And doing the base-rate comparison *before*
writing any mitigation code stopped a reflex "fix" (per-group thresholds) that
would have used protected attributes at inference and hidden a signal managers
arguably should see.

**What operationalizing this at team scale looks like**

- The audit is a **library, not a notebook** — same `audit_fairness` call, same
  report shape, same four-fifths + thin-slice rules across every people-data
  model, so results are comparable and reviewable.
- A failing audit triggers a **fixed investigation checklist** (base-rate ratio,
  score-vs-flag, proxy ablation) before any mitigation is proposed. "Is it the
  model or the world" is answered with evidence every time.
- Mitigations that touch protected attributes at inference need **explicit
  sign-off** from legal + a domain owner, not an engineer's judgement call.
- Every model ships with its audit doc and `scripts/audit_fairness.py`; the
  HRBP-facing surface shows per-group flag rates and group sizes at the point of
  use, so disparate impact is visible in the product, not only in a repo.
- Characterisation tests on fairness numbers are standard — drift in a fairness
  metric should break a build, the same as a failing unit test.

---

## Phase 5 — Explainability Layer

**AI-native practice used**

This is the phase where the LLM finally appears in the product — and the
practice was to make its non-involvement in the decision *enforceable*, not just
stated. Built with Claude Code:

- `explain.py` turns a prediction into a structured `Explanation` via SHAP, with
  one-hot columns folded back to source features so the output is in the
  manager's vocabulary.
- `narrative.py` has two narrators behind one interface: a deterministic
  `TemplateNarrator` (the default and the CI path) and a `ClaudeNarrator` that
  calls the API. The app degrades to the template with no key, so the live demo
  never depends on a secret being set.
- The layer boundary is covered by **four tests**, each closing a different
  path: `Narrative` has no score field; the score never enters the prompt;
  `narrate()` never calls the model (monkeypatched to explode if it does); the
  narrator can't mutate the decision. "The LLM only narrates" went from a
  sentence in the brief to something CI fails on.

Claude also wrote the test suite for this phase (previewing Phase 6): the mock
Anthropic client, the fence-tolerant JSON parsing tests, the boundary tests.

**Why at this stage**

The moment an LLM is in the codebase is the moment someone can wire its output
back into a decision "just for this one case". Putting the boundary tests in the
same PR that introduces the LLM means that shortcut fails review from day one,
and the isolation claim in the brief has teeth for every phase after.

**What operationalizing this at team scale looks like**

- **A boundary test is a required artifact** wherever an LLM sits next to a
  model or a rule engine: prove the LLM's output can't reach the decision, by
  test, in the introducing PR.
- **Two implementations, one interface** is the house pattern for any LLM
  feature — a deterministic fallback that is the CI path and the
  no-credentials path, and the model call as an upgrade. Tests never hit the
  API; the live link never hard-depends on a key.
- The prompt is **built from a structured object** (`as_prompt_facts()`), not
  string-concatenated at the call site, so what the model does and doesn't see
  is reviewable in one place.
- LLM output is **parsed into a typed shape** with an explicit schema, not
  passed through as free text — the narrative can be wrong-in-wording but never
  wrong-in-structure.

---

## Phase 6 — Test Generation & Code Review

**AI-native practice used**

The suite (~94 tests, ~100% line coverage) was **generated by Claude Code in the
same PR as the code it covers** — not written afterward against a finished
module. Each phase's PR shipped its implementation and its tests together, and
the `claude-review` gate then reviewed both. `docs/test-strategy.md` lays out the
six kinds of test and what each is for: property/invariant, characterisation,
architectural-boundary, contract/doc-drift, error-path, and mocked-integration.

The distinctive loop this phase formalises: **a review finding becomes a
regression test.** The non-tree explainer crash, the self-contradictory Low-tier
narrative, the uniform-sign edge case — each was caught by `claude-review`, fixed,
*and* pinned with a test in the same commit, so the class of bug can't come back.

CI enforces three gates on every PR — Ruff, pytest with a 95% coverage floor
(headroom over the actual ~100%), and the Claude review. The coverage floor is
deliberately not set at 100%: it exists to fail a PR that adds an untested code
path, not to make anyone chase the last line.

**Why at this stage**

Generating tests with the code, under review, is what kept coverage from being a
number chased at the end. By Phase 6 the "test suite" already existed as a
byproduct of Phases 2–5; this phase's work was consolidation — an error-path
sweep (`tests/test_edge_cases.py`), the coverage gate, and writing down the
strategy so it's repeatable rather than tacit.

**What operationalizing this at team scale looks like**

- **Tests ship in the implementation PR, always** — a PR that adds a code path
  and no test fails the coverage gate. Reviewers don't approve "tests to
  follow".
- **Review findings are converted to tests, not just fixed.** A one-line "also
  add a regression test for this" norm on every review comment that describes a
  wrong behaviour.
- A **shared test-strategy doc** per repo naming the kinds of test in play, so a
  reviewer can ask "where's the boundary test / the characterisation test" by
  name.
- The AI **generates the first draft of the suite**; humans review it for the
  tests that matter (boundary, fairness, error paths) and delete the ones that
  only restate the implementation.
- Coverage floors set with headroom and revisited rarely — a floor that tracks
  the exact current number just generates churn.

---

## Phase 7 — Interface

**AI-native practice used**

Built the Streamlit dashboard (`app.py`) with the rendering logic factored into a
pure, importable module (`retention_risk/app_data.py`) that has no Streamlit
import. That split let Claude Code generate a real test for the UI: a headless
render via `streamlit.testing.v1.AppTest` that runs the whole page, switches
reports, and asserts no exception — plus unit tests on the view-assembly
functions. The UI is under the same coverage gate as the rest of the package.

The interface deliberately encodes the discovery brief's constraints:
- **Leads with the narrative, not the number.** Tier badge, plain-language
  summary, drivers, suggested step; the underlying probability is one line in a
  "how to read this" expander.
- **The HRBP view surfaces the fairness finding at the point of use** —
  per-protected-group flag rates *with group sizes*, and the four-fifths
  failures shown as an explicit review-required banner, exactly as
  `docs/fairness-audit.md` §6 requires.
- **Narration degrades gracefully in the product**: `get_narrator()` gives the
  template when no key is set, so the live link works with zero configuration.

**Why at this stage**

Deferring the UI until the pipeline was validated (an architecture principle in
`CLAUDE.md`) meant the dashboard is a thin presentation layer over already-tested
functions — `manager_card`, `hrbp_view` — rather than a place where logic
accretes. The AppTest render is cheap insurance that the wiring holds as those
functions change.

**What operationalizing this at team scale looks like**

- **UI logic lives in a Streamlit-free module**; the `.py` entrypoint is only
  layout. This is what makes a dashboard testable at all.
- **A headless render test per app** in CI — it catches API misuse and broken
  wiring that a human would only find by clicking around.
- Manager-facing surfaces for a model get a **framing review** (does it read as a
  verdict or a prompt?) alongside the code review — a checklist item, not a
  matter of taste.
- Fairness results are **rendered in the product**, not just filed in a repo, so
  the people using the tool see the caveats when they act.

---

## Phase 8 — Deployment & Documentation

**AI-native practice used**

Used Claude Code to produce the deployment surface — `README.md` as the repo's
front door, `DEPLOY.md` as an executable runbook, `runtime.txt` and a
version-bounded `requirements.txt` — and to keep the "CD" honest: the app is
never built into a binary artifact, it trains on cold start, and
`tests/test_app.py` renders the whole page headless in CI, so **a green `main` is
the deploy gate**. The one step Claude Code cannot take — the one-time
connect-and-deploy click in the Streamlit dashboard — is written down as two
minutes of instructions rather than left implicit.

The docs were written *as each phase happened* (`discovery-brief`, `data-framing`,
`model-card`, `fairness-audit`, `explainability`, `test-strategy`), each with a
regeneration command or a drift test so it can't quietly go stale. This final
phase only had to assemble the index and the runbook.

**Why at this stage**

Deferring the deploy runbook to the end would have meant reconstructing decisions
("why no committed model artifact?", "why does the app work without a key?")
from memory. Written now, with the code fresh, it's accurate. And gating deploy
on the same CI that gates every PR means "it works on `main`" and "it works
deployed" are the same claim.

**What operationalizing this at team scale looks like**

- **Deploy is a runbook in the repo**, versioned with the code, with the
  non-automatable steps explicitly called out — not tribal knowledge.
- The **build artifact is reproducible from source** (here: trained on boot) or
  built by CI, never hand-uploaded.
- A **headless render / smoke test in CI is the deploy gate** — "green main
  deploys" only holds if main is actually exercised.
- Docs carry a **regeneration command or a drift test**; a doc that can't be
  re-derived or checked is a doc that will lie.

---

## Closing — this repo as a template for AI-native delivery

The eight phases above are one project, but the shape is meant to transfer. What
made each stage "AI-native" was not that an AI wrote code — it was that a
judgement that usually stays tacit got **turned into an artifact a machine can
check**:

| Stage | The tacit thing | The artifact that enforces it |
|---|---|---|
| Discovery | "we know who this is for" | a structured brief, reviewed as a PR |
| Framing | "we won't use protected attributes" | `schema.py` + a leakage test in CI |
| Build | "the model is good enough" | the brief's thresholds as `passes_brief` checks |
| Fairness | "we'll audit it later" | `audit_fairness()` + characterisation tests |
| Explanation | "the LLM just narrates" | four boundary tests that fail if it doesn't |
| Test/review | "we have tests" | a coverage gate + review-finding-becomes-test |
| Interface | "it reads as a prompt, not a verdict" | a framing review + a headless render test |
| Deploy | "it works" | one CI gate for PRs and deploys alike |

**To run this across an org:** ship these as a template repo — the three CI
workflows, a `schema.py` skeleton, a `passes_brief`-style eval object, a
fairness-audit library, the boundary-test pattern, and a docs folder where every
file has a drift test. A new project then inherits the gates on day one, and a
reviewer (AI or human) can ask for each artifact *by name*. The AI generates the
first draft of all of it; people spend their review time on the artifacts that
carry real risk — the exclusion policy, the fairness method, the LLM boundary —
instead of on formatting and boilerplate.

That is the answer to "embed AI across the entire development lifecycle": not a
tool bolted onto one stage, but a habit of making each stage's judgement
executable, with the AI doing the drafting and the humans doing the deciding.
