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
