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
