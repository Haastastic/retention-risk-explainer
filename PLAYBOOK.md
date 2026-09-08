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
