"""Turn a SHAP :class:`Explanation` into a manager-facing narrative.

**Layer boundary.** This module is the only place the LLM is used, and it is
downstream of the risk decision. It receives an :class:`Explanation` whose
``risk_score`` / ``risk_tier`` are already set by the model, describes the
drivers in plain language, and suggests a conversation. It never calls the
model, never runs SHAP, and :class:`Narrative` has no score field — there is
nothing for the LLM to feed back into the decision.

Two implementations:

* :class:`TemplateNarrator` — deterministic, no network. Always available.
* :class:`ClaudeNarrator` — calls the Claude API for a more natural narrative.

:func:`get_narrator` returns Claude when ``ANTHROPIC_API_KEY`` is set (loaded
from ``.env`` if present), otherwise the template. CI always gets the template.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Protocol

from dotenv import load_dotenv

from retention_risk.explain import Explanation

load_dotenv()

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

NarrativeSource = Literal["template", "claude"]

# Plain-language phrasing for each driver, by direction. Keys are model features.
_DRIVER_PHRASES: dict[str, dict[str, str]] = {
    "enps": {
        "increases": "recent engagement-survey sentiment is low",
        "decreases": "engagement-survey sentiment is healthy",
    },
    "engagement_score": {
        "increases": "overall engagement has slipped",
        "decreases": "overall engagement is solid",
    },
    "workload_strain": {
        "increases": "workload and hours look heavy",
        "decreases": "workload looks manageable",
    },
    "manager_relationship": {
        "increases": "the manager relationship may need attention",
        "decreases": "the manager relationship looks strong",
    },
    "growth_opportunity": {
        "increases": "growth and development opportunities look thin",
        "decreases": "there is a clear growth path",
    },
    "recognition": {
        "increases": "recognition has been light lately",
        "decreases": "recognition has been consistent",
    },
    "comp_ratio": {
        "increases": "pay sits below the midpoint for the level",
        "decreases": "pay is competitive for the level",
    },
    "OverTime": {
        "increases": "sustained overtime",
        "decreases": "little overtime",
    },
    "YearsAtCompany": {
        "increases": "early tenure, when attrition risk runs higher",
        "decreases": "established tenure",
    },
    "StockOptionLevel": {
        "increases": "limited equity tie-in",
        "decreases": "meaningful equity tie-in",
    },
    "MonthlyIncome": {
        "increases": "compensation may be a sticking point",
        "decreases": "compensation looks solid",
    },
    "DistanceFromHome": {
        "increases": "a long commute",
        "decreases": "a short commute",
    },
    "JobSatisfaction": {
        "increases": "reported job satisfaction is low",
        "decreases": "reported job satisfaction is good",
    },
}

_ACTION_BY_DRIVER: dict[str, str] = {
    "enps": (
        "Ask open questions about what's felt frustrating lately and what "
        "would make the role better."
    ),
    "engagement_score": "Check in on what's energising vs. draining in the current work.",
    "workload_strain": (
        "Review the current workload together and see what can be reprioritised or shed."
    ),
    "manager_relationship": (
        "Make space for a candid 1:1 about how the working relationship is going."
    ),
    "growth_opportunity": "Talk through career goals and concrete next steps or stretch work.",
    "recognition": "Recognise recent contributions specifically, and ask if they feel seen.",
    "comp_ratio": "Flag for a compensation review and be honest about what's possible and when.",
    "MonthlyIncome": "Flag for a compensation review and set expectations on timing.",
    "OverTime": "Look at whether the overtime is sustainable and what's driving it.",
    "DistanceFromHome": "Explore whether flexibility on location or hours would help.",
}

_GENERIC_ACTION = "Have an informal check-in to understand how they're feeling about the role."


@dataclass
class Narrative:
    """Manager-facing output. Deliberately has no risk score — see module docstring."""

    summary: str
    drivers: list[str]
    suggested_action: str
    source: NarrativeSource


class Narrator(Protocol):
    def narrate(
        self, explanation: Explanation, *, employee_ref: str | None = None
    ) -> Narrative: ...


def _tier_sentence(tier: str, ref: str) -> str:
    if tier == "High":
        return f"{ref} is showing several signals associated with leaving in the next 6-12 months."
    if tier == "Medium":
        return f"{ref} is showing some early signals worth a check-in."
    return f"{ref} is not currently showing elevated retention risk."


def _phrase(feature: str, direction: str) -> str:
    entry = _DRIVER_PHRASES.get(feature)
    if entry:
        return entry[direction]
    verb = "raising" if direction == "increases" else "easing"
    return f"{feature.replace('_', ' ')} is {verb} the estimate"


class TemplateNarrator:
    """Deterministic narrative — no LLM, no network."""

    source: NarrativeSource = "template"

    def narrate(self, explanation: Explanation, *, employee_ref: str | None = None) -> Narrative:
        ref = employee_ref or "This person"
        raising = [c for c in explanation.top_contributions if c.direction == "increases"]
        easing = [c for c in explanation.top_contributions if c.direction == "decreases"]
        elevated = explanation.risk_tier != "Low"

        if elevated:
            drivers = [_phrase(c.feature, "increases").capitalize() for c in raising[:3]]
            if easing:
                drivers.append(f"On the other side, {_phrase(easing[0].feature, 'decreases')}.")
        else:
            # Low tier: lead with what's keeping risk down; note any lone risk
            # factor as a watch-item, not a headline.
            drivers = [_phrase(c.feature, "decreases").capitalize() for c in easing[:3]]
            if raising:
                drivers.append(
                    f"Worth keeping an eye on: {_phrase(raising[0].feature, 'increases')}."
                )

        top_feature = raising[0].feature if raising else None
        # For a Low-tier person, don't headline a risk driver — the tool isn't
        # flagging them. Point at what's keeping risk down instead.
        action = (
            _ACTION_BY_DRIVER.get(top_feature, _GENERIC_ACTION) if elevated else _GENERIC_ACTION
        )

        summary = _tier_sentence(explanation.risk_tier, ref)
        if raising and elevated:
            summary += " The biggest contributor is that " + _phrase(top_feature, "increases") + "."
        elif easing and not elevated:
            summary += " What's helping most: " + _phrase(easing[0].feature, "decreases") + "."

        return Narrative(
            summary=summary, drivers=drivers, suggested_action=action, source=self.source
        )


_CLAUDE_SYSTEM = (
    "You help a people-manager prepare for a retention conversation. You are given "
    "a risk tier and the main factors behind it, already computed. Do NOT invent, "
    "restate, or estimate any probability or score. Write for a non-technical "
    "manager. Frame everything as a conversation prompt, never a prediction of what "
    "the employee will do, and never a recommendation about pay, promotion, or "
    "performance ratings. Respond as JSON with keys: summary (2-3 sentences), "
    "drivers (array of 2-4 short plain-language bullets), suggested_action (one "
    "concrete, humane next step)."
)


class ClaudeNarrator:
    """Narrative via the Claude API. The score is never sent and never requested."""

    source: NarrativeSource = "claude"

    def __init__(self, *, model: str = DEFAULT_CLAUDE_MODEL, client: object | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic()

    def narrate(self, explanation: Explanation, *, employee_ref: str | None = None) -> Narrative:
        """Narrate via Claude, degrading to the template on any API or parse failure.

        A network error, rate limit, or a reply that isn't the expected JSON must
        not take down the manager-facing narrative — the caller still gets a
        usable :class:`Narrative`, just with ``source == "template"``.
        """
        try:
            return self._narrate_via_claude(explanation, employee_ref=employee_ref)
        except Exception:  # noqa: BLE001 - any API/parse failure degrades gracefully
            return TemplateNarrator().narrate(explanation, employee_ref=employee_ref)

    def _narrate_via_claude(
        self, explanation: Explanation, *, employee_ref: str | None = None
    ) -> Narrative:
        import json

        ref = employee_ref or "the employee"
        user = (
            f"Manager is reviewing {ref}.\n\n"
            f"{explanation.as_prompt_facts()}\n\n"
            "Write the JSON described in the system prompt."
        )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=600,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(block, "text", "") for block in message.content).strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1].removeprefix("json").strip()
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("drivers"), list):
            raise ValueError("narrative reply is not the expected JSON shape")
        return Narrative(
            summary=str(data["summary"]).strip(),
            drivers=[str(d).strip() for d in data["drivers"]],
            suggested_action=str(data["suggested_action"]).strip(),
            source=self.source,
        )


def get_narrator(*, prefer_claude: bool | None = None) -> Narrator:
    """Return a Claude narrator when an API key is available, else the template."""
    use_claude = (
        bool(os.environ.get("ANTHROPIC_API_KEY")) if prefer_claude is None else prefer_claude
    )
    if use_claude:
        model = os.environ.get("RETENTION_RISK_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        try:
            return ClaudeNarrator(model=model)
        except Exception:  # noqa: BLE001  (any import/config problem -> safe fallback)
            return TemplateNarrator()
    return TemplateNarrator()
