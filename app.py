"""Retention Risk Explainer — manager-facing Streamlit dashboard.

Run locally:  streamlit run app.py
Deploy:       see DEPLOY.md (Streamlit Community Cloud)

The model makes the prediction; this page only helps a manager act on it. The
narrative layer degrades to a deterministic template when no ANTHROPIC_API_KEY is
configured, so the live link always works.
"""

from __future__ import annotations

import os

import altair as alt
import streamlit as st

from retention_risk.app_data import (
    AppBundle,
    HrbpView,
    ManagerCard,
    build_bundle,
    hrbp_view,
    manager_card,
)
from retention_risk.narrative import get_narrator

st.set_page_config(page_title="Retention Risk Explainer", page_icon="🧭", layout="wide")

_TIER_COLOR = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}
_TIER_BLURB = {
    "High": "Worth a retention conversation this cycle.",
    "Medium": "Keep an eye on it; a light check-in is enough.",
    "Low": "No action indicated.",
}


@st.cache_resource(show_spinner="Training the model…")
def _bundle() -> AppBundle:
    return build_bundle()


@st.cache_resource(show_spinner="Explaining…")
def _card(_bundle: AppBundle, row: int, live: bool) -> ManagerCard:
    # _bundle is unhashable and stable for the session -> underscore-skip it;
    # `row` and `live` (narration mode) are the real cache key. Without this,
    # a fresh SHAP explainer is built on every unrelated widget rerun.
    return manager_card(_bundle, row, narrator=get_narrator())


@st.cache_resource(show_spinner="Auditing…")
def _hrbp(_bundle: AppBundle) -> HrbpView:
    # audit_fairness scores the whole population; do it once, not per rerun.
    return hrbp_view(_bundle)


def _tier_badge(tier: str) -> None:
    st.markdown(
        f"<span style='background:{_TIER_COLOR[tier]};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600'>{tier} risk</span>",
        unsafe_allow_html=True,
    )


def _ensure_api_key_in_env() -> bool:
    """Bridge a Streamlit-secrets key into os.environ so ``get_narrator`` sees it.

    ``retention_risk.narrative`` deliberately doesn't import streamlit, so it only
    reads ``os.environ``. If the deployment set ANTHROPIC_API_KEY solely via
    ``.streamlit/secrets.toml``, copy it across before the narrator is built —
    otherwise the sidebar would claim "live" while narration stays templated.
    Returns True if a key is available by either route.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        secret = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:  # noqa: BLE001 - no secrets.toml at all
        secret = ""
    if secret:
        os.environ["ANTHROPIC_API_KEY"] = str(secret)
        return True
    return False


def _narration_source_note() -> None:
    if _ensure_api_key_in_env():
        st.sidebar.success("Narratives: live (Claude)")
    else:
        st.sidebar.info(
            "Narratives: templated fallback\n\nSet `ANTHROPIC_API_KEY` for live narration."
        )


def manager_tab(bundle: AppBundle) -> None:
    order = {"High": 0, "Medium": 1, "Low": 2}
    roster = bundle.roster()
    roster = roster.assign(_o=roster["tier"].map(order)).sort_values(
        ["_o", "score"], ascending=[True, False]
    )
    # Options are display strings (keeps the AppTest harness simple); map back to row.
    row_by_text = {f"{r.label}  ·  {r.tier}": int(r.row) for r in roster.itertuples(index=False)}

    left, right = st.columns([1, 2])
    with left:
        st.caption("Your team, most-flagged first")
        pick = st.selectbox("Report", list(row_by_text), label_visibility="collapsed")

    card = _card(bundle, row_by_text[pick], _ensure_api_key_in_env())

    with right:
        _tier_badge(card.tier)
        st.caption(_TIER_BLURB[card.tier])

    st.subheader(card.label)
    st.write(card.narrative.summary)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**What's behind it**")
        for d in card.narrative.drivers:
            st.markdown(f"- {d}")
        st.info(f"**Suggested next step** — {card.narrative.suggested_action}")
    with c2:
        st.markdown("**Contribution to the estimate** (percentage points)")
        frame = card.driver_chart_frame()
        chart = (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X("contribution_pp:Q", title=None),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["increases", "decreases"],
                        range=["#c0392b", "#1e8449"],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["feature", "contribution_pp", "direction"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    with st.expander("How to read this"):
        st.markdown(
            "- The tier is the model's output; the narrative and suggested step are "
            "phrased for a conversation, not a verdict.\n"
            "- Bars are SHAP contributions in probability points — they sum "
            "(with a baseline) to the underlying estimate.\n"
            "- This score never feeds pay, promotion, or performance. It is a prompt "
            "to check in, nothing more."
        )
        st.caption(f"Underlying estimate: {card.score:.0%}  ·  model: {bundle.model.kind}")


def hrbp_tab(bundle: AppBundle) -> None:
    view = _hrbp(bundle)

    st.markdown("**Risk tier distribution across the population**")
    st.bar_chart(view.tier_counts)

    st.markdown("**High-risk flag rate by protected group**")
    st.caption(
        "Group size is shown alongside every rate — a small group's rate is noisy, "
        "not a finding. Full methodology in docs/fairness-audit.md."
    )
    tbl = view.flag_rate_table()
    st.dataframe(
        tbl.rename(
            columns={
                "high_flag_rate": "High flag rate",
                "mean_score": "Mean score",
                "sufficient_n": "n ≥ 30",
                "n": "Group size",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    _render_disparity_banner(view.fairness_failures())


# Attributes whose four-fifths failure is analysed and accepted for v1 in
# docs/fairness-audit.md. A failure on anything else is not covered by that
# analysis and must be treated as new.
_DOCUMENTED_DISPARITIES = {"MaritalStatus", "AgeBand"}


def _render_disparity_banner(failures: list[tuple[str, float]]) -> None:
    if not failures:
        st.success("All protected groups within the four-fifths rule on the High flag.")
        return

    known = [f for f in failures if f[0] in _DOCUMENTED_DISPARITIES]
    novel = [f for f in failures if f[0] not in _DOCUMENTED_DISPARITIES]

    if known:
        st.warning(
            "**Disparate-impact — documented finding.** These attributes fall "
            "outside the four-fifths rule on the High flag; `docs/fairness-audit.md` "
            "shows the disparity tracks real cohort attrition differences in this "
            "dataset. Still check flagged lists for cohort concentration before "
            "acting:\n\n" + "\n".join(f"- {a}: DI ratio {r:.2f}" for a, r in known)
        )
    if novel:
        st.error(
            "**Disparate-impact — not previously analysed.** These attributes fail "
            "the four-fifths rule and are *not* covered by the v1 fairness audit. "
            "Investigate before the tool is used for this population:\n\n"
            + "\n".join(f"- {a}: DI ratio {r:.2f}" for a, r in novel)
        )


def main() -> None:
    st.title("🧭 Retention Risk Explainer")
    st.caption(
        "An early, explainable read on which of your people are at elevated risk of "
        "leaving in the next 6–12 months — and how to have a useful conversation about it."
    )
    _narration_source_note()
    st.sidebar.caption(
        "Public IBM HR Analytics data + a synthetic engagement-survey overlay. "
        "Portfolio demo — not a basis for real personnel decisions."
    )

    bundle = _bundle()
    manager, hrbp = st.tabs(["Manager view", "HR business partner view"])
    with manager:
        manager_tab(bundle)
    with hrbp:
        hrbp_tab(bundle)


if __name__ == "__main__":
    main()
