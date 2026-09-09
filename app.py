"""Retention Risk Explainer — manager-facing Streamlit dashboard.

Run locally:  streamlit run app.py
Deploy:       see DEPLOY.md (Streamlit Community Cloud)

The model makes the prediction; this page only helps a manager act on it. The
narrative layer degrades to a deterministic template when no ANTHROPIC_API_KEY is
configured, so the live link always works.
"""

from __future__ import annotations

import os

import streamlit as st

from retention_risk.app_data import AppBundle, build_bundle, hrbp_view, manager_card
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


def _tier_badge(tier: str) -> None:
    st.markdown(
        f"<span style='background:{_TIER_COLOR[tier]};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600'>{tier} risk</span>",
        unsafe_allow_html=True,
    )


def _has_api_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("ANTHROPIC_API_KEY", ""))
    except Exception:  # noqa: BLE001 - no secrets.toml at all
        return False


def _narration_source_note() -> None:
    if _has_api_key():
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

    narrator = get_narrator()
    card = manager_card(bundle, row_by_text[pick], narrator=narrator)

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
        chart = card.driver_chart_frame().set_index("feature")["contribution_pp"]
        st.markdown("**Contribution to the estimate** (percentage points)")
        st.bar_chart(chart, horizontal=True, color="#c0392b")

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
    view = hrbp_view(bundle)

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

    flags = view.disparity_flags()
    if flags:
        st.warning(
            "**Disparate-impact review required.** The following groups fall outside "
            "the four-fifths rule on the High flag. This is a documented, accepted "
            "finding for v1 (the disparity tracks real cohort attrition differences "
            "in this dataset) — but the flagged lists should be checked for cohort "
            "concentration before a manager acts on them:\n\n" + "\n".join(f"- {f}" for f in flags)
        )
    else:
        st.success("All protected groups within the four-fifths rule on the High flag.")


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
