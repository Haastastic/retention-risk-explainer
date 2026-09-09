"""Dashboard: app_data assembly + a headless render of app.py via AppTest."""

from pathlib import Path

import numpy as np
import pytest

from retention_risk.app_data import (
    build_bundle,
    hrbp_view,
    manager_card,
    report_label,
)

APP_PY = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="session")
def bundle():
    return build_bundle(seed=42)


def test_report_label_is_stable_and_1_indexed():
    assert report_label(0) == "Report 01"
    assert report_label(463) == "Report 464"


def test_roster_covers_every_row_with_a_tier_and_score(bundle):
    roster = bundle.roster()
    assert len(roster) == len(bundle.dataset.X)
    assert set(roster["tier"]) <= {"High", "Medium", "Low"}
    assert roster["score"].between(0, 1).all()
    assert list(roster["row"]) == list(range(len(roster)))


def test_manager_card_is_manager_facing(bundle):
    hi = int(np.argmax(bundle.scores.to_numpy()))
    card = manager_card(bundle, hi)
    assert card.label == report_label(hi)
    assert card.tier == "High"
    # the narrative carries the message; no raw employee id anywhere in it
    assert "EmployeeNumber" not in card.narrative.summary
    assert card.narrative.suggested_action
    chart = card.driver_chart_frame()
    assert list(chart.columns) == ["feature", "contribution_pp", "direction"]
    assert len(chart) == 5


def test_manager_card_row_bounds(bundle):
    with pytest.raises(IndexError):
        manager_card(bundle, len(bundle.dataset.X))


def test_hrbp_view_surfaces_group_sizes_next_to_rates(bundle):
    view = hrbp_view(bundle)
    tbl = view.flag_rate_table()
    assert {"attribute", "group", "n", "high_flag_rate", "mean_score"} <= set(tbl.columns)
    assert (tbl["n"] > 0).all()
    # the known v1 finding is surfaced as (attribute, di_ratio), not hidden
    failures = dict(view.fairness_failures())
    assert "MaritalStatus" in failures and failures["MaritalStatus"] < 0.8
    assert "AgeBand" in failures and failures["AgeBand"] < 0.8


def test_disparity_banner_separates_documented_from_novel_failures():
    from app import _DOCUMENTED_DISPARITIES

    assert sorted(_DOCUMENTED_DISPARITIES) == ["AgeBand", "MaritalStatus"]


def test_app_renders_without_exception():
    """Headless render of the whole page — catches wiring / API misuse."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PY, default_timeout=90)
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("Retention Risk Explainer")
    assert len(at.tabs) == 2


def test_app_switching_reports_does_not_error():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PY, default_timeout=90)
    at.run()
    assert not at.exception
    # pick a different report from the selectbox and re-run
    other = at.selectbox[0].options[5]
    at.selectbox[0].set_value(other).run()
    assert not at.exception
