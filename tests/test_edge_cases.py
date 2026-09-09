"""Guardrails and error paths that the happy-path phase tests don't exercise.

Consolidated in Phase 6 so the coverage gate (>=95%) has something to hold onto
and the failure modes are documented as tests, not just as raises.
"""

import numpy as np
import pandas as pd
import pytest

from retention_risk import schema
from retention_risk.data import Dataset
from retention_risk.explain import Explanation, FeatureContribution, _onehot_source
from retention_risk.fairness import AttributeResult, FairnessReport, GroupResult
from retention_risk.model import RiskModel, train_baseline
from retention_risk.narrative import ClaudeNarrator, TemplateNarrator


# --- Dataset leakage guard ------------------------------------------------
def test_dataset_rejects_a_withheld_column_in_X(dataset):
    bad_X = dataset.X.copy()
    bad_X["Age"] = 30
    with pytest.raises(AssertionError, match="withheld columns leaked"):
        Dataset(
            X=bad_X,
            y=dataset.y,
            protected=dataset.protected,
            months_to_departure=dataset.months_to_departure,
        )


# --- RiskModel.load type check -----------------------------------------
def test_load_rejects_a_non_riskmodel_pickle(tmp_path):
    import joblib

    p = tmp_path / "not_a_model.joblib"
    joblib.dump({"hello": "world"}, p)
    with pytest.raises(TypeError, match="not a RiskModel"):
        RiskModel.load(p)


# --- explain internals -----------------------------------------------
def test_onehot_source_passes_through_an_unknown_column():
    # not a model feature and not a "<categorical>_" dummy -> returned as-is
    assert _onehot_source("totally_unknown_col") == "totally_unknown_col"


def test_explain_requires_a_background_sample(dataset):
    m = train_baseline(dataset.X, dataset.y, seed=1)
    m.background_ = None  # simulate a model pickled before the field existed
    from retention_risk.explain import explain_prediction

    with pytest.raises(RuntimeError, match="no background sample"):
        explain_prediction(m, dataset.X.iloc[[0]])


# --- narrative "no dominant factor" lines ----------------------------
def _explanation(tier: str, contributions: list[FeatureContribution]) -> Explanation:
    return Explanation(
        risk_score=0.5,
        risk_tier=tier,
        base_value=0.2,
        top_contributions=contributions,
        all_contributions={c.feature: c.shap for c in contributions},
    )


def test_elevated_tier_all_decreasing_uses_mixed_signal_line():
    exp = _explanation(
        "High",
        [FeatureContribution("engagement_score", 70.0, -0.3, "decreases")],
    )
    n = TemplateNarrator().narrate(exp)
    assert n.drivers == [TemplateNarrator._MIXED_SIGNAL]
    assert "biggest contributor" not in n.summary


def test_low_tier_all_increasing_uses_mixed_signal_line():
    exp = _explanation(
        "Low",
        [FeatureContribution("OverTime", "Yes", 0.4, "increases")],
    )
    n = TemplateNarrator().narrate(exp)
    assert n.drivers == [TemplateNarrator._MIXED_SIGNAL]


def test_elevated_tier_with_mixed_contributions_contrasts_the_two_sides():
    exp = _explanation(
        "High",
        [
            FeatureContribution("workload_strain", 85.0, 0.6, "increases"),
            FeatureContribution("engagement_score", 40.0, 0.2, "increases"),
            FeatureContribution("comp_ratio", 1.3, -0.15, "decreases"),
        ],
    )
    n = TemplateNarrator().narrate(exp)
    assert any(d.startswith("On the other side") for d in n.drivers)
    assert "biggest contributor" in n.summary


# --- ClaudeNarrator real-client construction --------------------------
def test_claude_narrator_builds_a_real_client_when_none_supplied(monkeypatch):
    created = {}

    class _FakeAnthropicModule:
        class Anthropic:
            def __init__(self):
                created["yes"] = True

    monkeypatch.setitem(__import__("sys").modules, "anthropic", _FakeAnthropicModule)
    ClaudeNarrator()
    assert created == {"yes": True}


# --- FairnessReport.passes true branch ------------------------------
def test_fairness_report_passes_when_every_decided_attribute_passes():
    groups = [
        GroupResult("Gender", "F", 100, 0.10, 0.2, float("nan"), True),
        GroupResult("Gender", "M", 100, 0.11, 0.2, 1.1, True),
    ]
    attr = AttributeResult("Gender", "F", 0.91, True, groups)
    report = FairnessReport(tier="High", min_group_size=30, n_rows=200, attributes=[attr])
    assert report.passes is True
    assert report.to_dict()["passes"] is True


def test_fairness_report_does_not_pass_with_no_decidable_attribute():
    attr = AttributeResult("Gender", None, float("nan"), None, [])
    report = FairnessReport(tier="High", min_group_size=30, n_rows=0, attributes=[attr])
    assert report.passes is False


# --- EvalReport.to_dict serialisation ------------------------------
def test_eval_report_to_dict_round_trips_the_metrics():
    from retention_risk.evaluate import EvalReport, TierStats

    report = EvalReport(
        model_kind="xgboost",
        n_test=300,
        positives_test=40,
        auc_roc=0.81,
        brier=0.12,
        cv_auc_mean=0.79,
        cv_auc_std=0.02,
        tiers=[
            TierStats("High", 30, 0.10, 0.5, 0.375),
            TierStats("Medium", 90, 0.30, 0.2, 0.45),
            TierStats("Low", 180, 0.60, 0.03, 0.175),
        ],
        checks={"auc_roc>=0.75": True},
    )
    d = report.to_dict()
    assert d["model_kind"] == "xgboost"
    assert d["auc_roc"] == 0.81
    assert set(d["tiers"]) == {"High", "Medium", "Low"}
    assert d["tiers"]["High"]["precision"] == 0.5
    assert d["passes_brief"] is True


# --- schema policy sanity (fast, no model) --------------------------
def test_withheld_columns_cover_label_and_proxies():
    w = schema.withheld_columns()
    assert {"Attrition", "left_within_horizon", "Age", "YearsSinceLastPromotion"} <= w
    assert w.isdisjoint(schema.MODEL_FEATURES)


def test_model_feature_matrix_is_all_present_in_a_built_row(dataset):
    row = dataset.X.iloc[[0]]
    assert list(row.columns) == list(schema.MODEL_FEATURES)
    assert not np.isnan(pd.to_numeric(row.select_dtypes("number").iloc[0], errors="coerce")).all()
