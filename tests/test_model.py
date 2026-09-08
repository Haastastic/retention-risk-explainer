"""RiskModel: fitting, scoring, tiers, persistence, and the decision-layer guardrails."""

import numpy as np
import pytest

from retention_risk.model import RiskModel, TierThresholds, train_baseline, train_xgboost


@pytest.fixture(scope="session")
def fitted_xgb(dataset):
    return train_xgboost(dataset.X, dataset.y, seed=1)


def test_threshold_assignment_is_monotone():
    th = TierThresholds(high=0.7, medium=0.4)
    tiers = th.assign(np.array([0.1, 0.4, 0.55, 0.7, 0.99]))
    assert list(tiers) == ["Low", "Medium", "Medium", "High", "High"]


def test_predict_proba_is_probability(fitted_xgb, dataset):
    proba = fitted_xgb.predict_proba(dataset.X)
    assert proba.shape == (len(dataset.X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tiers_partition_the_rows(fitted_xgb, dataset):
    tiers = fitted_xgb.predict_tier(dataset.X)
    assert sorted(np.unique(tiers).tolist()) == ["High", "Low", "Medium"]


def test_high_tier_is_the_smallest_and_highest_scoring(fitted_xgb, dataset):
    proba = fitted_xgb.predict_proba(dataset.X)
    tiers = fitted_xgb.assign_tier(proba)
    assert proba[tiers == "High"].min() >= proba[tiers == "Medium"].max()
    assert (tiers == "High").sum() < (tiers == "Low").sum()


def test_baseline_and_xgboost_both_learn_something(dataset):
    # AUC on train well above chance for both kinds
    from sklearn.metrics import roc_auc_score

    for train in (train_baseline, train_xgboost):
        m = train(dataset.X, dataset.y, seed=2)
        assert roc_auc_score(dataset.y, m.predict_proba(dataset.X)) > 0.7


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RiskModel().predict_proba(None)


def test_rejects_withheld_columns(dataset):
    bad = dataset.X.copy()
    bad["Age"] = 30
    with pytest.raises(ValueError, match="withheld"):
        train_baseline(bad, dataset.y)


def test_rejects_missing_feature_columns(dataset):
    with pytest.raises(ValueError, match="missing"):
        train_baseline(dataset.X.drop(columns=["engagement_score"]), dataset.y)


def test_invalid_quantiles_raise():
    with pytest.raises(ValueError):
        RiskModel(high_quantile=0.5, medium_quantile=0.6)


def test_save_load_roundtrip(fitted_xgb, dataset, tmp_path):
    path = tmp_path / "m.joblib"
    fitted_xgb.save(path)
    reloaded = RiskModel.load(path)
    np.testing.assert_allclose(
        reloaded.predict_proba(dataset.X), fitted_xgb.predict_proba(dataset.X)
    )
