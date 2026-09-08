"""Evaluation report: metric wiring, tier stats, and the brief checks."""

import numpy as np
import pytest

from retention_risk.evaluate import evaluate_model
from retention_risk.model import train_baseline, train_xgboost
from retention_risk.training import run_training, split_dataset


@pytest.fixture(scope="session")
def split(dataset):
    return split_dataset(dataset, seed=3)


@pytest.fixture(scope="session")
def xgb_report(dataset, split):
    X_train, X_test, y_train, y_test = split
    model = train_xgboost(X_train, y_train, seed=3)
    return evaluate_model(model, X_test, y_test)


def test_report_shape(xgb_report):
    assert xgb_report.model_kind == "xgboost"
    assert 0.0 <= xgb_report.auc_roc <= 1.0
    assert xgb_report.n_test > 0
    assert {t.tier for t in xgb_report.tiers} == {"High", "Medium", "Low"}


def test_tier_shares_sum_to_one_and_recall_partitions(xgb_report):
    assert sum(t.share for t in xgb_report.tiers) == pytest.approx(1.0)
    assert sum(t.recall for t in xgb_report.tiers) == pytest.approx(1.0, abs=1e-9)


def test_high_tier_is_within_volume_budget(xgb_report):
    high = next(t for t in xgb_report.tiers if t.tier == "High")
    assert high.share <= 0.15


def test_checks_keys_are_the_brief_metrics(xgb_report):
    assert set(xgb_report.checks) == {
        "auc_roc>=0.75",
        "high_precision>=0.30",
        "high_share<=0.15",
        "top2_recall>=0.50",
    }


def test_perfect_scores_pass_all_checks(dataset, split):
    X_train, X_test, y_train, y_test = split

    class _Perfect:
        kind = "oracle"

        def predict_proba(self, X):
            return y_test.to_numpy(dtype=float)

        def assign_tier(self, proba):
            # every real leaver (48 / 368 ~= 13%, within the volume budget) in High
            return np.where(proba >= 0.5, "High", "Low")

    report = evaluate_model(_Perfect(), X_test, y_test)
    assert report.auc_roc == pytest.approx(1.0)
    assert report.passes_brief is True
    high = next(t for t in report.tiers if t.tier == "High")
    assert high.precision == pytest.approx(1.0)
    assert high.recall == pytest.approx(1.0)


def test_oversized_high_tier_fails_the_volume_check(dataset, split):
    X_train, X_test, y_train, y_test = split

    class _Everyone:
        kind = "oracle"

        def predict_proba(self, X):
            return np.full(len(X), 0.9)

        def assign_tier(self, proba):
            return np.full(len(proba), "High")  # 100% of rows -> blows the 15% budget

    report = evaluate_model(_Everyone(), X_test, y_test)
    assert report.checks["high_share<=0.15"] is False
    assert report.passes_brief is False


def test_cross_validation_is_reported_when_requested(dataset, split):
    X_train, X_test, y_train, y_test = split
    model = train_baseline(X_train, y_train, seed=3)
    report = evaluate_model(model, X_test, y_test, X_cv=X_train, y_cv=y_train, cv_splits=3, seed=3)
    assert not np.isnan(report.cv_auc_mean)
    assert report.cv_auc_std >= 0.0


def test_full_training_run_meets_the_brief():
    result = run_training(split_seed=42)
    assert result.xgboost_report.passes_brief
    assert result.baseline_report.passes_brief
    assert result.best.kind in {"baseline", "xgboost"}
