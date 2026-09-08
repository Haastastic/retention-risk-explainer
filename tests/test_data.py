"""Dataset assembly: shape, determinism, target definition, no leakage."""

import numpy as np

from retention_risk import schema
from retention_risk.data import build_dataset, load_raw


def test_raw_loads_without_id_or_constant_columns():
    raw = load_raw()
    assert len(raw) == 1470
    assert schema.ID_OR_CONSTANT.isdisjoint(raw.columns)
    assert "Attrition" in raw.columns


def test_feature_matrix_holds_exactly_the_policy_columns(dataset):
    assert list(dataset.X.columns) == list(schema.MODEL_FEATURES)
    assert not dataset.X.isnull().to_numpy().any()


def test_no_withheld_column_leaks_into_features(dataset):
    assert set(dataset.X.columns).isdisjoint(schema.withheld_columns())


def test_protected_frame_is_kept_for_the_audit_only(dataset):
    assert {"Age", "Gender", "MaritalStatus", "AgeBand"} <= set(dataset.protected.columns)
    assert dataset.protected.index.equals(dataset.X.index)


def test_build_is_deterministic_for_a_fixed_seed():
    a, b = build_dataset(seed=7), build_dataset(seed=7)
    assert a.X.equals(b.X)
    assert a.y.equals(b.y)


def test_seed_changes_the_engagement_overlay_but_not_the_raw_label():
    a, b = build_dataset(seed=1), build_dataset(seed=2)
    assert not a.X["enps"].equals(b.X["enps"])
    # raw attrition rate is real data, seed-independent
    assert a.meta["raw_attrition_rate"] == b.meta["raw_attrition_rate"]


def test_target_is_leavers_within_the_horizon(dataset):
    raw = load_raw()
    leaver = (raw["Attrition"] == "Yes").to_numpy()
    positives = dataset.y.to_numpy().astype(bool)
    # every positive is a real leaver; some leavers are censored out by the horizon
    assert np.all(positives <= leaver)
    assert dataset.meta["n_positive"] < leaver.sum()
    assert dataset.meta["censored_leavers"] == int(leaver.sum() - positives.sum())


def test_shorter_horizon_never_increases_positives():
    wide = build_dataset(horizon_months=24)
    narrow = build_dataset(horizon_months=6)
    assert narrow.y.sum() <= wide.y.sum()


def test_months_to_departure_is_present_for_leavers_only(dataset):
    raw = load_raw()
    leaver = (raw["Attrition"] == "Yes").to_numpy()
    has_months = dataset.months_to_departure.notna().to_numpy()
    assert np.array_equal(has_months, leaver)
    vals = dataset.months_to_departure.dropna()
    assert vals.between(1.0, 36.0).all()


def test_engineered_features_stay_in_documented_ranges(dataset):
    x = dataset.X
    for col in [
        "engagement_score",
        "manager_relationship",
        "growth_opportunity",
        "workload_strain",
        "recognition",
    ]:
        assert x[col].between(0, 100).all()
    assert x["enps"].between(-100, 100).all()
    assert (x["comp_ratio"] > 0).all()
