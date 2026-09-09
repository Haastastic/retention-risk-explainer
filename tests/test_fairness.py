"""Fairness audit: DI arithmetic, thin-slice handling, and characterisation.

The characterisation tests pin the *current* audit result. If the model or data
changes such that these move, the test fails on purpose — go update
docs/fairness-audit.md and re-review, don't just bump the numbers.
"""

import numpy as np
import pandas as pd
import pytest

from retention_risk.fairness import (
    AUDIT_ATTRIBUTES,
    audit_fairness,
)
from retention_risk.training import run_training


@pytest.fixture(scope="session")
def audit(dataset):
    result = run_training(split_seed=42)
    return audit_fairness(result.best, dataset.X, dataset.protected)


# --- structure -------------------------------------------------------------
def test_report_covers_every_audit_attribute(audit):
    assert {a.attribute for a in audit.attributes} == set(AUDIT_ATTRIBUTES)
    assert audit.n_rows == 1470
    assert audit.tier == "High"


def test_group_flag_rates_are_probabilities(audit):
    for attr in audit.attributes:
        for g in attr.groups:
            assert 0.0 <= g.flag_rate <= 1.0
            assert 0.0 <= g.mean_score <= 1.0
            assert g.n > 0


def test_reference_group_is_the_lowest_flag_rate(audit):
    for attr in audit.attributes:
        if attr.reference_group is None:
            continue
        sufficient = [g for g in attr.groups if g.sufficient]
        ref = next(g for g in attr.groups if g.group == attr.reference_group)
        assert ref.flag_rate == min(g.flag_rate for g in sufficient)


def test_attribute_di_ratio_matches_min_over_max(audit):
    for attr in audit.attributes:
        sufficient = [g for g in attr.groups if g.sufficient]
        if len(sufficient) < 2:
            assert np.isnan(attr.di_ratio)
            continue
        rates = [g.flag_rate for g in sufficient]
        assert attr.di_ratio == pytest.approx(min(rates) / max(rates), rel=1e-6)
        assert attr.passes == (attr.di_ratio >= 0.8)


# --- thin slices ---------------------------------------------------------
def test_small_groups_are_excluded_from_the_decision(dataset):
    result = run_training(split_seed=42)
    # min_group_size above every AgeBand slice -> nothing decidable for it
    report = audit_fairness(result.best, dataset.X, dataset.protected, min_group_size=100_000)
    for attr in report.attributes:
        assert attr.passes is None
        assert attr.insufficient_groups
    assert report.passes is False  # no decidable attribute => not a pass


def test_x_and_protected_length_mismatch_raises(dataset):
    result = run_training(split_seed=42)
    with pytest.raises(ValueError, match="same number of rows"):
        audit_fairness(result.best, dataset.X, dataset.protected.head(10))


def test_unknown_protected_column_raises(dataset):
    result = run_training(split_seed=42)
    with pytest.raises(KeyError):
        audit_fairness(result.best, dataset.X, dataset.protected, attributes=("Ethnicity",))


# --- characterisation (see docs/fairness-audit.md §3) --------------------
# These pin the *qualitative* finding, not exact ratios: XGBoost probabilities
# differ slightly across library builds (Gender DI is ~0.87 locally, ~0.93 on the
# CI Python), which moves the decimals but not the pass/fail picture.
def test_gender_passes_four_fifths(audit):
    gender = next(a for a in audit.attributes if a.attribute == "Gender")
    assert gender.passes is True
    assert gender.di_ratio >= 0.80
    assert gender.di_ratio == pytest.approx(0.90, abs=0.12)  # ~0.85-0.95 across builds


def test_marital_status_and_age_fail_by_a_wide_margin(audit):
    for name in ("MaritalStatus", "AgeBand"):
        attr = next(a for a in audit.attributes if a.attribute == name)
        assert attr.passes is False
        assert attr.di_ratio < 0.60  # both sit near 0.35-0.45, nowhere near 0.80
    assert audit.passes is False


def test_flag_di_tracks_true_attrition_base_rate_ratio(dataset, audit):
    """The disparity is in the data: flag DI ~= true base-rate ratio per attribute."""
    y = dataset.y.to_numpy()
    prot = dataset.protected.reset_index(drop=True)
    for name in ("MaritalStatus", "AgeBand"):
        rates = pd.Series(y).groupby(prot[name].to_numpy()).mean()
        base_rate_ratio = rates.min() / rates.max()
        attr = next(a for a in audit.attributes if a.attribute == name)
        assert attr.di_ratio == pytest.approx(base_rate_ratio, abs=0.15)
